import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Union

import torch
import transformers
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.sparse.block_cache_sparse_dlm_patch import patch_model


eval_logger = logging.getLogger(__name__)


def _optional_number(value, cast):
    if value is None or str(value).lower() == "none":
        return None
    return cast(value)


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


@register_model("llada")
class LLaDA(LM):
    """Generation-only lm-eval adapter for LLaDA2.1."""

    def __init__(
        self,
        pretrained: str,
        batch_size: Union[int, str] = 1,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = "bfloat16",
        trust_remote_code: bool = True,
        attn_implementation: str = "sdpa",
        max_prompt_len: int = 32768,
        gen_length: int = 128,
        max_new_tokens: Optional[int] = None,
        block_length: int = 32,
        steps: int = 32,
        diffusion_steps: Optional[int] = None,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        threshold: float = 0.5,
        editing_threshold: float = 0.0,
        max_post_steps: int = 16,
        minimal_topk: int = 1,
        num_to_transfer: int = 1,
        mask_id: int = 156895,
        eos_id: int = 156892,
        sparse_dlm: bool = True,
        sparse_dlm_ratio: float = 0.5,
        sparse_dlm_top_k: int = 64,
        sparse_dlm_selection_interval: int = 4,
        sparse_dlm_dense_fallback_mask_count: int = 4,
        sparse_dlm_block_length: Optional[int] = None,
        query_sparse: bool = True,
        prefix_sparse: bool = True,
        prefix_token_budget: int = 256,
        prefix_chunk_size: int = 256,
        losa: bool = False,
        losa_active_topk: int = 5,
        show_samples: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning("Ignoring unsupported model arguments: %s", sorted(kwargs))

        batch_size = int(batch_size)
        if batch_size != 1:
            raise ValueError("LLaDA block-cache evaluation currently requires batch_size=1")

        accelerator = Accelerator(
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(weeks=52))]
        )
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            self._device = accelerator.device
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._device = torch.device(device)

        load_kwargs = {
            "trust_remote_code": trust_remote_code,
            "torch_dtype": get_dtype(dtype),
            "attn_implementation": attn_implementation,
        }
        if self._device.type != "cpu":
            load_kwargs["device_map"] = {"": str(self._device)}

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained, **load_kwargs
        ).eval()
        if self._device.type == "cpu":
            self.model.to(self._device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

        if _as_bool(sparse_dlm):
            patch_model(
                self.model,
                ratio=float(sparse_dlm_ratio),
                top_k=int(sparse_dlm_top_k),
                selection_interval=int(sparse_dlm_selection_interval),
                dense_fallback_mask_count=int(sparse_dlm_dense_fallback_mask_count),
                query_sparse=_as_bool(query_sparse),
                prefix_sparse=_as_bool(prefix_sparse),
                prefix_token_budget=int(prefix_token_budget),
                prefix_chunk_size=int(prefix_chunk_size),
                losa=_as_bool(losa),
                losa_active_topk=int(losa_active_topk),
            )
            eval_logger.info(
                "Applied block-cache SparseDLM: ratio=%s, top_k=%s, "
                "selection_interval=%s, dense_fallback_mask_count=%s, "
                "query_sparse=%s, prefix_sparse=%s, prefix_token_budget=%s, "
                "losa=%s, losa_active_topk=%s",
                sparse_dlm_ratio,
                sparse_dlm_top_k,
                sparse_dlm_selection_interval,
                sparse_dlm_dense_fallback_mask_count,
                query_sparse,
                prefix_sparse,
                prefix_token_budget,
                losa,
                losa_active_topk,
            )

        self.batch_size_per_gpu = batch_size
        self.max_prompt_len = int(max_prompt_len)
        self.gen_length = int(max_new_tokens or gen_length)
        self.block_length = int(sparse_dlm_block_length or block_length)
        self.steps = int(diffusion_steps or steps)
        self.temperature = float(temperature)
        self.top_p = _optional_number(top_p, float)
        self.top_k = _optional_number(top_k, int)
        self.threshold = float(threshold)
        self.editing_threshold = float(editing_threshold)
        self.max_post_steps = int(max_post_steps)
        self.minimal_topk = int(minimal_topk)
        self.num_to_transfer = int(num_to_transfer)
        self.mask_id = int(mask_id)
        self.eos_id = int(eos_id)
        self.show_samples = _as_bool(show_samples)
        self._generation_stats = {
            "generated_tokens": 0,
            "generation_time_seconds": 0.0,
            "generation_tokens_per_second": 0.0,
        }

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def get_model_info(self):
        """Expose generation throughput in the aggregated lm-eval result."""
        return dict(self._generation_stats)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    def _generate_one(self, context: str, gen_kwargs: dict) -> tuple[str, int]:
        gen_length = int(gen_kwargs.get("max_gen_toks", self.gen_length))
        temperature = gen_kwargs.get("temperature", self.temperature)
        if temperature is None:
            temperature = self.temperature
        model_limit = int(getattr(self.tokenizer, "model_max_length", 0))
        max_prompt_len = self.max_prompt_len
        if 0 < model_limit < 1_000_000_000:
            usable_length = (model_limit // self.block_length) * self.block_length
            max_prompt_len = min(max_prompt_len, usable_length - gen_length)
        if max_prompt_len <= 0:
            raise ValueError("gen_length leaves no room for the prompt")
        input_ids = self.tokenizer(
            context, return_tensors="pt", add_special_tokens=False
        ).input_ids[:, -max_prompt_len:]
        input_ids = input_ids.to(self.device)

        output_ids = self.model.generate(
            inputs=input_ids,
            eos_early_stop=True,
            gen_length=gen_length,
            block_length=self.block_length,
            steps=self.steps,
            temperature=float(temperature),
            top_p=_optional_number(gen_kwargs.get("top_p", self.top_p), float),
            top_k=_optional_number(gen_kwargs.get("top_k", self.top_k), int),
            threshold=self.threshold,
            editing_threshold=self.editing_threshold,
            max_post_steps=self.max_post_steps,
            minimal_topk=self.minimal_topk,
            num_to_transfer=self.num_to_transfer,
            mask_id=self.mask_id,
            eos_id=self.eos_id,
        )
        if hasattr(output_ids, "sequences"):
            output_ids = output_ids.sequences

        response = self.tokenizer.decode(
            output_ids[0].tolist(), skip_special_tokens=True
        )
        until = gen_kwargs.get("until", []) or []
        if isinstance(until, str):
            until = [until]
        for stop_sequence in until:
            response = response.split(stop_sequence, 1)[0]
        token_count = int((output_ids != self.eos_id).sum().item())
        return response, token_count

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        responses = []
        generated_tokens = 0
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        for request in tqdm(
            requests,
            disable=disable_tqdm or self.rank != 0,
            desc="Running generate_until requests",
        ):
            context, gen_kwargs = request.args
            response, token_count = self._generate_one(context, gen_kwargs)
            responses.append(response)
            generated_tokens += token_count
            self.cache_hook.add_partial(
                "generate_until", (context, gen_kwargs), response
            )
            if self.show_samples and self.rank == 0:
                print(f"Context:\n{context}\nResponse:\n{response}\n")

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        throughput = generated_tokens / elapsed if elapsed else 0.0
        self._generation_stats = {
            "generated_tokens": int(generated_tokens),
            "generation_time_seconds": float(elapsed),
            "generation_tokens_per_second": float(throughput),
        }
        if self.rank == 0:
            print(f"Time taken: {elapsed:.4f} seconds")
            print(f"Generated token num: {generated_tokens}")
            print(f"Generated token num per second: {throughput:.4f}")
        return responses

    def loglikelihood(self, requests):
        raise NotImplementedError("The LLaDA adapter supports generative tasks only")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("The LLaDA adapter supports generative tasks only")
