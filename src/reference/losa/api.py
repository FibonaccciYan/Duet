from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import types

import torch

from .attention_patch import install_paper_losa_attention
from .generation import (
    block_diffusion_generate,
    load_model_and_tokenizer,
    set_seed,
)
try:  # pragma: no cover - the packed MoE backend is optional
    from .moe_patch import patch_moe_experts
except Exception:  # pragma: no cover
    patch_moe_experts = None


DEFAULT_MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}


def _resolve_family(model: Any, model_name: str) -> str:
    actual = getattr(getattr(model, "config", None), "model_type", None)
    inferred = {"llada2_moe": "llada", "sdar": "sdar"}.get(actual)
    if model_name == "auto":
        if inferred is None:
            raise ValueError(f"Unsupported model_type: {actual!r}")
        return inferred
    if model_name not in {"llada", "sdar"} or inferred != model_name:
        raise ValueError(
            f"Requested {model_name!r}, but checkpoint model_type is {actual!r}"
        )
    return model_name


def _losa_generate(self, *args, **kwargs):
    inputs = kwargs.pop("inputs", args[0] if args else None)
    if inputs is None:
        raise ValueError("LoSA generation requires `inputs`")
    install_paper_losa_attention(self, self._losa_patch_family)
    return block_diffusion_generate(
        self,
        family=self._losa_patch_family,
        inputs=inputs,
        use_losa=True,
        **self._losa_generation_options,
        **kwargs,
    ).tokens


def patch_model(
    model: Any,
    model_name: str = "auto",
    *,
    page_size: int = 16,
    token_budget: int = 256,
    active_topk: int = 5,
    gqa_mode: str = "per_query_head",
    backend: str = "auto",
    trace_detail: bool = False,
    **_: Any,
):
    """Install paper-LoSA generation through sparse's model-patching contract."""
    family = _resolve_family(model, model_name)
    if page_size <= 0 or token_budget <= 0 or active_topk <= 0:
        raise ValueError("page_size, token_budget, and active_topk must be positive")
    if gqa_mode not in {"per_query_head", "group_mean", "group_max_score"}:
        raise ValueError(f"Unsupported LoSA GQA mode: {gqa_mode!r}")
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"Unsupported LoSA backend: {backend!r}")
    if not hasattr(model, "_losa_original_generate"):
        model._losa_original_generate = model.generate
        model.generate = types.MethodType(_losa_generate, model)
    model._losa_patch_family = family
    install_paper_losa_attention(model, family)
    model._losa_generation_options = {
        "losa_page_size": int(page_size),
        "losa_token_budget": int(token_budget),
        "losa_active_topk": int(active_topk),
        "losa_gqa_mode": gqa_mode,
        "losa_backend": backend,
        "losa_trace_detail": bool(trace_detail),
    }
    return model


@dataclass
class LoSARuntime:
    family: str
    model_path: str | None = None
    dtype: str | None = None
    attn_implementation: str = "sdpa"
    moe_expert_patch: bool = False
    losa_page_size: int = 16
    losa_token_budget: int = 256
    losa_active_topk: int = 5
    losa_gqa_mode: str = "per_query_head"
    losa_backend: str = "auto"
    losa_trace_detail: bool = False
    model: Any | None = field(init=False, default=None)
    tokenizer: Any | None = field(init=False, default=None)
    moe_patch_report: Any | None = field(init=False, default=None)

    def load(self):
        model_path = self.model_path or DEFAULT_MODEL_PATHS[self.family]
        dtype_name = self.dtype or ("bfloat16" if self.family == "llada" else "float16")
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.family,
            model_path=model_path,
            dtype=dtype_name,
            attn_implementation=self.attn_implementation,
        )
        if self.moe_expert_patch:
            if patch_moe_experts is None:
                raise RuntimeError("the optional Triton MoE backend is unavailable")
            self.moe_patch_report = patch_moe_experts(self.model, family=self.family)
        return self.model, self.tokenizer

    def generate(self, inputs: torch.Tensor, **kwargs):
        if self.model is None or self.tokenizer is None:
            self.load()
        assert self.model is not None
        install_paper_losa_attention(self.model, self.family)
        return block_diffusion_generate(
            self.model,
            family=self.family,
            inputs=inputs,
            use_losa=True,
            losa_page_size=self.losa_page_size,
            losa_token_budget=self.losa_token_budget,
            losa_active_topk=self.losa_active_topk,
            losa_gqa_mode=self.losa_gqa_mode,
            losa_backend=self.losa_backend,
            losa_trace_detail=self.losa_trace_detail,
            **kwargs,
        )
