"""Self-contained dense / LoSA-v2 block diffusion generation.

Model weights and their remote-code model classes are loaded from the supplied
checkpoint path.  All generation and LoSA implementation code lives in this
package.
"""

from __future__ import annotations

import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .attention_patch import install_losa_v2_attention as install_paper_losa_attention


DEFAULT_MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(
    family: str,
    *,
    model_path: str | None = None,
    dtype: str | None = None,
    attn_implementation: str = "sdpa",
):
    # Keep this in the shared loader so dense, LoSA, and FOCUS v2 get the same
    # checkpoint-remote-code compatibility behavior.
    try:
        from src.focus_v2.compat import install_runtime_compat

        install_runtime_compat()
    except ImportError:
        pass
    model_path = model_path or DEFAULT_MODEL_PATHS[family]
    dtype = dtype or ("bfloat16" if family == "llada" else "float16")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if family == "sdar" and getattr(config, "pad_token_id", None) is None:
        # Some SDAR checkpoints omit pad_token_id, but their remote modeling
        # code unconditionally reads it while constructing the embedding layer.
        config.pad_token_id = 151643
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=getattr(torch, dtype),
        attn_implementation=attn_implementation,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    actual = getattr(model.config, "model_type", None)
    if family == "llada" and actual != "llada2_moe":
        raise ValueError(f"expected llada2_moe, got {actual!r}")
    if family == "sdar" and actual != "sdar":
        raise ValueError(f"expected sdar, got {actual!r}")
    return model, tokenizer


def top_k_logits(logits, k):
    if k is None or int(k) <= 0:
        return logits
    values, _ = torch.topk(logits, min(int(k), logits.shape[-1]))
    return torch.where(logits < values[..., -1, None], torch.full_like(logits, -torch.inf), logits)


def top_p_logits(logits, p):
    if p is None or float(p) >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask = torch.scatter(torch.zeros_like(logits, dtype=torch.bool), -1, sorted_indices, sorted_mask)
    return logits.masked_fill(mask, -torch.inf)


def sample_with_confidence(logits, temperature=0.0, top_k=None, top_p=None):
    shape = logits.shape[:-1]
    vocab = logits.shape[-1]
    logits = logits.reshape(-1, vocab)
    if temperature is None or float(temperature) <= 0:
        token = logits.argmax(dim=-1)
        prob = F.softmax(logits.float(), dim=-1).gather(-1, token[:, None]).squeeze(-1)
        return token.view(*shape), prob.view(*shape)
    logits = logits / float(temperature)
    logits = top_k_logits(logits, top_k)
    logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits.float(), dim=-1)
    token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    prob = probs.gather(-1, token[:, None]).squeeze(-1)
    return token.view(*shape), prob.view(*shape)


def block_causal_mask(num_blocks, block_length, device, dtype, family):
    mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
    mask = mask.repeat_interleave(block_length, dim=0).repeat_interleave(block_length, dim=1)
    if family == "sdar":
        return mask.unsqueeze(0)
    return torch.zeros((1, 1, mask.shape[0], mask.shape[1]), dtype=dtype, device=device).masked_fill(
        ~mask.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min
    )


def all_visible_mask(query_length, key_length, device, dtype, family):
    if family == "sdar":
        return torch.ones((1, query_length, key_length), dtype=torch.bool, device=device)
    return torch.zeros((1, 1, query_length, key_length), dtype=dtype, device=device)


def legacy_prefix_cache(cache, prefix_length):
    if prefix_length <= 0:
        return ()
    return tuple(
        (
            key[:, :, :prefix_length, :].contiguous(),
            value[:, :, :prefix_length, :].contiguous(),
        )
        for key, value in cache.to_legacy_cache()
    )


@torch.no_grad()
def build_sdar_prefix_cache(
    model,
    x: torch.Tensor,
    prefix_length: int,
    position_ids: torch.Tensor,
    *,
    block_length: int,
    query_chunk_length: int = 512,
) -> tuple:
    """Build SDAR prefix KV without a full-length quadratic query batch.

    The old initialization forwarded ``x[:, :block_end]`` as one query batch.
    SDAR's original prefill branch materializes an explicit SDPA mask, so a
    32K prompt can OOM even though the subsequent LoSA steps are sparse.  This
    helper computes the same immutable block-causal prefix KV in query chunks;
    the current generation block is intentionally excluded and is rebuilt by
    the existing LoSA step loop.
    """
    if prefix_length <= 0:
        return ()
    if prefix_length % block_length:
        raise ValueError("SDAR prefix length must be block aligned")
    query_chunk_length = (int(query_chunk_length) // block_length) * block_length
    if query_chunk_length <= 0:
        raise ValueError("query_chunk_length must be at least one block")

    base = model.model
    cache = DynamicCache()
    device = x.device
    for chunk_start in range(0, prefix_length, query_chunk_length):
        chunk_end = min(chunk_start + query_chunk_length, prefix_length)
        cur_x = x[:, chunk_start:chunk_end]
        cur_pos = position_ids[:, chunk_start:chunk_end]
        query_blocks = torch.arange(chunk_start, chunk_end, device=device) // block_length
        key_blocks = torch.arange(0, chunk_end, device=device) // block_length
        # SDAR uses boolean masks (True=attend).  Earlier blocks are fully
        # visible; queries are also visible to every token in their own block.
        attention_mask = (key_blocks[None, :] <= query_blocks[:, None])[None]
        base(
            cur_x,
            attention_mask=attention_mask,
            position_ids=cur_pos,
            past_key_values=cache,
            use_cache=True,
            store_kv=True,
        )
    return cache.to_legacy_cache()


@torch.no_grad()
def build_llada_prefix_cache(
    model,
    x: torch.Tensor,
    prefix_length: int,
    position_ids: torch.Tensor,
    *,
    block_length: int,
    query_chunk_length: int = 1024,
) -> tuple:
    """Build LLaDA prefix KV without a full-window LM-head forward.

    The previous initialization forwarded the whole 32K window through
    ``model(...)`` and materialized logits for every token.  That dominated
    memory (~65 GiB allocated in the exact 32K stress).  Here we build the
    same immutable block-causal prefix KV in bounded query chunks, and the
    caller later evaluates only the current generation block.
    """
    if prefix_length <= 0:
        return ()
    if prefix_length % block_length:
        raise ValueError("LLaDA prefix length must be block aligned")
    query_chunk_length = (int(query_chunk_length) // block_length) * block_length
    if query_chunk_length <= 0:
        raise ValueError("query_chunk_length must be at least one block")

    base = model.model
    cache = DynamicCache()
    device = x.device
    dtype = next(base.parameters()).dtype
    for chunk_start in range(0, prefix_length, query_chunk_length):
        chunk_end = min(chunk_start + query_chunk_length, prefix_length)
        cur_x = x[:, chunk_start:chunk_end]
        cur_pos = position_ids[:, chunk_start:chunk_end]
        hidden_states = base.word_embeddings(cur_x)
        position_embeddings = base.rotary_emb(hidden_states, cur_pos)
        query_blocks = torch.arange(chunk_start, chunk_end, device=device) // block_length
        key_blocks = torch.arange(0, chunk_end, device=device) // block_length
        allowed = key_blocks[None, :] <= query_blocks[:, None]
        attention_mask = torch.zeros(
            (1, 1, chunk_end - chunk_start, chunk_end),
            dtype=dtype,
            device=device,
        ).masked_fill(~allowed[None, None], torch.finfo(dtype).min)
        for layer in base.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=cur_pos,
                past_key_value=cache,
                output_attentions=False,
                output_router_logits=False,
                use_cache=True,
                position_embeddings=position_embeddings,
            )[0]
    return cache.to_legacy_cache()


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    values = torch.full((steps,), base, dtype=torch.long)
    values[:remainder] += 1
    return values


def transfer_llada(
    model,
    block_tokens,
    old_block_tokens,
    prompt_mask,
    active_block_mask,
    active_logits,
    *,
    temperature,
    top_p,
    top_k,
    threshold,
    editing_threshold,
    num_to_transfer,
):
    x0, x0_p = sample_with_confidence(active_logits, temperature=temperature, top_p=top_p, top_k=top_k)
    mask_transfer = torch.zeros_like(x0, dtype=torch.bool)
    if active_block_mask.any():
        confidence = torch.where(active_block_mask, x0_p, -torch.inf)
        high_conf = (confidence[0] > threshold) & active_block_mask[0]
        if int(high_conf.sum().item()) >= num_to_transfer:
            mask_transfer[0] = high_conf
        else:
            count = min(int(num_to_transfer), int(active_block_mask.sum().item()))
            if count:
                mask_transfer[0, torch.topk(confidence[0], k=count).indices] = True
    editable = (~active_block_mask) & (~prompt_mask[None, :])
    edit_confidence = torch.where(editable, x0_p, -torch.inf)
    editing = ((edit_confidence[0] > editing_threshold) & editable[0] & (x0[0] != old_block_tokens[0]))
    transfer = mask_transfer | editing.unsqueeze(0)
    if transfer.any():
        block_tokens[transfer] = x0[transfer]
    return block_tokens, transfer


def transfer_sdar(block_tokens, active_block_mask, active_logits, *, step, num_transfer_tokens, temperature, top_p, top_k, strategy, threshold):
    x0, x0_p = sample_with_confidence(active_logits, temperature=temperature, top_p=top_p, top_k=top_k)
    transfer = torch.zeros_like(active_block_mask)
    available = torch.where(active_block_mask[0])[0]
    if available.numel() == 0:
        return block_tokens, transfer
    count = min(int(num_transfer_tokens[min(step, len(num_transfer_tokens) - 1)].item()), int(available.numel()))
    if count <= 0:
        count = 1
    if strategy == "sequential":
        selected = available[:count]
    else:
        confidence = torch.where(active_block_mask, x0_p, -torch.inf)[0]
        high = available[confidence[available] > threshold]
        selected = high if high.numel() >= count else torch.topk(confidence, k=count).indices
    transfer[0, selected] = True
    block_tokens[transfer] = x0[transfer]
    return block_tokens, transfer


@contextmanager
def paper_losa_context(
    model,
    *,
    states,
    prefix_length,
    page_size,
    token_budget,
    active_count,
    gqa_mode,
    backend,
    trace_detail,
):
    trace = []
    model._paper_losa_context = {
        "states": states,
        "prefix_length": int(prefix_length),
        "page_size": int(page_size),
        "token_budget": int(token_budget),
        "active_count": int(active_count),
        "gqa_mode": str(gqa_mode),
        "backend": str(backend),
        "trace_detail": bool(trace_detail),
        "trace": trace,
    }
    try:
        yield trace
    finally:
        model._paper_losa_context = None


def model_forward(model, family, input_ids, attention_mask, position_ids, *, prefix_cache=(), losa_context_kwargs=None, store_kv=True):
    if prefix_cache:
        return layerwise_cached_forward(
            model,
            family,
            input_ids,
            attention_mask,
            position_ids,
            prefix_cache=prefix_cache,
            losa_context_kwargs=losa_context_kwargs,
        )
    cache = DynamicCache.from_legacy_cache(prefix_cache) if prefix_cache else None
    kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
    }
    if family == "sdar":
        kwargs["store_kv"] = bool(store_kv)
    if losa_context_kwargs is None:
        return model(**kwargs), []
    with paper_losa_context(model, **losa_context_kwargs) as trace:
        return model(**kwargs), trace


def layerwise_cached_forward(
    model,
    family,
    input_ids,
    attention_mask,
    position_ids,
    *,
    prefix_cache,
    losa_context_kwargs=None,
    store_kv=True,
):
    """Forward only the current block with a prefix KV cache.

    LLaDA's top-level forward validates block-attention masks and rejects
    `[block, prefix+block]` masks.  The layer modules themselves support this
    shape, so this self-owned loop mirrors the standard Transformer stack while
    bypassing only the top-level mask validator.
    """

    base = model.model
    cache = DynamicCache.from_legacy_cache(prefix_cache)
    if family == "llada":
        hidden_states = base.word_embeddings(input_ids)
    else:
        hidden_states = base.embed_tokens(input_ids)
    position_embeddings = base.rotary_emb(hidden_states, position_ids)
    trace = []

    def run_layers():
        nonlocal hidden_states
        for layer in base.layers:
            if family == "llada":
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=cache,
                    output_attentions=False,
                    output_router_logits=False,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )[0]
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=cache,
                    output_attentions=False,
                    use_cache=True,
                    store_kv=store_kv,
                    position_embeddings=position_embeddings,
                )[0]

    if losa_context_kwargs is None:
        run_layers()
    else:
        with paper_losa_context(model, **losa_context_kwargs) as active_trace:
            run_layers()
            trace = list(active_trace)
    hidden_states = base.norm(hidden_states)
    logits = model.lm_head(hidden_states)
    return SimpleNamespace(logits=logits, past_key_values=cache), trace


@torch.no_grad()
def block_diffusion_generate(
    model,
    *,
    family,
    inputs,
    use_losa=False,
    gen_length=32,
    block_length=32,
    steps=32,
    temperature=0.0,
    top_p=None,
    top_k=None,
    threshold=None,
    editing_threshold=0.9,
    max_post_steps=16,
    minimal_topk=1,
    num_to_transfer=1,
    remasking_strategy="sequential",
    eos_early_stop=True,
    mask_id=None,
    eos_id=None,
    losa_page_size=16,
    losa_token_budget=256,
    losa_active_topk=5,
    losa_gqa_mode="per_query_head",
    losa_backend="auto",
    losa_trace_detail=False,
):
    if inputs.shape[0] != 1:
        raise ValueError("runtime currently supports batch_size=1")
    if use_losa:
        install_paper_losa_attention(model, family)
    device = model.device
    dtype = next(model.parameters()).dtype
    input_ids = inputs.to(device)
    if mask_id is None:
        mask_id = 156895 if family == "llada" else int(
            getattr(model.config, "mask_token_id", None) or 151669
        )
    if eos_id is None and family == "llada":
        configured_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", 156892)
        eos_id = int(
            configured_eos[0]
            if isinstance(configured_eos, (list, tuple))
            else configured_eos or 156892
        )
    prompt_length = input_ids.shape[1]
    gen_length = int(gen_length)
    if gen_length < 0:
        raise ValueError("gen_length must be non-negative")
    if gen_length == 0:
        return SimpleNamespace(tokens=input_ids[:, :0], trace=[])
    if family == "llada":
        if minimal_topk <= 0:
            raise ValueError("minimal_topk must be positive")
        steps = min(int(steps), gen_length // int(minimal_topk))
        if steps <= 0:
            raise ValueError("LLaDA requires at least one denoising step")
    else:
        steps = max(1, min(int(steps), int(block_length)))
    num_blocks = math.ceil((prompt_length + gen_length) / block_length)
    total_length = num_blocks * block_length
    x = torch.full((1, total_length), int(mask_id), dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    full_mask = block_causal_mask(num_blocks, block_length, device, dtype, family)
    prefill_blocks = prompt_length // block_length
    traces = []
    transfer_counts = get_num_transfer_tokens(block_length, steps)
    threshold = threshold if threshold is not None else (0.85 if family == "sdar" else 0.95)

    # The immutable prefix cache is finalized once at the end of every block and
    # reused by the next block.  v1 rebuilt all previous blocks from scratch.
    persistent_prefix_cache = None

    for block_idx in range(prefill_blocks, num_blocks):
        block_start = block_idx * block_length
        block_end = (block_idx + 1) * block_length
        current_window_end = block_end
        prompt_mask = torch.zeros(block_length, dtype=torch.bool, device=device)
        if block_start < prompt_length:
            prompt_mask[: min(prompt_length - block_start, block_length)] = True

        cur_x = x[:, :current_window_end]
        cur_mask = full_mask[..., :current_window_end, :current_window_end] if family == "llada" else full_mask[:, :current_window_end, :current_window_end]
        cur_pos = position_ids[:, :current_window_end]
        if persistent_prefix_cache is not None:
            prefix_cache = persistent_prefix_cache
        elif family == "sdar":
            # Avoid SDAR's full-window prefill OOM; this produces the same
            # block-causal prefix KV that the removed full forward provided.
            prefix_cache = build_sdar_prefix_cache(
                model, x, block_start, position_ids, block_length=block_length
            )
            # Compute only the initial current-block logits against that
            # prefix.  The all-visible mask uses SDAR's FlashAttention branch.
            outputs, _ = model_forward(
                model,
                family,
                x[:, block_start:block_end],
                all_visible_mask(block_length, block_end, device, dtype, family),
                position_ids[:, block_start:block_end],
                prefix_cache=prefix_cache,
                store_kv=False,
            )
        elif family == "llada":
            # Same bounded-prefix construction for LLaDA.  This avoids the
            # full 32K LM-head forward that produced ~65 GiB allocations.
            prefix_cache = build_llada_prefix_cache(
                model, x, block_start, position_ids, block_length=block_length
            )
            outputs, _ = model_forward(
                model,
                family,
                x[:, block_start:block_end],
                all_visible_mask(block_length, block_end, device, dtype, family),
                position_ids[:, block_start:block_end],
                prefix_cache=prefix_cache,
            )
        else:
            outputs, _ = model_forward(model, family, cur_x, cur_mask, cur_pos)
            prefix_cache = legacy_prefix_cache(outputs.past_key_values, block_start)
        states = {}
        step = 0
        post_steps = 0
        while True:
            old_block = x[:, block_start:block_end].clone()
            active_mask = old_block == int(mask_id)
            if family == "llada" and not active_mask.any():
                post_steps += 1
                if post_steps > int(max_post_steps):
                    break
            if family != "llada" and step >= steps:
                break
            if step == 0:
                trace = []
            else:
                block_input = x[:, block_start:block_end]
                key_length = block_start + block_length
                block_mask = all_visible_mask(block_length, key_length, device, dtype, family)
                block_pos = position_ids[:, block_start:block_end]
                losa_kwargs = None
                if use_losa and block_start > 0:
                    losa_kwargs = {
                        "states": states,
                        "prefix_length": block_start,
                        "page_size": losa_page_size,
                        "token_budget": losa_token_budget,
                        "active_count": min(losa_active_topk, block_length),
                        "gqa_mode": losa_gqa_mode,
                        "backend": losa_backend,
                        "trace_detail": losa_trace_detail,
                    }
                outputs, trace = model_forward(
                    model,
                    family,
                    block_input,
                    block_mask,
                    block_pos,
                    prefix_cache=prefix_cache,
                    losa_context_kwargs=losa_kwargs,
                    store_kv=False,
                )
            traces.extend({"block": block_idx, "step": step, **item} for item in trace)
            logits = outputs.logits[:, -block_length:, :]
            if family == "llada":
                new_block, _ = transfer_llada(
                    model,
                    x[:, block_start:block_end],
                    old_block,
                    prompt_mask,
                    active_mask,
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    threshold=threshold,
                    editing_threshold=float(editing_threshold),
                    num_to_transfer=num_to_transfer,
                )
                x[:, block_start:block_end] = new_block
                editing_happened = bool(
                    ((old_block != int(mask_id)) & (new_block != old_block)).any().item()
                )
                if not active_mask.any() and not editing_happened:
                    break
            else:
                new_block, _ = transfer_sdar(
                    x[:, block_start:block_end],
                    active_mask,
                    logits,
                    step=step,
                    num_transfer_tokens=transfer_counts,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    strategy=remasking_strategy,
                    threshold=threshold,
                )
                x[:, block_start:block_end] = new_block
            step += 1

        # Finalize the block's KV once its tokens are fixed.  This forward is
        # required because the last denoising forward saw the pre-transfer block.
        final_outputs, _ = model_forward(
            model,
            family,
            x[:, block_start:block_end],
            all_visible_mask(block_length, block_end, device, dtype, family),
            position_ids[:, block_start:block_end],
            prefix_cache=prefix_cache,
            store_kv=True,
        )
        persistent_prefix_cache = final_outputs.past_key_values.to_legacy_cache()

        if eos_early_stop and eos_id is not None:
            generated_part = x[0, prompt_length:block_end]
            if (generated_part == int(mask_id)).sum() == 0:
                eos_positions = (generated_part == int(eos_id)).nonzero(as_tuple=True)[0]
                if eos_positions.numel():
                    break

    generated = x[:, prompt_length : prompt_length + gen_length]
    if eos_early_stop and eos_id is not None:
        eos_positions = (generated[0] == int(eos_id)).nonzero(as_tuple=True)[0]
        if eos_positions.numel():
            generated = generated[:, : int(eos_positions[0].item()) + 1]
    return SimpleNamespace(tokens=generated, trace=traces)
