"""Block-diffusion generation driver for FOCUS v2."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from src.losa.generation import get_num_transfer_tokens, sample_with_confidence

from src.focus.algorithm import FocusDecodeState
from src.focus.generation import (
    _selected_llada_transfer,
    _selected_sdar_transfer,
)

from .model import focus_v2_forward


@torch.inference_mode()
def focus_v2_generate(
    model,
    *,
    family: str,
    inputs: torch.Tensor,
    alpha: float = 1.5,
    gen_length: int = 128,
    block_length: int = 32,
    steps: int = 32,
    temperature: float = 0.0,
    top_p=None,
    top_k=None,
    threshold=None,
    editing_threshold: float = 0.5,
    max_post_steps: int = 16,
    minimal_topk: int = 1,
    num_to_transfer: int = 1,
    remasking_strategy: str = "low_confidence_dynamic",
    eos_early_stop: bool = True,
    mask_id=None,
    eos_id=None,
):
    if inputs.shape[0] != 1:
        raise ValueError("FOCUS v2 currently requires batch_size=1")
    if alpha < 1:
        raise ValueError("FOCUS v2 alpha must be at least 1")
    if block_length <= 0 or gen_length < 0:
        raise ValueError("block_length must be positive and gen_length non-negative")
    if gen_length == 0:
        return SimpleNamespace(tokens=inputs[:, :0], trace=[])

    device = model.device
    input_ids = inputs.to(device)
    prompt_length = input_ids.shape[1]
    mask_id = int(mask_id if mask_id is not None else (156895 if family == "llada" else 151669))
    if threshold is None:
        threshold = 0.7 if family == "llada" else 0.95
    if eos_id is None:
        value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if value is not None:
            eos_id = int(value)
        elif family == "llada":
            eos_id = 156892

    if family == "llada":
        steps = min(int(steps), gen_length // max(1, int(minimal_topk)))
        if steps <= 0:
            raise ValueError("LLaDA requires at least one denoising step")
    else:
        steps = max(1, min(int(steps), int(block_length)))
    transfer_counts = get_num_transfer_tokens(block_length, steps).to(device)

    num_blocks = math.ceil((prompt_length + gen_length) / block_length)
    total_length = num_blocks * block_length
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids
    all_positions = torch.arange(total_length, device=device).unsqueeze(0)
    prefill_blocks = prompt_length // block_length
    prefix_cache = ()

    # Build an immutable, block-causal prefix a block at a time. Passing an
    # impossible mask id disables FOCUS selection during prefill.
    for block_idx in range(prefill_blocks):
        start = block_idx * block_length
        end = start + block_length
        dense = focus_v2_forward(
            model,
            family=family,
            input_ids=x[:, start:end],
            position_ids=all_positions[:, start:end],
            prefix_cache=prefix_cache,
            mask_id=-1,
            alpha=alpha,
            average_decoded_tokens=1.0,
            block_progress=-1,
        )
        prefix_cache = dense.cache.to_legacy_cache()

    traces = []
    for block_idx in range(prefill_blocks, num_blocks):
        start = block_idx * block_length
        end = start + block_length
        prompt_mask = torch.zeros(block_length, dtype=torch.bool, device=device)
        if start < prompt_length:
            prompt_mask[: min(prompt_length - start, block_length)] = True
        state = FocusDecodeState()
        post_steps = 0
        step = 0
        while True:
            old = x[:, start:end].clone()
            active = old == mask_id
            if family == "llada" and not bool(active.any().item()):
                post_steps += 1
                if post_steps > int(max_post_steps):
                    break
            if family == "sdar" and step >= steps:
                break

            result = focus_v2_forward(
                model,
                family=family,
                input_ids=x[:, start:end],
                position_ids=all_positions[:, start:end],
                prefix_cache=prefix_cache,
                mask_id=mask_id,
                alpha=alpha,
                average_decoded_tokens=state.average_decoded_tokens,
                block_progress=state.block_progress,
            )
            selected = result.positions
            state.block_progress = max(state.block_progress, int(selected.max().item()))
            if family == "llada":
                transfer, decoded, edited = _selected_llada_transfer(
                    x[:, start:end],
                    old,
                    active,
                    prompt_mask,
                    result.logits,
                    selected,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    threshold=threshold,
                    editing_threshold=editing_threshold,
                    num_to_transfer=num_to_transfer,
                )
                resolved = int(active.sum().item()) - int((x[:, start:end] == mask_id).sum().item())
                if bool(active.any().item()) and resolved <= 0:
                    raise RuntimeError("FOCUS v2 LLaDA mask-to-token step made no progress")
                state.record_transfer(resolved)
                if not bool(active.any().item()) and not edited:
                    break
            else:
                transfer, decoded = _selected_sdar_transfer(
                    x[:, start:end],
                    active,
                    result.logits,
                    selected,
                    count=transfer_counts[min(step, steps - 1)],
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    strategy=remasking_strategy,
                    threshold=threshold,
                )
                resolved = int(active.sum().item()) - int((x[:, start:end] == mask_id).sum().item())
                if bool(active.any().item()) and resolved <= 0:
                    raise RuntimeError("FOCUS v2 SDAR mask-to-token step made no progress")
                state.record_transfer(resolved)
            traces.append(
                {
                    "block": block_idx,
                    "step": step,
                    "active_before": int(active.sum().item()),
                    "retained": int(selected.numel()),
                    "transferred": int(transfer.sum().item()),
                }
            )
            step += 1

        if bool((x[:, start:end] == mask_id).any().item()):
            raise RuntimeError(f"FOCUS block {block_idx} still contains mask tokens")

        # Persist exact full-block K/V for the next block, matching the final
        # cache-only pass in the SDAR reference decoder.
        final = focus_v2_forward(
            model,
            family=family,
            input_ids=x[:, start:end],
            position_ids=all_positions[:, start:end],
            prefix_cache=prefix_cache,
            mask_id=-1,
            alpha=alpha,
            average_decoded_tokens=state.average_decoded_tokens,
            block_progress=state.block_progress,
        )
        prefix_cache = final.cache.to_legacy_cache()

        if eos_early_stop and eos_id is not None:
            generated = x[0, prompt_length:end]
            eos = torch.where(generated == int(eos_id))[0]
            if eos.numel():
                break

    generated = x[:, prompt_length : prompt_length + gen_length]
    if eos_early_stop and eos_id is not None:
        eos = torch.where(generated[0] == int(eos_id))[0]
        if eos.numel():
            generated = generated[:, : int(eos[0].item()) + 1]
    return SimpleNamespace(tokens=generated, trace=traces)


__all__ = ["focus_v2_generate"]
