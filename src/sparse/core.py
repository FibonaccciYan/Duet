"""Shared sparse controls and model patch dispatcher."""

import math

import torch

from .moe_expert_patch import patch_moe_experts
from .sdar_generate import entropy_from_logits, select_transfer


MODEL_TYPES = {
    "llada": {"llada2_moe"},
    "sdar": {"sdar"},
}


def _sample_with_confidence(model, logits, temperature, top_p, top_k):
    return model._sample_with_temperature_topk_topp(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def _select_positions(
    model,
    hidden_states,
    input_ids,
    mask_id,
    ratio,
    top_k,
    temperature=0.0,
    top_p=None,
    cached_positions=None,
    selection_step=0,
    selection_interval=1,
    dense_fallback_mask_count=0,
    minimum_mask_candidates=1,
    strategy="low_confidence_static",
    threshold=1.0,
    entropy_budget=None,
):
    mask = input_ids[0] == mask_id
    mask_count = int(mask.sum().item())
    if ratio >= 1.0 or mask_count <= dense_fallback_mask_count:
        return None

    candidate_count = min(
        max(int(minimum_mask_candidates), math.ceil(mask_count * ratio)),
        mask_count,
    )
    decoded = torch.where(~mask)[0]
    if (
        cached_positions is not None
        and selection_interval > 1
        and selection_step % selection_interval != 0
    ):
        old_masks = cached_positions[mask[cached_positions]]
        if old_masks.numel() >= candidate_count:
            return torch.cat((decoded, old_masks))

    mask_positions = torch.where(mask)[0]
    if strategy == "sequential":
        return torch.cat((decoded, mask_positions[:candidate_count]))

    mask_logits = model.lm_head(hidden_states[:, mask_positions, :])
    _, confidence = _sample_with_confidence(
        model,
        mask_logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    entropy = (
        entropy_from_logits(mask_logits, temperature, top_k, top_p)
        if strategy == "entropy_bounded"
        else None
    )
    selected = select_transfer(
        torch.ones_like(confidence, dtype=torch.bool),
        confidence,
        candidate_count,
        strategy,
        threshold,
        entropy=entropy,
        entropy_budget=entropy_budget,
    )
    return torch.cat((decoded, mask_positions[torch.where(selected[0])[0]]))


def _legacy_prefix_cache(cache, prefix_length):
    return tuple(
        (
            key[:, :, :prefix_length, :].contiguous(),
            value[:, :, :prefix_length, :].contiguous(),
        )
        for key, value in cache.to_legacy_cache()
    )


class _BlockDualCache:
    """Keep dense current-block KV and overwrite sparse query positions."""

    def __init__(self, key_values, prefix_lengths):
        self.key_cache = [key_states for key_states, _ in key_values]
        self.value_cache = [value_states for _, value_states in key_values]
        self.prefix_lengths = prefix_lengths
        self.positions = None

    def set_positions(self, positions):
        self.positions = positions

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # TODO(query-sparse): fuse selected KV writes into the attention path.
        if self.positions is None:
            raise RuntimeError("Sparse cache positions must be set before updating KV")
        positions = self.positions.to(device=key_states.device, dtype=torch.long)
        if key_states.shape[-2] != positions.numel():
            raise ValueError("Sparse KV length does not match selected positions")
        replace_positions = positions + self.prefix_lengths[layer_idx]
        self.key_cache[layer_idx].index_copy_(2, replace_positions, key_states)
        self.value_cache[layer_idx].index_copy_(2, replace_positions, value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def _dual_cache_from_dense(dense_cache, prefix_cache, block_start, block_end):
    legacy_cache = dense_cache.to_legacy_cache()
    if len(legacy_cache) != len(prefix_cache):
        raise ValueError("Dense and prefix caches must have the same layer count")
    key_values = []
    prefix_lengths = []
    for (dense_key, dense_value), (prefix_key, prefix_value) in zip(
        legacy_cache, prefix_cache
    ):
        key_values.append(
            (
                torch.cat((prefix_key, dense_key[:, :, block_start:block_end]), dim=2),
                torch.cat((prefix_value, dense_value[:, :, block_start:block_end]), dim=2),
            )
        )
        prefix_lengths.append(prefix_key.shape[-2])
    return _BlockDualCache(key_values, prefix_lengths)


def resolve_model_family(model, model_name="auto"):
    actual_type = getattr(model.config, "model_type", None)
    if model_name == "auto":
        for family, model_types in MODEL_TYPES.items():
            if actual_type in model_types:
                return family
        raise ValueError(f"Unsupported model_type: {actual_type!r}")
    if model_name not in MODEL_TYPES:
        raise ValueError(f"Unsupported model name: {model_name!r}")
    if actual_type not in MODEL_TYPES[model_name]:
        raise ValueError(
            f"Requested {model_name!r}, but checkpoint model_type is {actual_type!r}"
        )
    return model_name


def patch_model(
    model,
    model_name="auto",
    ratio=0.5,
    top_k=64,
    selection_interval=None,
    dense_fallback_mask_count=None,
    refresh_step=2,
    query_sparse=True,
    prefix_sparse=False,
    prefix_token_budget=256,
    prefix_chunk_size=256,
    losa=False,
    losa_active_topk=5,
    moe_expert_patch=True,
):
    """Enable requested sparse features through the matching model patch."""
    family = resolve_model_family(model, model_name)
    if family == "llada":
        from .block_cache_sparse_dlm_patch import patch_llada_model

        patch_llada_model(
            model,
            ratio=ratio,
            top_k=top_k,
            selection_interval=selection_interval or 4,
            dense_fallback_mask_count=(
                4 if dense_fallback_mask_count is None else dense_fallback_mask_count
            ),
            query_sparse=query_sparse,
            prefix_sparse=prefix_sparse,
            prefix_token_budget=prefix_token_budget,
            prefix_chunk_size=prefix_chunk_size,
            losa=losa,
            losa_active_topk=losa_active_topk,
        )
    else:
        if prefix_sparse or losa:
            raise ValueError("SDAR currently supports query_sparse only")
        from .sdar_block_diffusion_patch import patch_sdar_model

        patch_sdar_model(
            model,
            ratio=ratio,
            top_k=top_k,
            selection_interval=selection_interval or 1,
            dense_fallback_mask_count=(
                0 if dense_fallback_mask_count is None else dense_fallback_mask_count
            ),
            refresh_step=refresh_step,
            query_sparse=query_sparse,
        )

    if moe_expert_patch:
        patch_moe_experts(model)
    model._sparse_patch_family = family
    return model
