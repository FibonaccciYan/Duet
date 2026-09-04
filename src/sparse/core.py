"""Shared sparse controls and model patch dispatcher."""

import math

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal installs
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from .llada_moe_expert_patch import patch_moe_experts
from .sdar_generate import entropy_from_logits, select_transfer


MODEL_TYPES = {
    "llada": {"llada2_moe"},
    "sdar": {"sdar"},
}


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_kv_index_copy_kernel(
        key_cache,
        value_cache,
        key_states,
        value_states,
        positions,
        prefix_length,
        heads,
        selected,
        head_dim,
        cache_stride_0,
        cache_stride_1,
        cache_stride_2,
        cache_stride_3,
        value_cache_stride_0,
        value_cache_stride_1,
        value_cache_stride_2,
        value_cache_stride_3,
        state_stride_0,
        state_stride_1,
        state_stride_2,
        state_stride_3,
        value_state_stride_0,
        value_state_stride_1,
        value_state_stride_2,
        value_state_stride_3,
        total,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = offsets < total
        dim = offsets % head_dim
        rows = offsets // head_dim
        selected_idx = rows % selected
        rows = rows // selected
        head = rows % heads
        batch = rows // heads
        cache_position = tl.load(positions + selected_idx, mask=valid) + prefix_length
        key_cache_offsets = (
            batch * cache_stride_0
            + head * cache_stride_1
            + cache_position * cache_stride_2
            + dim * cache_stride_3
        )
        value_cache_offsets = (
            batch * value_cache_stride_0
            + head * value_cache_stride_1
            + cache_position * value_cache_stride_2
            + dim * value_cache_stride_3
        )
        key_state_offsets = (
            batch * state_stride_0
            + head * state_stride_1
            + selected_idx * state_stride_2
            + dim * state_stride_3
        )
        value_state_offsets = (
            batch * value_state_stride_0
            + head * value_state_stride_1
            + selected_idx * value_state_stride_2
            + dim * value_state_stride_3
        )
        tl.store(
            key_cache + key_cache_offsets,
            tl.load(key_states + key_state_offsets, mask=valid),
            mask=valid,
        )
        tl.store(
            value_cache + value_cache_offsets,
            tl.load(value_states + value_state_offsets, mask=valid),
            mask=valid,
        )


def _fused_kv_index_copy_(
    key_cache,
    value_cache,
    positions,
    key_states,
    value_states,
    prefix_length,
):
    if key_states.shape[-2] == 0:
        return
    use_triton = (
        _TRITON_AVAILABLE
        and key_states.ndim == 4
        and key_states.is_cuda
        and value_states.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and positions.is_cuda
        and key_states.device
        == value_states.device
        == key_cache.device
        == value_cache.device
        == positions.device
        and key_states.shape == value_states.shape
        and key_cache.shape == value_cache.shape
        and key_states.shape[:2] == key_cache.shape[:2]
        and key_states.shape[3] == key_cache.shape[3]
        and key_states.dtype == key_cache.dtype
        and value_states.dtype == value_cache.dtype
    )
    if not use_triton:
        replace_positions = positions + prefix_length
        key_cache.index_copy_(2, replace_positions, key_states)
        value_cache.index_copy_(2, replace_positions, value_states)
        return

    batch, heads, selected, head_dim = key_states.shape
    total = key_states.numel()
    _fused_kv_index_copy_kernel[(triton.cdiv(total, 256),)](
        key_cache,
        value_cache,
        key_states,
        value_states,
        positions,
        prefix_length,
        heads,
        selected,
        head_dim,
        *key_cache.stride(),
        *value_cache.stride(),
        *key_states.stride(),
        *value_states.stride(),
        total,
        BLOCK_SIZE=256,
    )


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
        self.positions = positions.to(
            device=self.key_cache[0].device,
            dtype=torch.long,
        ).contiguous()

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if self.positions is None:
            raise RuntimeError("Sparse cache positions must be set before updating KV")
        positions = self.positions
        if key_states.shape[-2] != positions.numel():
            raise ValueError("Sparse KV length does not match selected positions")
        _fused_kv_index_copy_(
            self.key_cache[layer_idx],
            self.value_cache[layer_idx],
            positions,
            key_states,
            value_states,
            self.prefix_lengths[layer_idx],
        )
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
    refresh_step=-1,
    selection_layer=5,
    deep_only_transfer=False,
    query_sparse=False,
    prefix_sparse=False,
    prefix_token_budget=256,
    prefix_chunk_size=256,
    losa=False,
    losa_active_topk=5,
    losa_score_mode="query",
    losa_key_samples=32,
    query_losa_union=False,
    moe_expert_patch=True,
):
    """Enable requested sparse features through the matching model patch."""
    family = resolve_model_family(model, model_name)
    if family == "llada":
        from .llada_patch import patch_llada_model

        patch_llada_model(
            model,
            ratio=ratio,
            top_k=top_k,
            selection_interval=selection_interval or 4,
            dense_fallback_mask_count=(
                4 if dense_fallback_mask_count is None else dense_fallback_mask_count
            ),
            selection_layer=selection_layer,
            query_sparse=query_sparse,
            prefix_sparse=prefix_sparse,
            prefix_token_budget=prefix_token_budget,
            prefix_chunk_size=prefix_chunk_size,
            losa=losa,
            losa_active_topk=losa_active_topk,
            losa_score_mode=losa_score_mode,
            losa_key_samples=losa_key_samples,
            query_losa_union=query_losa_union,
        )
        
        if moe_expert_patch:
            patch_moe_experts(model)
    else:
        from .sdar_patch import patch_sdar_model

        patch_sdar_model(
            model,
            ratio=ratio,
            top_k=top_k,
            selection_interval=selection_interval or 1,
            dense_fallback_mask_count=(
                0 if dense_fallback_mask_count is None else dense_fallback_mask_count
            ),
            refresh_step=refresh_step,
            selection_layer=selection_layer,
            deep_only_transfer=deep_only_transfer,
            query_sparse=query_sparse,
            prefix_sparse=prefix_sparse,
            prefix_token_budget=prefix_token_budget,
            prefix_chunk_size=prefix_chunk_size,
            losa=losa,
            losa_active_topk=losa_active_topk,
            losa_score_mode=losa_score_mode,
            losa_key_samples=losa_key_samples,
        )

    model._sparse_patch_family = family
    return model
