"""Sparse attention and prefix-cache operations shared by LLaDA and SDAR."""

import math
import os

import torch

try:
    import faster_hadamard_transform
except (ImportError, OSError):  # pragma: no cover - optional CUDA extension
    faster_hadamard_transform = None

from .core import _legacy_prefix_cache
from .triton_kernels import adamas_distances, attention_output_lse, losa_query_delta


ADAMAS_BUCKET_THRESHOLDS = {
    "llada2_moe": ((-1.73, 0.0, 1.72), (-2.74, 0.0, 2.69)),
    "sdar": ((-1.50, 0.0, 1.49), (-2.87, 0.0, 2.86)),
}


def _repeat_kv(hidden_states, num_key_value_groups):
    if num_key_value_groups == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(batch, heads, num_key_value_groups, seq_len, head_dim)
        .reshape(batch, heads * num_key_value_groups, seq_len, head_dim)
    )


def _losa_key_energy(prefix_key, num_key_value_groups, sample_count):
    """Estimate diagonal E[k^2] once; runtime scoring stays O(QHD)."""
    prefix_length = prefix_key.shape[2]
    if prefix_length > sample_count:
        indices = torch.arange(
            sample_count, device=prefix_key.device, dtype=torch.long
        ).mul_(prefix_length).div_(sample_count, rounding_mode="floor")
        prefix_key = prefix_key.index_select(2, indices)
    energy = prefix_key.float().square().mean(dim=2)
    return energy.repeat_interleave(num_key_value_groups, dim=1)[0]


def _attention_output_lse(
    query,
    key,
    value,
    attention_mask,
    num_key_value_groups,
    use_triton=True,
):
    """Attention returning the normalized output and row-wise LSE."""
    triton_result = (
        attention_output_lse(query, key, value, attention_mask)
        if use_triton
        else None
    )
    if triton_result is not None:
        return triton_result
    key = _repeat_kv(key, num_key_value_groups)
    value = _repeat_kv(value, num_key_value_groups)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores * (query.shape[-1] ** -0.5)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, -torch.inf)
        else:
            scores = scores + attention_mask
    lse = torch.logsumexp(scores.float(), dim=-1)
    attention_weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
        query.dtype
    )
    return torch.matmul(attention_weights, value), lse


def _merge_attention_states(prefix_output, prefix_lse, block_output, block_lse):
    prefix_lse = prefix_lse.float()
    block_lse = block_lse.float()
    total_lse = torch.logaddexp(prefix_lse, block_lse)
    prefix_scale = torch.exp(prefix_lse - total_lse).unsqueeze(-1)
    block_scale = torch.exp(block_lse - total_lse).unsqueeze(-1)
    output = prefix_output.float() * prefix_scale + block_output.float() * block_scale
    return output, total_lse


def _new_losa_state(query, block_length):
    batch, heads, _, head_dim = query.shape
    return {
        "previous_query": torch.zeros(
            batch, heads, block_length, head_dim, dtype=query.dtype, device=query.device
        ),
        "prefix_output": torch.zeros(
            batch, heads, block_length, head_dim, dtype=torch.float32, device=query.device
        ),
        "prefix_lse": torch.full(
            (batch, heads, block_length),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        ),
        "valid": torch.zeros(batch, block_length, dtype=torch.bool, device=query.device),
        "fully_valid": False,
    }


def _queue_losa_active_update(
    context,
    layer_idx,
    positions,
    query,
    active_indices,
    prefix_output,
    prefix_lse,
):
    active_positions = positions.index_select(0, active_indices)
    context["pending_losa_queries"].append(
        (layer_idx, active_positions, query.index_select(2, active_indices))
    )
    context["pending_losa"].append(
        (layer_idx, active_positions, prefix_output, prefix_lse)
    )


def _losa_active_indices(
    state,
    query,
    query_positions,
    active_topk,
    return_metadata=False,
    score_mode="query",
):
    positions = query_positions.to(device=query.device, dtype=torch.long)
    weights = state.get("key_energy") if score_mode.startswith("key_diag") else None
    if state.get("fully_valid", False) and state.get("use_triton_delta", True):
        delta = losa_query_delta(
            query, state["previous_query"], positions, weights=weights
        )
        if delta is not None:
            active = torch.topk(
                delta, k=min(int(active_topk), positions.numel())
            ).indices
            if return_metadata:
                return active, state["valid"].index_select(1, positions)[0], delta
            return active
    valid = state["valid"].index_select(1, positions)[0]
    missing = torch.where(~valid)[0]
    if missing.numel() == positions.numel():
        active = torch.arange(positions.numel(), device=query.device)
        if return_metadata:
            delta = torch.full((positions.numel(),), torch.nan, device=query.device)
            return active, valid, delta
        return active

    active = missing.tolist()
    remaining = max(0, min(int(active_topk), positions.numel()) - len(active))
    stable = None
    stable_delta = None
    if remaining or return_metadata:
        stable = torch.where(valid)[0]
        previous = state["previous_query"].index_select(2, positions[stable])
        difference = query[:, :, stable, :].float() - previous.float()
        if score_mode.startswith("key_diag") and weights is not None:
            stable_delta = difference.square().mul(weights[None, :, None, :]).mean(
                dim=(1, 3)
            )[0]
        else:
            stable_delta = difference.square().mean(dim=(1, 3))[0]
        if remaining:
            active.extend(
                stable[torch.topk(stable_delta, k=remaining).indices].tolist()
            )
    active = torch.tensor(active, dtype=torch.long, device=query.device)
    if return_metadata:
        delta = torch.full((positions.numel(),), torch.nan, device=query.device)
        delta.index_copy_(0, stable, stable_delta)
        return active, valid, delta
    return active


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rotary(query, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    query_rot = query_rot * cos + _rotate_half(query_rot) * sin
    return torch.cat((query_rot, query_pass), dim=-1)


def _hadamard_transform(x):
    size = x.shape[-1]
    if size <= 0 or size & (size - 1):
        raise ValueError(f"Adamas requires a power-of-two head dimension, got {size}")
    if (
        faster_hadamard_transform is not None
        and x.is_cuda
        and os.environ.get("SPARSE_DLM_FASTER_HADAMARD", "true").lower()
        not in {"0", "false", "no", "n"}
    ):
        return faster_hadamard_transform.hadamard_transform(
            x.contiguous(), inplace=False
        )
    leading_shape = x.shape[:-1]
    output = x
    width = 1
    while width < size:
        output = output.reshape(*leading_shape, -1, 2, width)
        left, right = output.unbind(dim=-2)
        output = torch.cat((left + right, left - right), dim=-1)
        width *= 2
    return output.reshape_as(x) / math.sqrt(size)


def _adamas_prefix_indices(
    query,
    key,
    token_budget,
    chunk_size=256,
    use_triton=True,
    bucket_thresholds=None,
):
    """Select a shared prefix KV set from the union of query candidates."""
    prefix_length = key.shape[-2]
    budget = min(int(token_budget), prefix_length)
    if budget >= prefix_length:
        return torch.arange(prefix_length, device=key.device)
    if budget <= 0:
        return torch.empty(0, dtype=torch.long, device=key.device)

    query_edges, key_edges = bucket_thresholds or (
        (-1.35, 0.0, 1.35),
        (-2.26, 0.0, 2.26),
    )
    query_code = torch.bucketize(
        _hadamard_transform(query), query.new_tensor(query_edges), out_int32=True
    )
    key_code = torch.bucketize(
        _hadamard_transform(key), key.new_tensor(key_edges), out_int32=True
    )
    batch_size, query_heads, query_length, head_dim = query_code.shape
    key_heads = key_code.shape[1]
    if batch_size != 1 or query_heads % key_heads:
        raise ValueError("Adamas prefix selection requires batch_size=1 and valid GQA heads")
    query_groups = query_code.reshape(
        batch_size,
        key_heads,
        query_heads // key_heads,
        query_length,
        head_dim,
    )

    local_budget = max(1, math.ceil(budget / (query_heads * query_length)))
    query_distances = query_indices = None
    scores = []
    for start in range(0, prefix_length, chunk_size):
        chunk = key_code[:, :, start : start + chunk_size]
        flat_distances = adamas_distances(query_code, chunk) if use_triton else None
        if flat_distances is None:
            distances = (
                query_groups[..., None, :] - chunk[:, :, None, None, :, :]
            ).abs().sum(dim=-1)
            flat_distances = distances[0].reshape(-1, chunk.shape[-2])
            scores.append(distances.amin(dim=(1, 2, 3)))
        else:
            scores.append(flat_distances.amin(dim=0, keepdim=True))
        chunk_length = chunk.shape[-2]
        chunk_distances = flat_distances
        chunk_indices = torch.arange(
            start, start + chunk_length, device=key.device
        ).expand_as(chunk_distances)
        if query_distances is not None:
            chunk_distances = torch.cat((query_distances, chunk_distances), dim=1)
            chunk_indices = torch.cat((query_indices, chunk_indices), dim=1)
        keep = min(local_budget, chunk_distances.shape[1])
        query_distances, local_indices = torch.topk(
            chunk_distances, keep, dim=1, largest=False
        )
        query_indices = chunk_indices.gather(1, local_indices)
    scores = torch.cat(scores, dim=-1)
    indices = torch.unique(query_indices.flatten())
    if indices.numel() < budget:
        remaining_scores = scores[0].float()
        remaining_scores[indices] = torch.inf
        fill = torch.topk(
            remaining_scores, budget - indices.numel(), largest=False
        ).indices
        indices = torch.cat((indices, fill))
    return indices.sort().values


def _compact_prefix_cache(
    model,
    cache,
    prefix_length,
    captured_queries,
    block_position_ids,
    token_budget,
    chunk_size,
    use_triton_adamas=True,
):
    prefix_cache = _legacy_prefix_cache(cache, prefix_length)
    if not prefix_cache:
        return prefix_cache, ()
    if prefix_length <= token_budget:
        indices = torch.arange(prefix_length, device=prefix_cache[0][0].device)
        return prefix_cache, tuple(indices for _ in prefix_cache)

    cos, sin = model.model.rotary_emb(captured_queries[0], block_position_ids)
    selection_layer = len(prefix_cache) - 1
    query = captured_queries[selection_layer]
    if query is None:
        raise RuntimeError("Failed to capture a layer query during dense refresh")
    key = prefix_cache[selection_layer][0]
    indices = _adamas_prefix_indices(
        _apply_rotary(query, cos, sin),
        key,
        token_budget,
        chunk_size,
        use_triton=use_triton_adamas,
        bucket_thresholds=ADAMAS_BUCKET_THRESHOLDS.get(model.config.model_type),
    )
    compact_cache = tuple(
        (
            key.index_select(2, indices).contiguous(),
            value.index_select(2, indices).contiguous(),
        )
        for key, value in prefix_cache
    )
    return compact_cache, tuple(indices for _ in prefix_cache)
