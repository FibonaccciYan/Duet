"""Sparse attention and prefix-cache operations shared by LLaDA and SDAR."""

import math

import torch

from .triton_kernels import (
    adamas_distances,
    attention_output_lse,
    fused_kv_index_copy_,
    losa_query_delta,
    rotary_embedding,
)


ADAMAS_BUCKET_THRESHOLDS = {
    "llada2_moe": ((-1.35, 0.0, 1.35), (-2.26, 0.0, 2.26)),
    "sdar": ((-1.50, 0.0, 1.49), (-2.87, 0.0, 2.86)),
}


def _sample_with_confidence(model, logits, temperature, top_p, top_k):
    return model._sample_with_temperature_topk_topp(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


def _prefix_from_dynamic_cache(cache, prefix_length):
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
        fused_kv_index_copy_(
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


def _attention_output_lse(query, key, value, attention_mask, _groups=None):
    """Run the canonical Triton prefix-attention implementation."""
    return attention_output_lse(query, key, value, attention_mask)


def _block_attention_output_lse(
    query, key, value, attention_mask, num_key_value_groups
):
    """Compute exact attention for the short current block."""
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
    if state.get("fully_valid", False):
        delta = losa_query_delta(
            query, state["previous_query"], positions, weights=weights
        )
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
        stable_delta = losa_query_delta(
            query.index_select(2, stable),
            state["previous_query"],
            positions.index_select(0, stable),
            weights=weights,
        )
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
    if query.is_cuda:
        return rotary_embedding(query, cos, sin)
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
    if not x.is_cuda:
        raise ValueError("Faster Hadamard requires a CUDA tensor")
    import faster_hadamard_transform

    return faster_hadamard_transform.hadamard_transform(
        x.contiguous(), inplace=False
    )


def _adamas_prefix_indices(
    query,
    key,
    token_budget,
    chunk_size=256,
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
    return _distance_prefix_indices(query_code, key_code, budget, chunk_size)


def _selection_stats(stats, length, budget, local_budget, union_size, selected_size,
                     strict_budget, bypassed=False):
    if stats is not None:
        stats.update(candidate_length=int(length), budget=int(budget),
                     local_budget=int(local_budget), union_size=int(union_size),
                     selected_size=int(selected_size), strict_budget=bool(strict_budget),
                     bypassed=bool(bypassed))


def _distance_prefix_indices(query, key, budget, chunk_size, *,
                             strict_budget=False, selection_stats=None):
    prefix_length = key.shape[-2]
    budget = max(0, min(int(budget), prefix_length))
    if budget == 0 or budget == prefix_length:
        indices = torch.arange(budget, device=key.device)
        _selection_stats(selection_stats, prefix_length, budget, 0, budget, budget,
                         strict_budget, bypassed=True)
        return indices
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    batch_size, query_heads, query_length, _ = query.shape
    key_heads = key.shape[1]
    if batch_size != 1 or query_heads % key_heads:
        raise ValueError("Adamas prefix selection requires batch_size=1 and valid GQA heads")
    local_budget = max(1, math.ceil(budget / (query_heads * query_length)))
    query_distances = query_indices = None
    scores = []
    for start in range(0, prefix_length, chunk_size):
        chunk = key[:, :, start : start + chunk_size]
        flat_distances = adamas_distances(query, chunk)
        scores.append(flat_distances.amin(dim=0, keepdim=True))
        chunk_length = chunk.shape[-2]
        if local_budget == 1:
            chunk_distances, chunk_indices = flat_distances.min(dim=1, keepdim=True)
            chunk_indices.add_(start)
            if query_distances is None:
                query_distances, query_indices = chunk_distances, chunk_indices
            else:
                replace = chunk_distances < query_distances
                query_distances = torch.where(
                    replace, chunk_distances, query_distances
                )
                query_indices = torch.where(replace, chunk_indices, query_indices)
            continue
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
    union_size = indices.numel()
    if indices.numel() < budget:
        remaining_scores = scores[0].float()
        remaining_scores[indices] = torch.inf
        fill = torch.topk(
            remaining_scores, budget - indices.numel(), largest=False
        ).indices
        indices = torch.cat((indices, fill))
    elif strict_budget and indices.numel() > budget:
        # Rank only union members; preserve the original candidate construction.
        keep = scores[0].index_select(0, indices).topk(budget, largest=False).indices
        indices = indices.index_select(0, keep)
    _selection_stats(selection_stats, prefix_length, budget, local_budget,
                     union_size, indices.numel(), strict_budget)
    return indices.sort().values


def _qk_prefix_indices(query, key, token_budget):
    """Select the same per-query union as Adamas using exact QK scores."""
    prefix_length = key.shape[-2]
    budget = min(int(token_budget), prefix_length)
    if budget >= prefix_length:
        return torch.arange(prefix_length, device=key.device)
    if budget <= 0:
        return torch.empty(0, dtype=torch.long, device=key.device)

    batch_size, query_heads, query_length, head_dim = query.shape
    key_heads = key.shape[1]
    if batch_size != 1 or query_heads % key_heads:
        raise ValueError("QK prefix selection requires batch_size=1 and valid GQA heads")
    grouped_query = query.reshape(
        key_heads, query_heads // key_heads * query_length, head_dim
    )
    scores = torch.bmm(grouped_query, key[0].transpose(1, 2)).reshape(
        query_heads * query_length, prefix_length
    )
    local_budget = max(1, math.ceil(budget / scores.shape[0]))
    query_indices = scores.topk(local_budget, dim=1).indices
    indices = torch.unique(query_indices.flatten())
    if indices.numel() < budget:
        remaining_scores = scores.amax(dim=0).float()
        remaining_scores[indices] = -torch.inf
        indices = torch.cat(
            (indices, remaining_scores.topk(budget - indices.numel()).indices)
        )
    return indices.sort().values


def _hadamard_qk_prefix_indices(query, key, token_budget):
    """Select by exact L1 distance between floating-point Hq and Hk."""
    prefix_length = key.shape[-2]
    budget = min(int(token_budget), prefix_length)
    if budget >= prefix_length:
        return torch.arange(prefix_length, device=key.device)
    if budget <= 0:
        return torch.empty(0, dtype=torch.long, device=key.device)
    return _distance_prefix_indices(
        _hadamard_transform(query),
        _hadamard_transform(key),
        budget,
        prefix_length,
    )


def _raw_l1_prefix_indices(query, key, token_budget, *, strict_budget=False,
                           selection_stats=None):
    """Select by exact L1 distance without a Hadamard transform."""
    prefix_length = key.shape[-2]
    budget = min(int(token_budget), prefix_length)
    if budget >= prefix_length:
        _selection_stats(selection_stats, prefix_length, budget, 0, prefix_length,
                         prefix_length, strict_budget, bypassed=True)
        return torch.arange(prefix_length, device=key.device)
    if budget <= 0:
        _selection_stats(selection_stats, prefix_length, 0, 0, 0, 0,
                         strict_budget, bypassed=True)
        return torch.empty(0, dtype=torch.long, device=key.device)
    return _distance_prefix_indices(query, key, budget, prefix_length,
                                    strict_budget=strict_budget, selection_stats=selection_stats)


def _prefix_indices(
    query,
    key,
    token_budget,
    chunk_size=256,
    bucket_thresholds=None,
    strict_budget=False,
    selection_stats=None,
):
    """Select prefix tokens with exact L1 distance in the learned Q/K basis."""
    return _raw_l1_prefix_indices(query, key, token_budget,
                                 strict_budget=strict_budget, selection_stats=selection_stats)


def _compact_prefix_cache(
    model,
    cache,
    prefix_length,
    captured_queries,
    block_position_ids,
    token_budget,
    chunk_size,
    previous_indices=None,
    previous_length=0,
    prefix_start_layer=0,
):
    family = "sdar" if model.config.model_type == "sdar" else "llada"
    strict = getattr(model.config, family + "_prefix_strict_budget", False)
    collector = getattr(model, "_prefix_selection_stats", None)
    # Selection only reads the prefix; gather directly from strided views.
    # Copying the entire history here duplicates KV before discarding most of it.
    prefix_cache = tuple(
        (key[:, :, :prefix_length], value[:, :, :prefix_length])
        for key, value in cache.to_legacy_cache()
    )
    if not prefix_cache:
        return prefix_cache, ()
    budget = min(int(token_budget), prefix_length)
    if prefix_length <= budget:
        indices = torch.arange(prefix_length, device=prefix_cache[0][0].device)
        if collector is not None:
            group = 2 if family == "sdar" and getattr(
                model.config, "sdar_prefix_share_layer_pairs", False) else 1
            for start in range(0, len(prefix_cache), group):
                stats = dict(layer=min(start + group, len(prefix_cache)) - 1,
                             prefix_length=prefix_length)
                _selection_stats(stats, prefix_length, budget, 0, prefix_length,
                                 prefix_length, strict, bypassed=True)
                collector.append(stats)
        return prefix_cache, tuple(indices for _ in prefix_cache)

    group_size = (
        2
        if model.config.model_type == "sdar"
        and getattr(model.config, "sdar_prefix_share_layer_pairs", False)
        else 1
    )
    if group_size == 2 and prefix_start_layer % 2:
        group_size = 1
    cos, sin = model.model.rotary_emb(
        captured_queries[group_size - 1], block_position_ids
    )
    compact_cache = []
    prefix_indices = []
    thresholds = ADAMAS_BUCKET_THRESHOLDS.get(model.config.model_type)
    for start in range(0, len(prefix_cache), group_size):
        representative = min(start + group_size, len(prefix_cache)) - 1
        if representative < prefix_start_layer:
            for key, value in prefix_cache[start : start + group_size]:
                compact_cache.append((key, value))
                prefix_indices.append(
                    torch.arange(prefix_length, device=key.device)
                )
            continue
        query = captured_queries[representative]
        if query is None:
            raise RuntimeError("Failed to capture a layer query during dense refresh")
        key = prefix_cache[representative][0]
        candidates = None
        if previous_indices is not None and previous_length < prefix_length:
            candidates = torch.cat(
                (
                    previous_indices[representative],
                    torch.arange(previous_length, prefix_length, device=key.device),
                )
            )
            key = key.index_select(2, candidates)
        stats = {} if collector is not None else None
        extra = {}
        # Keep the disabled path's call contract compatible with existing hooks.
        if strict or stats is not None:
            extra.update(strict_budget=strict, selection_stats=stats)
        indices = _prefix_indices(
            _apply_rotary(query, cos, sin),
            key,
            token_budget,
            chunk_size,
            bucket_thresholds=thresholds,
            **extra,
        )
        if stats is not None:
            stats.update(layer=representative, prefix_length=prefix_length)
            collector.append(stats)
        if candidates is not None:
            indices = candidates.index_select(0, indices)
        for key, value in prefix_cache[start : start + group_size]:
            compact_cache.append(
                (
                    key.index_select(2, indices).contiguous(),
                    value.index_select(2, indices).contiguous(),
                )
            )
            prefix_indices.append(indices)
    return tuple(compact_cache), tuple(prefix_indices)
