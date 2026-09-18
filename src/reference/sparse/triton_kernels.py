"""Triton kernels used by the sparse diffusion runtime."""

import torch
import triton
import triton.language as tl


_MOE_BLOCK_M = 32
_MOE_BLOCK_N = 128
_MOE_BLOCK_K = 64


@triton.jit
def _moe_route_kernel(
    expert_ids,
    counts,
    offsets,
    order,
    total_rows: tl.constexpr,
    num_experts: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    tl.store(counts + row, 0, mask=row < num_experts)
    tl.debug_barrier()

    valid = row < total_rows
    expert = tl.load(expert_ids + row, mask=valid, other=0).to(tl.int32)
    rank = tl.atomic_add(counts + expert, 1, mask=valid)
    tl.debug_barrier()

    destination = rank
    prefix = tl.zeros((BLOCK,), dtype=tl.int32)
    for expert_idx in range(num_experts):
        count = tl.load(counts + expert_idx)
        destination += tl.where(expert_idx < expert, count, 0)
        prefix += tl.where(expert_idx < row, count, 0)
    tl.store(order + destination, row, mask=valid)
    tl.store(offsets + row, prefix, mask=row <= num_experts)


def _route_moe(expert_ids, num_experts):
    total_rows = expert_ids.numel()
    counts = torch.empty(num_experts, dtype=torch.int32, device=expert_ids.device)
    offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=expert_ids.device)
    order = torch.empty(total_rows, dtype=torch.long, device=expert_ids.device)
    _moe_route_kernel[(1,)](
        expert_ids,
        counts,
        offsets,
        order,
        total_rows=total_rows,
        num_experts=num_experts,
        BLOCK=triton.next_power_of_2(max(total_rows, num_experts + 1)),
        num_warps=4,
    )
    return counts, offsets, order


@triton.jit
def _rotary_kernel(X, C, S, Y, s0, s1, s2, s3,
                   c0, c1, c2, t0, t1, t2,
                   H: tl.constexpr, L, D: tl.constexpr,
                   R: tl.constexpr, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    pos = i // D % L
    h = i // (D * L) % H
    b = i // (D * L * H)
    x = tl.load(X + b*s0 + h*s1 + pos*s2 + d*s3, i < N, other=0)
    partner = tl.where(d < R//2, d + R//2, d - R//2)
    v = tl.load(X + b*s0 + h*s1 + pos*s2 + partner*s3,
                (i < N) & (d < R), other=0)
    v = tl.where(d < R//2, -v, v)
    c = tl.load(C + b*c0 + pos*c1 + d*c2, (i < N) & (d < R), other=0)
    s = tl.load(S + b*t0 + pos*t1 + d*t2, (i < N) & (d < R), other=0)
    a = (x.to(tl.float32)*c.to(tl.float32)).to(x.dtype)
    z = (v.to(tl.float32)*s.to(tl.float32)).to(x.dtype)
    y = (a.to(tl.float32) + z.to(tl.float32)).to(x.dtype)
    tl.store(Y + i, tl.where(d < R, y, x), i < N)


def rotary_embedding(x, cos, sin):
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _rotary_kernel[(triton.cdiv(x.numel(), 256),)](
        x, cos, sin, output, *x.stride(), *cos.stride(), *sin.stride(),
        H=x.shape[1], L=x.shape[2], D=x.shape[3], R=cos.shape[-1],
        N=x.numel(), BLOCK=256, enable_fp_fusion=False,
    )
    return output


@triton.jit
def _adamas_distance_kernel(
    query,
    key,
    output,
    query_stride_1,
    query_stride_2,
    query_stride_3,
    key_stride_1,
    key_stride_2,
    key_stride_3,
    query_length: tl.constexpr,
    prefix_length: tl.constexpr,
    query_heads: tl.constexpr,
    key_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_row = tl.program_id(0)
    key_block = tl.program_id(1)
    query_head = query_row // query_length
    query_position = query_row % query_length
    key_head = query_head // (query_heads // key_heads)
    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, BLOCK_D)
    query_offsets = (
        query_head * query_stride_1
        + query_position * query_stride_2
        + offsets_d * query_stride_3
    )
    key_offsets = (
        key_head * key_stride_1
        + offsets_n[:, None] * key_stride_2
        + offsets_d[None, :] * key_stride_3
    )
    query_values = tl.load(query + query_offsets, mask=offsets_d < head_dim)
    key_values = tl.load(
        key + key_offsets,
        mask=(offsets_n[:, None] < prefix_length)
        & (offsets_d[None, :] < head_dim),
    )
    distances = tl.sum(
        tl.abs(query_values[None, :] - key_values), axis=1
    )
    output_offsets = query_row * prefix_length + offsets_n
    tl.store(
        output + output_offsets,
        distances,
        mask=offsets_n < prefix_length,
    )


@triton.jit
def _losa_query_delta_kernel(
    query,
    previous,
    positions,
    weights,
    output,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    query_stride_3,
    previous_stride_0,
    previous_stride_1,
    previous_stride_2,
    previous_stride_3,
    weight_stride_0,
    weight_stride_1,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    WEIGHTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    query_position = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    head = offsets // head_dim
    dim = offsets % head_dim
    valid = offsets < heads * head_dim
    previous_position = tl.load(positions + query_position)
    query_offsets = (
        head * query_stride_1
        + query_position * query_stride_2
        + dim * query_stride_3
    )
    previous_offsets = (
        head * previous_stride_1
        + previous_position * previous_stride_2
        + dim * previous_stride_3
    )
    current = tl.load(query + query_offsets, mask=valid).to(tl.float32)
    old = tl.load(previous + previous_offsets, mask=valid).to(tl.float32)
    squared = tl.where(valid, (current - old) * (current - old), 0.0)
    if WEIGHTED:
        weight = tl.load(
            weights + head * weight_stride_0 + dim * weight_stride_1,
            mask=valid,
        ).to(tl.float32)
        squared *= weight
    tl.store(output + query_position, tl.sum(squared) / (heads * head_dim))


@triton.jit
def _attention_output_lse_kernel(
    query,
    key,
    value,
    attention_mask,
    output,
    output_lse,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    query_stride_3,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    key_stride_3,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    value_stride_3,
    mask_stride_0,
    mask_stride_1,
    mask_stride_2,
    mask_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    lse_stride_0,
    lse_stride_1,
    lse_stride_2,
    scale,
    query_length: tl.constexpr,
    prefix_length: tl.constexpr,
    query_heads: tl.constexpr,
    key_heads: tl.constexpr,
    head_dim: tl.constexpr,
    MASK_BOOL: tl.constexpr,
    MASK_HEADS_ONE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    query_head = batch_head % query_heads
    batch = batch_head // query_heads
    key_head = query_head // (query_heads // key_heads)
    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_offsets = (
        batch * query_stride_0
        + query_head * query_stride_1
        + offsets_m[:, None] * query_stride_2
        + offsets_d[None, :] * query_stride_3
    )
    q = tl.load(
        query + query_offsets,
        mask=(offsets_m[:, None] < query_length)
        & (offsets_d[None, :] < head_dim),
        other=0.0,
    )
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, prefix_length, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offsets_n < prefix_length
        key_offsets = (
            batch * key_stride_0
            + key_head * key_stride_1
            + offsets_n[:, None] * key_stride_2
            + offsets_d[None, :] * key_stride_3
        )
        k = tl.load(
            key + key_offsets,
            mask=valid_n[:, None] & (offsets_d[None, :] < head_dim),
        )
        # Match the reference path: torch.matmul and the following scale
        # stay in the model dtype before softmax promotes scores to FP32.
        scores = (tl.dot(q, tl.trans(k)) * scale).to(q.dtype).to(tl.float32)
        mask_head = 0 if MASK_HEADS_ONE else query_head
        mask_offsets = (
            batch * mask_stride_0
            + mask_head * mask_stride_1
            + offsets_m[:, None] * mask_stride_2
            + offsets_n[None, :] * mask_stride_3
        )
        valid = (offsets_m[:, None] < query_length) & valid_n[None, :]
        mask_values = tl.load(attention_mask + mask_offsets, mask=valid)
        if MASK_BOOL:
            scores = tl.where(valid & mask_values, scores, -float("inf"))
        else:
            scores = tl.where(valid, scores + mask_values, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        value_offsets = (
            batch * value_stride_0
            + key_head * value_stride_1
            + offsets_n[:, None] * value_stride_2
            + offsets_d[None, :] * value_stride_3
        )
        v = tl.load(
            value + value_offsets,
            mask=valid_n[:, None] & (offsets_d[None, :] < head_dim),
        )
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probabilities.to(v.dtype), v
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = new_max

    output_offsets = (
        batch * output_stride_0
        + query_head * output_stride_1
        + offsets_m[:, None] * output_stride_2
        + offsets_d[None, :] * output_stride_3
    )
    tl.store(
        output + output_offsets,
        accumulator / running_sum[:, None],
        mask=(offsets_m[:, None] < query_length)
        & (offsets_d[None, :] < head_dim),
    )
    tl.store(
        output_lse
        + batch * lse_stride_0
        + query_head * lse_stride_1
        + offsets_m * lse_stride_2,
        running_max + tl.log(running_sum),
        mask=offsets_m < query_length,
    )


@triton.jit
def _block_causal_prefill_kernel(
    query,
    key,
    value,
    output,
    query_stride_0,
    query_stride_1,
    query_stride_2,
    query_stride_3,
    key_stride_0,
    key_stride_1,
    key_stride_2,
    key_stride_3,
    value_stride_0,
    value_stride_1,
    value_stride_2,
    value_stride_3,
    output_stride_0,
    output_stride_1,
    output_stride_2,
    output_stride_3,
    scale,
    query_length: tl.constexpr,
    prefix_length: tl.constexpr,
    query_heads: tl.constexpr,
    key_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_length: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    query_head = batch_head % query_heads
    batch = batch_head // query_heads
    key_head = query_head // (query_heads // key_heads)
    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_offsets = (
        batch * query_stride_0
        + query_head * query_stride_1
        + offsets_m[:, None] * query_stride_2
        + offsets_d[None, :] * query_stride_3
    )
    q = tl.load(
        query + query_offsets,
        mask=(offsets_m[:, None] < query_length)
        & (offsets_d[None, :] < head_dim),
        other=0.0,
    )
    running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    visible_length = (
        prefix_length
        - query_length
        + (offsets_m // block_length + 1) * block_length
    )

    visible_end = tl.minimum(
        prefix_length,
        prefix_length
        - query_length
        + ((query_block * BLOCK_M + BLOCK_M + block_length - 1) // block_length)
        * block_length,
    )
    for start_n in range(0, visible_end, BLOCK_N):
        offsets_n = start_n + tl.arange(0, BLOCK_N)
        valid_n = offsets_n < prefix_length
        key_offsets = (
            batch * key_stride_0
            + key_head * key_stride_1
            + offsets_n[:, None] * key_stride_2
            + offsets_d[None, :] * key_stride_3
        )
        k = tl.load(
            key + key_offsets,
            mask=valid_n[:, None] & (offsets_d[None, :] < head_dim),
        )
        scores = (tl.dot(q, tl.trans(k)) * scale).to(q.dtype).to(tl.float32)
        valid = (
            (offsets_m[:, None] < query_length)
            & valid_n[None, :]
            & (offsets_n[None, :] < visible_length[:, None])
        )
        scores = tl.where(valid, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        value_offsets = (
            batch * value_stride_0
            + key_head * value_stride_1
            + offsets_n[:, None] * value_stride_2
            + offsets_d[None, :] * value_stride_3
        )
        v = tl.load(
            value + value_offsets,
            mask=valid_n[:, None] & (offsets_d[None, :] < head_dim),
        )
        accumulator = accumulator * old_scale[:, None] + tl.dot(
            probabilities.to(v.dtype), v
        )
        running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
        running_max = new_max

    output_offsets = (
        batch * output_stride_0
        + query_head * output_stride_1
        + offsets_m[:, None] * output_stride_2
        + offsets_d[None, :] * output_stride_3
    )
    tl.store(
        output + output_offsets,
        accumulator / running_sum[:, None],
        mask=(offsets_m[:, None] < query_length)
        & (offsets_d[None, :] < head_dim),
    )


def adamas_distances(query_code, key_code):
    if not (
        query_code.is_cuda
        and key_code.is_cuda
        and query_code.device == key_code.device
        and query_code.ndim == key_code.ndim == 4
        and query_code.shape[0] == key_code.shape[0] == 1
        and query_code.shape[-1] == key_code.shape[-1]
    ):
        raise ValueError("Adamas Triton inputs must be compatible CUDA tensors")
    _, query_heads, query_length, head_dim = query_code.shape
    _, key_heads, prefix_length, _ = key_code.shape
    if query_heads % key_heads:
        raise ValueError("Adamas requires query heads divisible by key heads")
    output = torch.empty(
        query_heads * query_length,
        prefix_length,
        dtype=torch.int32 if query_code.dtype == torch.int32 else torch.float32,
        device=query_code.device,
    )
    block_n = 32
    _adamas_distance_kernel[
        (query_heads * query_length, triton.cdiv(prefix_length, block_n))
    ](
        query_code,
        key_code,
        output,
        *query_code.stride()[1:],
        *key_code.stride()[1:],
        query_length=query_length,
        prefix_length=prefix_length,
        query_heads=query_heads,
        key_heads=key_heads,
        head_dim=head_dim,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


def losa_query_delta(query, previous_query, positions, weights=None):
    if not (
        query.is_cuda
        and previous_query.is_cuda
        and positions.is_cuda
        and query.device == previous_query.device == positions.device
        and query.shape[0] == previous_query.shape[0] == 1
        and query.shape[1] == previous_query.shape[1]
        and query.shape[-1] == previous_query.shape[-1]
        and (
            weights is None
            or (
                weights.is_cuda
                and weights.device == query.device
                and weights.shape == (query.shape[1], query.shape[-1])
            )
        )
    ):
        raise ValueError("LoSA delta inputs must be compatible CUDA tensors")
    heads, query_length, head_dim = query.shape[1:]
    output = torch.empty(query_length, dtype=torch.float32, device=query.device)
    _losa_query_delta_kernel[(query_length,)](
        query,
        previous_query,
        positions,
        weights if weights is not None else query,
        output,
        *query.stride(),
        *previous_query.stride(),
        *(weights.stride() if weights is not None else (0, 0)),
        heads=heads,
        head_dim=head_dim,
        WEIGHTED=weights is not None,
        BLOCK=triton.next_power_of_2(heads * head_dim),
        num_warps=8,
    )
    return output


def attention_output_lse(query, key, value, attention_mask):
    if not (
        query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.is_cuda
        and value.is_cuda
        and attention_mask is not None
        and attention_mask.is_cuda
        and query.device == key.device == value.device == attention_mask.device
        and query.dtype == key.dtype == value.dtype
        and query.ndim == key.ndim == value.ndim == attention_mask.ndim == 4
        and query.shape[0] == key.shape[0] == value.shape[0]
        and key.shape == value.shape
        and query.shape[-1] == key.shape[-1]
        and query.shape[1] % key.shape[1] == 0
        and attention_mask.shape[0] == query.shape[0]
        and attention_mask.shape[1] in (1, query.shape[1])
        and attention_mask.shape[2] == query.shape[2]
        and attention_mask.shape[3] == key.shape[2]
    ):
        raise ValueError("LoSA attention inputs must be compatible CUDA tensors")
    batch, query_heads, query_length, head_dim = query.shape
    key_heads, prefix_length = key.shape[1:3]
    output = torch.empty_like(query)
    output_lse = torch.empty(
        batch,
        query_heads,
        query_length,
        dtype=torch.float32,
        device=query.device,
    )
    block_n = 64
    block_m = 16
    _attention_output_lse_kernel[
        (triton.cdiv(query_length, block_m), batch * query_heads)
    ](
        query,
        key,
        value,
        attention_mask,
        output,
        output_lse,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *attention_mask.stride(),
        *output.stride(),
        *output_lse.stride(),
        head_dim**-0.5,
        query_length=query_length,
        prefix_length=prefix_length,
        query_heads=query_heads,
        key_heads=key_heads,
        head_dim=head_dim,
        MASK_BOOL=attention_mask.dtype == torch.bool,
        MASK_HEADS_ONE=attention_mask.shape[1] == 1,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output, output_lse


def block_causal_prefill(query, key, value, block_length=32):
    if not (
        query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.is_cuda
        and value.is_cuda
        and query.device == key.device == value.device
        and query.dtype == key.dtype == value.dtype
        and query.ndim == key.ndim == value.ndim == 4
        and query.shape[0] == key.shape[0] == value.shape[0]
        and key.shape == value.shape
        and query.shape[-1] == key.shape[-1]
        and query.shape[1] % key.shape[1] == 0
    ):
        raise ValueError("prefill attention inputs must be compatible CUDA tensors")
    batch, query_heads, query_length, head_dim = query.shape
    key_heads, prefix_length = key.shape[1:3]
    if (
        block_length <= 0
        or query_length % block_length
        or (prefix_length - query_length) % block_length
    ):
        raise ValueError("prefill attention requires aligned blocks")
    output = torch.empty_like(query)
    block_m = 32
    _block_causal_prefill_kernel[
        (triton.cdiv(query_length, block_m), batch * query_heads)
    ](
        query,
        key,
        value,
        output,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        head_dim**-0.5,
        query_length=query_length,
        prefix_length=prefix_length,
        query_heads=query_heads,
        key_heads=key_heads,
        head_dim=head_dim,
        block_length=block_length,
        BLOCK_M=block_m,
        BLOCK_N=64,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


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


def fused_kv_index_copy_(
    key_cache,
    value_cache,
    positions,
    key_states,
    value_states,
    prefix_length,
):
    if key_states.shape[-2] == 0:
        return
    _, heads, selected, head_dim = key_states.shape
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


@triton.jit
def _rms_norm_kernel(
    x,
    weight,
    output,
    width: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    values = tl.load(
        x + row * width + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    variance = tl.sum(
        tl.where(mask, values * values, 0.0), axis=0
    ) / width
    weights = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output + row * width + offsets,
        values * tl.rsqrt(variance + eps) * weights,
        mask=mask,
    )


def rms_norm(hidden_states, weight, eps):
    hidden_states = hidden_states.contiguous()
    width = hidden_states.shape[-1]
    output = torch.empty_like(hidden_states)
    _rms_norm_kernel[(hidden_states.numel() // width,)](
        hidden_states,
        weight,
        output,
        width=width,
        eps=eps,
        BLOCK_SIZE=triton.next_power_of_2(width),
        num_warps=8 if width >= 2048 else 4,
    )
    return output


@triton.jit
def _fused_swiglu_kernel(
    x,
    gate_weight,
    up_weight,
    output,
    rows: tl.constexpr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offsets_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, hidden_size, BLOCK_K):
        hidden = tl.load(
            x + offsets_m[:, None] * hidden_size + start_k + offsets_k[None, :],
            mask=offsets_m[:, None] < rows,
            other=0.0,
        )
        weight_mask = offsets_n[None, :] < intermediate_size
        gate_values = tl.load(
            gate_weight
            + offsets_n[None, :] * hidden_size
            + start_k
            + offsets_k[:, None],
            mask=weight_mask,
            other=0.0,
        )
        up_values = tl.load(
            up_weight
            + offsets_n[None, :] * hidden_size
            + start_k
            + offsets_k[:, None],
            mask=weight_mask,
            other=0.0,
        )
        gate += tl.dot(hidden, gate_values)
        up += tl.dot(hidden, up_values)
    gate = gate.to(tl.float16)
    up = up.to(tl.float16)
    activated = gate * tl.sigmoid(gate.to(tl.float32)) * up
    tl.store(
        output + offsets_m[:, None] * intermediate_size + offsets_n[None, :],
        activated,
        mask=(offsets_m[:, None] < rows)
        & (offsets_n[None, :] < intermediate_size),
    )


def fused_swiglu(hidden_states, gate_weight, up_weight):
    hidden_size = hidden_states.shape[-1]
    intermediate_size = gate_weight.shape[0]
    rows = hidden_states.numel() // hidden_size
    output = torch.empty(
        (*hidden_states.shape[:-1], intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _fused_swiglu_kernel[(1, triton.cdiv(intermediate_size, 64))](
        hidden_states,
        gate_weight,
        up_weight,
        output,
        rows=rows,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
    )
    return output


@triton.jit
def _moe_gate_up_kernel(
    x_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    activated_ptr,
    tile_experts_ptr,
    tile_offsets_ptr,
    offsets_ptr,
    counts_ptr,
    total_rows,
    hidden_size,
    intermediate_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_id = tl.load(tile_experts_ptr + pid)
    tile_m = pid - tl.load(tile_offsets_ptr + expert_id)
    pid_n = tl.program_id(1)

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + expert_id)
    start = tl.load(offsets_ptr + expert_id)
    row_ids = start + rows
    row_mask = (rows < count) & (row_ids < total_rows)
    col_mask = cols < intermediate_size

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(hidden_size, BLOCK_K)):
        ks = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = ks < hidden_size
        x_ptrs = x_ptr + row_ids[:, None] * hidden_size + ks[None, :]
        x_tile = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        gate_ptrs = (
            gate_weight_ptr
            + expert_id * intermediate_size * hidden_size
            + cols[:, None] * hidden_size
            + ks[None, :]
        )
        up_ptrs = (
            up_weight_ptr
            + expert_id * intermediate_size * hidden_size
            + cols[:, None] * hidden_size
            + ks[None, :]
        )
        gate_tile = tl.load(
            gate_ptrs,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        up_tile = tl.load(
            up_ptrs,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        gate_acc += tl.dot(x_tile, tl.trans(gate_tile), out_dtype=tl.float32)
        up_acc += tl.dot(x_tile, tl.trans(up_tile), out_dtype=tl.float32)

    gate_values = gate_acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    up_values = up_acc.to(x_ptr.dtype.element_ty).to(tl.float32)
    activated = gate_values * tl.sigmoid(gate_values) * up_values
    out_ptrs = activated_ptr + row_ids[:, None] * intermediate_size + cols[None, :]
    tl.store(out_ptrs, activated, mask=row_mask[:, None] & col_mask[None, :])


@triton.jit
def _moe_down_kernel(
    activated_ptr,
    down_weight_ptr,
    out_ptr,
    order_ptr,
    tile_experts_ptr,
    tile_offsets_ptr,
    offsets_ptr,
    counts_ptr,
    total_rows,
    hidden_size,
    intermediate_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    expert_id = tl.load(tile_experts_ptr + pid)
    tile_m = pid - tl.load(tile_offsets_ptr + expert_id)
    pid_n = tl.program_id(1)

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + expert_id)
    start = tl.load(offsets_ptr + expert_id)
    row_ids = start + rows
    row_mask = (rows < count) & (row_ids < total_rows)
    col_mask = cols < hidden_size

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(intermediate_size, BLOCK_K)):
        ks = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = ks < intermediate_size
        activated = tl.load(
            activated_ptr + row_ids[:, None] * intermediate_size + ks[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weight_ptrs = (
            down_weight_ptr
            + expert_id * hidden_size * intermediate_size
            + cols[:, None] * intermediate_size
            + ks[None, :]
        )
        weights = tl.load(
            weight_ptrs,
            mask=col_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(activated, tl.trans(weights), out_dtype=tl.float32)

    tl.store(
        out_ptr
        + tl.load(order_ptr + row_ids, mask=row_mask, other=0)[:, None] * hidden_size
        + cols[None, :],
        acc,
        mask=row_mask[:, None] & col_mask[None, :],
    )


def _triton_moe_infer(self, x, topk_ids, topk_weight):
    topk_ids = topk_ids.reshape(-1, topk_ids.shape[-1])
    topk_weight = topk_weight.reshape_as(topk_ids)
    token_count, top_k = topk_ids.shape
    if token_count == 0:
        return x.new_empty((0, x.shape[-1]))

    gate_weight = self._llada_moe_gate_weight
    up_weight = self._llada_moe_up_weight
    down_weight = self._llada_moe_down_weight
    num_experts, intermediate_size, hidden_size = gate_weight.shape
    flat_ids = topk_ids.reshape(-1)
    if flat_ids.numel() <= 256:
        counts, offsets, order = _route_moe(flat_ids, num_experts)
    else:
        counts = torch.bincount(flat_ids, minlength=num_experts)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        order = torch.argsort(flat_ids)
    sorted_tokens = x.index_select(0, order // top_k).contiguous()
    total_rows = sorted_tokens.shape[0]
    # Launch only occupied expert tiles instead of padding every expert to the
    # most-routed expert's size.
    tile_counts = torch.div(
        counts + _MOE_BLOCK_M - 1, _MOE_BLOCK_M, rounding_mode="floor"
    )
    tile_offsets = torch.cat((tile_counts.new_zeros(1), tile_counts.cumsum(0)))
    total_tiles = int(tile_offsets[-1].item())
    tile_experts = torch.repeat_interleave(tile_counts, output_size=total_tiles)

    activated = torch.empty(
        (total_rows, intermediate_size), dtype=torch.float32, device=x.device
    )
    _moe_gate_up_kernel[
        (total_tiles, triton.cdiv(intermediate_size, _MOE_BLOCK_N))
    ](
        sorted_tokens,
        gate_weight,
        up_weight,
        activated,
        tile_experts,
        tile_offsets,
        offsets,
        counts,
        total_rows,
        hidden_size,
        intermediate_size,
        BLOCK_M=_MOE_BLOCK_M,
        BLOCK_N=_MOE_BLOCK_N,
        BLOCK_K=_MOE_BLOCK_K,
    )

    routed_out = torch.empty(
        (total_rows, hidden_size), dtype=x.dtype, device=x.device
    )
    _moe_down_kernel[
        (total_tiles, triton.cdiv(hidden_size, _MOE_BLOCK_N))
    ](
        activated,
        down_weight,
        routed_out,
        order,
        tile_experts,
        tile_offsets,
        offsets,
        counts,
        total_rows,
        hidden_size,
        intermediate_size,
        BLOCK_M=_MOE_BLOCK_M,
        BLOCK_N=_MOE_BLOCK_N,
        BLOCK_K=_MOE_BLOCK_K,
        num_warps=8,
    )

    return (
        routed_out.view(token_count, top_k, hidden_size)
        .to(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(-1))
        .sum(dim=1)
        .to(x.dtype)
    )
