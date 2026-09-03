"""Small shared Triton kernels for sparse diffusion reference paths."""

import os

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - minimal CPU installs
    triton = None
    tl = None
    TRITON_AVAILABLE = False

TRITON_ENABLED = TRITON_AVAILABLE and os.environ.get(
    "SPARSE_DLM_TRITON", "true"
).lower() not in {"0", "false", "no", "n"}
ADAMAS_ENABLED = TRITON_ENABLED and os.environ.get(
    "SPARSE_DLM_TRITON_ADAMAS", "true"
).lower() not in {"0", "false", "no", "n"}
LOSA_DELTA_ENABLED = TRITON_ENABLED and os.environ.get(
    "SPARSE_DLM_TRITON_LOSA_DELTA", "true"
).lower() not in {"0", "false", "no", "n"}
LOSA_ATTENTION_ENABLED = TRITON_ENABLED and os.environ.get(
    "SPARSE_DLM_TRITON_LOSA_ATTENTION", "true"
).lower() not in {"0", "false", "no", "n"}


if TRITON_AVAILABLE:

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
        output,
        query_stride_0,
        query_stride_1,
        query_stride_2,
        query_stride_3,
        previous_stride_0,
        previous_stride_1,
        previous_stride_2,
        previous_stride_3,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
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


def adamas_distances(query_code, key_code):
    if not (
        ADAMAS_ENABLED
        and query_code.is_cuda
        and key_code.is_cuda
        and query_code.device == key_code.device
        and query_code.ndim == key_code.ndim == 4
        and query_code.shape[0] == key_code.shape[0] == 1
        and query_code.shape[-1] == key_code.shape[-1]
    ):
        return None
    _, query_heads, query_length, head_dim = query_code.shape
    _, key_heads, prefix_length, _ = key_code.shape
    if query_heads % key_heads:
        return None
    output = torch.empty(
        query_heads * query_length,
        prefix_length,
        dtype=torch.int64,
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


def losa_query_delta(query, previous_query, positions):
    if not (
        LOSA_DELTA_ENABLED
        and query.is_cuda
        and previous_query.is_cuda
        and positions.is_cuda
        and query.device == previous_query.device == positions.device
        and query.shape[0] == previous_query.shape[0] == 1
        and query.shape[1] == previous_query.shape[1]
        and query.shape[-1] == previous_query.shape[-1]
    ):
        return None
    heads, query_length, head_dim = query.shape[1:]
    output = torch.empty(query_length, dtype=torch.float32, device=query.device)
    _losa_query_delta_kernel[(query_length,)](
        query,
        previous_query,
        positions,
        output,
        *query.stride(),
        *previous_query.stride(),
        heads=heads,
        head_dim=head_dim,
        BLOCK=triton.next_power_of_2(heads * head_dim),
        num_warps=8,
    )
    return output


def attention_output_lse(query, key, value, attention_mask):
    if not (
        LOSA_ATTENTION_ENABLED
        and query.is_cuda
        and query.dtype == torch.float16
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
        # Compact Prefix Sparse caches are at most the default 256-token
        # budget. PyTorch is already fast there and preserves the combined
        # Prefix+LoSA greedy trajectory at long context.
        and key.shape[2] > 256
    ):
        return None
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
