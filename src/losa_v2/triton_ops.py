"""Triton fast paths for LoSA v2.

The kernels intentionally keep scheduling simple.  QUEST scores are written to
a fixed-size workspace and PyTorch performs the final top-k.  Sparse prefix
attention consumes the padded page table directly, avoiding the eager
gather/materialization of selected K/V.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU-only environments
    triton = None
    tl = None


def triton_available() -> bool:
    return triton is not None and torch.cuda.is_available()


if triton is not None:

    @triton.jit
    def _locality_score_kernel(
        query_ptr,
        previous_ptr,
        output_ptr,
        elements_per_row,
        stride_qr,
        stride_pr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < elements_per_row
        query = tl.load(
            query_ptr + row * stride_qr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        previous = tl.load(
            previous_ptr + row * stride_pr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        delta = query - previous
        score = tl.sum(delta * delta, axis=0) / elements_per_row
        tl.store(output_ptr + row, score)


    @triton.jit
    def _quest_group_mean_score_kernel(
        q_ptr,
        k_min_ptr,
        k_max_ptr,
        score_ptr,
        num_pages,
        stride_qa,
        stride_qh,
        stride_qd,
        stride_mp,
        stride_mh,
        stride_md,
        stride_sa,
        stride_sh,
        stride_sp,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_kv = tl.program_id(0)
        page_block = tl.program_id(1)
        row = row_kv // NUM_KV_HEADS
        kv_head = row_kv % NUM_KV_HEADS

        d = tl.arange(0, BLOCK_D)
        d_mask = d < HEAD_DIM
        q_mean = tl.zeros((BLOCK_D,), tl.float32)
        for group_offset in range(GROUP_SIZE):
            q_head = kv_head * GROUP_SIZE + group_offset
            q = tl.load(
                q_ptr + row * stride_qa + q_head * stride_qh + d * stride_qd,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)
            q_mean += q
        q_mean /= GROUP_SIZE
        q_positive = tl.maximum(q_mean, 0.0)
        q_negative = tl.minimum(q_mean, 0.0)

        pages = page_block * BLOCK_P + tl.arange(0, BLOCK_P)
        page_mask = pages < num_pages
        metadata_offset = (
            pages[:, None] * stride_mp
            + kv_head * stride_mh
            + d[None, :] * stride_md
        )
        load_mask = page_mask[:, None] & d_mask[None, :]
        k_min = tl.load(k_min_ptr + metadata_offset, mask=load_mask, other=0.0).to(
            tl.float32
        )
        k_max = tl.load(k_max_ptr + metadata_offset, mask=load_mask, other=0.0).to(
            tl.float32
        )
        score = tl.sum(
            k_max * q_positive[None, :] + k_min * q_negative[None, :],
            axis=1,
        )
        tl.store(
            score_ptr
            + row * stride_sa
            + kv_head * stride_sh
            + pages * stride_sp,
            score,
            mask=page_mask,
        )


    @triton.jit
    def _paged_prefix_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        pages_ptr,
        out_ptr,
        lse_ptr,
        prefix_length,
        softmax_scale,
        stride_qa,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_pt,
        stride_ps,
        stride_oa,
        stride_oh,
        stride_od,
        stride_la,
        stride_lh,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
        MAX_PAGES: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // GROUP_SIZE

        d = tl.arange(0, BLOCK_D)
        d_mask = d < HEAD_DIM
        q = tl.load(
            q_ptr + row * stride_qa + q_head * stride_qh + d * stride_qd,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)

        running_max = tl.full((1,), -float("inf"), tl.float32)
        running_sum = tl.zeros((1,), tl.float32)
        accumulator = tl.zeros((BLOCK_D,), tl.float32)
        max_tokens = MAX_PAGES * PAGE_SIZE

        for token_start in tl.range(0, max_tokens, BLOCK_N, num_stages=1):
            packed_token = token_start + tl.arange(0, BLOCK_N)
            page_slot = packed_token // PAGE_SIZE
            token_in_page = packed_token % PAGE_SIZE
            page = tl.load(
                pages_ptr + kv_head * stride_pt + page_slot * stride_ps,
                mask=page_slot < MAX_PAGES,
                other=-1,
            )
            token = page * PAGE_SIZE + token_in_page
            token_valid = (
                (packed_token < max_tokens)
                & (page >= 0)
                & (token < prefix_length)
            )

            kv_offset = (
                token[:, None] * stride_kt
                + kv_head * stride_kh
                + d[None, :] * stride_kd
            )
            load_mask = token_valid[:, None] & d_mask[None, :]
            key = tl.load(k_ptr + kv_offset, mask=load_mask, other=0.0).to(
                tl.float32
            )
            score = tl.sum(key * q[None, :], axis=1) * softmax_scale
            score = tl.where(token_valid, score, -float("inf"))

            tile_max = tl.max(score, axis=0)
            next_max = tl.maximum(running_max, tile_max)
            old_scale = tl.exp(running_max - next_max)
            probability = tl.exp(score - next_max)
            probability = tl.where(token_valid, probability, 0.0)

            value = tl.load(v_ptr + kv_offset, mask=load_mask, other=0.0).to(
                tl.float32
            )
            accumulator = accumulator * old_scale + tl.sum(
                probability[:, None] * value, axis=0
            )
            running_sum = running_sum * old_scale + tl.sum(probability, axis=0)
            running_max = next_max

        output = accumulator / running_sum
        tl.store(
            out_ptr + row * stride_oa + q_head * stride_oh + d * stride_od,
            output,
            mask=d_mask,
        )
        scalar_offset = tl.arange(0, 1)
        tl.store(
            lse_ptr + row * stride_la + q_head * stride_lh + scalar_offset,
            running_max + tl.log(running_sum),
        )


    @triton.jit
    def _dense_attention_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        lse_ptr,
        key_length,
        softmax_scale,
        stride_qa,
        stride_qh,
        stride_qd,
        stride_kt,
        stride_kh,
        stride_kd,
        stride_oa,
        stride_oh,
        stride_od,
        stride_la,
        stride_lh,
        GROUP_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        q_head = tl.program_id(1)
        kv_head = q_head // GROUP_SIZE
        d = tl.arange(0, BLOCK_D)
        n = tl.arange(0, BLOCK_N)
        d_mask = d < HEAD_DIM
        n_mask = n < key_length
        q = tl.load(
            q_ptr + row * stride_qa + q_head * stride_qh + d * stride_qd,
            mask=d_mask,
            other=0.0,
        ).to(tl.float32)
        kv_offset = (
            n[:, None] * stride_kt
            + kv_head * stride_kh
            + d[None, :] * stride_kd
        )
        load_mask = n_mask[:, None] & d_mask[None, :]
        key = tl.load(k_ptr + kv_offset, mask=load_mask, other=0.0).to(tl.float32)
        score = tl.sum(key * q[None, :], axis=1) * softmax_scale
        score = tl.where(n_mask, score, -float("inf"))
        maximum = tl.max(score, axis=0)
        probability = tl.exp(score - maximum)
        probability = tl.where(n_mask, probability, 0.0)
        denominator = tl.sum(probability, axis=0)
        value = tl.load(v_ptr + kv_offset, mask=load_mask, other=0.0).to(
            tl.float32
        )
        output = tl.sum(probability[:, None] * value, axis=0) / denominator
        tl.store(
            out_ptr + row * stride_oa + q_head * stride_oh + d * stride_od,
            output,
            mask=d_mask,
        )
        scalar_offset = tl.arange(0, 1)
        tl.store(
            lse_ptr + row * stride_la + q_head * stride_lh + scalar_offset,
            maximum + tl.log(denominator),
        )



    @triton.jit
    def _mark_selected_pages_kernel(
        selected_ptr,
        page_mask_ptr,
        total_entries,
        num_pages,
        stride_sr,
        stride_sh,
        stride_sk,
        stride_mh,
        stride_mp,
        NUM_SELECTED_HEADS: tl.constexpr,
        PAGE_BUDGET: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total_entries
        per_row = NUM_SELECTED_HEADS * PAGE_BUDGET
        row_remainder = offsets % per_row
        selected_head = row_remainder // PAGE_BUDGET
        selected_slot = row_remainder % PAGE_BUDGET
        row = offsets // per_row
        page = tl.load(
            selected_ptr
            + row * stride_sr
            + selected_head * stride_sh
            + selected_slot * stride_sk,
            mask=valid,
            other=0,
        )
        kv_head = selected_head // GROUP_SIZE
        valid &= (page >= 0) & (page < num_pages)
        tl.atomic_xchg(
            page_mask_ptr + kv_head * stride_mh + page * stride_mp,
            1,
            mask=valid,
        )


    @triton.jit
    def _compact_page_mask_kernel(
        page_mask_ptr,
        output_ptr,
        num_pages,
        max_pages,
        stride_mh,
        stride_mp,
        stride_oh,
        stride_op,
        BLOCK_P: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        pages = tl.arange(0, BLOCK_P)
        valid_page = pages < num_pages
        selected = tl.load(
            page_mask_ptr + kv_head * stride_mh + pages * stride_mp,
            mask=valid_page,
            other=0,
        ).to(tl.int32)
        position = tl.cumsum(selected, axis=0) - 1
        store_mask = valid_page & (selected != 0) & (position < max_pages)
        tl.store(
            output_ptr + kv_head * stride_oh + position * stride_op,
            pages,
            mask=store_mask,
        )


def locality_scores_triton(
    query: torch.Tensor,
    previous_query: torch.Tensor,
) -> torch.Tensor:
    if not triton_available():
        raise RuntimeError("Triton CUDA support is unavailable")
    if query.shape != previous_query.shape or query.ndim != 3:
        raise ValueError("query and previous_query must share [B,H,D] shape")
    query = query.contiguous()
    previous_query = previous_query.contiguous()
    rows = query.shape[0]
    elements_per_row = query.shape[1] * query.shape[2]
    block = triton.next_power_of_2(elements_per_row)
    if block > 65536:
        raise ValueError("locality Triton row is too large")
    output = torch.empty(rows, device=query.device, dtype=torch.float32)
    _locality_score_kernel[(rows,)](
        query,
        previous_query,
        output,
        elements_per_row,
        query.stride(0),
        previous_query.stride(0),
        BLOCK=block,
        num_warps=8,
    )
    return output


def select_active_rows_triton(
    query: torch.Tensor,
    previous_query: torch.Tensor,
    active_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    locality = locality_scores_triton(query, previous_query)
    count = min(int(active_count), query.shape[0])
    active = locality.topk(count, sorted=False).indices.sort().values
    return active, locality



def quest_group_mean_scores_triton(
    q: torch.Tensor,
    k_min: torch.Tensor,
    k_max: torch.Tensor,
) -> torch.Tensor:
    if not triton_available():
        raise RuntimeError("Triton CUDA support is unavailable")
    if not (q.is_cuda and k_min.is_cuda and k_max.is_cuda):
        raise ValueError("QUEST Triton inputs must be CUDA tensors")
    if q.ndim != 3 or k_min.ndim != 3 or k_max.shape != k_min.shape:
        raise ValueError("invalid QUEST tensor shapes")
    rows, num_q_heads, head_dim = q.shape
    num_pages, num_kv_heads, metadata_dim = k_min.shape
    if head_dim != metadata_dim or num_q_heads % num_kv_heads:
        raise ValueError("incompatible QUEST Q/metadata layouts")
    if head_dim > 256:
        raise ValueError("QUEST Triton supports head_dim <= 256")

    q = q.contiguous()
    k_min = k_min.contiguous()
    k_max = k_max.contiguous()
    scores = torch.empty(
        (rows, num_kv_heads, num_pages),
        device=q.device,
        dtype=torch.float32,
    )
    block_d = triton.next_power_of_2(head_dim)
    block_p = 16
    grid = (rows * num_kv_heads, triton.cdiv(num_pages, block_p))
    _quest_group_mean_score_kernel[grid](
        q,
        k_min,
        k_max,
        scores,
        num_pages,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_min.stride(0),
        k_min.stride(1),
        k_min.stride(2),
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=num_q_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_P=block_p,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return scores


def compact_prefix_attention_triton(
    q_active: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    compact_pages: torch.Tensor,
    *,
    page_size: int,
    prefix_length: int,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not triton_available():
        raise RuntimeError("Triton CUDA support is unavailable")
    if not all(
        tensor.is_cuda
        for tensor in (q_active, k_prefix, v_prefix, compact_pages)
    ):
        raise ValueError("paged attention Triton inputs must be CUDA tensors")
    rows, num_q_heads, head_dim = q_active.shape
    num_kv_heads = k_prefix.shape[1]
    if num_q_heads % num_kv_heads:
        raise ValueError("Hq must be divisible by Hkv")
    if head_dim > 256:
        raise ValueError("paged attention Triton supports head_dim <= 256")
    if page_size <= 0 or page_size > 64:
        raise ValueError("paged attention Triton supports page_size in [1, 64]")
    if compact_pages.shape[0] != num_kv_heads:
        raise ValueError("page table KV-head dimension mismatch")

    q_active = q_active.contiguous()
    k_prefix = k_prefix.contiguous()
    v_prefix = v_prefix.contiguous()
    compact_pages = compact_pages.contiguous()
    output = torch.empty_like(q_active)
    lse = torch.empty(
        (rows, num_q_heads), device=q_active.device, dtype=torch.float32
    )
    block_d = triton.next_power_of_2(head_dim)
    max_pages = compact_pages.shape[1]
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    grid = (rows, num_q_heads)
    _paged_prefix_attention_kernel[grid](
        q_active,
        k_prefix,
        v_prefix,
        compact_pages,
        output,
        lse,
        prefix_length,
        scale,
        q_active.stride(0),
        q_active.stride(1),
        q_active.stride(2),
        k_prefix.stride(0),
        k_prefix.stride(1),
        k_prefix.stride(2),
        compact_pages.stride(0),
        compact_pages.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        GROUP_SIZE=num_q_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        MAX_PAGES=max_pages,
        BLOCK_D=block_d,
        BLOCK_N=16,
        num_warps=4,
    )
    return output, lse


def dense_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not triton_available():
        raise RuntimeError("Triton CUDA support is unavailable")
    if q.ndim != 3 or k.ndim != 3 or k.shape != v.shape:
        raise ValueError("q/k/v must use [tokens,heads,dim] layout")
    rows, num_q_heads, head_dim = q.shape
    key_length, num_kv_heads, kv_dim = k.shape
    if head_dim != kv_dim or num_q_heads % num_kv_heads:
        raise ValueError("incompatible dense attention layouts")
    if key_length <= 0 or key_length > 128:
        raise ValueError("Triton dense fast path supports 1..128 KV tokens")
    if head_dim > 256:
        raise ValueError("Triton dense fast path supports head_dim <= 256")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)
    lse = torch.empty((rows, num_q_heads), device=q.device, dtype=torch.float32)
    block_d = triton.next_power_of_2(head_dim)
    block_n = triton.next_power_of_2(key_length)
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    _dense_attention_kernel[(rows, num_q_heads)](
        q,
        k,
        v,
        output,
        lse,
        key_length,
        scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        GROUP_SIZE=num_q_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=8,
    )
    return output, lse


def compact_selected_pages_triton(
    selected_pages: torch.Tensor,
    num_prefix_pages: int,
    *,
    num_kv_heads: int | None = None,
    group_size: int | None = None,
) -> torch.Tensor:
    if not triton_available():
        raise RuntimeError("Triton CUDA support is unavailable")
    if selected_pages.ndim != 3 or not selected_pages.is_cuda:
        raise ValueError("selected_pages must be CUDA [A,H,K]")
    rows, selected_heads, page_budget = selected_pages.shape
    if num_kv_heads is None:
        num_kv_heads = selected_heads
    if selected_heads % num_kv_heads:
        raise ValueError("selected heads must be divisible by KV heads")
    physical_group = selected_heads // num_kv_heads
    max_pages = min(
        num_prefix_pages, rows * page_budget * physical_group
    )
    selected_pages = selected_pages.contiguous()
    page_mask = torch.zeros(
        (num_kv_heads, num_prefix_pages),
        device=selected_pages.device,
        dtype=torch.int32,
    )
    output = torch.full(
        (num_kv_heads, max_pages),
        -1,
        device=selected_pages.device,
        dtype=selected_pages.dtype,
    )
    total_entries = selected_pages.numel()
    block = 256
    _mark_selected_pages_kernel[(triton.cdiv(total_entries, block),)](
        selected_pages,
        page_mask,
        total_entries,
        num_prefix_pages,
        selected_pages.stride(0),
        selected_pages.stride(1),
        selected_pages.stride(2),
        page_mask.stride(0),
        page_mask.stride(1),
        NUM_SELECTED_HEADS=selected_heads,
        PAGE_BUDGET=page_budget,
        GROUP_SIZE=physical_group,
        BLOCK=block,
        num_warps=4,
    )
    block_p = triton.next_power_of_2(num_prefix_pages)
    if block_p > 65536:
        raise ValueError("union Triton supports at most 65536 prefix pages")
    _compact_page_mask_kernel[(num_kv_heads,)](
        page_mask,
        output,
        num_prefix_pages,
        max_pages,
        page_mask.stride(0),
        page_mask.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_P=block_p,
        num_warps=8,
    )
    return output


def use_triton_backend(backend: str, *tensors: torch.Tensor) -> bool:
    if backend not in {"auto", "torch", "triton"}:
        raise ValueError(f"unsupported LoSA v2 backend: {backend}")
    if backend == "torch":
        return False
    available = triton_available() and all(tensor.is_cuda for tensor in tensors)
    if backend == "triton" and not available:
        raise RuntimeError("LoSA v2 Triton backend requested but unavailable")
    return available


__all__ = [
    "compact_prefix_attention_triton",
    "compact_selected_pages_triton",
    "dense_attention_triton",
    "locality_scores_triton",
    "quest_group_mean_scores_triton",
    "select_active_rows_triton",
    "triton_available",
    "use_triton_backend",
]
