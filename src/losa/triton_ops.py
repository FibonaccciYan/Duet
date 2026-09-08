"""Triton kernels for paper-faithful LoSA.

The first kernel covers the semantic core that differs from the current n33
implementation: active query rows attend to the QUEST union pages, not to the
full prefix.  It is a correctness-oriented kernel; performance tuning comes
after it matches the PyTorch oracle.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import guard for CPU-only environments
    triton = None
    tl = None

from .operators import QuestSelection, build_union_pages


if triton is not None:

    @triton.jit
    def _losa_union_prefix_attention_kernel(
        q,
        k,
        v,
        union_pages,
        out,
        out_lse,
        q_stride_0,
        q_stride_1,
        q_stride_2,
        k_stride_0,
        k_stride_1,
        k_stride_2,
        v_stride_0,
        v_stride_1,
        v_stride_2,
        page_stride_0,
        page_stride_1,
        out_stride_0,
        out_stride_1,
        out_stride_2,
        lse_stride_0,
        lse_stride_1,
        scale,
        active_rows: tl.constexpr,
        query_heads: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        prefix_length: tl.constexpr,
        page_size: tl.constexpr,
        max_union_pages: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        row = tl.program_id(0)
        q_head = tl.program_id(1)
        group = query_heads // kv_heads
        kv_head = q_head // group
        offsets_d = tl.arange(0, BLOCK_D)
        q_vec = tl.load(
            q + row * q_stride_0 + q_head * q_stride_1 + offsets_d * q_stride_2,
            mask=offsets_d < head_dim,
            other=0.0,
        )
        running_max = tl.full((), -float("inf"), tl.float32)
        running_sum = tl.full((), 0.0, tl.float32)
        accumulator = tl.zeros((BLOCK_D,), tl.float32)

        for page_slot in range(0, max_union_pages):
            page = tl.load(
                union_pages + kv_head * page_stride_0 + page_slot * page_stride_1
            )
            for token_base in range(0, page_size, BLOCK_T):
                token_offsets = token_base + tl.arange(0, BLOCK_T)
                token = page * page_size + token_offsets
                valid_token = (page >= 0) & (token_offsets < page_size) & (token < prefix_length)
                k_offsets = (
                    token[:, None] * k_stride_0
                    + kv_head * k_stride_1
                    + offsets_d[None, :] * k_stride_2
                )
                k_tile = tl.load(
                    k + k_offsets,
                    mask=valid_token[:, None] & (offsets_d[None, :] < head_dim),
                    other=0.0,
                )
                scores = tl.sum(k_tile * q_vec[None, :], axis=1) * scale
                scores = tl.where(valid_token, scores, -float("inf"))
                tile_max = tl.max(scores, axis=0)
                new_max = tl.maximum(running_max, tile_max)
                probs = tl.exp(scores - new_max)
                old_scale = tl.exp(running_max - new_max)
                v_offsets = (
                    token[:, None] * v_stride_0
                    + kv_head * v_stride_1
                    + offsets_d[None, :] * v_stride_2
                )
                v_tile = tl.load(
                    v + v_offsets,
                    mask=valid_token[:, None] & (offsets_d[None, :] < head_dim),
                    other=0.0,
                )
                accumulator = accumulator * old_scale + tl.sum(
                    probs[:, None].to(v_tile.dtype) * v_tile, axis=0
                )
                running_sum = running_sum * old_scale + tl.sum(probs, axis=0)
                running_max = new_max

        tl.store(
            out + row * out_stride_0 + q_head * out_stride_1 + offsets_d * out_stride_2,
            accumulator / running_sum,
            mask=offsets_d < head_dim,
        )
        tl.store(out_lse + row * lse_stride_0 + q_head * lse_stride_1, running_max + tl.log(running_sum))


def padded_union_pages(
    selection: QuestSelection,
    *,
    num_query_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Return [Hkv, max_union_pages] pages padded with -1."""

    unions = build_union_pages(
        selection,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
    )
    max_pages = max(int(union.pages.numel()) for union in unions)
    pages = torch.full(
        (num_kv_heads, max_pages),
        -1,
        dtype=torch.long,
        device=selection.pages.device,
    )
    for union in unions:
        pages[union.kv_head, : union.pages.numel()] = union.pages
    return pages


def losa_union_prefix_attention_triton(
    q_active: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    selection: QuestSelection,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton active-prefix LoSA attention over QUEST union pages."""

    if triton is None:
        raise RuntimeError("triton is not available")
    if not (
        q_active.is_cuda
        and k_prefix.is_cuda
        and v_prefix.is_cuda
        and selection.pages.is_cuda
        and q_active.device == k_prefix.device == v_prefix.device == selection.pages.device
    ):
        raise ValueError("all LoSA Triton inputs must be CUDA tensors on the same device")
    if q_active.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q_active must be fp16 or bf16")
    if k_prefix.shape != v_prefix.shape:
        raise ValueError("k_prefix and v_prefix shapes differ")
    if q_active.shape[-1] != k_prefix.shape[-1] or q_active.shape[1] % k_prefix.shape[1]:
        raise ValueError("incompatible Q/K/V head shapes")

    active_rows, query_heads, head_dim = q_active.shape
    prefix_length, kv_heads, _ = k_prefix.shape
    union_pages = padded_union_pages(
        selection,
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
    ).contiguous()
    out = torch.empty_like(q_active)
    out_lse = torch.empty((active_rows, query_heads), dtype=torch.float32, device=q_active.device)
    block_d = triton.next_power_of_2(head_dim)
    block_t = min(32, triton.next_power_of_2(selection.page_size))
    _losa_union_prefix_attention_kernel[(active_rows, query_heads)](
        q_active,
        k_prefix,
        v_prefix,
        union_pages,
        out,
        out_lse,
        *q_active.stride(),
        *k_prefix.stride(),
        *v_prefix.stride(),
        *union_pages.stride(),
        *out.stride(),
        *out_lse.stride(),
        head_dim**-0.5,
        active_rows=active_rows,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        prefix_length=prefix_length,
        page_size=selection.page_size,
        max_union_pages=union_pages.shape[1],
        BLOCK_D=block_d,
        BLOCK_T=block_t,
        num_warps=4,
    )
    return out, out_lse
