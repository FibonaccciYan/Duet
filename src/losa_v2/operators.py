"""LoSA v2 fixed-shape operators.

The v2 path keeps the paper semantics used by `src.losa`, but removes dynamic
union construction and full-state cloning from the per-layer hot loop.  Page
selection returns a fixed-width compact page table; unused slots contain -1 and
are masked in attention.  No GPU->CPU synchronization is required to launch the
attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .triton_ops import (
    compact_prefix_attention_triton,
    compact_selected_pages_triton,
    dense_attention_triton,
    quest_group_mean_scores_triton,
    select_active_rows_triton,
    use_triton_backend,
)


@dataclass(frozen=True)
class GQAMode:
    PER_QUERY_HEAD = "per_query_head"
    GROUP_MEAN = "group_mean"
    GROUP_MAX_SCORE = "group_max_score"


@dataclass(frozen=True)
class V2PageMetadata:
    k_min: torch.Tensor  # [P, Hkv, D]
    k_max: torch.Tensor  # [P, Hkv, D]
    prefix_length: int
    page_size: int

    @property
    def num_pages(self) -> int:
        return int(self.k_min.shape[0])


@dataclass
class V2LayerState:
    query: torch.Tensor
    prefix_output: torch.Tensor
    prefix_lse: torch.Tensor
    metadata: V2PageMetadata | None


@dataclass
class V2AttentionResult:
    output: torch.Tensor
    lse: torch.Tensor
    prefix_output: torch.Tensor
    prefix_lse: torch.Tensor
    block_output: torch.Tensor
    block_lse: torch.Tensor
    next_state: V2LayerState
    active_rows: torch.Tensor
    locality: torch.Tensor
    initialized_dense: bool
    compact_pages: torch.Tensor | None


def _kv_head_map(num_q: int, num_kv: int, device: torch.device) -> torch.Tensor:
    if num_q % num_kv:
        raise ValueError("Hq must be divisible by Hkv")
    return torch.arange(num_q, device=device) // (num_q // num_kv)


def validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be [tokens, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError("K and V shapes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K head dimensions differ")
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    if k.shape[0] == 0:
        raise ValueError("empty KV is unsupported")


def build_page_metadata(k_prefix: torch.Tensor, page_size: int) -> V2PageMetadata:
    if k_prefix.ndim != 3:
        raise ValueError("k_prefix must be [L,Hkv,D]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    length, num_kv, dim = k_prefix.shape
    if length == 0:
        raise ValueError("prefix must be non-empty")
    pages = math.ceil(length / page_size)
    padded_len = pages * page_size
    if padded_len == length:
        padded = k_prefix
    else:
        pad = k_prefix.new_zeros((padded_len - length, num_kv, dim))
        padded = torch.cat((k_prefix, pad), dim=0)
    view = padded.reshape(pages, page_size, num_kv, dim)
    valid = torch.arange(padded_len, device=k_prefix.device).reshape(pages, page_size) < length
    k_min = view.masked_fill(~valid[:, :, None, None], float("inf")).amin(dim=1)
    k_max = view.masked_fill(~valid[:, :, None, None], float("-inf")).amax(dim=1)
    return V2PageMetadata(k_min, k_max, length, page_size)


def locality_scores(query: torch.Tensor, previous_query: torch.Tensor) -> torch.Tensor:
    if query.shape != previous_query.shape:
        raise ValueError("query and previous_query shapes differ")
    return (query.float() - previous_query.float()).square().mean(dim=(1, 2))


def select_active_rows(
    query: torch.Tensor,
    previous_query: torch.Tensor,
    active_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if active_count <= 0:
        raise ValueError("active_count must be positive")
    count = min(active_count, query.shape[0])
    locality = locality_scores(query, previous_query)
    active = locality.topk(count, sorted=False).indices.sort().values
    return active, locality


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense GQA attention returning normalized output and natural-log LSE."""
    validate_qkv(q, k, v)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    kv_head = _kv_head_map(q.shape[1], k.shape[1], q.device)
    k_exp = k[:, kv_head, :].float()
    v_exp = v[:, kv_head, :].float()
    scores = torch.einsum("bhd,nhd->bhn", q.float(), k_exp) * scale
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhn,nhd->bhd", probs, v_exp)
    return out.to(q.dtype), lse


def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    block_output: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_lse = torch.logaddexp(prefix_lse.float(), block_lse.float())
    prefix_weight = torch.exp(prefix_lse.float() - total_lse)
    block_weight = torch.exp(block_lse.float() - total_lse)
    out = prefix_output.float() * prefix_weight[..., None]
    out += block_output.float() * block_weight[..., None]
    return out.to(prefix_output.dtype), total_lse


def quest_page_scores_v2(
    q: torch.Tensor,
    metadata: V2PageMetadata,
    mode: str = GQAMode.GROUP_MEAN,
) -> torch.Tensor:
    """Return page scores without constructing a dynamic union.

    For GROUP_MEAN, the mean query is computed first.  Unlike v1, v2 does not
    materialize the unused per-query-head score matrix for this mode.
    """
    if mode not in {GQAMode.PER_QUERY_HEAD, GQAMode.GROUP_MEAN, GQAMode.GROUP_MAX_SCORE}:
        raise ValueError(f"unsupported GQA mode: {mode}")
    num_q = q.shape[1]
    num_kv = metadata.k_min.shape[1]
    group = num_q // num_kv
    if mode == GQAMode.GROUP_MEAN:
        qbar = q.float().reshape(q.shape[0], num_kv, group, q.shape[-1]).mean(dim=2)
        positive = torch.bmm(
            qbar.clamp_min(0).transpose(0, 1).contiguous(),
            metadata.k_max.float().permute(1, 2, 0),
        )
        negative = torch.bmm(
            qbar.clamp_max(0).transpose(0, 1).contiguous(),
            metadata.k_min.float().permute(1, 2, 0),
        )
        scores = positive + negative
        return scores.transpose(0, 1).contiguous()

    kv_head = _kv_head_map(num_q, num_kv, q.device)
    k_min = metadata.k_min[:, kv_head, :].float()
    k_max = metadata.k_max[:, kv_head, :].float()
    scores = torch.einsum("bhd,phd->bhp", q.float().clamp_min(0), k_max)
    scores += torch.einsum("bhd,phd->bhp", q.float().clamp_max(0), k_min)
    if mode == GQAMode.PER_QUERY_HEAD:
        return scores
    return scores.reshape(q.shape[0], num_kv, group, metadata.num_pages).amax(dim=2)


def compact_selected_pages(
    selected_pages: torch.Tensor,
    num_prefix_pages: int,
    *,
    num_kv_heads: int | None = None,
    group_size: int | None = None,
) -> torch.Tensor:
    """Compact selected pages into [Hkv,max_pages] without host synchronization.

    All active rows/query heads are unioned into their physical KV head, matching
    v1's `build_union_pages`.  Unused compact slots contain -1.  The width is an
    upper bound (`min(P, rows*page_budget)`), so no `.item()` is needed.
    """
    if selected_pages.ndim != 3:
        raise ValueError("selected_pages must have shape [A,H,K]")
    rows, heads, page_budget = selected_pages.shape
    if num_kv_heads is None:
        num_kv_heads = heads
    if heads % num_kv_heads:
        raise ValueError("Hq must be divisible by Hkv")
    selections_per_kv_head = heads // num_kv_heads
    max_pages = min(
        num_prefix_pages, rows * page_budget * selections_per_kv_head
    )
    device = selected_pages.device
    mask_flat = torch.zeros(num_kv_heads * num_prefix_pages, dtype=torch.bool, device=device)
    if num_kv_heads != heads or group_size is not None:
        kv_for_q = _kv_head_map(heads, num_kv_heads, device)
        target_kv = kv_for_q[None, :, None].expand(rows, heads, page_budget)
        flat_index = (target_kv * num_prefix_pages + selected_pages).reshape(-1)
    else:
        offsets = torch.arange(num_kv_heads, device=device)[:, None] * num_prefix_pages
        flat_index = (selected_pages + offsets[None, :, :]).reshape(-1)
    mask_flat.scatter_(0, flat_index, True)
    mask = mask_flat.view(num_kv_heads, num_prefix_pages)

    positions = mask.to(torch.int32).cumsum(dim=1) - 1
    scatter_idx = torch.where(mask, positions, torch.full_like(positions, max_pages))
    page_ids = torch.arange(num_prefix_pages, device=device, dtype=selected_pages.dtype)
    values = torch.where(
        mask,
        page_ids[None, :].expand_as(mask),
        torch.full_like(page_ids, -1)[None, :].expand_as(mask),
    )
    # Allocate one overflow slot so invalid entries can be scattered without a
    # data-dependent compact width or host sync.
    buffer = torch.full((num_kv_heads, max_pages + 1), -1, dtype=selected_pages.dtype, device=device)
    buffer.scatter_(1, scatter_idx, values)
    return buffer[:, :max_pages]

def compact_prefix_attention(
    q_active: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    compact_pages: torch.Tensor,
    *,
    page_size: int,
    prefix_length: int,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if compact_pages.ndim != 2:
        raise ValueError("compact_pages must be [Hkv,max_pages]")
    scale = q_active.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    rows, num_q, dim = q_active.shape
    num_kv = k_prefix.shape[1]
    group = num_q // num_kv
    max_pages = compact_pages.shape[1]
    page_valid = compact_pages >= 0
    safe_pages = compact_pages.clamp_min(0)
    raw_token_idx = safe_pages[:, :, None] * page_size + torch.arange(page_size, device=q_active.device)[None, None, :]
    token_valid = page_valid[:, :, None] & (raw_token_idx < prefix_length)
    token_idx = raw_token_idx.clamp_max(prefix_length - 1)
    kv_head_idx = torch.arange(num_kv, device=q_active.device)[:, None, None].expand_as(token_idx)
    k_sel = k_prefix[token_idx, kv_head_idx, :].float().reshape(num_kv, max_pages * page_size, dim)
    v_sel = v_prefix[token_idx, kv_head_idx, :].float().reshape(num_kv, max_pages * page_size, dim)
    token_valid = token_valid.reshape(num_kv, 1, max_pages * page_size).expand(
        num_kv, rows * group, max_pages * page_size
    )

    qg = q_active.view(rows, num_kv, group, dim).permute(1, 0, 2, 3).reshape(num_kv, rows * group, dim).float()
    scores = torch.bmm(qg, k_sel.transpose(1, 2)) * scale
    scores = scores.masked_fill(~token_valid, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.bmm(probs, v_sel)
    out = out.view(num_kv, rows, group, dim).permute(1, 0, 2, 3).reshape(rows, num_q, dim)
    lse = lse.view(num_kv, rows, group).permute(1, 0, 2).reshape(rows, num_q)
    return out.to(q_active.dtype), lse


def losa_v2_attention_step(
    query: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    *,
    page_size: int,
    token_budget: int,
    active_count: int,
    mode: str = GQAMode.GROUP_MEAN,
    backend: str = "auto",
    metadata: V2PageMetadata | None = None,
    previous_state: V2LayerState | None = None,
) -> V2AttentionResult:
    validate_qkv(query, k_prefix, v_prefix)
    validate_qkv(query, k_block, v_block)
    if k_prefix.shape[1:] != k_block.shape[1:]:
        raise ValueError("prefix and block KV layouts differ")
    if token_budget <= 0 or page_size <= 0 or active_count <= 0:
        raise ValueError("token_budget, page_size, and active_count must be positive")

    num_rows, num_q, dim = query.shape
    num_kv = k_prefix.shape[1]
    group = num_q // num_kv
    step_triton_enabled = use_triton_backend(
        backend, query, k_block, v_block
    )
    if step_triton_enabled and k_block.shape[0] <= 128:
        block_output, block_lse = dense_attention_triton(
            query, k_block, v_block
        )
    else:
        block_output, block_lse = dense_attention(query, k_block, v_block)

    if previous_state is None:
        if metadata is None:
            metadata = build_page_metadata(k_prefix, page_size)
        prefix_output, prefix_lse = dense_attention(query, k_prefix, v_prefix)
        active_rows = torch.arange(num_rows, device=query.device)
        locality = torch.zeros(num_rows, dtype=torch.float32, device=query.device)
        compact_pages = None
        initialized = True
    else:
        if previous_state.query.shape != query.shape:
            raise ValueError("previous v2 state shape differs")
        metadata = previous_state.metadata
        if metadata is None or metadata.prefix_length != k_prefix.shape[0] or metadata.page_size != page_size:
            metadata = build_page_metadata(k_prefix, page_size)
        triton_enabled = step_triton_enabled and all(
            tensor.is_cuda
            for tensor in (
                previous_state.query,
                metadata.k_min,
                metadata.k_max,
            )
        )
        if triton_enabled:
            active_rows, locality = select_active_rows_triton(
                query, previous_state.query, active_count
            )
        else:
            active_rows, locality = select_active_rows(
                query, previous_state.query, active_count
            )
        q_active = query.index_select(0, active_rows).contiguous()
        if triton_enabled and mode == GQAMode.GROUP_MEAN:
            scores = quest_group_mean_scores_triton(
                q_active, metadata.k_min, metadata.k_max
            )
        else:
            scores = quest_page_scores_v2(q_active, metadata, mode=mode)
        page_budget = min(metadata.num_pages, math.ceil(token_budget / page_size))
        selected = scores.topk(page_budget, dim=-1).indices
        num_kv_heads = num_kv
        group_size = group
        if triton_enabled:
            compact_pages = compact_selected_pages_triton(
                selected,
                metadata.num_pages,
                num_kv_heads=num_kv_heads,
                group_size=group_size,
            )
        else:
            compact_pages = compact_selected_pages(
                selected,
                metadata.num_pages,
                num_kv_heads=num_kv_heads,
                group_size=group_size,
            )
        if triton_enabled:
            prefix_output, prefix_lse = compact_prefix_attention_triton(
                q_active,
                k_prefix,
                v_prefix,
                compact_pages,
                page_size=page_size,
                prefix_length=metadata.prefix_length,
            )
        else:
            prefix_output, prefix_lse = compact_prefix_attention(
                q_active,
                k_prefix,
                v_prefix,
                compact_pages,
                page_size=page_size,
                prefix_length=metadata.prefix_length,
            )
        state_prefix_output = previous_state.prefix_output
        state_prefix_lse = previous_state.prefix_lse
        state_prefix_output.index_copy_(0, active_rows, prefix_output)
        state_prefix_lse.index_copy_(0, active_rows, prefix_lse)
        prefix_output = state_prefix_output
        prefix_lse = state_prefix_lse
        initialized = False

    output, lse = merge_attention_states(prefix_output, prefix_lse, block_output, block_lse)
    next_state = previous_state
    if next_state is None:
        next_state = V2LayerState(
            query=query.contiguous(),
            prefix_output=prefix_output.contiguous(),
            prefix_lse=prefix_lse.contiguous(),
            metadata=metadata,
        )
    else:
        next_state.query = query.contiguous()
        next_state.metadata = metadata
    return V2AttentionResult(
        output=output,
        lse=lse,
        prefix_output=prefix_output,
        prefix_lse=prefix_lse,
        block_output=block_output,
        block_lse=block_lse,
        next_state=next_state,
        active_rows=active_rows,
        locality=locality,
        initialized_dense=initialized,
        compact_pages=compact_pages,
    )
