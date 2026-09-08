"""PyTorch reference implementation of paper-faithful LoSA.

The implementation is intentionally small and explicit.  Tensor layout is:

    Q:        [B, Hq, D]        current block queries
    K/V pref: [L, Hkv, D]       immutable prefix KV
    K/V blk:  [B, Hkv, D]       current block KV

For t=0 LoSA initializes the cached prefix state with dense prefix attention.
For t>0 it:

1. ranks query rows by query MSE drift and chooses active rows A;
2. runs QUEST page selection only for active rows;
3. builds the physical KV-head union of selected pages;
4. computes active prefix attention over the union pages;
5. reuses cached prefix output/LSE for stable rows;
6. computes exact current-block attention and online-softmax merges the two.

This file is the correctness oracle for the Triton kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch


class GQAMode(str, Enum):
    """How QUEST page selection handles grouped-query attention."""

    PER_QUERY_HEAD = "per_query_head"
    GROUP_MEAN = "group_mean"
    GROUP_MAX_SCORE = "group_max_score"


@dataclass(frozen=True)
class QuestPageMetadata:
    k_min: torch.Tensor  # [num_pages, Hkv, D]
    k_max: torch.Tensor  # [num_pages, Hkv, D]
    prefix_length: int
    page_size: int

    @property
    def num_pages(self) -> int:
        return int(self.k_min.shape[0])


@dataclass(frozen=True)
class QuestSelection:
    pages: torch.Tensor  # [active_rows, Hq or Hkv, selected_pages]
    mode: GQAMode
    page_size: int
    token_budget: int


@dataclass(frozen=True)
class KVHeadUnion:
    kv_head: int
    query_heads: torch.Tensor
    pages: torch.Tensor


@dataclass(frozen=True)
class LoSAPrefixState:
    query: torch.Tensor  # [B, Hq, D]
    output: torch.Tensor  # [B, Hq, D]
    lse: torch.Tensor  # [B, Hq]
    metadata: QuestPageMetadata | None = None


@dataclass(frozen=True)
class LoSAResult:
    output: torch.Tensor
    lse: torch.Tensor
    prefix_output: torch.Tensor
    prefix_lse: torch.Tensor
    block_output: torch.Tensor
    block_lse: torch.Tensor
    next_state: LoSAPrefixState
    active_rows: torch.Tensor
    locality: torch.Tensor
    selection: QuestSelection | None
    unions: tuple[KVHeadUnion, ...]
    initialized_dense: bool
    backend: str


def _kv_head_map(num_query_heads: int, num_kv_heads: int, device: torch.device) -> torch.Tensor:
    if num_query_heads % num_kv_heads:
        raise ValueError("Hq must be divisible by Hkv")
    return torch.arange(num_query_heads, device=device) // (num_query_heads // num_kv_heads)


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must be [tokens, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError("k and v must have identical shape")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head_dim differ")
    if q.shape[1] % k.shape[1]:
        raise ValueError("Hq must be divisible by Hkv")
    if k.shape[0] == 0:
        raise ValueError("empty KV is not supported")


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense attention with GQA expansion and FP32 softmax statistics."""

    _validate_qkv(q, k, v)
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    kv_head = _kv_head_map(q.shape[1], k.shape[1], q.device)
    expanded_k = k[:, kv_head, :]
    expanded_v = v[:, kv_head, :]
    scores = torch.einsum("bhd,nhd->bhn", q.float(), expanded_k.float()) * scale
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhn,nhd->bhd", probs, expanded_v.float())
    return out.to(q.dtype), lse


def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    block_output: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact online-softmax merge of two disjoint KV partitions."""

    if prefix_output.shape != block_output.shape:
        raise ValueError("output shapes differ")
    if prefix_lse.shape != block_lse.shape or prefix_lse.shape != prefix_output.shape[:-1]:
        raise ValueError("LSE shapes do not match")
    total_lse = torch.logaddexp(prefix_lse.float(), block_lse.float())
    prefix_weight = torch.exp(prefix_lse.float() - total_lse)
    block_weight = torch.exp(block_lse.float() - total_lse)
    out = prefix_output.float() * prefix_weight[..., None]
    out += block_output.float() * block_weight[..., None]
    return out.to(prefix_output.dtype), total_lse


def build_page_metadata(k_prefix: torch.Tensor, page_size: int) -> QuestPageMetadata:
    if k_prefix.ndim != 3:
        raise ValueError("k_prefix must be [L, Hkv, D]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    prefix_length, num_kv_heads, head_dim = k_prefix.shape
    if prefix_length == 0:
        raise ValueError("prefix must be non-empty")
    num_pages = math.ceil(prefix_length / page_size)
    padded_length = num_pages * page_size
    if padded_length == prefix_length:
        padded = k_prefix
    else:
        pad = k_prefix.new_zeros((padded_length - prefix_length, num_kv_heads, head_dim))
        padded = torch.cat((k_prefix, pad), dim=0)
    page_view = padded.reshape(num_pages, page_size, num_kv_heads, head_dim)
    valid = torch.arange(padded_length, device=k_prefix.device).reshape(num_pages, page_size)
    valid = valid < prefix_length
    k_min = page_view.masked_fill(~valid[:, :, None, None], float("inf")).amin(dim=1)
    k_max = page_view.masked_fill(~valid[:, :, None, None], float("-inf")).amax(dim=1)
    return QuestPageMetadata(k_min, k_max, prefix_length, page_size)


def quest_page_scores(
    q: torch.Tensor,
    metadata: QuestPageMetadata,
    mode: GQAMode | str = GQAMode.PER_QUERY_HEAD,
) -> torch.Tensor:
    """QUEST upper-bound page scores.

    Returns [B, Hq, P] for per-query-head mode, otherwise [B, Hkv, P].
    """

    mode = GQAMode(mode)
    if q.ndim != 3 or q.shape[-1] != metadata.k_min.shape[-1]:
        raise ValueError("q must be [B, Hq, D] matching metadata")
    num_query_heads = q.shape[1]
    num_kv_heads = metadata.k_min.shape[1]
    kv_head = _kv_head_map(num_query_heads, num_kv_heads, q.device)

    k_min = metadata.k_min[:, kv_head, :].float()
    k_max = metadata.k_max[:, kv_head, :].float()
    native = torch.einsum("bhd,phd->bhp", q.float().clamp_min(0), k_max)
    native += torch.einsum("bhd,phd->bhp", q.float().clamp_max(0), k_min)

    if mode is GQAMode.PER_QUERY_HEAD:
        return native

    group = num_query_heads // num_kv_heads
    native = native.reshape(q.shape[0], num_kv_heads, group, metadata.num_pages)
    if mode is GQAMode.GROUP_MAX_SCORE:
        return native.amax(dim=2)

    grouped_q = q.reshape(q.shape[0], num_kv_heads, group, q.shape[-1]).float().mean(dim=2)
    return torch.einsum("bhd,phd->bhp", grouped_q.clamp_min(0), metadata.k_max.float()) + torch.einsum(
        "bhd,phd->bhp", grouped_q.clamp_max(0), metadata.k_min.float()
    )


def select_pages(
    q: torch.Tensor,
    metadata: QuestPageMetadata,
    token_budget: int,
    *,
    mode: GQAMode | str = GQAMode.PER_QUERY_HEAD,
    always_include_last_page: bool = False,
) -> QuestSelection:
    """Run QUEST page selection for each supplied query row."""

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    mode = GQAMode(mode)
    page_budget = min(metadata.num_pages, math.ceil(token_budget / metadata.page_size))
    scores = quest_page_scores(q, metadata, mode)
    if page_budget == metadata.num_pages:
        pages = torch.arange(metadata.num_pages, device=q.device).expand(*scores.shape[:-1], -1)
    elif always_include_last_page:
        if page_budget == 1:
            pages = torch.full((*scores.shape[:-1], 1), metadata.num_pages - 1, device=q.device)
        else:
            pages = scores[..., :-1].topk(page_budget - 1, dim=-1).indices
            pages = torch.cat((pages, torch.full_like(pages[..., :1], metadata.num_pages - 1)), dim=-1)
        pages = pages.sort(dim=-1).values
    else:
        pages = scores.topk(page_budget, dim=-1).indices.sort(dim=-1).values
    return QuestSelection(pages.to(torch.long), mode, metadata.page_size, token_budget)


def build_union_pages(
    selection: QuestSelection,
    *,
    num_query_heads: int,
    num_kv_heads: int,
) -> tuple[KVHeadUnion, ...]:
    """Build LoSA's physical KV-head union pages from active-row QUEST selections."""

    group = num_query_heads // num_kv_heads
    unions: list[KVHeadUnion] = []
    for kv_head in range(num_kv_heads):
        query_heads = torch.arange(kv_head * group, (kv_head + 1) * group, device=selection.pages.device)
        if selection.mode is GQAMode.PER_QUERY_HEAD:
            selected = selection.pages[:, query_heads, :]
        else:
            selected = selection.pages[:, kv_head : kv_head + 1, :]
        pages = torch.unique(selected.reshape(-1), sorted=True)
        unions.append(KVHeadUnion(kv_head, query_heads, pages))
    return tuple(unions)


def sparse_prefix_attention_on_union(
    q_active: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    selection: QuestSelection,
    *,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[KVHeadUnion, ...]]:
    """Paper LoSA active-prefix attention over the union of QUEST-selected pages.

    Important: after the union is built, every active query head in the same
    physical KV group attends to the whole union for that KV head.  We do not
    mask each row back down to its own individual QUEST pages.
    """

    _validate_qkv(q_active, k_prefix, v_prefix)
    scale = q_active.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    num_query_heads = q_active.shape[1]
    num_kv_heads = k_prefix.shape[1]
    kv_head_for_q = _kv_head_map(num_query_heads, num_kv_heads, q_active.device)
    unions = build_union_pages(selection, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads)
    token_pages = torch.arange(k_prefix.shape[0], device=q_active.device) // selection.page_size

    output = torch.empty_like(q_active)
    lse = torch.empty(q_active.shape[:-1], dtype=torch.float32, device=q_active.device)
    for h in range(num_query_heads):
        kv_head = int(kv_head_for_q[h])
        union_pages = unions[kv_head].pages
        visible = (token_pages[:, None] == union_pages[None, :]).any(dim=1)
        k = k_prefix[visible, kv_head, :]
        v = v_prefix[visible, kv_head, :]
        scores = torch.einsum("bd,nd->bn", q_active[:, h, :].float(), k.float()) * scale
        row_lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        row_out = torch.einsum("bn,nd->bd", probs, v.float())
        output[:, h, :] = row_out.to(output.dtype)
        lse[:, h] = row_lse
    return output, lse, unions


def locality_scores(query: torch.Tensor, previous_query: torch.Tensor) -> torch.Tensor:
    if query.shape != previous_query.shape:
        raise ValueError("query and previous_query shapes differ")
    return (query.float() - previous_query.float()).square().mean(dim=(1, 2))


def select_active_rows(query: torch.Tensor, previous_query: torch.Tensor, active_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if active_count <= 0:
        raise ValueError("active_count must be positive")
    count = min(active_count, query.shape[0])
    locality = locality_scores(query, previous_query)
    active = locality.topk(count, sorted=False).indices.sort().values
    return active, locality


def _normalize_backend(backend: str) -> str:
    backend = str(backend).lower()
    if backend not in {"torch", "triton", "auto"}:
        raise ValueError(f"unsupported LoSA backend: {backend!r}")
    return backend


def _can_use_triton(q_active: torch.Tensor, k_prefix: torch.Tensor, v_prefix: torch.Tensor) -> bool:
    if not (q_active.is_cuda and k_prefix.is_cuda and v_prefix.is_cuda):
        return False
    if q_active.dtype not in (torch.float16, torch.bfloat16):
        return False
    try:
        from .triton_ops import triton
    except Exception:
        return False
    return triton is not None


def sparse_prefix_attention_dispatch(
    q_active: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    selection: QuestSelection,
    *,
    backend: str = "torch",
) -> tuple[torch.Tensor, torch.Tensor, tuple[KVHeadUnion, ...], str]:
    """Run active-prefix LoSA attention with an explicit backend switch."""

    backend = _normalize_backend(backend)
    if backend == "auto":
        backend = "triton" if _can_use_triton(q_active, k_prefix, v_prefix) else "torch"

    if backend == "triton":
        from .triton_ops import losa_union_prefix_attention_triton

        output, lse = losa_union_prefix_attention_triton(q_active, k_prefix, v_prefix, selection)
        unions = build_union_pages(
            selection,
            num_query_heads=q_active.shape[1],
            num_kv_heads=k_prefix.shape[1],
        )
        return output, lse, unions, "triton"

    output, lse, unions = sparse_prefix_attention_on_union(q_active, k_prefix, v_prefix, selection)
    return output, lse, unions, "torch"


def losa_attention_step(
    query: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_block: torch.Tensor,
    v_block: torch.Tensor,
    *,
    page_size: int,
    token_budget: int,
    active_count: int,
    mode: GQAMode | str = GQAMode.PER_QUERY_HEAD,
    always_include_last_page: bool = False,
    backend: str = "torch",
    metadata: QuestPageMetadata | None = None,
    previous_state: LoSAPrefixState | None = None,
) -> LoSAResult:
    """One layer, one denoising-step LoSA reference."""

    _validate_qkv(query, k_prefix, v_prefix)
    _validate_qkv(query, k_block, v_block)
    if k_prefix.shape[1:] != k_block.shape[1:]:
        raise ValueError("prefix and block KV layouts differ")
    block_output, block_lse = dense_attention(query, k_block, v_block)
    selection = None
    unions: tuple[KVHeadUnion, ...] = ()
    locality = torch.zeros(query.shape[0], dtype=torch.float32, device=query.device)
    actual_backend = "torch"

    if previous_state is None:
        prefix_output, prefix_lse = dense_attention(query, k_prefix, v_prefix)
        active_rows = torch.arange(query.shape[0], device=query.device)
        initialized_dense = True
    else:
        if previous_state.query.shape != query.shape:
            raise ValueError("previous LoSA state shape differs from current query")
        if metadata is None:
            metadata = previous_state.metadata
        if metadata is None or metadata.prefix_length != k_prefix.shape[0] or metadata.page_size != page_size:
            metadata = build_page_metadata(k_prefix, page_size)
        active_rows, locality = select_active_rows(query, previous_state.query, active_count)
        q_active = query.index_select(0, active_rows).contiguous()
        selection = select_pages(
            q_active,
            metadata,
            token_budget,
            mode=mode,
            always_include_last_page=always_include_last_page,
        )
        active_prefix_output, active_prefix_lse, unions, actual_backend = sparse_prefix_attention_dispatch(
            q_active,
            k_prefix,
            v_prefix,
            selection,
            backend=backend,
        )
        prefix_output = previous_state.output.clone()
        prefix_lse = previous_state.lse.clone()
        prefix_output.index_copy_(0, active_rows, active_prefix_output)
        prefix_lse.index_copy_(0, active_rows, active_prefix_lse)
        initialized_dense = False

    output, lse = merge_attention_states(prefix_output, prefix_lse, block_output, block_lse)
    next_state = LoSAPrefixState(
        query.contiguous(),
        prefix_output.contiguous(),
        prefix_lse.contiguous(),
        metadata,
    )
    return LoSAResult(
        output=output,
        lse=lse,
        prefix_output=prefix_output,
        prefix_lse=prefix_lse,
        block_output=block_output,
        block_lse=block_lse,
        next_state=next_state,
        active_rows=active_rows,
        locality=locality,
        selection=selection,
        unions=unions,
        initialized_dense=initialized_dense,
        backend=actual_backend,
    )
