"""PyTorch reference implementation for LoSA and QUEST (Adapted).

The reference path intentionally favors explicit sparse semantics over speed.  It is
used as the correctness oracle for the Triton kernels and documents every tensor
contract that the model integration must satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class QuestPageMetadata:
    """Per-page bounding boxes used by the QUEST selector.

    ``k_min`` and ``k_max`` use the physical KV-head layout ``[P, H_kv, D]``.
    The final page may be partially filled; ``prefix_length`` is the authoritative
    valid-token count.
    """

    k_min: torch.Tensor
    k_max: torch.Tensor
    prefix_length: int
    page_size: int

    @property
    def num_pages(self) -> int:
        return int(self.k_min.shape[0])


@dataclass(frozen=True)
class SparsePagePlan:
    """Union-based sparse attention plan.

    ``selected_pages`` preserves the exact per-query/per-head QUEST result.
    ``union_pages`` is the compact list loaded by the attention kernel, while
    ``row_page_mask`` prevents a query from attending to pages selected only by
    another query.  This distinction is essential for reproducing QUEST (Adapted):
    replacing the row mask with an all-true mask changes the algorithm.
    """

    selected_pages: torch.Tensor
    union_pages: torch.Tensor
    row_page_mask: torch.Tensor
    page_size: int

    @property
    def union_size(self) -> int:
        return int(self.union_pages.numel())


def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [tokens, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have identical shapes, got {k.shape} and {v.shape}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dimensions differ")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("the number of query heads must be divisible by KV heads")


def _query_to_kv_head(num_q_heads: int, num_kv_heads: int, device: torch.device) -> torch.Tensor:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("the number of query heads must be divisible by KV heads")
    group_size = num_q_heads // num_kv_heads
    return torch.arange(num_q_heads, device=device) // group_size


def build_quest_page_metadata(k: torch.Tensor, page_size: int) -> QuestPageMetadata:
    """Build the coordinate-wise min/max key summaries used by QUEST."""

    if k.ndim != 3:
        raise ValueError("k must have shape [prefix_length, kv_heads, head_dim]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    prefix_length, num_kv_heads, head_dim = k.shape
    if prefix_length == 0:
        raise ValueError("QUEST requires a non-empty prefix")

    num_pages = math.ceil(prefix_length / page_size)
    padded_length = num_pages * page_size
    if padded_length != prefix_length:
        padding = k.new_zeros((padded_length - prefix_length, num_kv_heads, head_dim))
        padded = torch.cat((k, padding), dim=0)
    else:
        padded = k

    pages = padded.reshape(num_pages, page_size, num_kv_heads, head_dim)
    valid = torch.arange(padded_length, device=k.device).reshape(num_pages, page_size)
    valid = valid < prefix_length
    k_min = pages.masked_fill(~valid[:, :, None, None], float("inf")).amin(dim=1)
    k_max = pages.masked_fill(~valid[:, :, None, None], float("-inf")).amax(dim=1)
    return QuestPageMetadata(k_min, k_max, prefix_length, page_size)


def quest_page_scores(q: torch.Tensor, metadata: QuestPageMetadata) -> torch.Tensor:
    """Compute QUEST's query-dependent upper bound for every prefix page.

    For each page ``p`` the score is

    ``sum_d q[d] * (k_max[p,d] if q[d] >= 0 else k_min[p,d])``.

    The softmax scale is omitted because it is a positive constant and therefore
    cannot change the page ranking.
    """

    if q.ndim != 3:
        raise ValueError("q must have shape [query_tokens, query_heads, head_dim]")
    if metadata.k_min.shape != metadata.k_max.shape:
        raise ValueError("QUEST min/max metadata shapes differ")
    if q.shape[-1] != metadata.k_min.shape[-1]:
        raise ValueError("q and metadata head dimensions differ")

    num_q_heads = q.shape[1]
    num_kv_heads = metadata.k_min.shape[1]
    kv_head = _query_to_kv_head(num_q_heads, num_kv_heads, q.device)
    k_min = metadata.k_min[:, kv_head, :]
    k_max = metadata.k_max[:, kv_head, :]
    positive_q = torch.clamp_min(q.float(), 0)
    negative_q = torch.clamp_max(q.float(), 0)
    return torch.einsum("mhd,phd->mhp", positive_q, k_max.float()) + torch.einsum(
        "mhd,phd->mhp", negative_q, k_min.float()
    )


def select_quest_pages(
    q: torch.Tensor,
    metadata: QuestPageMetadata,
    token_budget: int,
    *,
    always_include_last_page: bool = False,
) -> torch.Tensor:
    """Return sorted QUEST page ids with shape ``[M, H_q, K_pages]``.

    ``always_include_last_page`` reproduces the original autoregressive QUEST
    implementation.  LoSA operates on a fixed prefix and does not document this
    special case, so its paper-faithful default is ``False``.
    """

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    page_budget = min(metadata.num_pages, math.ceil(token_budget / metadata.page_size))
    scores = quest_page_scores(q, metadata)

    if page_budget == metadata.num_pages:
        pages = torch.arange(metadata.num_pages, device=q.device, dtype=torch.long)
        return pages.expand(q.shape[0], q.shape[1], -1)

    if always_include_last_page:
        if page_budget == 1:
            selected = torch.full(
                (*scores.shape[:-1], 1), metadata.num_pages - 1, device=q.device, dtype=torch.long
            )
        else:
            selected = scores[..., :-1].topk(page_budget - 1, dim=-1).indices
            last = torch.full_like(selected[..., :1], metadata.num_pages - 1)
            selected = torch.cat((selected, last), dim=-1)
    else:
        selected = scores.topk(page_budget, dim=-1).indices
    return selected.sort(dim=-1).values


def build_sparse_page_plan(selected_pages: torch.Tensor, page_size: int) -> SparsePagePlan:
    """Convert per-row page ids into a union plus an exact membership mask."""

    if selected_pages.ndim != 3:
        raise ValueError("selected_pages must have shape [query_tokens, query_heads, pages]")
    if selected_pages.numel() == 0:
        raise ValueError("selected_pages cannot be empty")
    union_pages = torch.unique(selected_pages.reshape(-1), sorted=True)
    membership = (selected_pages[..., :, None] == union_pages).any(dim=-2)
    return SparsePagePlan(selected_pages, union_pages, membership, page_size)


def dense_partition_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense non-causal attention returning normalized output and natural-log LSE."""

    _validate_qkv(q, k, v)
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    kv_head = _query_to_kv_head(q.shape[1], k.shape[1], q.device)
    k_expanded = k[:, kv_head, :]
    v_expanded = v[:, kv_head, :]
    scores = torch.einsum("mhd,nhd->mhn", q.float(), k_expanded.float()) * scale
    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("mhn,nhd->mhd", probabilities, v_expanded.float())
    return output.to(q.dtype), lse


def sparse_page_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: SparsePagePlan,
    *,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact reference for union-loaded, row-masked sparse page attention."""

    _validate_qkv(q, k, v)
    if plan.row_page_mask.shape[:2] != q.shape[:2]:
        raise ValueError("plan rows do not match q tokens and heads")
    if plan.row_page_mask.shape[-1] != plan.union_size:
        raise ValueError("plan membership width must equal union size")

    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    kv_head = _query_to_kv_head(q.shape[1], k.shape[1], q.device)
    k_expanded = k[:, kv_head, :]
    v_expanded = v[:, kv_head, :]
    scores = torch.einsum("mhd,nhd->mhn", q.float(), k_expanded.float()) * scale

    token_pages = torch.arange(k.shape[0], device=k.device) // plan.page_size
    union_slot = torch.searchsorted(plan.union_pages, token_pages)
    in_union = union_slot < plan.union_size
    safe_slot = union_slot.clamp_max(plan.union_size - 1)
    in_union = in_union & (plan.union_pages[safe_slot] == token_pages)
    row_mask = plan.row_page_mask[..., safe_slot] & in_union
    scores = scores.masked_fill(~row_mask, float("-inf"))
    if torch.any(torch.isneginf(scores).all(dim=-1)):
        raise ValueError("each query/head row must select at least one valid token")

    lse = torch.logsumexp(scores, dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("mhn,nhd->mhd", probabilities, v_expanded.float())
    return output.to(q.dtype), lse


def merge_attention_states(
    first_output: torch.Tensor,
    first_lse: torch.Tensor,
    second_output: torch.Tensor,
    second_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two normalized attention partitions using their log normalizers."""

    if first_output.shape != second_output.shape:
        raise ValueError("attention output shapes differ")
    if first_lse.shape != second_lse.shape or first_lse.shape != first_output.shape[:-1]:
        raise ValueError("LSE tensors must match the attention row dimensions")

    total_lse = torch.logaddexp(first_lse.float(), second_lse.float())
    first_weight = torch.exp(first_lse.float() - total_lse)
    second_weight = torch.exp(second_lse.float() - total_lse)
    output = first_output.float() * first_weight[..., None]
    output = output + second_output.float() * second_weight[..., None]
    return output.to(first_output.dtype), total_lse


def select_active_queries(q: torch.Tensor, previous_q: torch.Tensor, active_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Select LoSA active token rows by mean query MSE over heads and channels."""

    if q.shape != previous_q.shape or q.ndim != 3:
        raise ValueError("q and previous_q must have identical [tokens, heads, dim] shapes")
    if not 1 <= active_count <= q.shape[0]:
        raise ValueError("active_count must be in [1, query_tokens]")
    locality = (q.float() - previous_q.float()).square().mean(dim=(1, 2))
    active = locality.topk(active_count, sorted=False).indices.sort().values
    return active, locality
