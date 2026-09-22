"""Raw L1 selector with fused reductions; original union/fill/tie rules."""
import math
import torch
import triton
import triton.language as tl
from src.reference.sparse.sparse_ops import _raw_l1_prefix_indices as reference, _selection_stats


@triton.jit
def _distance_minima(Q, K, M, I, S,
                    qs1, qs2, qs3, ks1, ks2, ks3,
                    QLEN: tl.constexpr, QHEADS: tl.constexpr, KHEADS: tl.constexpr,
                    LENGTH: tl.constexpr, DIM: tl.constexpr,
                    CHUNKS: tl.constexpr, BD: tl.constexpr,
                    BQ: tl.constexpr = 4, BN: tl.constexpr = 32):
    group, chunk = tl.program_id(0), tl.program_id(1)
    row = group * BQ + tl.arange(0, BQ)
    token = chunk * BN + tl.arange(0, BN)
    dim = tl.arange(0, BD)
    qh, qp = row // QLEN, row % QLEN
    kh = qh // (QHEADS // KHEADS)
    q = tl.load(Q + qh[:, None] * qs1 + qp[:, None] * qs2 + dim[None, :] * qs3,
                mask=(row[:, None] < QHEADS * QLEN) & (dim[None, :] < DIM), other=0)
    k = tl.load(K + kh[:, None, None] * ks1 + token[None, :, None] * ks2
                + dim[None, None, :] * ks3,
                mask=(row[:, None, None] < QHEADS * QLEN)
                     & (token[None, :, None] < LENGTH) & (dim[None, None, :] < DIM),
                other=0)
    # Match the reference operand dtype and dimension-reduction expression.
    dist = tl.sum(tl.abs(q[:, None, :] - k), axis=2)
    dist = tl.where((row[:, None] < QHEADS * QLEN) & (token[None, :] < LENGTH),
                    dist, float("inf"))
    minimum = tl.min(dist, axis=1)
    index = tl.min(tl.where(dist == minimum[:, None], token[None, :], 2147483647), axis=1)
    tl.store(M + row * CHUNKS + chunk, minimum, mask=row < QHEADS * QLEN)
    tl.store(I + row * CHUNKS + chunk, index, mask=row < QHEADS * QLEN)
    score = tl.min(dist, axis=0)
    tl.store(S + group * LENGTH + token, score, mask=token < LENGTH)


def prefix_indices(query, key, token_budget, chunk_size=256, bucket_thresholds=None,
                   strict_budget=False, selection_stats=None, *, selector="raw_l1"):
    if selector == "qk_tc":
        from .qk_tc_prefix import prefix_indices as tc_indices
        return tc_indices(query, key, token_budget, chunk_size, bucket_thresholds,
                          strict_budget, selection_stats)
    if selector == "qk":
        from .qk_prefix import prefix_indices as qk_indices
        return qk_indices(query, key, token_budget, chunk_size, bucket_thresholds,
                          strict_budget, selection_stats)
    if selector != "raw_l1":
        from src.reference.sparse.sparse_ops import _prefix_indices
        return _prefix_indices(query, key, token_budget, chunk_size, bucket_thresholds,
                               strict_budget, selection_stats, selector=selector)
    if selection_stats is not None:
        selection_stats.update(selector="raw_l1", score_definition="raw_l1_legacy")
    length = key.shape[-2]
    budget = min(int(token_budget), length)
    rows = query.shape[1] * query.shape[2]
    if (budget <= 0 or budget >= length or length > 2048 or query.shape[-1] != 128
            or math.ceil(budget / rows) != 1
            or not query.is_cuda or query.dtype not in (torch.float16, torch.bfloat16)):
        return reference(query, key, token_budget, strict_budget=strict_budget,
                         selection_stats=selection_stats)
    if query.shape[0] != 1 or query.shape[1] % key.shape[1]:
        raise ValueError("Raw L1 requires batch size one and valid GQA")
    chunks, groups = triton.cdiv(length, 32), triton.cdiv(rows, 4)
    minima = torch.empty((rows, chunks), device=query.device, dtype=torch.float32)
    indices = torch.empty((rows, chunks), device=query.device, dtype=torch.int64)
    scores = torch.empty((groups, length), device=query.device, dtype=torch.float32)
    _distance_minima[(groups, chunks)](
        query, key, minima, indices, scores, *query.stride()[1:], *key.stride()[1:],
        QLEN=query.shape[2], QHEADS=query.shape[1], KHEADS=key.shape[1],
        LENGTH=length, DIM=query.shape[-1], CHUNKS=chunks,
        BD=triton.next_power_of_2(query.shape[-1]), num_warps=8)
    # Keep PyTorch's original unique and fill-topk behavior, including ties.
    best_chunk = minima.min(dim=1, keepdim=True).indices
    selected = torch.unique(indices.gather(1, best_chunk).flatten())
    union_size = selected.numel()
    if selected.numel() < budget:
        remaining = scores.amin(dim=0)
        remaining[selected] = torch.inf
        fill = torch.topk(remaining, budget - selected.numel(), largest=False).indices
        selected = torch.cat((selected, fill))
    elif strict_budget and selected.numel() > budget:
        keep = scores.amin(dim=0).index_select(0, selected).topk(
            budget, largest=False).indices
        selected = selected.index_select(0, keep)
    _selection_stats(selection_stats, length, budget, 1, union_size,
                     selected.numel(), strict_budget)
    return selected.sort().values
