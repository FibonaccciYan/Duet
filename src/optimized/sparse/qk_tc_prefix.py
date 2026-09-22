"""Opt-in Tensor Core QK: FP16/BF16 inputs, FP32 accumulation, no TF32.

One tile stays inside a physical KV head, including when Qlen is not aligned.
Per-query candidates, candidate union, fill, strict trim and final position
ordering are preserved. Arithmetic is not bitwise equivalent to qk's tree sum.
"""
import math
import torch
import triton
import triton.language as tl
from src.reference.sparse.qk_prefix import (
    validate, prefix_indices as reference, rank, finish_union)

SCORE_DEFINITION = "negative_unscaled_dot_tensorcore_fp32_accum"


@triton.jit
def _tile(Q, K, M, I, S, Full,
          qh_stride, qt_stride, qd_stride, kh_stride, kt_stride, kd_stride,
          QLEN: tl.constexpr, GROUP: tl.constexpr, LENGTH: tl.constexpr,
          QTILES: tl.constexpr, CHUNKS: tl.constexpr, LOCAL: tl.constexpr,
          BM: tl.constexpr = 16, BN: tl.constexpr = 64,
          STORE_FULL: tl.constexpr = False):
    kv, qb, kb = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = qb * BM + tl.arange(0, BM)
    n = kb * BN + tl.arange(0, BN)
    d = tl.arange(0, 128)
    valid_q = r < GROUP * QLEN
    qh = kv * GROUP + r // QLEN
    qt = r % QLEN
    q = tl.load(Q + qh[:, None]*qh_stride + qt[:, None]*qt_stride
                + d[None, :]*qd_stride, mask=valid_q[:, None], other=0)
    k = tl.load(K + kv*kh_stride + d[:, None]*kd_stride + n[None, :]*kt_stride,
                mask=n[None, :] < LENGTH, other=0)
    score = -tl.dot(q, k, acc=tl.zeros((BM, BN), tl.float32),
                    input_precision="ieee", out_dtype=tl.float32)
    score = tl.where(valid_q[:, None] & (n[None, :] < LENGTH),
                     score, float("inf"))
    global_row = kv * GROUP * QLEN + r
    if STORE_FULL:
        tl.store(Full + global_row[:, None]*LENGTH + n[None, :], score,
                 mask=valid_q[:, None] & (n[None, :] < LENGTH))
    tl.store(S + (kv*QTILES + qb)*LENGTH + n, tl.min(score, 0), mask=n < LENGTH)
    # Each tile contributes local_budget candidates. Their global top-k is
    # exactly the per-query top-k; invalid tail slots are sorted after valid ones.
    for slot in tl.static_range(LOCAL):
        minimum = tl.min(score, 1)
        index = tl.min(tl.where((score == minimum[:, None]) & (n[None, :] < LENGTH),
                                n[None, :], 2147483647), 1)
        offset = global_row*(CHUNKS*LOCAL) + kb*LOCAL + slot
        tl.store(M + offset, minimum, mask=valid_q)
        tl.store(I + offset, index, mask=valid_q)
        score = tl.where(n[None, :] == index[:, None], float("inf"), score)


def eligible(query, key, budget):
    rows = query.shape[1]*query.shape[2]
    local = max(1, math.ceil(budget/rows))
    return (query.is_cuda and key.is_cuda and query.dtype == key.dtype
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.shape[-1] == 128 and 0 < budget < key.shape[2] <= 32768
            and local <= 8
            and torch.cuda.get_device_capability(query.device)[0] >= 8)


def summaries(query, key, local_budget=1, *, return_scores=False):
    """Return tile candidates and global-score partials; full scores only for audit."""
    validate(query, key)
    length = key.shape[2]
    if not (query.is_cuda and key.is_cuda and query.dtype == key.dtype
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.shape[-1] == 128 and 0 < length <= 32768
            and 1 <= local_budget <= 8
            and torch.cuda.get_device_capability(query.device)[0] >= 8):
        raise ValueError("Unsupported Tensor Core summary shape")
    hq, qlen, _ = query.shape[1:]
    hkv = key.shape[1]
    group = hq // hkv
    qtiles, chunks = triton.cdiv(group*qlen, 16), triton.cdiv(length, 64)
    m = torch.empty((hq*qlen, chunks*local_budget), device=query.device, dtype=torch.float32)
    i = torch.empty(m.shape, device=query.device, dtype=torch.int64)
    s = torch.empty((hkv*qtiles, length), device=query.device, dtype=torch.float32)
    full = torch.empty((hq*qlen,length), device=query.device, dtype=torch.float32) if return_scores else m
    kernel = _tile[(hkv, qtiles, chunks)](
        query, key, m, i, s, full, *query.stride()[1:], *key.stride()[1:],
        QLEN=qlen, GROUP=group, LENGTH=length, QTILES=qtiles, CHUNKS=chunks,
        LOCAL=local_budget, STORE_FULL=return_scores, num_warps=4, num_stages=2)
    return m, i, s, full if return_scores else None, kernel


def prefix_indices(query, key, token_budget, chunk_size=256, bucket_thresholds=None,
                   strict_budget=False, selection_stats=None):
    validate(query,key)
    budget = max(0,min(int(token_budget),key.shape[2]))
    if not eligible(query,key,budget):
        out = reference(query,key,token_budget,chunk_size,bucket_thresholds,
                        strict_budget,selection_stats)
        if selection_stats is not None:
            selection_stats.update(selector="qk_tc", execution_backend="qk_reference")
        return out
    local = max(1,math.ceil(budget/(query.shape[1]*query.shape[2])))
    m,i,s,_,_ = summaries(query,key,local)
    if local == 1:
        # Chunks are in token order; min selects the earliest tied chunk.
        candidates = i.gather(1,m.min(dim=1,keepdim=True).indices)
    else:
        _, candidates = rank(m,i,local)
    result = finish_union(candidates,s.amin(0),budget,strict_budget,selection_stats,local)
    if selection_stats is not None:
        selection_stats.update(selector="qk_tc",score_definition=SCORE_DEFINITION,
                               execution_backend="tensorcore_fp32_accum")
    return result
