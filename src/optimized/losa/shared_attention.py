"""Experimental shared-KV tiles across active rows and GQA heads.

All rows consume the unchanged physical-KV-head union. FP32 dot inputs
and FP32 probabilities avoid the old candidate's BF16 probability rounding.
Tensor Core arithmetic can still change reduction order; explicit opt-in.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _shared(Q, K, V, P, O, L,
            QR: tl.constexpr, QH: tl.constexpr, QD: tl.constexpr,
            KT: tl.constexpr, KH: tl.constexpr, KD: tl.constexpr,
            VT: tl.constexpr, VH: tl.constexpr, VD: tl.constexpr,
            PH: tl.constexpr, PS: tl.constexpr,
            A: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
            D: tl.constexpr, N: tl.constexpr, PAGE: tl.constexpr,
            CAP: tl.constexpr, CB: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr,
            SCALE: tl.constexpr, PRECISION: tl.constexpr,
            SPLITS: tl.constexpr, SPAN: tl.constexpr, PAD_M: tl.constexpr):
    head, tile = tl.program_id(0), tl.program_id(1)
    split = tl.program_id(2)
    m = tile * BM + tl.arange(0, BM)
    d = tl.arange(0, BD)
    row, qh = m // G, head * G + m % G
    q = tl.load(Q + row[:, None]*QR + qh[:, None]*QH + d[None, :]*QD,
                (row[:, None] < A) & (d[None, :] < D), other=0).to(tl.float32)
    slots = tl.arange(0, CB)
    ids = tl.load(P + head*PH + slots*PS, slots < CAP, other=-1)
    tokens = tl.sum((ids >= 0).to(tl.int32)) * PAGE
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, BD), tl.float32)
    for start in tl.range(split*SPAN, tl.minimum(tokens, (split+1)*SPAN), BN, num_stages=1):
        n = start + tl.arange(0, BN)
        page = tl.load(P + head*PH + (n//PAGE)*PS, n//PAGE < CAP, other=-1)
        token = page*PAGE + n%PAGE
        valid = (n < tokens) & (page >= 0) & (token < N)
        k = tl.load(K + token[None, :]*KT + head*KH + d[:, None]*KD,
                    valid[None, :] & (d[:, None] < D), other=0).to(tl.float32)
        score = tl.dot(q, k, input_precision=PRECISION) * SCALE
        score = tl.where(valid[None, :], score, -float("inf"))
        next_max = tl.maximum(maximum, tl.max(score, axis=1))
        safe = tl.where(next_max == -float("inf"), 0., next_max)
        alpha = tl.exp(maximum-safe)
        prob = tl.exp(score-safe[:, None])
        prob = tl.where(valid[None, :], prob, 0.)
        value = tl.load(V + token[:, None]*VT + head*VH + d[None, :]*VD,
                        valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        acc = acc*alpha[:, None] + tl.dot(prob, value, input_precision=PRECISION)
        denom = denom*alpha + tl.sum(prob, axis=1)
        maximum = next_max
    safe_denom = tl.where(denom > 0, denom, 1.)
    if SPLITS == 1:
        oi = row*H + qh
    else:
        oi = (head*SPLITS+split)*PAD_M + m
    tl.store(O + oi[:, None]*D + d[None, :],
             acc/safe_denom[:, None], (row[:, None] < A) & (d[None, :] < D))
    tl.store(L + oi, tl.where(denom > 0, maximum+tl.log(safe_denom),
                                    -float("inf")), row < A)


def attention(q, k, v, pages, *, page_size, prefix_length,
              softmax_scale=None, workspace=None, query_tile=16,
              token_tile=32, precision="tf32x3", num_warps=4, splits=4):
    a, h, d = q.shape
    hk = k.shape[1]
    if h % hk or d > 256 or query_tile not in (16, 32):
        raise ValueError("unsupported shared attention layout")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("unsupported dtype")
    out = (torch.empty_like(q) if workspace is None
           else workspace.get("prefix_out", q.shape, q))
    lse = (torch.empty((a, h), device=q.device, dtype=torch.float32) if workspace is None
           else workspace.get("prefix_lse", (a, h), q, torch.float32))
    pad_m = triton.next_power_of_2(a*(h//hk))
    span = triton.cdiv(pages.shape[1]*page_size, splits*token_tile)*token_tile
    part, part_lse = out, lse
    if splits > 1:
        shape = (hk, splits, pad_m, d)
        lshape = (hk, splits, pad_m)
        part = (torch.empty(shape, device=q.device, dtype=torch.float32) if workspace is None
                else workspace.get("shared_part", shape, q, torch.float32))
        part_lse = (torch.empty(lshape, device=q.device, dtype=torch.float32) if workspace is None
                    else workspace.get("shared_part_lse", lshape, q, torch.float32))
    _shared[(hk, triton.cdiv(a*(h//hk), query_tile), splits)](
        q,k,v,pages,part,part_lse,*q.stride(),*k.stride(),*v.stride(),*pages.stride(),
        A=a,H=h,G=h//hk,D=d,N=prefix_length,PAGE=page_size,CAP=pages.shape[1],
        CB=triton.next_power_of_2(pages.shape[1]),BM=query_tile,BN=token_tile,
        BD=triton.next_power_of_2(d),
        SCALE=d**-.5 if softmax_scale is None else softmax_scale,
        PRECISION=precision,SPLITS=splits,SPAN=span,PAD_M=pad_m,
        num_warps=num_warps,enable_fp_fusion=False)
    if splits > 1:
        from src.kernels.optimized.losa_tensorcore import _merge
        _merge[(hk,a*(h//hk))](
            part,part_lse,out,lse,A=a,H=h,G=h//hk,D=d,SPLITS=splits,
            BM=pad_m,BS=triton.next_power_of_2(splits),
            BD=triton.next_power_of_2(d),num_warps=4)
    return out,lse
