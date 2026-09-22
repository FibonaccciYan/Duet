"""Small-candidate QK kernel, with QK reference fallback (never Raw L1)."""
import math
import torch
import triton
import triton.language as tl
from src.reference.sparse.qk_prefix import prefix_indices as reference, validate, finish_union


@triton.jit
def _qk_distance_minima(Q, K, M, I, S,
                       qs1, qs2, qs3, ks1, ks2, ks3,
                       QLEN: tl.constexpr, QHEADS: tl.constexpr, KHEADS: tl.constexpr,
                       LENGTH: tl.constexpr, DIM: tl.constexpr, CHUNKS: tl.constexpr,
                       BQ: tl.constexpr = 4, BN: tl.constexpr = 32):
    group, chunk = tl.program_id(0), tl.program_id(1)
    row = group*BQ + tl.arange(0, BQ)
    token = chunk*BN + tl.arange(0, BN)
    dim = tl.arange(0, 128)
    qh, qp = row//QLEN, row%QLEN
    kh = qh//(QHEADS//KHEADS)
    q = tl.load(Q+qh[:, None]*qs1+qp[:, None]*qs2+dim[None, :]*qs3,
                mask=row[:, None]<QHEADS*QLEN, other=0).to(tl.float32)
    k = tl.load(K+kh[:, None, None]*ks1+token[None, :, None]*ks2
                +dim[None, None, :]*ks3,
                mask=(row[:, None, None]<QHEADS*QLEN)&(token[None, :, None]<LENGTH),
                other=0).to(tl.float32)
    products = q[:, None, :]*k
    # Adjacent pair tree, identical to reference.distances; fusion disabled at launch.
    a = tl.sum(tl.reshape(products, (BQ, BN, 64, 2)), 3)
    b = tl.sum(tl.reshape(a, (BQ, BN, 32, 2)), 3)
    c = tl.sum(tl.reshape(b, (BQ, BN, 16, 2)), 3)
    d = tl.sum(tl.reshape(c, (BQ, BN, 8, 2)), 3)
    e = tl.sum(tl.reshape(d, (BQ, BN, 4, 2)), 3)
    f = tl.sum(tl.reshape(e, (BQ, BN, 2, 2)), 3)
    distance = -tl.sum(f, 2)
    distance = tl.where((row[:, None]<QHEADS*QLEN)&(token[None, :]<LENGTH),
                        distance, float("inf"))
    minimum = tl.min(distance, 1)
    index = tl.min(tl.where(distance==minimum[:, None],token[None, :],2147483647),1)
    tl.store(M+row*CHUNKS+chunk,minimum,mask=row<QHEADS*QLEN)
    tl.store(I+row*CHUNKS+chunk,index,mask=row<QHEADS*QLEN)
    tl.store(S+group*LENGTH+token,tl.min(distance,0),mask=token<LENGTH)


def summaries(query, key):
    rows = query.shape[1]*query.shape[2]
    length = key.shape[2]
    chunks, groups = triton.cdiv(length,32), triton.cdiv(rows,4)
    m = torch.empty((rows,chunks),device=query.device,dtype=torch.float32)
    i = torch.empty((rows,chunks),device=query.device,dtype=torch.int64)
    s = torch.empty((groups,length),device=query.device,dtype=torch.float32)
    _qk_distance_minima[(groups,chunks)](
        query,key,m,i,s,*query.stride()[1:],*key.stride()[1:],
        QLEN=query.shape[2],QHEADS=query.shape[1],KHEADS=key.shape[1],
        LENGTH=length,DIM=128,CHUNKS=chunks,num_warps=8,enable_fp_fusion=False)
    return m,i,s


def prefix_indices(query, key, token_budget, chunk_size=256, bucket_thresholds=None,
                   strict_budget=False, selection_stats=None):
    validate(query,key)
    length = key.shape[2]
    budget = max(0,min(int(token_budget),length))
    rows = query.shape[1]*query.shape[2]
    eligible = (query.is_cuda and key.is_cuda and query.dtype==key.dtype
                and query.dtype in (torch.float16,torch.bfloat16)
                and query.shape[-1]==128 and 0<budget<length<=2048
                and math.ceil(budget/rows)==1)
    if not eligible:
        return reference(query,key,token_budget,chunk_size,bucket_thresholds,
                         strict_budget,selection_stats)
    m,i,s = summaries(query,key)
    chunks = m.min(dim=1,keepdim=True).indices
    candidates = i.gather(1,chunks)
    return finish_union(candidates,s.amin(0),budget,strict_budget,selection_stats,1)
