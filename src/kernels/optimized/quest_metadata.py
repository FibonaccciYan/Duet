"""Exact simultaneous page extrema; no score or selection-policy changes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _extrema(K,LO,HI,ST:tl.constexpr,SH:tl.constexpr,SD:tl.constexpr,
             LENGTH:tl.constexpr,HEADS:tl.constexpr,DIM:tl.constexpr,
             PAGE:tl.constexpr,BP:tl.constexpr,BD:tl.constexpr):
    page,head=tl.program_id(0),tl.program_id(1)
    offset=tl.arange(0,BP)
    dim=tl.arange(0,BD)
    token=page*PAGE+offset
    valid=(offset<PAGE)&(token<LENGTH)
    key=tl.load(K+token[:,None]*ST+head*SH+dim[None,:]*SD,
                mask=valid[:,None]&(dim[None,:]<DIM),other=0.).to(tl.float32)
    lo=tl.min(tl.where(valid[:,None],key,float("inf")),axis=0)
    hi=tl.max(tl.where(valid[:,None],key,-float("inf")),axis=0)
    # torch.amin/amax propagate NaNs.
    has_nan=tl.sum((valid[:,None]&(key!=key)).to(tl.int32),axis=0)>0
    lo=tl.where(has_nan,float("nan"),lo)
    hi=tl.where(has_nan,float("nan"),hi)
    output=(page*HEADS+head)*DIM+dim
    tl.store(LO+output,lo,mask=dim<DIM)
    tl.store(HI+output,hi,mask=dim<DIM)


def extrema(key,page_size):
    if key.ndim!=3 or not key.is_cuda or page_size<=0 or key.shape[0]<=0:
        raise ValueError("expected nonempty CUDA [tokens,heads,dim] and positive page size")
    length,heads,dim=key.shape
    shape=(triton.cdiv(length,page_size),heads,dim)
    lo=torch.empty(shape,device=key.device,dtype=key.dtype)
    hi=torch.empty_like(lo)
    _extrema[(shape[0],heads)](key,lo,hi,*key.stride(),LENGTH=length,HEADS=heads,
                              DIM=dim,PAGE=page_size,BP=triton.next_power_of_2(page_size),
                              BD=triton.next_power_of_2(dim),num_warps=4)
    return lo,hi
