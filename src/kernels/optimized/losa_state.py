"""Experimental fused LoSA state update and prefix/block softmax merge.

Selection remains external. Active rows overwrite prefix state, inactive rows
retain it. All rows update previous Q. Natural-log softmax states are merged.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def _finish(Q,PQ,PO,PL,AO,AL,ROWS,BO,BL,O,L,
            H:tl.constexpr,D:tl.constexpr,A:tl.constexpr,BA:tl.constexpr,
            BD:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    ai=tl.arange(0,BA)
    selected=tl.load(ROWS+ai,ai<A,other=-1)
    match=(selected==row)&(ai<A)
    found=tl.sum(match.to(tl.int32),0)>0
    index=tl.sum(tl.where(match,ai,0),0)
    d=tl.arange(0,BD)
    pos=(row*H+head)*D+d
    old=tl.load(PO+pos,d<D,other=0).to(tl.float32)
    new=tl.load(AO+(index*H+head)*D+d,(d<D)&found,other=0).to(tl.float32)
    prefix=tl.where(found,new,old)
    pl=tl.load(PL+row*H+head)
    al=tl.load(AL+index*H+head,found,other=0)
    pl=tl.where(found,al,pl)
    bl=tl.load(BL+row*H+head)
    maximum=tl.maximum(pl,bl)
    total=maximum+libdevice.log1p(libdevice.exp(-tl.abs(pl-bl)))
    total=tl.where((pl==bl)&(tl.abs(pl)==float("inf")),pl,total)
    pw=libdevice.exp(pl-total)
    bw=libdevice.exp(bl-total)
    block=tl.load(BO+pos,d<D,other=0).to(tl.float32)
    result=prefix*pw+block*bw
    tl.store(PO+pos,prefix,d<D)
    tl.store(PL+row*H+head,pl)
    tl.store(O+pos,result,d<D)
    tl.store(L+row*H+head,total)
    q=tl.load(Q+pos,d<D,other=0)
    tl.store(PQ+pos,q,d<D)

def finish(query,state,active_rows,active_output,active_lse,block_output,block_lse):
    tensors=(query,state.query,state.prefix_output,state.prefix_lse,
             active_output,active_lse,active_rows,block_output,block_lse)
    if not all(t.is_cuda and t.is_contiguous() for t in tensors):
        raise ValueError('fused state requires contiguous CUDA tensors')
    r,h,d=query.shape
    out=torch.empty_like(state.prefix_output)
    lse=torch.empty_like(state.prefix_lse)
    _finish[(r,h)](*tensors,out,lse,H=h,D=d,A=active_rows.numel(),
                  BA=triton.next_power_of_2(active_rows.numel()),
                  BD=triton.next_power_of_2(d),num_warps=4,enable_fp_fusion=False)
    return out,lse
