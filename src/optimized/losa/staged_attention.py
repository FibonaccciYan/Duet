"""Pack the exact per-KV-head union once, share it across all query programs.

The bounded staging area is union-sized, never full-prefix-sized. Attention
keeps the v5 16-token FP32 reduction/online-softmax order and output dtypes.
"""
import torch
import triton
import triton.language as tl
from .workspace import Workspace


@triton.jit
def _pack(K,V,P,PK,PV,VALID,COUNT,
          KT:tl.constexpr,KH:tl.constexpr,KD:tl.constexpr,
          VT:tl.constexpr,VH:tl.constexpr,VD:tl.constexpr,
          PH:tl.constexpr,PS:tl.constexpr,N,
          PAGE:tl.constexpr,CAP:tl.constexpr,D:tl.constexpr,
          BD:tl.constexpr,CB:tl.constexpr):
    head, slot=tl.program_id(0),tl.program_id(1)
    page=tl.load(P+head*PH+slot*PS)
    n=tl.arange(0,PAGE)
    d=tl.arange(0,BD)
    token=page*PAGE+n
    valid=(page>=0)&(token<N)
    key=tl.load(K+token[:,None]*KT+head*KH+d[None,:]*KD,
                valid[:,None]&(d[None,:]<D),other=0.)
    value=tl.load(V+token[:,None]*VT+head*VH+d[None,:]*VD,
                  valid[:,None]&(d[None,:]<D),other=0.)
    offset=((head*CAP+slot)*PAGE+n[:,None])*D+d[None,:]
    tl.store(PK+offset,key,d[None,:]<D)
    tl.store(PV+offset,value,d[None,:]<D)
    tl.store(VALID+(head*CAP+slot)*PAGE+n,valid)
    if slot==0:
        ids=tl.load(P+head*PH+tl.arange(0,CB)*PS,tl.arange(0,CB)<CAP,other=-1)
        tl.store(COUNT+head,tl.sum((ids>=0).to(tl.int32))*PAGE)


@triton.jit
def _attention(Q,K,V,VALID,COUNT,O,L,
               QR:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,
               H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
               CAP_T:tl.constexpr,BD:tl.constexpr,SCALE):
    row,head=tl.program_id(0),tl.program_id(1)
    kv=head//G
    d=tl.arange(0,BD)
    q=tl.load(Q+row*QR+head*QH+d*QD,d<D,other=0.).to(tl.float32)
    count=tl.load(COUNT+kv)
    m=tl.full((1,),-float("inf"),tl.float32)
    z=tl.zeros((1,),tl.float32)
    acc=tl.zeros((BD,),tl.float32)
    for start in tl.range(0,count,16,num_stages=1):
        n=start+tl.arange(0,16)
        valid=tl.load(VALID+kv*CAP_T+n,n<CAP_T,other=0).to(tl.int1)
        key=tl.load(K+(kv*CAP_T+n[:,None])*D+d[None,:],
                    valid[:,None]&(d[None,:]<D),other=0.).to(tl.float32)
        score=tl.sum(key*q[None,:],axis=1)*SCALE
        score=tl.where(valid,score,-float("inf"))
        nm=tl.maximum(m,tl.max(score,axis=0))
        alpha=tl.exp(m-nm)
        p=tl.exp(score-nm)
        p=tl.where(valid,p,0.)
        value=tl.load(V+(kv*CAP_T+n[:,None])*D+d[None,:],
                      valid[:,None]&(d[None,:]<D),other=0.).to(tl.float32)
        acc=acc*alpha+tl.sum(p[:,None]*value,axis=0)
        z=z*alpha+tl.sum(p,axis=0)
        m=nm
    tl.store(O+(row*H+head)*D+d,acc/z,d<D)
    tl.store(L+row*H+head+tl.arange(0,1),m+tl.log(z))


def attention(q,k,v,pages,*,page_size,prefix_length,softmax_scale=None,workspace=None):
    workspace=Workspace() if workspace is None else workspace
    a,h,d=q.shape; hk=k.shape[1];cap=pages.shape[1];tokens=cap*page_size
    pk=workspace.get("packed_k",(hk,tokens,d),k)
    pv=workspace.get("packed_v",(hk,tokens,d),v)
    valid=workspace.get("packed_valid",(hk,tokens),q,torch.bool)
    count=workspace.get("packed_count",(hk,),q,torch.int32)
    out=workspace.get("prefix_out",q.shape,q)
    lse=workspace.get("prefix_lse",(a,h),q,torch.float32)
    _pack[(hk,cap)](k,v,pages,pk,pv,valid,count,*k.stride(),*v.stride(),*pages.stride(),
                    N=prefix_length,PAGE=page_size,CAP=cap,D=d,
                    BD=triton.next_power_of_2(d),CB=triton.next_power_of_2(cap),num_warps=4)
    _attention[(a,h)](q,pk,pv,valid,count,out,lse,*q.stride(),
                       H=h,G=h//hk,D=d,CAP_T=tokens,BD=triton.next_power_of_2(d),
                       SCALE=d**-.5 if softmax_scale is None else softmax_scale,num_warps=4)
    return out,lse
