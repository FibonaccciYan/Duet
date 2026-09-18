"""Shared GQA query tile with explicit v5 FP32 arithmetic order (D=128).

The v5 target reduction is tied to its four-warp D128 kernel. Tests must gate
this specialization; other shapes use the original kernel.
"""
import torch
import triton
import triton.language as tl
from .workspace import Workspace


@triton.jit
def _sum_prob(x,BQ:tl.constexpr):
    y=tl.sum(tl.reshape(x,(BQ,2,8)),axis=1)
    y=tl.sum(tl.reshape(y,(BQ,4,2)),axis=2)
    y=tl.sum(tl.reshape(y,(BQ,2,2)),axis=1)
    return tl.sum(y,axis=1)


@triton.jit
def _sum_value(p,v,BQ:tl.constexpr):
    lo,hi=tl.split(tl.permute(tl.reshape(v,(2,8,128)),(1,2,0)))
    p0,p1=tl.split(tl.permute(tl.reshape(p,(BQ,2,8)),(0,2,1)))
    y=tl.fma(p0[:,:,None],lo[None,:,:],p1[:,:,None]*hi[None,:,:])
    y=tl.sum(tl.reshape(y,(BQ,4,2,128)),axis=2)
    y=tl.sum(tl.reshape(y,(BQ,2,2,128)),axis=1)
    return tl.sum(y,axis=1)


@triton.jit
def _run(Q,K,V,P,O,L,
         QR:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,
         KT:tl.constexpr,KH:tl.constexpr,KD:tl.constexpr,
         VT:tl.constexpr,VH:tl.constexpr,VD:tl.constexpr,
         PH:tl.constexpr,PS:tl.constexpr,A:tl.constexpr,H:tl.constexpr,
         G:tl.constexpr,N,SCALE,PAGE:tl.constexpr,CAP:tl.constexpr,
         CB:tl.constexpr,BQ:tl.constexpr):
    kv,tile=tl.program_id(0),tl.program_id(1)
    qi=tile*BQ+tl.arange(0,BQ)
    row=qi//G;head=kv*G+qi%G
    d=tl.arange(0,16)*8
    q0=tl.load(Q+row[:,None]*QR+head[:,None]*QH+d[None,:]*QD,row[:,None]<A,other=0.).to(tl.float32)
    q1=tl.load(Q+row[:,None]*QR+head[:,None]*QH+(d[None,:]+1)*QD,row[:,None]<A,other=0.).to(tl.float32)
    q2=tl.load(Q+row[:,None]*QR+head[:,None]*QH+(d[None,:]+2)*QD,row[:,None]<A,other=0.).to(tl.float32)
    q3=tl.load(Q+row[:,None]*QR+head[:,None]*QH+(d[None,:]+3)*QD,row[:,None]<A,other=0.).to(tl.float32)
    q4=tl.load(Q+row[:,None]*QR+head[:,None]*QH+(d[None,:]+4)*QD,row[:,None]<A,other=0.).to(tl.float32)
    q5=tl.load(Q+row[:,None]*QR+head[:,None]*QH+(d[None,:]+5)*QD,row[:,None]<A,other=0.).to(tl.float32)
    q6=tl.load(Q+row[:,None]*QR+head[:,None]*QH+(d[None,:]+6)*QD,row[:,None]<A,other=0.).to(tl.float32)
    q7=tl.load(Q+row[:,None]*QR+head[:,None]*QH+(d[None,:]+7)*QD,row[:,None]<A,other=0.).to(tl.float32)
    slots=tl.arange(0,CB)
    ids=tl.load(P+kv*PH+slots*PS,slots<CAP,other=-1)
    tokens=tl.sum((ids>=0).to(tl.int32))*PAGE
    m=tl.full((BQ,),-float("inf"),tl.float32)
    z=tl.zeros((BQ,),tl.float32)
    acc=tl.zeros((BQ,128),tl.float32)
    for start in tl.range(0,tokens,16,num_stages=1):
        n=start+tl.arange(0,16)
        page=tl.load(P+kv*PH+(n//PAGE)*PS,n//PAGE<CAP,other=-1)
        token=page*PAGE+n%PAGE
        valid=(n<tokens)&(page>=0)&(token<N)
        base=token[:,None]*KT+kv*KH+d[None,:]*KD
        k0=tl.load(K+base,valid[:,None],other=0.).to(tl.float32)
        k1=tl.load(K+base+KD,valid[:,None],other=0.).to(tl.float32)
        k2=tl.load(K+base+2*KD,valid[:,None],other=0.).to(tl.float32)
        k3=tl.load(K+base+3*KD,valid[:,None],other=0.).to(tl.float32)
        k4=tl.load(K+base+4*KD,valid[:,None],other=0.).to(tl.float32)
        k5=tl.load(K+base+5*KD,valid[:,None],other=0.).to(tl.float32)
        k6=tl.load(K+base+6*KD,valid[:,None],other=0.).to(tl.float32)
        k7=tl.load(K+base+7*KD,valid[:,None],other=0.).to(tl.float32)
        score=tl.fma(k0[None,:,:],q0[:,None,:],k1[None,:,:]*q1[:,None,:])
        score=tl.fma(k2[None,:,:],q2[:,None,:],score)
        score=tl.fma(k3[None,:,:],q3[:,None,:],score)
        score=tl.fma(k4[None,:,:],q4[:,None,:],score)
        score=tl.fma(k5[None,:,:],q5[:,None,:],score)
        score=tl.fma(k6[None,:,:],q6[:,None,:],score)
        score=tl.fma(k7[None,:,:],q7[:,None,:],score)
        score=tl.sum(tl.reshape(score,(BQ,16,2,8)),axis=2)
        score=tl.sum(tl.reshape(score,(BQ,16,2,4)),axis=2)
        score=tl.sum(tl.reshape(score,(BQ,16,2,2)),axis=2)
        score=tl.sum(score,axis=2)*SCALE
        score=tl.where(valid[None,:],score,-float("inf"))
        nm=tl.maximum(m,tl.max(score,axis=1))
        alpha=tl.exp(m-nm)
        p=tl.exp(score-nm[:,None])
        p=tl.where(valid[None,:],p,0.)
        vd=tl.arange(0,128)
        value=tl.load(V+token[:,None]*VT+kv*VH+vd[None,:]*VD,
                      valid[:,None],other=0.).to(tl.float32)
        acc=tl.fma(acc,alpha[:,None],_sum_value(p,value,BQ))
        z=tl.fma(z,alpha,_sum_prob(p,BQ))
        m=nm
    d2=tl.arange(0,128)
    tl.store(O+(row[:,None]*H+head[:,None])*128+d2[None,:],acc/z[:,None],row[:,None]<A)
    tl.store(L+row*H+head,m+tl.log(z),row<A)


def attention(q,k,v,pages,*,page_size,prefix_length,softmax_scale=None,
              workspace=None,query_tile=2,num_warps=4):
    a,h,d=q.shape;hk=k.shape[1]
    if d!=128 or h%hk:raise ValueError("D128 GQA required")
    ws=Workspace() if workspace is None else workspace
    out=ws.get("prefix_out",q.shape,q)
    lse=ws.get("prefix_lse",(a,h),q,torch.float32)
    _run[(hk,triton.cdiv(a*(h//hk),query_tile))](
        q,k,v,pages,out,lse,*q.stride(),*k.stride(),*v.stride(),*pages.stride(),
        A=a,H=h,G=h//hk,N=prefix_length,SCALE=d**-.5 if softmax_scale is None else softmax_scale,
        PAGE=page_size,CAP=pages.shape[1],CB=triton.next_power_of_2(pages.shape[1]),
        BQ=query_tile,num_warps=num_warps)
    return out,lse
