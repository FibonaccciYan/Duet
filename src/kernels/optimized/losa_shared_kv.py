"""Experimental SIMD query-head grouping; unchanged FP32 attention reductions."""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q,K,V,P,O,L,
               QR:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,
               KT:tl.constexpr,KH:tl.constexpr,KD:tl.constexpr,
               VT:tl.constexpr,VH:tl.constexpr,VD:tl.constexpr,
               PH:tl.constexpr,PS:tl.constexpr,
               H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,N:tl.constexpr,
               PAGE:tl.constexpr,CAP:tl.constexpr,CB:tl.constexpr,
               BQ:tl.constexpr,BD:tl.constexpr,SCALE:tl.constexpr):
    row=tl.program_id(0)
    group=tl.program_id(1)
    heads=group*BQ+tl.arange(0,BQ)
    kv_head=(group*BQ)//G
    d=tl.arange(0,BD)
    q=tl.load(Q+row*QR+heads[:,None]*QH+d[None,:]*QD,
              mask=(heads[:,None]<H)&(d[None,:]<D),other=0.).to(tl.float32)
    slots=tl.arange(0,CB)
    ids=tl.load(P+kv_head*PH+slots*PS,mask=slots<CAP,other=-1)
    valid_pages=tl.sum((ids>=0).to(tl.int32))
    max_tokens=valid_pages*PAGE
    m=tl.full((BQ,),-float("inf"),tl.float32)
    l=tl.zeros((BQ,),tl.float32)
    acc=tl.zeros((BQ,BD),tl.float32)
    for start in tl.range(0,max_tokens,16,num_stages=1):
        packed=start+tl.arange(0,16)
        page=tl.load(P+kv_head*PH+(packed//PAGE)*PS,
                     mask=packed//PAGE<CAP,other=-1)
        token=page*PAGE+packed%PAGE
        valid=(packed<max_tokens)&(page>=0)&(token<N)
        key=tl.load(K+token[:,None]*KT+kv_head*KH+d[None,:]*KD,
                    mask=valid[:,None]&(d[None,:]<D),other=0.).to(tl.float32)
        score=tl.sum(q[:,None,:]*key[None,:,:],axis=2)*SCALE
        score=tl.where(valid[None,:],score,-float("inf"))
        new_m=tl.maximum(m,tl.max(score,axis=1))
        old_scale=tl.exp(m-new_m)
        prob=tl.exp(score-new_m[:,None])
        prob=tl.where(valid[None,:],prob,0.)
        value=tl.load(V+token[:,None]*VT+kv_head*VH+d[None,:]*VD,
                      mask=valid[:,None]&(d[None,:]<D),other=0.).to(tl.float32)
        acc=acc*old_scale[:,None]+tl.sum(prob[:,:,None]*value[None,:,:],axis=1)
        l=l*old_scale+tl.sum(prob,axis=1)
        m=new_m
    tl.store(O+(row*H+heads[:,None])*D+d[None,:],acc/l[:,None],
             mask=(heads[:,None]<H)&(d[None,:]<D))
    tl.store(L+row*H+heads,m+tl.log(l),mask=heads<H)


def attention(q,k,v,pages,*,page_size,prefix_length,softmax_scale=None,query_tile=2):
    rows,heads,dim=q.shape
    group=heads//k.shape[1]
    if heads%k.shape[1] or group%query_tile:
        raise ValueError("query tile must divide the GQA group")
    out=torch.empty((rows,heads,dim),device=q.device,dtype=q.dtype)
    lse=torch.empty((rows,heads),device=q.device,dtype=torch.float32)
    _attention[(rows,triton.cdiv(heads,query_tile))](
        q,k,v,pages,out,lse,*q.stride(),*k.stride(),*v.stride(),*pages.stride(),
        H=heads,G=group,D=dim,N=prefix_length,PAGE=page_size,CAP=pages.shape[1],
        CB=triton.next_power_of_2(pages.shape[1]),BQ=query_tile,
        BD=triton.next_power_of_2(dim),SCALE=dim**-0.5 if softmax_scale is None else softmax_scale,
        num_warps=4)
    return out,lse
