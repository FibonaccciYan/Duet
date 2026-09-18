"""Experimental paged union attention: same support, Tensor Core arithmetic.

Each physical KV head serves all active rows and all its GQA query heads.
Selection is external and unchanged. Split softmax states use natural-log LSE.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q,K,V,P,T,L,
             QR:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,
             KT:tl.constexpr,KH:tl.constexpr,KD:tl.constexpr,
             VT:tl.constexpr,VH:tl.constexpr,VD:tl.constexpr,
             PH:tl.constexpr,PS:tl.constexpr,
             A:tl.constexpr,H:tl.constexpr,HK:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
             LENGTH:tl.constexpr,PAGE:tl.constexpr,CAP:tl.constexpr,
             SPLITS:tl.constexpr,SPAN:tl.constexpr,
             BM:tl.constexpr,BN:tl.constexpr,BD:tl.constexpr,SCALE:tl.constexpr):
    head,split=tl.program_id(0),tl.program_id(1)
    m=tl.arange(0,BM)
    d=tl.arange(0,BD)
    q=tl.load(Q+(m//G)[:,None]*QR+(head*G+m%G)[:,None]*QH+d[None,:]*QD,
              mask=(m[:,None]<A*G)&(d[None,:]<D),other=0.)
    maximum=tl.full((BM,),-float("inf"),tl.float32)
    normalizer=tl.zeros((BM,),tl.float32)
    acc=tl.zeros((BM,BD),tl.float32)
    for offset in range(split*SPAN,(split+1)*SPAN,BN):
        n=offset+tl.arange(0,BN)
        page=tl.load(P+head*PH+(n//PAGE)*PS,mask=n<CAP*PAGE,other=-1)
        token=page*PAGE+n%PAGE
        valid=(n<CAP*PAGE)&(page>=0)&(token<LENGTH)
        k=tl.load(K+token[None,:]*KT+head*KH+d[:,None]*KD,
                  mask=valid[None,:]&(d[:,None]<D),other=0.)
        score=tl.dot(q,k).to(tl.float32)*SCALE
        score=tl.where(valid[None,:],score,-float("inf"))
        new_max=tl.maximum(maximum,tl.max(score,axis=1))
        safe_max=tl.where(new_max==-float("inf"),0.,new_max)
        alpha=tl.exp(maximum-safe_max)
        probability=tl.exp(score-safe_max[:,None])
        probability=tl.where(valid[None,:],probability,0.)
        v=tl.load(V+token[:,None]*VT+head*VH+d[None,:]*VD,
                  mask=valid[:,None]&(d[None,:]<D),other=0.)
        acc=acc*alpha[:,None]+tl.dot(probability.to(v.dtype),v)
        normalizer=normalizer*alpha+tl.sum(probability,axis=1)
        maximum=new_max
    denominator=tl.where(normalizer>0,normalizer,1.)
    base=(head*SPLITS+split)*BM
    tl.store(T+(base+m[:,None])*D+d[None,:],acc/denominator[:,None],
             mask=(m[:,None]<A*G)&(d[None,:]<D))
    tl.store(L+base+m,tl.where(normalizer>0,maximum+tl.log(denominator),-float("inf")),
             mask=m<A*G)


@triton.jit
def _merge(T,L,O,Z,A:tl.constexpr,H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
           SPLITS:tl.constexpr,BM:tl.constexpr,BS:tl.constexpr,BD:tl.constexpr):
    head,m=tl.program_id(0),tl.program_id(1)
    split=tl.arange(0,BS)
    d=tl.arange(0,BD)
    base=(head*SPLITS+split)*BM+m
    lse=tl.load(L+base,mask=split<SPLITS,other=-float("inf"))
    max_lse=tl.max(lse,axis=0)
    safe=tl.where(max_lse==-float("inf"),0.,max_lse)
    weights=tl.exp(lse-safe)
    total=tl.sum(weights,axis=0)
    part=tl.load(T+base[:,None]*D+d[None,:],mask=(split[:,None]<SPLITS)&(d[None,:]<D),other=0.)
    result=tl.sum(part*weights[:,None],axis=0)/tl.where(total>0,total,1.)
    row=m//G
    qhead=head*G+m%G
    tl.store(O+(row*H+qhead)*D+d,result,mask=d<D)
    tl.store(Z+row*H+qhead,tl.where(total>0,max_lse+tl.log(total),-float("inf")))


def attention(q,k,v,pages,*,page_size,prefix_length,softmax_scale=None,splits=4):
    if q.dtype not in (torch.float16,torch.bfloat16):
        raise ValueError("Tensor Core candidate requires fp16/bf16")
    rows,heads,dim=q.shape
    hk=k.shape[1]
    if heads%hk or splits<=0 or pages.shape[1]<=0:
        raise ValueError("invalid GQA/page/split layout")
    group=heads//hk
    bm=max(16,triton.next_power_of_2(rows*group))
    bd=max(16,triton.next_power_of_2(dim))
    span=triton.cdiv(pages.shape[1]*page_size,splits*64)*64
    part=torch.empty((hk,splits,bm,dim),device=q.device,dtype=torch.float32)
    lse=torch.empty((hk,splits,bm),device=q.device,dtype=torch.float32)
    out=torch.empty((rows,heads,dim),device=q.device,dtype=q.dtype)
    final_lse=torch.empty((rows,heads),device=q.device,dtype=torch.float32)
    _partial[(hk,splits)](
        q,k,v,pages,part,lse,*q.stride(),*k.stride(),*v.stride(),*pages.stride(),
        A=rows,H=heads,HK=hk,G=group,D=dim,LENGTH=prefix_length,PAGE=page_size,
        CAP=pages.shape[1],SPLITS=splits,SPAN=span,BM=bm,BN=64,BD=bd,
        SCALE=dim**-0.5 if softmax_scale is None else softmax_scale,num_warps=4,num_stages=1)
    _merge[(hk,rows*group)](
        part,lse,out,final_lse,A=rows,H=heads,G=group,D=dim,SPLITS=splits,
        BM=bm,BS=triton.next_power_of_2(splits),BD=bd,num_warps=4)
    return out,final_lse
