"""Shared KV loads with explicit independent reference-order FP32 recurrences."""
import torch
import triton
import triton.language as tl


@triton.jit
def _sum16(x):
    # v5 four-warp reduction: fold the two register groups first,
    # then combine adjacent lane/warp pairs in the eight-way reduction.
    y=tl.sum(tl.reshape(x,(2,8)),axis=0)
    y=tl.sum(tl.reshape(y,(4,2)),axis=1)
    y=tl.sum(tl.reshape(y,(2,2)),axis=0)
    return tl.sum(y,axis=0)

@triton.jit
def _sum16v(prob,value,BD:tl.constexpr):
    low,high=tl.split(tl.permute(tl.reshape(value,(2,8,BD)),(1,2,0)))
    p0,p1=tl.split(tl.trans(tl.reshape(prob,(2,8))))
    y=tl.fma(p0[:,None],low,p1[:,None]*high)
    y=tl.sum(tl.reshape(y,(4,2,BD)),axis=1)
    y=tl.sum(tl.reshape(y,(2,2,BD)),axis=0)
    return tl.sum(y,axis=0)

@triton.jit
def _shared_scalar(Q,K,V,P,O,L, QR:tl.constexpr,QH:tl.constexpr,QD:tl.constexpr,
                   KT:tl.constexpr,KH:tl.constexpr,KD:tl.constexpr,
                   VT:tl.constexpr,VH:tl.constexpr,VD:tl.constexpr,
                   PH:tl.constexpr,PS:tl.constexpr,A:tl.constexpr,H:tl.constexpr,
                   G:tl.constexpr,D:tl.constexpr,N,SCALE,PAGE:tl.constexpr,
                   CAP:tl.constexpr,CB:tl.constexpr,BD:tl.constexpr,SHARE:tl.constexpr):
    head,tile=tl.program_id(0),tl.program_id(1)
    d=tl.arange(0,BD)
    slots=tl.arange(0,CB)
    page_ids=tl.load(P+head*PH+slots*PS,mask=slots<CAP,other=-1)
    tokens=tl.sum((page_ids>=0).to(tl.int32))*PAGE
    if SHARE > 0:
        row0=(tile*SHARE+0)//G
        qh0=head*G+(tile*SHARE+0)%G
        q0=tl.load(Q+row0*QR+qh0*QH+d*QD,mask=(row0<A)&(d<D),other=0.).to(tl.float32)
        m0=tl.full((1,),-float("inf"),tl.float32)
        z0=tl.zeros((1,),tl.float32)
        acc0=tl.zeros((BD,),tl.float32)
    if SHARE > 1:
        row1=(tile*SHARE+1)//G
        qh1=head*G+(tile*SHARE+1)%G
        q1=tl.load(Q+row1*QR+qh1*QH+d*QD,mask=(row1<A)&(d<D),other=0.).to(tl.float32)
        m1=tl.full((1,),-float("inf"),tl.float32)
        z1=tl.zeros((1,),tl.float32)
        acc1=tl.zeros((BD,),tl.float32)
    if SHARE > 2:
        row2=(tile*SHARE+2)//G
        qh2=head*G+(tile*SHARE+2)%G
        q2=tl.load(Q+row2*QR+qh2*QH+d*QD,mask=(row2<A)&(d<D),other=0.).to(tl.float32)
        m2=tl.full((1,),-float("inf"),tl.float32)
        z2=tl.zeros((1,),tl.float32)
        acc2=tl.zeros((BD,),tl.float32)
    if SHARE > 3:
        row3=(tile*SHARE+3)//G
        qh3=head*G+(tile*SHARE+3)%G
        q3=tl.load(Q+row3*QR+qh3*QH+d*QD,mask=(row3<A)&(d<D),other=0.).to(tl.float32)
        m3=tl.full((1,),-float("inf"),tl.float32)
        z3=tl.zeros((1,),tl.float32)
        acc3=tl.zeros((BD,),tl.float32)
    if SHARE > 4:
        row4=(tile*SHARE+4)//G
        qh4=head*G+(tile*SHARE+4)%G
        q4=tl.load(Q+row4*QR+qh4*QH+d*QD,mask=(row4<A)&(d<D),other=0.).to(tl.float32)
        m4=tl.full((1,),-float("inf"),tl.float32)
        z4=tl.zeros((1,),tl.float32)
        acc4=tl.zeros((BD,),tl.float32)
    if SHARE > 5:
        row5=(tile*SHARE+5)//G
        qh5=head*G+(tile*SHARE+5)%G
        q5=tl.load(Q+row5*QR+qh5*QH+d*QD,mask=(row5<A)&(d<D),other=0.).to(tl.float32)
        m5=tl.full((1,),-float("inf"),tl.float32)
        z5=tl.zeros((1,),tl.float32)
        acc5=tl.zeros((BD,),tl.float32)
    if SHARE > 6:
        row6=(tile*SHARE+6)//G
        qh6=head*G+(tile*SHARE+6)%G
        q6=tl.load(Q+row6*QR+qh6*QH+d*QD,mask=(row6<A)&(d<D),other=0.).to(tl.float32)
        m6=tl.full((1,),-float("inf"),tl.float32)
        z6=tl.zeros((1,),tl.float32)
        acc6=tl.zeros((BD,),tl.float32)
    if SHARE > 7:
        row7=(tile*SHARE+7)//G
        qh7=head*G+(tile*SHARE+7)%G
        q7=tl.load(Q+row7*QR+qh7*QH+d*QD,mask=(row7<A)&(d<D),other=0.).to(tl.float32)
        m7=tl.full((1,),-float("inf"),tl.float32)
        z7=tl.zeros((1,),tl.float32)
        acc7=tl.zeros((BD,),tl.float32)
    for start in tl.range(0,tokens,16,num_stages=1):
        packed=start+tl.arange(0,16)
        page=tl.load(P+head*PH+(packed//PAGE)*PS,mask=packed//PAGE<CAP,other=-1)
        token=page*PAGE+packed%PAGE
        valid=(packed<tokens)&(page>=0)&(token<N)
        key=tl.load(K+token[:,None]*KT+head*KH+d[None,:]*KD,mask=valid[:,None]&(d[None,:]<D),other=0.).to(tl.float32)
        if SHARE > 0:
            score0=tl.sum(key*q0[None,:],axis=1)*SCALE
            score0=tl.where(valid,score0,-float("inf"))
            nm0=tl.maximum(m0,tl.max(score0,axis=0))
            alpha0=tl.exp(m0-nm0)
            p0=tl.exp(score0-nm0)
            p0=tl.where(valid,p0,0.)
        if SHARE > 1:
            score1=tl.sum(key*q1[None,:],axis=1)*SCALE
            score1=tl.where(valid,score1,-float("inf"))
            nm1=tl.maximum(m1,tl.max(score1,axis=0))
            alpha1=tl.exp(m1-nm1)
            p1=tl.exp(score1-nm1)
            p1=tl.where(valid,p1,0.)
        if SHARE > 2:
            score2=tl.sum(key*q2[None,:],axis=1)*SCALE
            score2=tl.where(valid,score2,-float("inf"))
            nm2=tl.maximum(m2,tl.max(score2,axis=0))
            alpha2=tl.exp(m2-nm2)
            p2=tl.exp(score2-nm2)
            p2=tl.where(valid,p2,0.)
        if SHARE > 3:
            score3=tl.sum(key*q3[None,:],axis=1)*SCALE
            score3=tl.where(valid,score3,-float("inf"))
            nm3=tl.maximum(m3,tl.max(score3,axis=0))
            alpha3=tl.exp(m3-nm3)
            p3=tl.exp(score3-nm3)
            p3=tl.where(valid,p3,0.)
        if SHARE > 4:
            score4=tl.sum(key*q4[None,:],axis=1)*SCALE
            score4=tl.where(valid,score4,-float("inf"))
            nm4=tl.maximum(m4,tl.max(score4,axis=0))
            alpha4=tl.exp(m4-nm4)
            p4=tl.exp(score4-nm4)
            p4=tl.where(valid,p4,0.)
        if SHARE > 5:
            score5=tl.sum(key*q5[None,:],axis=1)*SCALE
            score5=tl.where(valid,score5,-float("inf"))
            nm5=tl.maximum(m5,tl.max(score5,axis=0))
            alpha5=tl.exp(m5-nm5)
            p5=tl.exp(score5-nm5)
            p5=tl.where(valid,p5,0.)
        if SHARE > 6:
            score6=tl.sum(key*q6[None,:],axis=1)*SCALE
            score6=tl.where(valid,score6,-float("inf"))
            nm6=tl.maximum(m6,tl.max(score6,axis=0))
            alpha6=tl.exp(m6-nm6)
            p6=tl.exp(score6-nm6)
            p6=tl.where(valid,p6,0.)
        if SHARE > 7:
            score7=tl.sum(key*q7[None,:],axis=1)*SCALE
            score7=tl.where(valid,score7,-float("inf"))
            nm7=tl.maximum(m7,tl.max(score7,axis=0))
            alpha7=tl.exp(m7-nm7)
            p7=tl.exp(score7-nm7)
            p7=tl.where(valid,p7,0.)
        value=tl.load(V+token[:,None]*VT+head*VH+d[None,:]*VD,mask=valid[:,None]&(d[None,:]<D),other=0.).to(tl.float32)
        if SHARE > 0:
            acc0=acc0*alpha0+_sum16v(p0,value,BD)
            z0=z0*alpha0+_sum16(p0)
            m0=nm0
        if SHARE > 1:
            acc1=acc1*alpha1+_sum16v(p1,value,BD)
            z1=z1*alpha1+_sum16(p1)
            m1=nm1
        if SHARE > 2:
            acc2=acc2*alpha2+_sum16v(p2,value,BD)
            z2=z2*alpha2+_sum16(p2)
            m2=nm2
        if SHARE > 3:
            acc3=acc3*alpha3+_sum16v(p3,value,BD)
            z3=z3*alpha3+_sum16(p3)
            m3=nm3
        if SHARE > 4:
            acc4=acc4*alpha4+_sum16v(p4,value,BD)
            z4=z4*alpha4+_sum16(p4)
            m4=nm4
        if SHARE > 5:
            acc5=acc5*alpha5+_sum16v(p5,value,BD)
            z5=z5*alpha5+_sum16(p5)
            m5=nm5
        if SHARE > 6:
            acc6=acc6*alpha6+_sum16v(p6,value,BD)
            z6=z6*alpha6+_sum16(p6)
            m6=nm6
        if SHARE > 7:
            acc7=acc7*alpha7+_sum16v(p7,value,BD)
            z7=z7*alpha7+_sum16(p7)
            m7=nm7
    scalar=tl.arange(0,1)
    if SHARE > 0:
        tl.store(O+(row0*H+qh0)*D+d,acc0/z0,mask=(row0<A)&(d<D))
        tl.store(L+row0*H+qh0+scalar,m0+tl.log(z0),mask=row0<A)
    if SHARE > 1:
        tl.store(O+(row1*H+qh1)*D+d,acc1/z1,mask=(row1<A)&(d<D))
        tl.store(L+row1*H+qh1+scalar,m1+tl.log(z1),mask=row1<A)
    if SHARE > 2:
        tl.store(O+(row2*H+qh2)*D+d,acc2/z2,mask=(row2<A)&(d<D))
        tl.store(L+row2*H+qh2+scalar,m2+tl.log(z2),mask=row2<A)
    if SHARE > 3:
        tl.store(O+(row3*H+qh3)*D+d,acc3/z3,mask=(row3<A)&(d<D))
        tl.store(L+row3*H+qh3+scalar,m3+tl.log(z3),mask=row3<A)
    if SHARE > 4:
        tl.store(O+(row4*H+qh4)*D+d,acc4/z4,mask=(row4<A)&(d<D))
        tl.store(L+row4*H+qh4+scalar,m4+tl.log(z4),mask=row4<A)
    if SHARE > 5:
        tl.store(O+(row5*H+qh5)*D+d,acc5/z5,mask=(row5<A)&(d<D))
        tl.store(L+row5*H+qh5+scalar,m5+tl.log(z5),mask=row5<A)
    if SHARE > 6:
        tl.store(O+(row6*H+qh6)*D+d,acc6/z6,mask=(row6<A)&(d<D))
        tl.store(L+row6*H+qh6+scalar,m6+tl.log(z6),mask=row6<A)
    if SHARE > 7:
        tl.store(O+(row7*H+qh7)*D+d,acc7/z7,mask=(row7<A)&(d<D))
        tl.store(L+row7*H+qh7+scalar,m7+tl.log(z7),mask=row7<A)

def attention(q,k,v,pages,*,page_size,prefix_length,softmax_scale=None,workspace=None,share=2,num_warps=1):
    a,h,d=q.shape
    hk=k.shape[1];g=h//hk
    if h%hk or share not in (1,2,4,8):raise ValueError("unsupported layout")
    out=torch.empty_like(q) if workspace is None else workspace.get("prefix_out",q.shape,q)
    lse=torch.empty((a,h),device=q.device,dtype=torch.float32) if workspace is None else workspace.get("prefix_lse",(a,h),q,torch.float32)
    _shared_scalar[(hk,triton.cdiv(a*g,share))](q,k,v,pages,out,lse,*q.stride(),*k.stride(),*v.stride(),*pages.stride(),
        A=a,H=h,G=g,D=d,N=prefix_length,SCALE=d**-.5 if softmax_scale is None else softmax_scale,
        PAGE=page_size,CAP=pages.shape[1],CB=triton.next_power_of_2(pages.shape[1]),BD=triton.next_power_of_2(d),SHARE=share,num_warps=num_warps)
    return out,lse
