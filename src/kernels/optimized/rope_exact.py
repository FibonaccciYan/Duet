"""Fused Q/K RoPE preserving eager multiply rounding before addition.

Small inference shapes only; input/output strides and full rotary dimensions
are explicit. No attention support, sampling or row-selection changes.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _rope(Q,K,C,S,OQ,OK,
          QB:tl.constexpr,QH:tl.constexpr,QR:tl.constexpr,QD:tl.constexpr,
          KB:tl.constexpr,KH:tl.constexpr,KR:tl.constexpr,KD:tl.constexpr,
          CB:tl.constexpr,CR:tl.constexpr,CD:tl.constexpr,
          SB:tl.constexpr,SR:tl.constexpr,SD:tl.constexpr,
          OQB:tl.constexpr,OQH:tl.constexpr,OQR:tl.constexpr,OQD:tl.constexpr,
          OKB:tl.constexpr,OKH:tl.constexpr,OKR:tl.constexpr,OKD:tl.constexpr,
          NQ:tl.constexpr,NK:tl.constexpr,R:tl.constexpr,D:tl.constexpr,TILE:tl.constexpr):
    b=tl.program_id(0)
    ids=tl.program_id(1)*TILE+tl.arange(0,TILE)
    width=R*D
    head=ids//width
    row=(ids%width)//D
    d=ids%D
    partner=(d+D//2)%D
    isq=head<NQ
    kh=head-NQ
    q=tl.load(Q+b*QB+head*QH+row*QR+d*QD,(head<NQ),other=0)
    qr=tl.load(Q+b*QB+head*QH+row*QR+partner*QD,(head<NQ),other=0)
    k=tl.load(K+b*KB+kh*KH+row*KR+d*KD,(kh>=0)&(kh<NK),other=0)
    kr=tl.load(K+b*KB+kh*KH+row*KR+partner*KD,(kh>=0)&(kh<NK),other=0)
    x=tl.where(isq,q,k)
    y=tl.where(isq,qr,kr)
    y=tl.where(d<D//2,-y,y)
    c=tl.load(C+b*CB+row*CR+d*CD,head<NQ+NK,other=0)
    s=tl.load(S+b*SB+row*SR+d*SD,head<NQ+NK,other=0)
    # Match two standalone dtype-rounded torch multiplies and the final add.
    a=(x.to(tl.float32)*c.to(tl.float32)).to(x.dtype).to(tl.float32)
    v=(y.to(tl.float32)*s.to(tl.float32)).to(x.dtype).to(tl.float32)
    out=a+v
    tl.store(OQ+b*OQB+head*OQH+row*OQR+d*OQD,out,head<NQ)
    tl.store(OK+b*OKB+kh*OKH+row*OKR+d*OKD,out,(kh>=0)&(kh<NK))

def apply(q,k,cos,sin):
    if (q.dtype not in (torch.float16,torch.bfloat16,torch.float32) or q.ndim!=4 or k.ndim!=4 or cos.ndim!=3 or sin.shape!=cos.shape
        or q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:]
        or cos.shape!=(q.shape[0],q.shape[2],q.shape[3])
        or q.shape[-1]%2 or q.dtype!=k.dtype or q.dtype!=cos.dtype or q.dtype!=sin.dtype
        or q.shape[-2] == 0 or not all(t.is_cuda and t.device == q.device for t in (q,k,cos,sin))):
        raise ValueError('unsupported RoPE layout or dtype')
    oq=torch.empty_like(q);ok=torch.empty_like(k)
    b,h,r,d=q.shape;hk=k.shape[1]
    _rope[(b,triton.cdiv((h+hk)*r*d,256))](q,k,cos,sin,oq,ok,
        *q.stride(),*k.stride(),*cos.stride(),*sin.stride(),*oq.stride(),*ok.stride(),
        NQ=h,NK=hk,R=r,D=d,TILE=256,num_warps=4,enable_fp_fusion=False)
    return oq,ok
