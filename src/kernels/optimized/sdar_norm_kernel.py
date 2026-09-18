"""Experimental inference RMSNorm preserving FlashAttention arithmetic."""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X,Y,W,SX:tl.constexpr,N:tl.constexpr,EPS:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0)
    cols=tl.arange(0,B)
    x=tl.load(X+row*SX+cols,mask=cols<N,other=0.).to(tl.float32)
    xbar=tl.where(cols<N,x,0.)
    var=tl.sum(xbar*xbar,axis=0)/N
    rstd=1/tl.sqrt(var+EPS)
    w=tl.load(W+cols,mask=cols<N,other=0.).to(tl.float32)
    y=(x*rstd)*w
    tl.store(Y+row*N+cols,y,mask=cols<N)


def norm(x,weight,eps,num_warps=4):
    shape=x.shape
    flat=x.reshape(-1,shape[-1])
    if flat.stride(-1)!=1:
        flat=flat.contiguous()
    out=torch.empty(flat.shape,device=x.device,dtype=x.dtype)
    _norm[(flat.shape[0],)](flat,out,weight,SX=flat.stride(0),N=shape[-1],
                          EPS=eps,B=triton.next_power_of_2(shape[-1]),num_warps=num_warps)
    return out.reshape(shape)
