"""Reuse the installed RMSNorm JIT kernel and its selected launch configuration."""
import torch
import triton
from flash_attn.ops.triton.layer_norm import _layer_norm_fwd, _layer_norm_fwd_1pass_kernel

_raw = _layer_norm_fwd_1pass_kernel
while not hasattr(_raw, "src"):
    _raw = _raw.fn
_configs = {}


def norm(x, weight, eps):
    shape = x.shape
    flat = x.reshape(-1,shape[-1])
    if flat.stride(-1)!=1:
        flat=flat.contiguous()
    weight=weight.contiguous()
    rows,dim=flat.shape
    key=(flat.device,flat.dtype,weight.dtype,flat.stride(),rows,dim,float(eps))
    if key not in _configs:
        result=_layer_norm_fwd(flat,weight,None,eps,is_rms_norm=True)[0]
        config=_layer_norm_fwd_1pass_kernel.best_config
        _configs[key]=dict(num_warps=config.num_warps,
                           num_stages=config.num_stages,num_ctas=config.num_ctas)
        return result.reshape(shape)
    output=torch.empty_like(flat)
    rstd=torch.empty(rows,device=x.device,dtype=torch.float32)
    _raw[(rows,)](
        flat,output,weight,None,None,None,None,None,None,None,None,None,None,None,rstd,
        flat.stride(0),output.stride(0),0,0,0,0,
        rows,dim,eps,0.0,
        IS_RMS_NORM=True,BLOCK_N=triton.next_power_of_2(dim),
        HAS_RESIDUAL=False,STORE_RESIDUAL_OUT=False,HAS_BIAS=False,
        HAS_DROPOUT=False,STORE_DROPOUT_MASK=False,HAS_ROWSCALE=False,
        HAS_X1=False,HAS_W1=False,HAS_B1=False,**_configs[key],
    )
    return output.reshape(shape)
