"""Opt-in CUDA warp-per-query, CTA-shared KV kernel (D128/page16)."""
from pathlib import Path
import os
import torch
from .workspace import Workspace

_extension = None


def prepare():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load
        root=Path(__file__).parent/"csrc"
        os.environ.setdefault("MAX_JOBS","2")
        _extension=load(name="losa_optimized_shared_cuda",
                        sources=[str(root/"binding.cpp"),str(root/"shared.cu")],
                        extra_cuda_cflags=["-O3"],extra_cflags=["-O3"],verbose=False)
    return _extension


def attention(q,k,v,pages,*,page_size,prefix_length,softmax_scale=None,
              workspace=None,share=2):
    if q.shape[-1]!=128 or page_size!=16 or share not in (1,2,4):
        raise ValueError("CUDA sharing supports D128/page16/share1,2,4")
    if q.dtype not in (torch.float16,torch.bfloat16) or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError("matching fp16/bf16 QKV required")
    ws=Workspace() if workspace is None else workspace
    out=ws.get("prefix_out",q.shape,q)
    lse=ws.get("prefix_lse",q.shape[:2],q,torch.float32)
    prepare().run(q,k,v,pages,out,lse,prefix_length,
                  128**-.5 if softmax_scale is None else softmax_scale,share)
    return out,lse
