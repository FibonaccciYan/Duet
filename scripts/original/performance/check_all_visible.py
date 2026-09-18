"""Test backend effects of a semantically redundant all-visible mask."""
import json
import torch
import torch.nn.functional as F
from src.kernels.optimized.gqa_view import attention

torch.manual_seed(42)
for dtype in (torch.float16,torch.bfloat16):
    for heads,kv_heads in ((16,4),(32,8)):
        for length in (8192,32768):
            q=torch.randn(1,heads,32,128,device="cuda",dtype=dtype)
            k=torch.randn(1,kv_heads,length,128,device="cuda",dtype=dtype)
            v=torch.randn_like(k)
            mask=torch.ones(1,32,length,device="cuda",dtype=torch.bool)
            functions={
                "masked":lambda:attention(q,k,v,mask,128**-0.5),
                "unmasked":lambda:attention(q,k,v,None,128**-0.5),
                "native_gqa":lambda:F.scaled_dot_product_attention(q,k,v,enable_gqa=True),
            }
            ref=functions["masked"]()
            for name,fn in functions.items():
                out=fn()
                for _ in range(3):fn()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):fn()
                end.record();end.synchronize()
                print(json.dumps(dict(dtype=str(dtype),heads=heads,length=length,name=name,
                                      ms=start.elapsed_time(end)/10,bitwise=torch.equal(ref,out),
                                      max_error=float((ref-out).abs().max()))),flush=True)
