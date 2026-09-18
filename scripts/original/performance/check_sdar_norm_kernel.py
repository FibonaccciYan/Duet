import json
import time
import torch
from flash_attn.ops.triton.layer_norm import rms_norm_fn
from src.kernels.optimized.sdar_norm_launch import norm

def ms(fn):
    for _ in range(5):fn()
    times=[]
    for _ in range(5):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):fn()
        torch.cuda.synchronize();times.append((time.perf_counter()-start)*10)
    return sorted(times)[2]

torch.manual_seed(17)
for dtype in (torch.float16,torch.bfloat16):
    for dim in (128,4096):
        weight=torch.randn(dim,device="cuda",dtype=dtype)
        for rows in (1,5,17,32,544,1024):
            x=torch.randn(rows,dim,device="cuda",dtype=dtype)
            ref=lambda:rms_norm_fn(x,weight,bias=None,eps=1e-6)
            expected=ref()
            candidates=[]
            norm(x,weight,1e-6)
            for seed in range(3):
                x=torch.randn_like(x)
                expected=ref()
                actual=norm(x,weight,1e-6)
                candidates.append(dict(input_case=seed,equal=torch.equal(actual,expected),
                                       max_error=float((actual-expected).abs().max())))
            assert all(r["equal"] for r in candidates),(dtype,dim,rows,candidates)
            old,new=ms(ref),ms(lambda:norm(x,weight,1e-6))
            print(json.dumps(dict(dtype=str(dtype),dim=dim,rows=rows,candidates=candidates,
                                  old_ms=old,new_ms=new,speedup=old/new)),flush=True)
