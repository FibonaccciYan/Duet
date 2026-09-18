import json
import torch
import torch.nn.functional as F
from src.kernels.optimized.gqa_view import attention

torch.manual_seed(42)
for dtype in (torch.bfloat16, torch.float16):
    for length in (8192, 16384, 32768):
        for rows in (5, 17, 32):
            q = torch.randn(1, rows, 16, 128, device="cuda", dtype=dtype).transpose(1,2)
            k = torch.randn(1, length, 4, 128, device="cuda", dtype=dtype).transpose(1,2)
            v = torch.randn_like(k)
            mask = torch.zeros(1, 1, rows, length, device="cuda", dtype=dtype)
            old = lambda: F.scaled_dot_product_attention(q, k.repeat_interleave(4,1),
                        v.repeat_interleave(4,1), attn_mask=mask, scale=128**-0.5)
            new = lambda: attention(q,k,v,mask,128**-0.5)
            a,b = old(),new()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            def ms(fn):
                for _ in range(3): fn()
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):fn()
                end.record();end.synchronize()
                return begin.elapsed_time(end)/20
            before,after=ms(old),ms(new)
            print(json.dumps(dict(dtype=str(dtype),prefix=length,rows=rows,
                                  bitwise_equal=True,old_ms=before,new_ms=after,
                                  speedup=before/after)),flush=True)
