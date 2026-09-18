import json
import torch
from src.optimized.losa.operators import dense_attention_reference as dense_attention
from src.kernels.optimized.grouped_fp32 import attention

torch.manual_seed(42)
for dtype in (torch.float16,torch.bfloat16):
    for heads,kv in ((16,4),(32,8)):
        for length in (8192,16384,32768):
            q=torch.randn(32,heads,128,device="cuda",dtype=dtype)
            k=torch.randn(kv,length,128,device="cuda",dtype=dtype).transpose(0,1)
            v=torch.randn_like(k)
            a,al=dense_attention(q,k,v); b,bl=attention(q,k,v)
            def ms(fn):
                for _ in range(3):fn()
                s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(10):fn()
                e.record();e.synchronize()
                return s.elapsed_time(e)/10
            before,after=ms(lambda:dense_attention(q,k,v)),ms(lambda:attention(q,k,v))
            print(json.dumps(dict(dtype=str(dtype),heads=heads,length=length,
                                  output_equal=torch.equal(a,b),lse_equal=torch.equal(al,bl),
                                  output_max=float((a-b).abs().max()),
                                  lse_max=float((al-bl).abs().max()),
                                  old_ms=before,new_ms=after,speedup=before/after)),flush=True)
