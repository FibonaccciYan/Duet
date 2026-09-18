import json
import torch
from src.optimized.losa.operators import dense_attention
from src.kernels.optimized.flash_lse import attention

torch.manual_seed(42)
for dtype in (torch.float16, torch.bfloat16):
    for length in (8192,16384,32768):
        q=torch.randn(32,16,128,device="cuda",dtype=dtype)
        k=torch.randn(4,length,128,device="cuda",dtype=dtype).transpose(0,1)
        v=torch.randn_like(k)
        a,al=dense_attention(q,k,v)
        b,bl=attention(q,k,v)
        torch.testing.assert_close(b,a,rtol=0.02,atol=0.002)
        torch.testing.assert_close(bl,al,rtol=0,atol=0.0001)
        def ms(fn):
            for _ in range(3):fn()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(10):fn()
            end.record();end.synchronize()
            return begin.elapsed_time(end)/10
        old,new=ms(lambda:dense_attention(q,k,v)),ms(lambda:attention(q,k,v))
        print(json.dumps(dict(dtype=str(dtype),length=length,old_ms=old,new_ms=new,
                              speedup=old/new,output_max=float((a-b).abs().max()),
                              lse_max=float((al-bl).abs().max()))),flush=True)
