"""Change launch layout only; keep the original attention reduction code."""
import json
import torch
from src.optimized.losa.triton_ops import compact_prefix_attention_triton as attention

torch.manual_seed(23)
def ms(fn):
    for _ in range(3):fn()
    values=[]
    for _ in range(3):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):fn()
        b.record();b.synchronize();values.append(a.elapsed_time(b)/10)
    return sorted(values)[1]

for dtype in (torch.float16,torch.bfloat16):
    for heads,kv in ((16,4),(32,8)):
        for capacity in (8,40,80,320):
            length=32768
            q=torch.randn(5,heads,128,device="cuda",dtype=dtype)
            k=torch.randn(kv,length,128,device="cuda",dtype=dtype).transpose(0,1)
            v=torch.randn_like(k)
            pages=torch.full((kv,capacity),-1,device="cuda",dtype=torch.long)
            for h in range(kv):
                count=max(1,capacity-h)
                pages[h,:count]=torch.linspace(0,length//16-1,count,device="cuda").long()
            kw=dict(page_size=16,prefix_length=length)
            ref,ref_lse=attention(q,k,v,pages,num_warps=4,**kw)
            baseline=ms(lambda:attention(q,k,v,pages,num_warps=4,**kw))
            for warps in (1,2,4,8):
                out,lse=attention(q,k,v,pages,num_warps=warps,**kw)
                equal=torch.equal(out,ref) and torch.equal(lse,ref_lse)
                latency=ms(lambda:attention(q,k,v,pages,num_warps=warps,**kw))
                print(json.dumps(dict(dtype=str(dtype),heads=heads,capacity=capacity,warps=warps,
                                      equal=equal,baseline_ms=baseline,latency_ms=latency,
                                      speedup=baseline/latency,
                                      lse_error=float((lse-ref_lse).abs().max()))),flush=True)
