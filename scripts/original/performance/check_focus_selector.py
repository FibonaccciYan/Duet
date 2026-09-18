import json
import time
import torch
from src.reference.focus.algorithm import select_retained_positions as reference
from src.optimized.focus.algorithm import select_retained_positions as candidate

torch.manual_seed(7)
cases=0
for dtype in (torch.float16,torch.bfloat16,torch.float32):
    for length in (1,5,17,32,64):
        for pattern in ("all","alternating","random","none"):
            tokens=torch.arange(length,device="cuda")
            mask=(torch.ones(length,device="cuda",dtype=torch.bool) if pattern=="all" else
                  tokens%2==0 if pattern=="alternating" else
                  torch.rand(length,device="cuda")>0.4 if pattern=="random" else
                  torch.zeros(length,device="cuda",dtype=torch.bool))
            tokens=tokens.masked_fill(mask,99)
            count=int(mask.sum())
            a=torch.randn(count,device="cuda",dtype=dtype)
            for equal in (False,True):
                b=a.clone() if equal else torch.randn_like(a)
                for progress in (-1,length//2,length):
                    kw=dict(alpha=1.5,average_decoded_tokens=2.0,block_progress=progress)
                    expected=reference(tokens,99,a,b,**kw)
                    actual=candidate(tokens,99,a,b,**kw)
                    assert torch.equal(expected,actual),(dtype,length,pattern,equal,progress)
                    cases+=1
tokens=torch.full((32,),99,device="cuda",dtype=torch.long)
a=torch.randn(32,device="cuda");b=torch.randn_like(a)
kw=dict(alpha=1.5,average_decoded_tokens=1.0,block_progress=7)
def ms(fn):
    for _ in range(5):fn()
    samples=[]
    for _ in range(5):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(100):fn()
        torch.cuda.synchronize();samples.append((time.perf_counter()-start)*10)
    return sorted(samples)[2]
old=ms(lambda:reference(tokens,99,a,b,**kw))
new=ms(lambda:candidate(tokens,99,a,b,**kw))
print(json.dumps(dict(cases=cases,all_positions_equal=True,old_ms=old,new_ms=new,speedup=old/new)),flush=True)
