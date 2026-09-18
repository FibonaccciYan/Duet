"""Exact selector comparisons: random, ties, partial tiles, GQA and fallback."""
import json
import time
import torch
from src.reference.sparse.sparse_ops import _raw_l1_prefix_indices
from src.optimized.sparse.prefix import prefix_indices


def ms(fn):
    for _ in range(2): fn()
    samples=[]
    for _ in range(5):
        torch.cuda.synchronize()
        start=time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return sorted(samples)[2]


torch.manual_seed(7)
for dtype in (torch.bfloat16, torch.float16):
    for length, qlen, hq, hk, budget, ties in (
        (8192,32,16,4,256,False), (16384,32,16,4,256,False),
        (32768,32,16,4,256,False), (257,3,8,2,16,False),
        (128,32,16,4,64,True), (1024,32,32,8,256,False),
        (127,2,8,2,32,False), (63,32,16,4,256,False),
    ):
        q = torch.randn(1,qlen,hq,128,device="cuda",dtype=dtype).transpose(1,2)
        k = torch.randn(1,length,hk,128,device="cuda",dtype=dtype).transpose(1,2)
        if ties:
            q.zero_();k.zero_()
        old=lambda:_raw_l1_prefix_indices(q,k,budget)
        new=lambda:prefix_indices(q,k,budget)
        a,b=old(),new()
        assert torch.equal(a,b), (length,qlen,dtype, a.tolist(),b.tolist())
        before,after=ms(old),ms(new)
        print(json.dumps(dict(dtype=str(dtype),length=length,qlen=qlen,hq=hq,
                              budget=budget,ties=ties,indices_equal=True,
                              old_ms=before,new_ms=after,speedup=before/after)),flush=True)
