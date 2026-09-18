"""Bitwise output/LSE and graph-kernel latency for union staging."""
import json
from pathlib import Path
import torch
from src.optimized.losa.triton_ops import compact_prefix_attention_triton as reference
from src.optimized.losa.staged_attention import attention
from src.optimized.losa.workspace import Workspace


def bench(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    a,b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    a.record()
    for _ in range(100):
        g.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)/100


rows=[]
with torch.inference_mode():
    for dtype,h,hk in [(torch.float16,32,8),(torch.bfloat16,16,4)]:
        for length in (257,8192,32512):
            torch.manual_seed(11)
            q=torch.randn(5,h,128,device="cuda",dtype=dtype)
            k=torch.randn(hk,length,128,device="cuda",dtype=dtype).transpose(0,1)
            v=torch.randn_like(k)
            cap=min(80,(length+15)//16)
            for count in (min(16,cap),cap):
                p=torch.full((hk,cap),-1,device="cuda",dtype=torch.int64)
                p[:,:count]=torch.linspace(0,(length-1)//16,count,device="cuda").long()[None,:]
                kw=dict(page_size=16,prefix_length=length)
                a,al=reference(q,k,v,p,**kw)
                old=bench(lambda:reference(q,k,v,p,**kw))
                ws=Workspace()
                o,l=attention(q,k,v,p,workspace=ws,**kw)
                new=bench(lambda:attention(q,k,v,p,workspace=ws,**kw))
                row=dict(dtype=str(dtype),length=length,count=count,
                         equal=torch.equal(o,a) and torch.equal(l,al),
                         error=(o-a).abs().max().item(),lse=(l-al).abs().max().item(),
                         old_ms=old,new_ms=new)
                rows.append(row);print(row,flush=True)
    Path("results/losa_optimized_20260917/staged_attention_sweep.json").write_text(json.dumps(rows,indent=2))
