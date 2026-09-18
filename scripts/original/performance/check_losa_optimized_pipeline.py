"""Reference-bitwise shared attention: stochastic/boundary tests and timings."""
import json
from pathlib import Path
import torch
from src.optimized.losa.triton_ops import compact_prefix_attention_triton as reference
from src.optimized.losa.shared_pipeline import attention
from src.optimized.losa.workspace import Workspace


def bench(fn):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): fn()
    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    a.record()
    for _ in range(80): graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)/80


results=[]
with torch.inference_mode():
    for dtype,h,hk in [(torch.float16,32,8),(torch.bfloat16,16,4)]:
        for length in (257,8192,16384,32512):
            for seed in (11,42,20260917):
                torch.manual_seed(seed)
                q=torch.randn(5,h,128,device="cuda",dtype=dtype)
                if seed==42: q=q*4
                if seed==20260917:q.zero_()  # exact-score ties / uniform softmax
                k=torch.randn(hk,length,128,device="cuda",dtype=dtype).transpose(0,1)
                v=torch.randn_like(k)
                cap=min(80,(length+15)//16)
                pages=torch.full((hk,cap),-1,device="cuda",dtype=torch.int64)
                for head in range(hk):
                    count=max(1,cap-head*3)
                    pages[head,:count]=torch.linspace(0,(length-1)//16,count,device="cuda").long()
                kw=dict(page_size=16,prefix_length=length)
                a,al=reference(q,k,v,pages,**kw)
                for share,warps in ((2,1),(4,1),(8,1)):
                    ws=Workspace()
                    b,bl=attention(q,k,v,pages,score_share=share,value_share=warps,workspace=ws,**kw)
                    equal=torch.equal(a,b) and torch.equal(al,bl)
                    record=dict(dtype=str(dtype),length=length,seed=seed,share=share,
                                warps=warps,equal=equal,error=(a-b).abs().max().item(),
                                lse_error=(al-bl).abs().max().item())
                    if seed==11:
                        record["old_ms"]=bench(lambda:reference(q,k,v,pages,**kw))
                        record["new_ms"]=bench(lambda:attention(q,k,v,pages,score_share=share,
                                                               value_share=warps,workspace=ws,**kw))
                    results.append(record)
                    print(record,flush=True)
    path=Path("results/losa_optimized_20260917/shared_pipeline_validation.json")
    path.write_text(json.dumps(results,indent=2))
    print("TOTAL",len(results),"BITWISE",sum(x["equal"] for x in results),flush=True)
