import json
import torch
from src.optimized.losa.triton_ops import compact_prefix_attention_triton as reference
from src.kernels.optimized.losa_shared_kv import attention

torch.manual_seed(17)
for dtype in (torch.float16,torch.bfloat16):
    for heads,kv in ((16,4),(32,8)):
        for length in (8192,16384,32768):
            q=torch.randn(5,heads,128,device="cuda",dtype=dtype)
            k=torch.randn(kv,length,128,device="cuda",dtype=dtype).transpose(0,1)
            v=torch.randn_like(k)
            pages=torch.full((kv,80),-1,device="cuda",dtype=torch.long)
            for h in range(kv):
                count=40+h*5
                pages[h,:count]=torch.linspace(0,length//16-1,count,device="cuda").long()
            kw=dict(page_size=16,prefix_length=length)
            a,al=reference(q,k,v,pages,**kw)
            def ms(fn):
                for _ in range(3):fn()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20):fn()
                end.record();end.synchronize()
                return start.elapsed_time(end)/20
            old=ms(lambda:reference(q,k,v,pages,**kw))
            for tile in (1,2,4):
                b,bl=attention(q,k,v,pages,query_tile=tile,**kw)
                equal=torch.equal(a,b) and torch.equal(al,bl)
                new=ms(lambda:attention(q,k,v,pages,query_tile=tile,**kw))
                print(json.dumps(dict(dtype=str(dtype),heads=heads,length=length,tile=tile,
                                      equal=equal,output_max=float((a-b).abs().max()),
                                      lse_max=float((al-bl).abs().max()),old_ms=old,new_ms=new,
                                      speedup=old/new)),flush=True)
