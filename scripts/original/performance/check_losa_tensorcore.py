import json
import torch
from src.optimized.losa.triton_ops import compact_prefix_attention_triton as reference
from src.kernels.optimized.losa_tensorcore import attention

torch.manual_seed(42)
def ms(fn):
    for _ in range(3):fn()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):fn()
    b.record();b.synchronize()
    return a.elapsed_time(b)/20

for dtype in (torch.float16,torch.bfloat16):
    for heads,hk in ((16,4),(32,8)):
        for length,cap in ((8192,40),(16384,80),(32761,320)):
            q=torch.randn(5,heads,128,device="cuda",dtype=dtype)
            k=torch.randn(hk,length,128,device="cuda",dtype=dtype).transpose(0,1)
            v=torch.randn_like(k)
            pages=torch.full((hk,cap),-1,device="cuda",dtype=torch.long)
            for h in range(hk):
                n=max(1,cap-h*2)
                pages[h,:n]=torch.linspace(0,(length-1)//16,n,device="cuda").long()
            kw=dict(page_size=16,prefix_length=length)
            a,al=reference(q,k,v,pages,**kw)
            old=ms(lambda:reference(q,k,v,pages,**kw))
            for splits in (1,4,8):
                b,bl=attention(q,k,v,pages,splits=splits,**kw)
                torch.testing.assert_close(a,b,atol=0.002,rtol=0.02)
                torch.testing.assert_close(al,bl,atol=0.0001,rtol=0)
                new=ms(lambda:attention(q,k,v,pages,splits=splits,**kw))
                print(json.dumps(dict(dtype=str(dtype),heads=heads,length=length,cap=cap,splits=splits,
                                      old_ms=old,new_ms=new,speedup=old/new,
                                      output_max=float((a-b).abs().max()),
                                      lse_max=float((al-bl).abs().max()))),flush=True)
