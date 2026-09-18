import json
import torch
from src.kernels.optimized.rope_exact import apply

def ref(q,k,c,s):
 def rotate(x):return torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),-1)
 return q*c.unsqueeze(1)+rotate(q)*s.unsqueeze(1),k*c.unsqueeze(1)+rotate(k)*s.unsqueeze(1)

def ms(fn):
 for _ in range(4):fn()
 a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
 a.record()
 for _ in range(100):fn()
 b.record();b.synchronize();return a.elapsed_time(b)/100

torch.manual_seed(42)
for dtype in (torch.float16,torch.bfloat16,torch.float32):
 for rows in (1,5,17,32):
  for layout in ('transpose','split'):
   qkv=torch.randn(1,rows,48,128,device='cuda',dtype=dtype)
   q,k,_=qkv.split((32,8,8),dim=2)
   if layout=='transpose':q=q.contiguous();k=k.contiguous()
   q=q.transpose(1,2);k=k.transpose(1,2)
   phase=torch.randn(1,rows,128,device='cuda',dtype=dtype)
   c,s=phase.cos(),phase.sin()
   a,b=ref(q,k,c,s);x,y=apply(q,k,c,s)
   assert torch.equal(a,x) and torch.equal(b,y),(dtype,rows,layout,float((a-x).abs().max()))
   old=ms(lambda:ref(q,k,c,s));new=ms(lambda:apply(q,k,c,s))
   print(json.dumps(dict(dtype=str(dtype),rows=rows,layout=layout,bitwise=True,old_ms=old,new_ms=new,speedup=old/new)),flush=True)
