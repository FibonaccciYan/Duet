import json
from types import SimpleNamespace
import torch
from src.kernels.optimized.losa_state import finish
from src.optimized.losa.operators import merge_attention_states

torch.manual_seed(23)
for dtype in (torch.float16,torch.bfloat16,torch.float32):
 for heads in (16,32):
  for count in (1,5,32):
   q=torch.randn(32,heads,128,device='cuda',dtype=dtype)
   po=torch.randn_like(q); pl=torch.randn(32,heads,device='cuda')*10
   rows=torch.randperm(32,device='cuda')[:count].contiguous()
   ao=torch.randn(count,heads,128,device='cuda',dtype=dtype)
   al=torch.randn(count,heads,device='cuda')*10
   bo=torch.randn_like(q); bl=torch.randn_like(pl)*10
   state=SimpleNamespace(query=torch.zeros_like(q),prefix_output=po.clone(),prefix_lse=pl.clone())
   refpo=po.clone();refpl=pl.clone()
   refpo.index_copy_(0,rows,ao);refpl.index_copy_(0,rows,al)
   expected,expected_lse=merge_attention_states(refpo,refpl,bo,bl)
   out,lse=finish(q,state,rows,ao,al,bo,bl)
   assert torch.equal(state.query,q)
   assert torch.equal(state.prefix_output,refpo)
   assert torch.equal(state.prefix_lse,refpl)
   assert torch.equal(out,expected) and torch.equal(lse,expected_lse)
   torch.testing.assert_close(out,expected,atol=0.002 if dtype==torch.bfloat16 else 0.0002,rtol=0.01)
   torch.testing.assert_close(lse,expected_lse,atol=4e-6,rtol=1e-6)
   print(json.dumps(dict(dtype=str(dtype),heads=heads,active=count,bitwise_output=torch.equal(out,expected),bitwise_lse=torch.equal(lse,expected_lse),max_abs=float((out-expected).abs().max()))),flush=True)

# Infinite softmax states preserve the reference's NaN/output conventions.
q=torch.randn(4,2,8,device='cuda')
pl=torch.tensor([[float('-inf'),float('inf')],[float('-inf'),0.],[0.,float('inf')],[1.,2.]],device='cuda')
bl=pl.clone()
rows=torch.tensor([3],device='cuda'); ao=q[3:].clone(); al=pl[3:].clone()
state=SimpleNamespace(query=torch.zeros_like(q),prefix_output=q.clone(),prefix_lse=pl.clone())
expected,el=merge_attention_states(q,pl,q,bl)
out,lse=finish(q,state,rows,ao,al,q,bl)
torch.testing.assert_close(out,expected,equal_nan=True)
torch.testing.assert_close(lse,el,equal_nan=True)
print('INFINITE_STATES_PASS',flush=True)
