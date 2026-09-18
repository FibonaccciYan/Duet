"""Same-input all-visible attention and selector-input audit for FOCUS v5."""
from types import SimpleNamespace
import json
import torch
from src.optimized.focus.model import _attention_once,_make_cache

torch.manual_seed(42)
for dtype in (torch.float16,torch.bfloat16):
 for family,heads,hkv in [('llada',16,4),('sdar',32,8)]:
  dim=128
  for rows in (1,5,32):
   hidden=torch.randn(1,rows,heads*dim,device='cuda',dtype=dtype)
   cos=torch.ones(1,rows,dim,device='cuda',dtype=dtype);sin=torch.zeros_like(cos)
   def kv(x):return x[...,:hkv*dim]
   identity=lambda x:x
   attn=SimpleNamespace(layer_idx=0,head_dim=dim,num_heads=heads,num_key_value_heads=hkv,
     config=SimpleNamespace(use_qk_norm=False),q_proj=identity,k_proj=kv,v_proj=kv,
     q_norm=identity,k_norm=identity,o_proj=identity,dense=identity,
     query_key_value=lambda x:torch.cat((x,kv(x),kv(x)),dim=-1),scaling=dim**-.5)
   prefix=((torch.randn(1,hkv,8192,dim,device='cuda',dtype=dtype),
            torch.randn(1,hkv,8192,dim,device='cuda',dtype=dtype)),)
   a,b=_make_cache(prefix),_make_cache(prefix)
   mask=torch.ones(1,1,rows,8192+rows,device='cuda',dtype=torch.bool)
   q,k,out=_attention_once(attn,hidden,family=family,position_embeddings=(cos,sin),
        attention_mask=mask,past_key_value=a,store_kv=True,attention_backend='sdpa')
   q2,k2,out2=_attention_once(attn,hidden,family=family,position_embeddings=(cos,sin),
        attention_mask=None,past_key_value=b,store_kv=True,attention_backend='flash')
   assert torch.equal(q,q2) and torch.equal(k,k2)
   assert all(torch.equal(x,y) for x,y in zip(a[0],b[0]))
   torch.testing.assert_close(out,out2,atol=.02 if dtype==torch.bfloat16 else .003,rtol=.03)
   print(json.dumps(dict(dtype=str(dtype),family=family,rows=rows,max_abs=float((out-out2).abs().max()),
                         selector_inputs_equal=True,cache_equal=True)),flush=True)
