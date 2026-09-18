"""Validate block visibility including a prior prefix, independent of generation."""
import json
import torch
import torch.nn.functional as F
from src.reference.sparse.triton_kernels import block_causal_prefill

torch.manual_seed(42)
for dtype in (torch.float16,torch.bfloat16):
    for prefix,rows in ((0,32),(0,96),(64,96),(128,256)):
        q=torch.randn(1,16,rows,128,device="cuda",dtype=dtype)
        k=torch.randn(1,4,prefix+rows,128,device="cuda",dtype=dtype)
        v=torch.randn_like(k)
        qb=(prefix+torch.arange(rows,device="cuda"))//32
        kb=torch.arange(prefix+rows,device="cuda")//32
        allowed=kb[None,:]<=qb[:,None]
        result=block_causal_prefill(q,k,v,32)
        reference=F.scaled_dot_product_attention(q,k.repeat_interleave(4,1),
                    v.repeat_interleave(4,1),attn_mask=allowed[None,None],is_causal=False)
        torch.testing.assert_close(result,reference,atol=0.01,rtol=0.03)
        # Future blocks must have no influence on the first block's output.
        changed=v.clone()
        changed[:,:,prefix+32:]=300
        perturbed=block_causal_prefill(q,k,changed,32)
        assert torch.equal(result[:,:,:32],perturbed[:,:,:32])
        print(json.dumps(dict(dtype=str(dtype),prefix=prefix,rows=rows,
                              future_visibility_test=True,
                              max_error=float((result-reference).abs().max()))),flush=True)
