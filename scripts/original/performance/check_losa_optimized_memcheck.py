"""Small real-shape bounds checks for compute-sanitizer, including partial pages."""
import torch
from src.optimized.losa.triton_ops import compact_prefix_attention_triton as reference
from src.optimized.losa.shared_pipeline import attention
from src.optimized.losa.workspace import Workspace

with torch.inference_mode():
    for dtype,h,hk in [(torch.float16,32,8),(torch.bfloat16,16,4)]:
        for rows in (1,5):
            for length in (17,257):
                torch.manual_seed(51)
                q=torch.randn(rows,h,128,device="cuda",dtype=dtype)
                k=torch.randn(hk,length,128,device="cuda",dtype=dtype).transpose(0,1)
                v=torch.randn_like(k)
                count=(length+15)//16
                pages=torch.arange(count,device="cuda")[None,:].expand(hk,count).contiguous()
                ws=Workspace()
                expected,lse=reference(q,k,v,pages,page_size=16,prefix_length=length)
                actual,al=attention(q,k,v,pages,page_size=16,prefix_length=length,workspace=ws)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                torch.testing.assert_close(al,lse,atol=0,rtol=0)
    torch.cuda.synchronize()
print("MEMCHECK_CASES_EXACT",flush=True)
