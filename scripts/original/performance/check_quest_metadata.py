import json
import torch
from src.optimized.losa.operators import build_page_metadata
from src.kernels.optimized.quest_metadata import extrema

torch.manual_seed(17)
def ms(fn):
    for _ in range(3):fn()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(20):fn()
    b.record();b.synchronize()
    return a.elapsed_time(b)/20

for dtype in (torch.float16,torch.bfloat16,torch.float32):
    for length,page in ((1,16),(17,16),(257,7),(8192,16),(16384,16),(32768,16)):
        key=torch.randn(8,length+32,128,device="cuda",dtype=dtype)[:,:length].transpose(0,1)
        ref=build_page_metadata(key,page)
        lo,hi=extrema(key,page)
        assert torch.equal(ref.k_min,lo) and torch.equal(ref.k_max,hi)
        old=ms(lambda:build_page_metadata(key,page));new=ms(lambda:extrema(key,page))
        print(json.dumps(dict(dtype=str(dtype),length=length,page=page,equal=True,
                              old_ms=old,new_ms=new,speedup=old/new)),flush=True)
    key=torch.randn(17,4,128,device="cuda",dtype=dtype)
    key[0,0,0]=torch.nan
    key[16,0,1]=torch.inf
    ref=build_page_metadata(key,16);lo,hi=extrema(key,16)
    torch.testing.assert_close(lo,ref.k_min,rtol=0,atol=0,equal_nan=True)
    torch.testing.assert_close(hi,ref.k_max,rtol=0,atol=0,equal_nan=True)
print("NONFINITE_PROPAGATION_OK",flush=True)
