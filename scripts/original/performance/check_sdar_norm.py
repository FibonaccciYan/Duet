import json
import time
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.kernels.optimized.sdar_norm import install

model,_=load_model_and_tokenizer("sdar",model_path="/data0/ysy/models/SDAR-8B-Chat-b32")
print("PATCHED",install(model),flush=True)
torch.manual_seed(42)
def ms(fn):
    for _ in range(5):fn()
    times=[]
    for _ in range(5):
        torch.cuda.synchronize();t=time.perf_counter()
        for _ in range(100):fn()
        torch.cuda.synchronize();times.append((time.perf_counter()-t)*10)
    return sorted(times)[2]

with torch.inference_mode():
    for norm in (model.model.layers[0].input_layernorm,model.model.layers[0].self_attn.q_norm):
        dim=norm.weight.numel()
        for rows in (5,17,32):
            shape=(1,rows,dim) if dim==4096 else (1,rows,32,dim)
            x=torch.randn(*shape,device="cuda",dtype=torch.float16)
            for _ in range(4):
                x=torch.randn_like(x)
                a=norm._versioned_norm_original(x);b=norm(x)
                assert torch.equal(a,b)
            old,new=ms(lambda:norm._versioned_norm_original(x)),ms(lambda:norm(x))
            print(json.dumps(dict(dim=dim,rows=rows,bitwise=True,old_ms=old,new_ms=new,
                                  speedup=old/new)),flush=True)
