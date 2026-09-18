import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.kernels.optimized.sdar_qkv import prepare, project

model,_=load_model_and_tokenizer("sdar",model_path="/data0/ysy/models/SDAR-8B-Chat-b32")
print("PREPARED",prepare(model),flush=True)
torch.manual_seed(42)
with torch.inference_mode():
    for index in (0,17,35):
        a=model.model.layers[index].self_attn
        for rows in (1,5,16,32,256):
            x=torch.randn(1,rows,model.config.hidden_size,device="cuda",dtype=torch.float16)
            q=a.q_norm(a.q_proj(x).view(1,rows,-1,a.head_dim)).transpose(1,2)
            k=a.k_norm(a.k_proj(x).view(1,rows,-1,a.head_dim)).transpose(1,2)
            v=a.v_proj(x).view(1,rows,-1,a.head_dim).transpose(1,2)
            actual=project(a,x)
            print("QKV",index,rows,
                  [torch.equal(old,new) for old,new in zip((q,k,v),actual)],
                  [float((old-new).abs().max()) for old,new in zip((q,k,v),actual)],flush=True)
            for old,new in zip((q,k,v),actual):
                torch.testing.assert_close(old,new,atol=0,rtol=0)
