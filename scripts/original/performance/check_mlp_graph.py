import argparse
import json
import time
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.reference.losa.moe_patch import patch_moe_experts
from src.kernels.optimized.small_moe import install_small_moe
from src.kernels.optimized.mlp_graph import ExactShapeGraph

p=argparse.ArgumentParser()
p.add_argument("--family",choices=["llada","sdar"],required=True)
args=p.parse_args()
path="/data0/ysy/models/LLaDA2.1-mini" if args.family=="llada" else "/data0/ysy/models/SDAR-8B-Chat-b32"
model,_=load_model_and_tokenizer(args.family,model_path=path)
patch_moe_experts(model,family=args.family)
install_small_moe(model)

def check(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b), (a.shape,float((a-b).abs().max()))
    elif a is not None:
        for x,y in zip(a,b):check(x,y)

def ms(fn):
    for _ in range(2):fn()
    values=[]
    for _ in range(5):
        torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(10):fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*100)
    return sorted(values)[2]

torch.manual_seed(42)
with torch.inference_mode():
    for index in (1,len(model.model.layers)-1):
        mlp=model.model.layers[index].mlp
        for rows in (5,32):
            x=torch.randn(1,rows,model.config.hidden_size,device="cuda",dtype=next(model.parameters()).dtype)
            graph=ExactShapeGraph(mlp.forward,x)
            previous_ids=None
            routing_changes=0
            saved=None
            for step in range(5):
                x=torch.randn_like(x)
                eager=mlp(x)
                actual=graph(x)
                check(eager,actual)
                if saved is not None: check(saved[0],saved[1])
                if isinstance(eager,tuple):
                    ids=eager[1][1]
                    if previous_ids is not None:routing_changes+=int(not torch.equal(previous_ids,ids))
                    previous_ids=ids.clone()
                    saved=(actual[0],actual[0].clone())
                else:
                    saved=(actual,actual.clone())
            if args.family=="llada":assert routing_changes>0
            before,after=ms(lambda:mlp(x)),ms(lambda:graph(x))
            print(json.dumps(dict(family=args.family,layer=index,rows=rows,
                                  bitwise=True,routing_changes=routing_changes,
                                  old_ms=before,new_ms=after,speedup=before/after,
                                  capture_seconds=graph.capture_seconds)),flush=True)
            del graph
