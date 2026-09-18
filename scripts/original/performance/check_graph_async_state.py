"""Delay all correctness reads until after a replay sequence.

This deliberately avoids torch.equal/assert_close inside the replay loop:
their implicit synchronization can hide an asynchronous state bug.
"""
import argparse
import json
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.optimized.sparse import patch_model
from src.reference.sparse.sparse_ops import _BlockDualCache
from src.kernels.optimized.sparse_layer_graph import SparseLayerGraph

p=argparse.ArgumentParser()
p.add_argument("--steps",type=int,default=12)
p.add_argument("--prefix",type=int,default=256)
args=p.parse_args()
model,_=load_model_and_tokenizer("sdar",model_path="/data0/ysy/models/SDAR-8B-Chat-b32")
patch_model(model,model_name="sdar",query_sparse=True,prefix_sparse=True)
model._sdar_decode_attention=True
torch.manual_seed(17)
with torch.inference_mode():
    index=6
    layer=model.model.layers[index]
    layer._sparse_graph_sync_probe=False
    rows=17
    k=torch.randn(1,8,args.prefix+32,128,device="cuda",dtype=torch.float16)
    v=torch.randn_like(k)
    def new_cache():
        return _BlockDualCache([(k.clone(),v.clone()) for _ in range(index+1)],
                               [args.prefix]*(index+1))
    hidden_inputs=[torch.randn(1,rows,4096,device="cuda",dtype=torch.float16) for _ in range(args.steps)]
    positions=[torch.randperm(32,device="cuda")[:rows].sort().values for _ in range(args.steps)]
    def options(cache,i):
        cache.set_positions(positions[i])
        absolute=positions[i][None]+8192
        return dict(attention_mask=None,position_ids=absolute,past_key_value=cache,
                    output_attentions=False,use_cache=True,store_kv=True,
                    position_embeddings=model.model.rotary_emb(hidden_inputs[i],absolute))
    eager_cache=new_cache()
    expected=[]
    for i in range(args.steps):
        out=layer(hidden_inputs[i],**options(eager_cache,i))[0]
        expected.append((out.clone(),eager_cache.key_cache[index].clone(),
                         eager_cache.value_cache[index].clone()))
    graph_cache=new_cache()
    graph=SparseLayerGraph(layer,hidden_inputs[0],options(graph_cache,0))
    actual=[]
    for i in range(args.steps):
        out=graph(hidden_inputs[i],**options(graph_cache,i))[0]
        actual.append((out,graph_cache.key_cache[index].clone(),
                       graph_cache.value_cache[index].clone()))
        # Encourage allocator reuse without adding a synchronization point.
        transient=torch.empty_like(hidden_inputs[i])
        transient.fill_(i)
        del transient
    torch.cuda.synchronize()
    failures=[]
    for i,(ref,got) in enumerate(zip(expected,actual)):
        for name,a,b in zip(("hidden","key","value"),ref,got):
            if not torch.equal(a,b):
                failures.append(dict(step=i,tensor=name,max_error=float((a-b).abs().max())))
    print(json.dumps(dict(steps=args.steps,prefix=args.prefix,failures=failures,
                          deferred_checks=True)),flush=True)
    assert not failures
model._sdar_decode_attention=False
