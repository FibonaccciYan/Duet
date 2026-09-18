import json
import time
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.optimized.sparse import patch_model
from src.reference.sparse.sparse_ops import _BlockDualCache
from src.kernels.optimized.sparse_layer_graph import SparseLayerGraph

model,_=load_model_and_tokenizer("sdar",model_path="/data0/ysy/models/SDAR-8B-Chat-b32")
patch_model(model,model_name="sdar",query_sparse=True,prefix_sparse=True)
model._sdar_decode_attention=True
torch.manual_seed(42)
def ms(fn):
    for _ in range(3):fn()
    values=[]
    for _ in range(5):
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(10):fn()
        torch.cuda.synchronize();values.append((time.perf_counter()-start)*100)
    return sorted(values)[2]

with torch.inference_mode():
    for index in (6,35):
        layer=model.model.layers[index]
        for rows in (5,17,32):
            prefix=256
            key=torch.randn(1,8,prefix+32,128,device="cuda",dtype=torch.float16)
            value=torch.randn_like(key)
            def make_cache():
                return _BlockDualCache([(key.clone(),value.clone()) for _ in range(index+1)],
                                       [prefix]*(index+1))
            eager_cache,graph_cache=make_cache(),make_cache()
            hidden=torch.randn(1,rows,4096,device="cuda",dtype=torch.float16)
            pos=torch.arange(rows,device="cuda")
            def kw(cache,positions):
                cache.set_positions(positions)
                absolute=positions[None]+8192
                cos,sin=model.model.rotary_emb(hidden,absolute)
                return dict(attention_mask=None,position_ids=absolute,past_key_value=cache,
                            output_attentions=False,use_cache=True,store_kv=True,
                            position_embeddings=(cos,sin))
            ek,gk=kw(eager_cache,pos),kw(graph_cache,pos)
            graph=SparseLayerGraph(layer,hidden,gk)
            for step in range(5):
                hidden=torch.randn_like(hidden)
                pos=torch.randperm(32,device="cuda")[:rows].sort().values
                ek,gk=kw(eager_cache,pos),kw(graph_cache,pos)
                a=layer(hidden,**ek)[0]
                b=graph(hidden,**gk)[0]
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                assert torch.equal(eager_cache.key_cache[index],graph_cache.key_cache[index])
                assert torch.equal(eager_cache.value_cache[index],graph_cache.value_cache[index])
            before,after=ms(lambda:layer(hidden,**ek)),ms(lambda:graph(hidden,**gk))
            print(json.dumps(dict(layer=index,rows=rows,bitwise=True,kv_bitwise=True,
                                  old_ms=before,new_ms=after,speedup=before/after,
                                  capture_seconds=graph.capture_seconds)),flush=True)
            del graph,eager_cache,graph_cache
model._sdar_decode_attention=False
