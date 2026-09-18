import json
import torch
from transformers.cache_utils import DynamicCache
from src.reference.losa.generation import load_model_and_tokenizer
from src.optimized.dense.api import patch_model
from src.kernels.optimized.sparse_layer_graph import SparseLayerGraph

model,_=load_model_and_tokenizer("sdar",model_path="/data0/ysy/models/SDAR-8B-Chat-b32")
patch_model(model,model_name="sdar")
model._sdar_decode_attention=True
torch.manual_seed(42)
with torch.inference_mode():
    for index in (0,35):
        for length in (256,8192):
            key=torch.randn(1,8,length,128,device="cuda",dtype=torch.float16)
            value=torch.randn_like(key)
            cache=DynamicCache.from_legacy_cache(tuple((key,value) for _ in range(index+1)))
            hidden=torch.randn(1,32,4096,device="cuda",dtype=torch.float16)
            def options():
                pos=torch.arange(length,length+32,device="cuda")[None]
                return dict(past_key_value=cache,position_ids=pos,
                            position_embeddings=model.model.rotary_emb(hidden,pos),
                            attention_mask=torch.ones(1,32,length+32,device="cuda",dtype=torch.bool),
                            use_cache=True,store_kv=False,output_attentions=False)
            kw=options()
            layer=model.model.layers[index]
            graph=SparseLayerGraph(layer,hidden,kw)
            for _ in range(3):
                hidden=torch.randn_like(hidden)
                kw=options()
                eager=layer(hidden,**kw)[0]
                actual=graph(hidden,**kw)[0]
                assert torch.equal(eager,actual)
                assert torch.equal(cache[index][0],key) and torch.equal(cache[index][1],value)
            print(json.dumps(dict(layer=index,length=length,bitwise=True,
                                  prefix_unchanged=True,capture_seconds=graph.capture_seconds)),flush=True)
            del graph,cache
model._sdar_decode_attention=False
