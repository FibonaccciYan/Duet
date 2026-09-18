"""Compare full eager/graph generation by deferred layer-output snapshots."""
import argparse
import json
from pathlib import Path
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.runtime import patch_method
from src.kernels.optimized.sparse_layer_graph import install, stats
from scripts.original.performance.versioned_metadata import save_manifest

p=argparse.ArgumentParser()
p.add_argument("--mode",choices=["dense_optimized","sparse_optimized"],default="sparse_optimized")
p.add_argument("--output",required=True)
args=p.parse_args()
model,tokenizer=load_model_and_tokenizer("sdar",model_path="/data0/ysy/models/SDAR-8B-Chat-b32")
kw=dict(moe_expert_patch=True)
if args.mode=="sparse_optimized":kw.update(query_sparse=True,prefix_sparse=True,prefix_token_budget=256)
patch_method(model,args.mode,model_name="sdar",**kw)
save_manifest(args.output,args)
record=json.loads(Path("/data0/gs/losa_accuracy_prep/data/longbench_all21_full/gov_report.jsonl").read_text().splitlines()[115])
raw=tokenizer(record["prompt"],return_tensors="pt").input_ids
ids=torch.cat((raw[:,:4096],raw[:,-4096:]),dim=1).cuda()
options=dict(inputs=ids,gen_length=64,block_length=32,steps=32,temperature=0.,
             threshold=0.95,remasking_strategy="low_confidence_dynamic",mask_id=151669,
             eos_early_stop=False)
records=[]
handles=[]
for index,layer in enumerate(model.model.layers):
    def hook(module,a,kw,out,index=index):
        hidden=a[0] if a else kw["hidden_states"]
        if hidden.shape[1]>32:return
        pos=kw.get("position_ids")
        records.append((index,hidden.clone(),out[0].clone(),
                        None if pos is None else pos.clone(),bool(kw.get("store_kv"))))
    handles.append(layer.register_forward_hook(hook,with_kwargs=True))

with torch.inference_mode():
    # Warm ordinary kernels before either measured trajectory.
    torch.manual_seed(42)
    model.generate(**options)
    torch.cuda.synchronize()
    records.clear()
    torch.manual_seed(42)
    eager_tokens=model.generate(**options).clone()
    torch.cuda.synchronize()
    eager=records[:]
    records.clear()
    install(model,sync_probe=False)
    torch.manual_seed(42)
    graph_tokens=model.generate(**options).clone()
    torch.cuda.synchronize()
    captured=records[:]
    records.clear()
    first=None
    for i,(a,b) in enumerate(zip(eager,captured)):
        layer,ah,ao,ap,ac=a
        bl,bh,bo,bp,bc=b
        same_shape=ah.shape==bh.shape and ao.shape==bo.shape
        input_equal=same_shape and torch.equal(ah,bh)
        output_equal=same_shape and torch.equal(ao,bo)
        positions_equal=ap is None and bp is None or (
            ap is not None and bp is not None and torch.equal(ap,bp))
        if layer!=bl or not input_equal or not output_equal or not positions_equal or ac!=bc:
            first=dict(call=i,layer=layer,graph_layer=bl,input_equal=input_equal,
                       output_equal=output_equal,positions_equal=positions_equal,
                       eager_shape=list(ah.shape),graph_shape=list(bh.shape),
                       output_max=float((ao-bo).abs().max()) if same_shape else None,
                       positions=None if ap is None else ap.cpu().tolist(),
                       store_kv=[ac,bc])
            break
    result=dict(mode=args.mode,eager_calls=len(eager),graph_calls=len(captured),
                tokens_equal=torch.equal(eager_tokens,graph_tokens),
                first_difference=first,graph_stats=stats(model))
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
for handle in handles:handle.remove()
