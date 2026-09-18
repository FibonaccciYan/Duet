"""Cache-only passes must preserve every layer's K/V bit for bit."""
import argparse
import json
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.reference.losa.moe_patch import patch_moe_experts
from src.kernels.optimized.small_moe import install_small_moe
from src.optimized.focus.model import focus_optimized_forward
from src.kernels.optimized.append_cache import AppendCache

p=argparse.ArgumentParser()
p.add_argument("--family",choices=["llada","sdar"],required=True)
args=p.parse_args()
path="/data0/ysy/models/LLaDA2.1-mini" if args.family=="llada" else "/data0/ysy/models/SDAR-8B-Chat-b32"
model,tokenizer=load_model_and_tokenizer(args.family,model_path=path)
patch_moe_experts(model,family=args.family)
install_small_moe(model)
record=json.loads(open("/data0/gs/losa_accuracy_prep/data/longbench_all21_full/gov_report.jsonl").readline())
ids=tokenizer(record["prompt"],return_tensors="pt").input_ids[:,:128].cuda()
positions=torch.arange(128,device="cuda")[None]
counts={"head":0,"last_mlp":0}
def count_head(*_):counts["head"]+=1
def count_mlp(*_):counts["last_mlp"]+=1
handles=[model.lm_head.register_forward_hook(count_head),
         model.model.layers[-1].mlp.register_forward_hook(count_mlp)]
with torch.inference_mode():
    prefix=()
    append_cache=AppendCache(initial_capacity=32)
    for start in range(0,128,32):
        kw=dict(family=args.family,input_ids=ids[:,start:start+32],
                position_ids=positions[:,start:start+32],prefix_cache=prefix,
                mask_id=-1,alpha=1.5,average_decoded_tokens=1.0,block_progress=-1)
        full=focus_optimized_forward(model,**kw)
        before=dict(counts)
        cached=focus_optimized_forward(model,**kw,cache_only=True,append_cache=append_cache)
        assert cached.logits is None and before==counts
        for a,b in zip(full.cache.to_legacy_cache(),cached.cache.to_legacy_cache()):
            assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
        prefix=cached.cache.to_legacy_cache()
        print("CACHE_ONLY_KV_BITWISE",args.family,start,"head_and_last_mlp_skipped",flush=True)
for handle in handles:handle.remove()
