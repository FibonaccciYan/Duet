"""Locate prefill differences before accepting batched FOCUS prefill."""
import json
import torch
from src.reference.losa.generation import load_model_and_tokenizer, build_llada_prefix_cache
from src.reference.losa.moe_patch import patch_moe_experts
from src.optimized.focus.model import focus_optimized_forward

model, tokenizer = load_model_and_tokenizer("llada", model_path="/data0/ysy/models/LLaDA2.1-mini")
patch_moe_experts(model, family="llada")
record = json.loads(open("/data0/gs/losa_accuracy_prep/data/longbench_all21_full/gov_report.jsonl").readline())
ids = tokenizer(record["prompt"], return_tensors="pt").input_ids[:, :256].cuda()
positions = torch.arange(256, device="cuda")[None]
with torch.inference_mode():
    cache = ()
    for start in range(0, 256, 32):
        result = focus_optimized_forward(
            model, family="llada", input_ids=ids[:, start:start+32],
            position_ids=positions[:, start:start+32], prefix_cache=cache,
            mask_id=-1, alpha=1.5, average_decoded_tokens=1.0, block_progress=-1,
        )
        cache = result.cache.to_legacy_cache()
    for chunk in (32, 256):
        other = build_llada_prefix_cache(model, ids, 256, positions, block_length=32, query_chunk_length=chunk)
        errors = []
        for layer, (a,b) in enumerate(zip(cache, other)):
            errors.append(dict(layer=layer, k_equal=torch.equal(a[0],b[0]),
                               v_equal=torch.equal(a[1],b[1]),
                               k_max=float((a[0].float()-b[0].float()).abs().max()),
                               v_max=float((a[1].float()-b[1].float()).abs().max())))
        print(json.dumps(dict(chunk=chunk, errors=errors)), flush=True)
