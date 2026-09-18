"""Same-process dense/sparse comparison with explicit measured boundaries.

LLaDA steady-step boundary: cached forward through transfer return (includes
sampling and transfer writes). SDAR: denoise through select_transfer return;
the final token scatter is outside that interval and is reported as such.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.runtime import patch_method
from src.kernels.optimized.small_moe import install_small_moe
from scripts.original.performance.versioned_metadata import save_manifest

p=argparse.ArgumentParser()
p.add_argument("--exact-rope", action="store_true", help="Apply fused RoPE equally to dense and sparse (SDAR experiment).")
p.add_argument("--full-step-timing", action="store_true", help="SDAR whole loop iteration through token scatter; reports finalize separately.")
p.add_argument("--modes", nargs="+", choices=["dense_optimized","sparse_optimized","dense_optimized_rope","sparse_optimized_rope"], default=["dense_optimized","sparse_optimized"])
p.add_argument("--family",choices=["llada","sdar"],required=True)
p.add_argument("--contexts",type=int,nargs="+",default=[8192,16384,32768])
p.add_argument("--gen-length",type=int,default=128)
p.add_argument("--repeats",type=int,default=3)
p.add_argument("--index",type=int,default=115)
p.add_argument("--output",required=True)
p.add_argument("--profile-step",type=int,default=-1,
               help="Diagnostic profiler on this zero-based denoise invocation; timings are not benchmarks.")
p.add_argument("--direct-norm",action="store_true")
p.add_argument("--layer-graphs",action="store_true",
               help="Apply the same decoder graph mechanism to dense and sparse.")
p.add_argument("--verify-layer-graphs",action="store_true",
               help="Compare each replay against eager from identical state; diagnostic timings only.")
p.add_argument("--sync-graph-probe",action=argparse.BooleanOptionalAction,default=True,
               help="Keep synchronization protection enabled. --no-sync-graph-probe is a known-divergent diagnostic only.")
p.add_argument("--graph-sync-mode",choices=["both","before","after"],default="both")
p.add_argument("--eager-gate",action="store_true",
               help="Compare replay tokens and step counts to same-process eager runs.")
p.add_argument("--deterministic",action="store_true",
               help="Diagnostic deterministic algorithms and cuBLAS workspace, applied to both paths.")
args=p.parse_args()
if args.full_step_timing and args.family != "sdar":
    p.error("full-step timing currently supports SDAR only")
if args.deterministic:
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.use_deterministic_algorithms(True)
path="/data0/ysy/models/LLaDA2.1-mini" if args.family=="llada" else "/data0/ysy/models/SDAR-8B-Chat-b32"
model,tokenizer=load_model_and_tokenizer(args.family,model_path=path)
if args.direct_norm:
    from src.kernels.optimized.sdar_norm import install
    print("DIRECT_NORM_PATCHES",install(model),flush=True)
manifest=save_manifest(args.output,args)
data=Path("/data0/gs/losa_accuracy_prep/data/longbench_all21_full/gov_report.jsonl")
record=json.loads(data.read_text().splitlines()[args.index])
ids=tokenizer(record["prompt"],return_tensors="pt").input_ids
options=dict(gen_length=args.gen_length,block_length=32,steps=32,temperature=0.0,eos_early_stop=False)
options.update(dict(threshold=0.7,editing_threshold=0.5,num_to_transfer=1)
               if args.family=="llada" else
               dict(threshold=0.95,remasking_strategy="low_confidence_dynamic",mask_id=151669))
rows=[]


def run(mode,inputs,use_graphs=None):
    use_graphs=args.layer_graphs if use_graphs is None else use_graphs
    from src.kernels.optimized.rope_runtime import restore as restore_rope, install as install_rope
    restore_rope(model)
    # Rebind the original driver before installing a versioned per-model clone.
    if hasattr(model,"_sparse_original_generate"):
        model.generate=model._sparse_original_generate
    kwargs=dict(moe_expert_patch=True)
    if mode.startswith("sparse_"):
        kwargs.update(query_sparse=True,prefix_sparse=True,prefix_token_budget=256)
    patch_method(model,mode.removesuffix("_rope"),model_name=args.family,**kwargs)
    install_small_moe(model)
    use_rope=args.exact_rope or mode.endswith("_rope")
    rope_patches=install_rope(model) if use_rope else 0
    from src.kernels.optimized.sparse_layer_graph import install as install_graphs, stats
    if use_graphs:
        install_graphs(model,verify=args.verify_layer_graphs,sync_probe=args.sync_graph_probe,
                       sync_mode=args.graph_sync_mode)
        from src.kernels.optimized.sparse_layer_graph import _forward
        import types
        for layer in model.model.layers:
            layer.forward=types.MethodType(_forward,layer)
    else:
        for layer in model.model.layers:
            if hasattr(layer,"_sparse_graph_original"):
                layer.forward=layer._sparse_graph_original
    graph_before=stats(model)
    driver=inspect.unwrap(model.generate)
    namespace=driver.__globals__
    forward_events,step_events,pending=[],[],[]
    invocation=0
    full_clock=None
    restores=[]
    def wrap_forward(function):
        def measured(*a,**kw):
            nonlocal invocation
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            if invocation==args.profile_step:
                with torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True) as profiler:
                    result=function(*a,**kw)
                stem=f"{args.output}.{mode}.step{invocation}"
                Path(stem+".txt").write_text(
                    profiler.key_averages(group_by_input_shape=True).table(
                        sort_by="self_cuda_time_total",row_limit=50)
                    +"\nCPU\n"+profiler.key_averages().table(sort_by="self_cpu_time_total",row_limit=40))
            else:
                result=function(*a,**kw)
            invocation+=1
            end.record()
            forward_events.append((begin,end))
            pending[:]=[begin]
            return result
        return measured
    def replace(ns,name,value):
        restores.append((ns,name,ns[name]))
        ns[name]=value
    if args.family=="llada":
        replace(namespace,"_cached_forward",wrap_forward(namespace["_cached_forward"]))
        transfer_ns=namespace
        transfer_name="_transfer_tokens"
    else:
        original_block=namespace["block_diffusion_generate"]
        if args.full_step_timing:
            from scripts.original.performance.full_step_timing import StepClock, instrument_sdar
            full_clock=StepClock()
            original_block=instrument_sdar(original_block,full_clock)
        transfer_ns=inspect.unwrap(original_block).__globals__
        transfer_name="select_transfer"
        def block(*a,**kw):
            denoise=kw.get("denoise_fn")
            if denoise is None:
                def denoise(*,model,block_tokens,attention_mask,position_ids,past_key_values,**unused):
                    return model(block_tokens,attention_mask=attention_mask,position_ids=position_ids,
                                 past_key_values=past_key_values,use_cache=True,store_kv=False).logits,None
            kw["denoise_fn"]=wrap_forward(denoise)
            return original_block(*a,**kw)
        replace(namespace,"block_diffusion_generate",block)
    transfer=transfer_ns[transfer_name]
    def measured_transfer(*a,**kw):
        output=transfer(*a,**kw)
        if pending:
            end=torch.cuda.Event(enable_timing=True);end.record()
            step_events.append((pending.pop(),end))
        return output
    replace(transfer_ns,transfer_name,measured_transfer)
    try:
        torch.manual_seed(42);torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            output=model.generate(inputs=inputs,**options)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
    finally:
        for ns,name,value in reversed(restores):ns[name]=value
    forward_ms=[b.elapsed_time(e) for b,e in forward_events]
    step_ms=[b.elapsed_time(e) for b,e in step_events]
    assert forward_ms and len(forward_ms)==len(step_ms)
    return dict(mode=mode,family=args.family,moe_backend="small_moe",
                exact_rope=use_rope, rope_patches=rope_patches,
                full_iteration_timing=None if full_clock is None else full_clock.rows(),
                layer_graphs=use_graphs,graph_stats=stats(model),
                new_capture_seconds=stats(model)["capture_seconds"]-graph_before["capture_seconds"],
                direct_norm_count=getattr(model, "_versioned_direct_norm_count", 0),
                manifest=manifest,options=options,seconds=elapsed,
                forward_ms=forward_ms,step_ms=step_ms,step_count=len(step_ms),
                boundary="steady_cached_forward_through_transfer_return" if args.family=="llada"
                else "denoise_through_selection_return_excludes_final_token_scatter",
                tokens=output.cpu().tolist())


for length in args.contexts:
    capacity=getattr(model.config,"max_position_embeddings",None)
    if capacity is not None and length+args.gen_length>capacity:
        raise ValueError("prompt plus generation exceeds configured window")
    assert ids.shape[1]>=length
    inputs=torch.cat((ids[:,:length//2],ids[:,-(length-length//2):]),dim=1).cuda()
    token_hash=hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest()
    references={}
    if args.eager_gate:
        for mode in args.modes:
            run(mode,inputs,use_graphs=False)
            references[mode]=run(mode,inputs,use_graphs=False)
            print("EAGER_REFERENCE",length,mode,references[mode]["step_count"],flush=True)
        Path(args.output+".eager.json").write_text(json.dumps(references,indent=2))
    for mode in args.modes:
        result=run(mode,inputs)
        print("WARM",length,mode,result["seconds"],flush=True)
    for repeat in range(args.repeats):
        modes=args.modes if repeat%2==0 else list(reversed(args.modes))
        for mode in modes:
            result=run(mode,inputs)
            if args.eager_gate:
                ref=references[mode]
                result["eager_tokens_equal"]=result["tokens"]==ref["tokens"]
                result["eager_steps_equal"]=result["step_count"]==ref["step_count"]
            result.update(prompt_tokens=length,repeat=repeat,sample_index=args.index,prompt_sha256=token_hash)
            rows.append(result)
            Path(args.output).write_text(json.dumps(rows,indent=2))
            print("PAIRED",length,repeat,mode,result["seconds"],result["step_count"],
                  statistics.median(result["forward_ms"]),statistics.median(result["step_ms"]),flush=True)
            if args.eager_gate:
                assert result["eager_tokens_equal"] and result["eager_steps_equal"], (
                    mode,result["eager_tokens_equal"],ref["step_count"],result["step_count"])
