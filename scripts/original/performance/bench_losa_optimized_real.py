"""Optimized eager/graph ablations on exact NarrativeQA IDs; optional audits."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import time
import torch
from src.optimized.losa.api import LoSAOptimizedRuntime

p=argparse.ArgumentParser()
p.add_argument("--family",choices=["sdar","llada"],required=True)
p.add_argument("--length",type=int,default=8192)
p.add_argument("--gen",type=int,default=256)
p.add_argument("--sample",type=int,default=0)
p.add_argument("--modes",nargs="+",choices=["eager","shared","graph","shared_graph"],default=["eager","graph"])
p.add_argument("--audit",action="store_true")
p.add_argument("--verify-graph",action="store_true")
p.add_argument("--repeats",type=int,default=2)
p.add_argument("--output",required=True)
args=p.parse_args()
OUT=Path(args.output);OUT.parent.mkdir(parents=True,exist_ok=True)
rt=LoSAOptimizedRuntime(family=args.family);model,tokenizer=rt.load()
attr="self_attn" if args.family=="sdar" else "attention"
originals=[getattr(layer,attr)._paper_losa_dense_forward for layer in model.model.layers]
data=Path("/data0/gs/dataset_preparation/narrativeqa_speed80_20260917/dataset")
r=json.loads((data/("sdar" if args.family=="sdar" else "llada21")/f"{args.length}.jsonl").read_text().splitlines()[args.sample])
ids=r["input_ids"];n=min(len(ids),32768-args.gen)
if len(ids)>n:ids=ids[:n//2]+ids[-(n-n//2):]
inputs=torch.tensor([ids],device="cuda")
opts=dict(family=args.family,gen_length=args.gen,block_length=32,steps=32,
          temperature=0.,eos_early_stop=False,use_losa=True,
          losa_gqa_mode="group_mean",losa_page_size=16,losa_token_budget=256,
          losa_active_topk=5,losa_backend="triton",losa_fused_state=True)
opts.update(dict(threshold=.95,remasking_strategy="low_confidence_dynamic",mask_id=151669)
            if args.family=="sdar" else dict(threshold=.7,editing_threshold=.5,num_to_transfer=1))
all_results=[]
refs={}
for mode in args.modes:
    generation=importlib.import_module("src.optimized.losa.generation")
    patch=importlib.import_module("src.optimized.losa.attention_patch")
    stepname="losa_optimized_attention_step"
    for layer,original in zip(model.model.layers,originals):
        att=getattr(layer,attr);att.forward=original
        for name in ("_paper_losa_dense_forward","_paper_losa_model"):
            if hasattr(att,name):delattr(att,name)
    model._losa_optimized_graph_replay=mode in ("graph","shared_graph")
    model._losa_optimized_shared_kv=mode in ("shared","shared_graph")
    model._losa_optimized_verify_graph=args.verify_graph
    original_forward=generation.model_forward
    original_step=getattr(patch,stepname)
    selections=[];events=[];pending=[];step_events=[]
    transfer_name="transfer_sdar" if args.family=="sdar" else "transfer_llada"
    transfer=getattr(generation,transfer_name)
    def audited(*a,**kw):
        out=original_step(*a,**kw)
        if not out.initialized_dense:
            selections.append((out.active_rows.clone(),out.compact_pages.clone()))
        return out
    def measured(*a,**kw):
        ctx=kw.get("losa_context_kwargs")
        phase=("sparse" if ctx["states"] else "initialize") if ctx else "dense"
        begin,end=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
        begin.record();out=original_forward(*a,**kw);end.record()
        events.append((phase,begin,end))
        pending[:]=[(phase,begin)]
        return out
    def timed_transfer(*a,**kw):
        out=transfer(*a,**kw)
        if pending:
            phase,begin=pending.pop();end=torch.cuda.Event(enable_timing=True);end.record()
            step_events.append((phase,begin,end))
        return out
    if args.audit:setattr(patch,stepname,audited)
    # Warm each mode with the same actual request; includes compilation/capture.
    torch.manual_seed(42);torch.cuda.synchronize()
    t=time.perf_counter()
    warm=generation.block_diffusion_generate(model,inputs=inputs,**opts)
    torch.cuda.synchronize();warm_seconds=time.perf_counter()-t
    print("WARM",mode,warm_seconds,flush=True)
    generation.model_forward=measured
    setattr(generation,transfer_name,timed_transfer)
    try:
        for repeat in range(args.repeats):
            selections.clear();events.clear();pending.clear();step_events.clear()
            torch.manual_seed(42);torch.cuda.synchronize();t=time.perf_counter()
            out=generation.block_diffusion_generate(model,inputs=inputs,**opts)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            token_ids=out.tokens.cpu().tolist()
            selection_cpu=[(x.cpu().tolist(),y.cpu().tolist()) for x,y in selections]
            row=dict(mode=mode,repeat=repeat,seconds=elapsed,tokens=token_ids,
                     trace=out.trace,selections=selection_cpu if args.audit else None,
                     phase_ms={phase:[a.elapsed_time(b) for p,a,b in events if p==phase]
                               for phase in ("dense","initialize","sparse")},
                     step_ms={phase:[a.elapsed_time(b) for p,a,b in step_events if p==phase]
                              for phase in ("dense","initialize","sparse")},
                     graph_stats=dict(getattr(model,"_losa_optimized_graph_stats",{})),
                     source_id=r["source_id"],prompt_tokens=n,options=opts,
                     family=args.family,warm_seconds=warm_seconds,
                     diagnostic_timing=args.audit or args.verify_graph,
                     input_sha256=hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest())
            if mode=="eager":
                refs[repeat]=(token_ids,out.trace,selection_cpu)
            if repeat in refs:
                ref=refs[repeat]
                row.update(tokens_equal=token_ids==ref[0],trace_equal=out.trace==ref[1],
                           selections_equal=selection_cpu==ref[2] if args.audit else None)
            all_results.append(row);OUT.write_text(json.dumps(all_results,indent=2))
            print("RESULT",mode,repeat,elapsed,{k:statistics.median(v) if v else None for k,v in row["phase_ms"].items()},
                  row.get("tokens_equal"),row.get("selections_equal"),row["graph_stats"],flush=True)
    finally:
        generation.model_forward=original_forward
        setattr(generation,transfer_name,transfer)
        setattr(patch,stepname,original_step)
        model._losa_optimized_graph=None
