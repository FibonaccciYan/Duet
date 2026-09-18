"""Real-prompt sparse/FOCUS version comparison, with forward+transfer timing."""
import argparse
import hashlib
import inspect
import importlib
import json
from pathlib import Path
import statistics
import time
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.kernels.optimized.small_moe import install_small_moe, restore_small_moe
from scripts.original.performance.versioned_metadata import save_manifest

p = argparse.ArgumentParser()
p.add_argument("--exact-rope", action="store_true", help="SDAR public fused RoPE experiment.")
p.add_argument("--profile", action="store_true")
p.add_argument("--profile-step", type=int, default=2, help="One-based denoise forward to profile; default skips cache setup.")
p.add_argument("--method", choices=["sparse", "focus"], required=True)
p.add_argument("--focus-attention-backend", choices=["auto", "sdpa", "flash"], default="auto")
p.add_argument("--numerical-experiment",action="store_true",
               help="Report numerical-backend trajectory differences; not correctness acceptance.")
p.add_argument("--contexts", nargs="+", type=int, default=[8192, 16384, 32768])
p.add_argument("--repeats", type=int, default=2)
p.add_argument("--index", type=int, default=115)
p.add_argument("--output", required=True)
p.add_argument("--legacy-moe", action="store_true",
               help="Only for historical ablation; default matches the MoE backend.")
p.add_argument("--gen-length", type=int, default=32)
p.add_argument("--family", choices=["llada", "sdar"], default="llada")
p.add_argument("--remasking-strategy", choices=["low_confidence_static", "low_confidence_dynamic"],
               default="low_confidence_dynamic")
p.add_argument("--audit-selectors", action="store_true",
               help="Record actual query and prefix indices; audit timings are not speed results.")
p.add_argument("--maskless-focus", action="store_true",
               help="Experimental all-visible mask elision; production backend unchanged.")
p.add_argument("--mlp-graphs", action="store_true",
               help="Public FFN graph option, also available in the matched dense benchmark.")
p.add_argument("--sparse-layer-graphs", action="store_true",
               help="Experimental SDAR BlockDualCache layer replay; does not establish matched dense speedup.")
p.add_argument("--unsafe-unsynchronized-layer-graphs",action="store_true",
               help="Known-divergent diagnostic only; not valid for speedup acceptance.")
args = p.parse_args()
if args.method == "focus" and not args.numerical_experiment:
    p.error("experimental batched-prefill focus requires --numerical-experiment")
data = Path("/data0/gs/losa_accuracy_prep/data/longbench_all21_full/gov_report.jsonl")
record = json.loads(data.read_text().splitlines()[args.index])
model_path = "/data0/ysy/models/LLaDA2.1-mini" if args.family == "llada" else "/data0/ysy/models/SDAR-8B-Chat-b32"
model, tokenizer = load_model_and_tokenizer(args.family, model_path=model_path)
if args.family == "sdar":
    from src.kernels.optimized.sdar_norm import install as install_direct_norm
    model._versioned_direct_norm_count = install_direct_norm(model)
manifest_path = save_manifest(args.output, args)
tokens = tokenizer(record["prompt"], return_tensors="pt").input_ids
if args.method == "sparse":
    patch_model = importlib.import_module("src.optimized.sparse").patch_model
    patch_model(model, model_name=args.family, query_sparse=True, prefix_sparse=True,
                prefix_token_budget=256, moe_expert_patch=True)
else:
    from src.reference.losa.moe_patch import patch_moe_experts
    patch_moe_experts(model, family=args.family)
if args.mlp_graphs:
    from src.kernels.optimized.mlp_graph import install as install_mlp_graphs
    install_small_moe(model)
    print("MLP_GRAPHS_INSTALLED", install_mlp_graphs(model, rows=(32,)), flush=True)
options = dict(gen_length=args.gen_length, block_length=32, steps=32, temperature=0.0,
               eos_early_stop=False)
options.update(dict(threshold=0.7, editing_threshold=0.5, num_to_transfer=1)
               if args.family == "llada" else
               dict(threshold=0.95, remasking_strategy=args.remasking_strategy, mask_id=151669))
results = []
from rouge_score import rouge_scorer
scorer=rouge_scorer.RougeScorer(["rougeL"],use_stemmer=True)


def run(new, ids):
    if args.exact_rope:
        from src.kernels.optimized.rope_runtime import install as install_rope, restore as restore_rope
        restore_rope(model)
        if new:
            model._versioned_rope_patch_count=install_rope(model)
    from src.kernels.optimized.mlp_graph import stats as graph_stats
    graph_before = graph_stats(model)
    from src.kernels.optimized.sparse_layer_graph import stats as layer_graph_stats
    if args.sparse_layer_graphs and new:
        from src.kernels.optimized.sparse_layer_graph import install as install_layer_graphs
        install_layer_graphs(model,sync_probe=not args.unsafe_unsynchronized_layer_graphs)
    # Existing graphs are bypassed for the original-version correctness oracle.
    for layer in model.model.layers:
        if hasattr(layer,"_sparse_graph_original"):
            if new:
                from src.kernels.optimized.sparse_layer_graph import _forward
                import types
                layer.forward=types.MethodType(_forward,layer)
            else:
                layer.forward=layer._sparse_graph_original
    layer_before=layer_graph_stats(model)
    focus_model = None
    if args.method == "focus" and new and args.maskless_focus:
        focus_model = importlib.import_module("src.optimized.focus.model")
        original_mask = focus_model._attention_mask
        focus_model._attention_mask = lambda *a, **kw: None
    if args.method == "sparse":
        model.generate = model._sparse_optimized_generate if new else model._sparse_original_generate
    if new or not args.legacy_moe:
        install_small_moe(model)
    else:
        restore_small_moe(model)
    module = importlib.import_module(
        ("src.reference.sparse.llada_patch" if args.family == "llada" else "src.reference.sparse.sdar_patch") if args.method == "sparse"
        else "src.optimized.focus.generation" if new else "src.reference.focus.generation")
    forward_name = ("_cached_forward" if args.family == "llada" else "_sparse_cached_forward") if args.method == "sparse" else ("focus_optimized_forward" if new else "focus_forward")
    transfer_name = ("_transfer_tokens" if args.family == "llada" else "select_transfer") if args.method == "sparse" else ("_selected_llada_transfer" if args.family == "llada" else "_selected_sdar_transfer")
    if args.method == "sparse":
        namespace = inspect.unwrap(model.generate).__globals__
    else:
        namespace = module.__dict__
    transfer_namespace = module.__dict__ if args.method == "sparse" and args.family == "sdar" else namespace
    base_forward, transfer = namespace[forward_name], transfer_namespace[transfer_name]
    forward = base_forward
    forwards, steps, pending, selected, prefixes = [], [], [], [], []
    phase_events=[]
    denoise_index=0
    prefill_module=None
    if args.method == "focus" and new:
        prefill_module=importlib.import_module("src.optimized.focus.prefill")
        original_prefill=prefill_module.build_prefix
        def measured_prefill(*a, **kw):
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record()
            result=original_prefill(*a, **kw)
            end.record()
            phase_events.append(("prefill",begin,end))
            return result
        prefill_module.build_prefix=measured_prefill
    denoise_seen=False
    base_compactor = namespace.get("_compact_prefix_cache")
    if args.audit_selectors and args.method == "sparse":
        from src.kernels.optimized.function_binding import _bind_globals
        select = base_forward.__globals__["_select_positions"]
        def record_select(*a, **kw):
            out = select(*a, **kw)
            selected.append(None if out is None else out.clone())
            return out
        forward, _ = _bind_globals(base_forward, {"_select_positions": record_select})
        def record_prefix(*a, **kw):
            out = base_compactor(*a, **kw)
            prefixes.append(tuple(x.clone() for x in out[1]))
            return out
        namespace["_compact_prefix_cache"] = record_prefix

    def measure_forward(*a, **kw):
        nonlocal denoise_seen, denoise_index
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        is_step = args.method == "sparse" or kw.get("mask_id", -1) != -1
        if is_step:
            denoise_index += 1
        if args.profile and new and is_step and denoise_index == args.profile_step:
            measure_forward.profiled = True
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as prof:
                out = forward(*a, **kw)
            Path(args.output + ".profile.txt").write_text(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=35)
                + "\nCPU\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))
        else:
            out = forward(*a, **kw)
        end.record()
        phase = "denoise" if is_step else ("finalize" if denoise_seen else "prefill")
        phase_events.append((phase,begin,end))
        denoise_seen |= is_step
        if is_step:
            forwards.append((begin, end))
            pending[:] = [begin]
            if args.audit_selectors and args.method == "focus":
                selected.append(out.positions.clone())
        return out

    def measure_transfer(*a, **kw):
        out = transfer(*a, **kw)
        if pending:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            steps.append((pending.pop(), end))
        return out

    namespace[forward_name] = measure_forward
    transfer_namespace[transfer_name] = measure_transfer
    try:
        torch.manual_seed(42)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            if args.method == "sparse":
                output = model.generate(inputs=ids, **options)
                trace = []
            else:
                output = getattr(module, "focus_optimized_generate" if new else "focus_generate")(
                    model, family=args.family, inputs=ids, alpha=1.5,
                    **({"attention_backend": args.focus_attention_backend} if new else {}),
                    **options)
                trace, output = output.trace, output.tokens
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    finally:
        if prefill_module is not None:
            prefill_module.build_prefix=original_prefill
        namespace[forward_name] = base_forward
        transfer_namespace[transfer_name] = transfer
        if base_compactor is not None:
            namespace["_compact_prefix_cache"] = base_compactor
        if focus_model is not None:
            focus_model._attention_mask = original_mask
    if args.audit_selectors and args.method == "sparse":
        assert prefixes, "prefix audit hook did not run"
        trace = dict(query_positions=[None if x is None else x.cpu().tolist() for x in selected],
                     prefix_indices=[[x.cpu().tolist() for x in layer] for layer in prefixes])
    elif args.audit_selectors and args.method == "focus":
        assert selected, "FOCUS retained-position audit did not run"
        trace = dict(events=trace, retained_positions=[x.cpu().tolist() for x in selected])
    prediction=tokenizer.decode(output[0],skip_special_tokens=True)
    answers=record.get("answers") or [record.get("answer","")]
    if isinstance(answers,str):answers=[answers]
    score=max((scorer.score(str(answer),prediction)["rougeL"].fmeasure for answer in answers),default=0.)
    return dict(version=("sparse_optimized" if new else "sparse") if args.method == "sparse"
                else ("focus_optimized" if new else "focus"),
                prediction=prediction,diagnostic_rougeL=score,quality_acceptance=False,
                moe_backend="small_moe" if new or not args.legacy_moe else "legacy_packed",
                exact_rope=bool(new and args.exact_rope),
                direct_norm_count=getattr(model, "_versioned_direct_norm_count", 0),
                manifest=manifest_path,
                family=args.family, generation_options=options,
                mlp_graphs=args.mlp_graphs, mlp_graph_stats=graph_stats(model),
                mlp_graph_capture_seconds=graph_stats(model)["capture_seconds"]-graph_before["capture_seconds"],
                sparse_layer_graphs=bool(args.sparse_layer_graphs and new),
                layer_graph_stats=layer_graph_stats(model),
                layer_graph_capture_seconds=layer_graph_stats(model)["capture_seconds"]-layer_before["capture_seconds"],
                selector_audit=args.audit_selectors,
                maskless_focus=bool(new and args.maskless_focus),
                focus_attention_backend=args.focus_attention_backend if new else "sdpa",
                elapsed_seconds=elapsed, forward_ms=[b.elapsed_time(e) for b,e in forwards],
                phase_forward_ms={phase:[b.elapsed_time(e) for p,b,e in phase_events if p==phase]
                                  for phase in ("prefill","denoise","finalize")},
                forward_transfer_ms=[b.elapsed_time(e) for b,e in steps],
                token_ids=output.cpu().tolist(), trace=trace)


for length in args.contexts:
    capacity = getattr(model.config, "max_position_embeddings", None)
    if capacity is not None and length + args.gen_length > capacity:
        raise ValueError(f"prompt {length} + generation {args.gen_length} exceeds {capacity}")
    assert tokens.shape[1] >= length
    ids = torch.cat((tokens[:, :length//2], tokens[:, -(length-length//2):]), dim=1).cuda()
    for new in (False, True):
        warm = run(new, ids)
        print("WARM", length, warm["version"], warm["elapsed_seconds"], flush=True)
    for repeat in range(args.repeats):
        pair = {}
        for new in ((False, True) if repeat % 2 == 0 else (True, False)):
            row = run(new, ids)
            row.update(prompt_tokens=length, generation_budget=args.gen_length, repeat=repeat,
                       sample_index=args.index, dataset=str(data),
                       dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                       profile="LLaDA2.1-Q-threshold0.7-editing0.5" if args.family == "llada" else "SDAR-"+args.remasking_strategy)
            pair[new] = row
            results.append(row)
            Path(args.output).write_text(json.dumps(results, indent=2))
            print("MEASURED", length, row["version"], repeat, row["elapsed_seconds"],
                  statistics.median(row["forward_transfer_ms"]) if row["forward_transfer_ms"] else None,
                  flush=True)
        tokens_equal=pair[False]["token_ids"] == pair[True]["token_ids"]
        trace_equal=pair[False]["trace"] == pair[True]["trace"]
        if not args.numerical_experiment:
            assert tokens_equal, "tokens diverged"
            assert trace_equal, "selection trace diverged"
        print("TOKENS_TRACE_COMPARISON",length,repeat,tokens_equal,trace_equal,
              "DIAGNOSTIC_ROUGE",pair[False]["diagnostic_rougeL"],pair[True]["diagnostic_rougeL"],flush=True)
