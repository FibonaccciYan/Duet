"""NarrativeQA exact-ID FOCUS/dense matrix. No shared source mutations."""
import argparse
import ast
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import sys
import textwrap
import time
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("family", choices=("llada", "sdar"))
parser.add_argument("method", choices=("dense", "focus"))
parser.add_argument("--refresh", action="store_true")
parser.add_argument("--timing-only", action="store_true")
args = parser.parse_args()
family, method = args.family, args.method
ROOT = Path("/data0/gs/dataset_preparation/narrativeqa_speed80_20260917/dataset")
OUT = Path("results/narrative_focus_optimized_complete_20260917") / (family + "_" + method)
OUT.mkdir(parents=True, exist_ok=True)


class Clock:
    def __init__(self, profile=False):
        self.rows, self.profile, self.profiled = [], profile, set()
        self.count = 0
        self.block = None
        self.counts = {}

    @contextlib.contextmanager
    def span(self, name):
        a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        start_rows, start_count = len(self.rows), self.count
        profile_this = (self.profile and name == "loop_iteration"
                        and self.count >= 1 and not self.profiled)
        if profile_this:
            self.profiled.add("single_step")
            torch.cuda.synchronize()
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True)
            profiler.__enter__()
        a.record()
        with (torch.profiler.record_function("phase/"+name) if self.profile
              else contextlib.nullcontext()):
            yield
        b.record()
        if name == "loop_iteration":
            names = [r[0] for r in self.rows[start_rows:]]
            if self.count == start_count:
                name = "loop_finalize_or_stop"
            else:
                kind = next((x for x in ("focus_initial", "focus_retained", "focus_full",
                        "cached_forward", "dense_initial", "dense_forward") if x in names), "initial_transfer")
                name = "step/" + kind
        self.rows.append((name, a, b))
        if profile_this:
            torch.cuda.synchronize()
            profiler.__exit__(None,None,None)
            profiler.export_chrome_trace(str(OUT / f"{length}.single_step.trace.json"))
            (OUT / f"{length}.single_step.txt").write_text(profiler.key_averages(
                group_by_input_shape=True).table(sort_by="self_cuda_time_total",row_limit=120))

    def wrap(self, fn, name):
        def call(*a, **kw):
            label = name
            if name == "model_forward":
                ctx = kw.get("losa_context_kwargs")
                if ctx:
                    if not ctx["states"]:
                        label = "losa_initialize"
                    elif getattr(a[0], "_losa_optimized_primed_states", None) is not ctx["states"]:
                        label = "losa_prime"
                    else:
                        graph = getattr(a[0], "_losa_optimized_graph", None)
                        label = ("losa_replay" if graph is not None and graph.states is ctx["states"]
                                 else "losa_capture")
                else:
                    label = "finalize" if kw.get("store_kv", False) else "block_initial"
            elif name == "prefill_or_block_refresh":
                label = "prefill" if self.block is None else "block_refresh"
            if name == "focus_forward":
                label = "finalize" if kw.get("cache_only", False) else (
                    "focus_initial" if self.counts.get(str(self.block),0)==0 else "focus_forward")
            if label == "dense_forward" and self.counts.get(str(self.block), 0) == 0:
                label = "dense_initial"
            with self.span(label):
                result = fn(*a, **kw)
            if label == "focus_forward":
                _, begin, end = self.rows[-1]
                label = "focus_full" if result.positions.numel()==kw["input_ids"].shape[1] else "focus_retained"
                self.rows[-1] = (label,begin,end)
            if name == "transfer":
                self.count += 1
                self.counts[str(self.block)] = self.counts.get(str(self.block), 0) + 1
            return result
        return call

    def result(self):
        groups = {}
        for name, a, b in self.rows:
            groups.setdefault(name, []).append(a.elapsed_time(b))
        return dict(steps=self.count, steps_per_block=self.counts, phases={k:dict(count=len(v),total_ms=sum(v),
                    median_ms=statistics.median(v),values_ms=v) for k,v in groups.items()})


def clone(fn, clock, driver=False):
    f = inspect.unwrap(fn)
    if hasattr(f, "__func__"):
        f = f.__func__
    ns = dict(f.__globals__)
    for key, label in {
        "model_forward":"model_forward", "_cached_forward":"cached_forward",
        "_dense_block_prefill":"prefill_or_block_refresh",
        "build_sdar_prefix_cache":"prefill", "build_llada_prefix_cache":"prefill",
        "transfer_sdar":"transfer", "transfer_llada":"transfer",
        "focus_optimized_forward":"focus_forward", "_selected_llada_transfer":"transfer",
        "_selected_sdar_transfer":"transfer",
        "_transfer_tokens":"transfer", "select_transfer":"transfer",
    }.items():
        if key in ns:
            ns[key] = clock.wrap(ns[key], label)
    if "block_diffusion_generate" in ns and inspect.unwrap(ns["block_diffusion_generate"]) is not f:
        ns["block_diffusion_generate"] = clone(ns["block_diffusion_generate"], clock)
    if method == "focus":
        def timed_prefill(*a, **kw):
            from src.optimized.focus.prefill import build_prefix
            with clock.span("prefill"):
                return build_prefix(*a, **kw)
        ns["_timed_prefill"] = timed_prefill
    ns["_trim_to_first_eos"] = lambda generated, eos_id: generated
    ns["_clock"] = clock
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fun = tree.body[0]
    fun.decorator_list = []
    if method == "focus":
        for n in ast.walk(fun):
            if isinstance(n, ast.ImportFrom) and n.module == "prefill":
                # Replace the private clone import with the instrumented binding.
                n.module = "src.optimized.focus.prefill"
                n.level = 0
        class PrefillCalls(ast.NodeTransformer):
            def visit_Call(self,node):
                self.generic_visit(node)
                if isinstance(node.func,ast.Name) and node.func.id == "build_prefix":
                    node.func.id = "_timed_prefill"
                return node
        fun = PrefillCalls().visit(fun)
        tree.body[0] = fun
    if method == "dense" and family == "sdar" and f.__name__ == "block_diffusion_generate":
        def measured_model(callable_, *a, **kw):
            label = (("prefill" if clock.block is None else "finalize") if kw.get("store_kv") else "dense_forward")
            return clock.wrap(callable_, label)(*a, **kw)
        ns["_measured_model"] = measured_model
        class Calls(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                direct = isinstance(node.func, ast.Name) and node.func.id == "model"
                base = (isinstance(node.func, ast.Attribute) and node.func.attr == "model"
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "model")
                if direct or base:
                    node.args.insert(0, node.func)
                    node.func = ast.Name(id="_measured_model", ctx=ast.Load())
                return node
        fun = Calls().visit(fun)
        tree.body[0] = fun
    # The LLaDA dense first transfer is outside its iterative loop. Capture
    # from the current-block snapshot through token writeback as a full step.
    if method == "dense" and family == "llada":
        for node in list(ast.walk(fun)):
            if not (isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                    and node.target.id == "block_idx"):
                continue
            starts=[i for i,x in enumerate(node.body) if isinstance(x,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=="old_block_tokens" for t in x.targets)]
            ends=[i for i,x in enumerate(node.body) if isinstance(x,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=="prefix_indices" for t in x.targets)]
            assert len(starts)==len(ends)==1 and starts[0]<ends[0]
            a,b=starts[0],ends[0]
            first=ast.parse('with _clock.span("step/dense_initial"):\n pass').body[0]
            first.body=node.body[a:b]
            node.body[a:b]=[first]
    for block in ast.walk(fun):
        if isinstance(block, ast.For) and isinstance(block.target, ast.Name) and block.target.id in ("block_idx", "num_block"):
            block.body.insert(0, ast.parse(f"_clock.block = {block.target.id}").body[0])
    for node in ast.walk(fun):
        is_loop = isinstance(node, ast.While) or (
            isinstance(node, ast.For) and isinstance(node.target, ast.Name)
            and node.target.id in ("step", "_"))
        if is_loop:
            wrap = ast.parse('with _clock.span("loop_iteration"):\n pass').body[0]
            wrap.body = node.body
            node.body = [wrap]
            break
    exec(compile(ast.fix_missing_locations(tree), "<narrative-clock>", "exec"), ns)
    return torch.inference_mode()(ns[f.__name__])


if method == "dense":
    from src.optimized.dense.api import DenseOptimizedRuntime
    rt = DenseOptimizedRuntime(family=family)
else:
    from src.optimized.focus.api import FocusOptimizedRuntime
    rt = FocusOptimizedRuntime(family=family, attention_backend=("flash" if family == "llada" else "sdpa"))
model, tokenizer = rt.load()
options = dict(gen_length=256, block_length=32, steps=32, temperature=0.,
               eos_early_stop=False)
options.update(dict(threshold=.7,editing_threshold=.5,num_to_transfer=1)
               if family=="llada" else dict(threshold=.95,remasking_strategy="low_confidence_dynamic",mask_id=151669))
if method == "dense":
    original = model.generate
    # Same output-trimming policy as referenced Sparse benchmark.
    f = inspect.unwrap(original)
    if hasattr(f, "__func__"): f = f.__func__
    import types
    ns = {**f.__globals__, "_trim_to_first_eos": lambda generated,eos_id:generated}
    c = types.FunctionType(f.__code__, ns, f.__name__, f.__defaults__, f.__closure__)
    c.__kwdefaults__ = f.__kwdefaults__
    original = torch.inference_mode()(c)
    def plain(ids): return original(model,inputs=ids,**options)
else:
    from src.optimized.focus.generation import focus_optimized_generate as original
    options.update(family=family,alpha=1.5,attention_backend=("flash" if family == "llada" else "sdpa"))
    def plain(ids): return original(model,inputs=ids,**options).tokens

assert method in ("dense","focus")
if method == "dense":
    assert model.config.dense_sdpa_backend == "efficient_attention"

for length in (8192,16384,32768):
    file = ROOT / ("llada21" if family=="llada" else "sdar") / f"{length}.jsonl"
    record = json.loads(file.open().readline())
    ids = record["input_ids"]
    assert len(ids) == length  # strict prompt length; 32K total is 33024
    inputs=torch.tensor([ids],device="cuda")
    entry=dict(family=family,method=method,nominal_length=length,prompt_tokens=len(ids),
               source_id=record["source_id"],sample_index=record["sample_index"],
               input_sha256=hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
               options=options,dense_backend=getattr(model.config,"dense_sdpa_backend",None),
               warmup_seconds=[],seconds=[],run_graph_stats=[],
               runtime="dense_optimized_efficient" if method=="dense" else "focus_optimized",
               graph_replay=False,shared_kv=False)
    entry["declared_max_position_embeddings"]=model.config.max_position_embeddings
    reference=None
    refresh = "--refresh" in sys.argv
    previous = OUT/f"{length}.json"
    if refresh:
        entry = json.loads(previous.read_text())
        reference = torch.tensor(entry["tokens"], device="cuda")
        torch.manual_seed(42)
        warmed = plain(inputs)
        torch.cuda.synchronize()
        assert torch.equal(warmed, reference)
    for rep in range(0 if refresh else 5):
        torch.manual_seed(42);torch.cuda.synchronize();t=time.perf_counter()
        tokens=plain(inputs);torch.cuda.synchronize();elapsed=time.perf_counter()-t
        assert tokens.shape==(1,256)
        if reference is None:reference=tokens.clone()
        assert torch.equal(reference,tokens)
        entry["warmup_seconds" if rep<2 else "seconds"].append(elapsed)
        entry["run_graph_stats"].append(dict(getattr(model,"_losa_optimized_graph_stats",{})))
        print(family,method,length,"RUN",rep,elapsed,flush=True)
    for detail in ((False,) if "--timing-only" in sys.argv else (False, True)):
        clock=Clock(detail)
        fn=clone(original,clock)
        torch.manual_seed(42)
        output=fn(model,inputs=inputs,**options)
        tokens=output.tokens if hasattr(output,"tokens") else output
        torch.cuda.synchronize()
        assert torch.equal(tokens,reference),"instrumented tokens differ"
        entry["profile" if detail else "timing"]=clock.result()
        entry["profile" if detail else "timing"]["graph_stats"]=dict(getattr(model,"_losa_optimized_graph_stats",{}))
    entry["tokens"]=reference.cpu().tolist()
    entry["total_tokens"]=len(ids)+256
    entry["exceeds_config_window"]=len(ids)+256>model.config.max_position_embeddings
    (OUT/f"{length}.json").write_text(json.dumps(entry,indent=2))
    print("SAVED",family,method,length,entry["timing"]["steps"],flush=True)
