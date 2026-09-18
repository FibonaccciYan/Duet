"""NarrativeQA exact-ID LoSA/dense matrix. No shared source mutations."""
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
parser.add_argument("method", choices=("dense", "losa"))
parser.add_argument("--refresh", action="store_true")
parser.add_argument("--timing-only", action="store_true")
args = parser.parse_args()
family, method = args.family, args.method
ROOT = Path("/data0/gs/dataset_preparation/narrativeqa_speed80_20260917/dataset")
OUT = Path("results/narrative_losa_optimized_dense_20260917") / (family + "_" + method)
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
        a.record()
        yield
        b.record()
        if name == "loop_iteration":
            names = [r[0] for r in self.rows[start_rows:]]
            if self.count == start_count:
                name = "loop_finalize_or_stop"
            else:
                kind = next((x for x in ("losa_replay", "losa_capture", "losa_prime", "losa_initialize", "cached_forward",
                                         "dense_forward") if x in names), "initial_transfer")
                name = "step/" + kind
        self.rows.append((name, a, b))

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
            with self.span(label):
                if self.profile and label in ("cached_forward", "dense_forward", "losa_replay", "losa_prime", "losa_initialize") and label not in self.profiled:
                    self.profiled.add(label)
                    with torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as p:
                        result = fn(*a, **kw)
                    p.export_chrome_trace(str(OUT / f"{length}.{label}.trace.json"))
                    (OUT / f"{length}.{label}.txt").write_text(
                        p.key_averages(group_by_input_shape=True).table(
                            sort_by="self_cuda_time_total", row_limit=100))
                else:
                    result = fn(*a, **kw)
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
        "_transfer_tokens":"transfer", "select_transfer":"transfer",
    }.items():
        if key in ns:
            ns[key] = clock.wrap(ns[key], label)
    if "block_diffusion_generate" in ns and inspect.unwrap(ns["block_diffusion_generate"]) is not f:
        ns["block_diffusion_generate"] = clone(ns["block_diffusion_generate"], clock)
    ns["_trim_to_first_eos"] = lambda generated, eos_id: generated
    ns["_clock"] = clock
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fun = tree.body[0]
    fun.decorator_list = []
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
    from src.optimized.losa.api import LoSAOptimizedRuntime
    rt = LoSAOptimizedRuntime(family=family, losa_gqa_mode="group_mean")
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
    from src.optimized.losa.generation import block_diffusion_generate as original
    model._losa_optimized_graph_replay = True
    model._losa_optimized_shared_kv = False
    model._losa_optimized_verify_graph = False
    options.update(family=family,use_losa=True,losa_gqa_mode="group_mean",
                   losa_token_budget=256,losa_active_topk=5,losa_page_size=16,
                   losa_fused_state=True)
    def plain(ids): return original(model,inputs=ids,**options).tokens

for length in (8192,16384,32768):
    file = ROOT / ("llada21" if family=="llada" else "sdar") / f"{length}.jsonl"
    record = json.loads(file.open().readline())
    ids = record["input_ids"]
    if length+256>32768:
        ids=ids[:16256]+ids[-16256:]
    inputs=torch.tensor([ids],device="cuda")
    entry=dict(family=family,method=method,nominal_length=length,prompt_tokens=len(ids),
               source_id=record["source_id"],sample_index=record["sample_index"],
               input_sha256=hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
               options=options,dense_backend=getattr(model.config,"dense_sdpa_backend",None),
               warmup_seconds=[],seconds=[],run_graph_stats=[],
               runtime="dense_optimized_efficient" if method=="dense" else "losa_optimized",
               graph_replay=method!="dense",shared_kv=False)
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
    (OUT/f"{length}.json").write_text(json.dumps(entry,indent=2))
    print("SAVED",family,method,length,entry["timing"]["steps"],flush=True)
