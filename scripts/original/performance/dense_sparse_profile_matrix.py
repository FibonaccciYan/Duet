"""Matched dense/Sparse phase matrix. Run each method/context in a fresh process."""
import argparse
import ast
import contextlib
import gc
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import textwrap
import time
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.runtime import method_source_directory, patch_method
from src.kernels.optimized.function_binding import _bind_globals
from scripts.original.performance.profile_focus_losa_parts import Probe
import scripts.original.performance.profile_focus_losa_parts as probe_module

p = argparse.ArgumentParser()
p.add_argument("--family", choices=["sdar", "llada"], required=True)
p.add_argument("--method", choices=["dense_optimized", "sparse_optimized"], required=True)
p.add_argument("--index", type=int, default=59)
p.add_argument("--contexts", nargs="+", type=int, default=[8192, 16384, 32736])
args = p.parse_args()
OUT = Path("results/versioned_optimization_20260916/dense_sparse_profile_matrix")
OUT.mkdir(parents=True, exist_ok=True)
probe_module.OUT = OUT


def phase(name, body):
    n = ast.parse(f"with _probe.phase({name!r}):\n pass").body[0]
    n.body = body
    return n


def transfer_clone(function, probe):
    f = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fn = tree.body[0]
    fn.decorator_list = []
    groups = []
    label = "phase/selection"
    for s in fn.body:
        t = ast.unparse(s)
        if "_sample_with_confidence(" in t:
            label = "phase/sampling"
        elif t.startswith("if logit_positions is None"):
            label = "phase/selection"
        elif t.startswith("if transfer.any()"):
            label = "phase/scatter"
        if isinstance(s, ast.Return):
            label = "phase/transfer_return"
        if not groups or groups[-1][0] != label:
            groups.append((label, []))
        groups[-1][1].append(s)
    fn.body = [phase(n, body) for n, body in groups]
    ns = {**f.__globals__, "_probe": probe}
    exec(compile(ast.fix_missing_locations(tree), "<transfer-parts>", "exec"), ns)
    return ns[f.__name__]


def driver_clone(function, probe, family):
    f = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fn = tree.body[0]
    fn.decorator_list = []
    variable = "step" if family == "sdar" else "_"
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == variable]
    assert len(loops) == 1, (family, len(loops))
    loop = loops[0]
    groups = []
    label = "phase/mask_and_stop"
    for s in loop.body:
        t = ast.unparse(s)
        if family == "sdar":
            if t.startswith("logit_positions ="):
                label = "phase/model_forward"
            elif t.startswith("(x0, x0_p) ="):
                label = "phase/sampling"
            elif t.startswith("if logit_positions is not None"):
                label = "phase/expand_predictions"
            elif t.startswith("confidence ="):
                label = "phase/selection"
            elif t.startswith("cur_x[transfer_index] ="):
                label = "phase/scatter"
        else:
            if t.startswith("block_input ="):
                label = "phase/input_setup"
            elif "_cached_forward(" in t:
                label = "phase/model_forward"
            elif t.startswith("selection_state['step']"):
                label = "phase/transfer_setup"
            elif "_transfer_tokens(" in t:
                label = "phase/transfer_total"
            elif t.startswith("if query_sparse"):
                label = "phase/check_and_writeback"
        if not groups or groups[-1][0] != label:
            groups.append((label, []))
        groups[-1][1].append(s)
    assert any(n == "phase/model_forward" for n, _ in groups)
    outer = ast.parse(f"with _probe.step({variable}):\n pass").body[0]
    outer.body = [phase(n, b) for n, b in groups]
    loop.body = [outer]
    ns = {**f.__globals__, "_probe": probe}
    if family == "llada":
        ns["_transfer_tokens"] = transfer_clone(ns["_transfer_tokens"], probe)
    exec(compile(ast.fix_missing_locations(tree), "<step-matrix>", "exec"), ns)
    return ns[f.__name__]


def module_hooks(model, probe):
    handles, stack = [], {}
    labels = {"self_attn", "attention", "mlp", "input_layernorm",
              "post_attention_layernorm", "q_norm", "k_norm", "query_layernorm",
              "key_layernorm", "norm", "lm_head", "o_proj", "dense", "query_key_value",
              "gate_proj", "up_proj", "down_proj"}
    for path, module in model.named_modules():
        label = path.split(".")[-1]
        if label not in labels:
            continue
        def pre(m, a, label=label):
            c = (torch.profiler.record_function("module/" + label)
                 if probe.active else contextlib.nullcontext())
            stack.setdefault(id(m), []).append(c)
            c.__enter__()
        def post(m, a, output):
            stack[id(m)].pop().__exit__(None, None, None)
        handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post)]
    return handles


path = "/data0/ysy/models/" + ("LLaDA2.1-mini" if args.family == "llada" else "SDAR-8B-Chat-b32")
model, tokenizer = load_model_and_tokenizer(args.family, model_path=path)
patch_options = dict(moe_expert_patch=True)
if args.method == "sparse_optimized":
    patch_options.update(query_sparse=True, prefix_sparse=True, prefix_token_budget=256)
patch_method(model, args.method, model_name=args.family, **patch_options)
data = Path("/data0/gs/losa_accuracy_prep/data/longbench_all21_full/gov_report.jsonl")
record = json.loads(data.read_text().splitlines()[args.index])
raw = tokenizer(record["prompt"], return_tensors="pt").input_ids
options = dict(gen_length=32, block_length=32, steps=32, temperature=0., eos_early_stop=False)
options.update(dict(threshold=.7, editing_threshold=.5, num_to_transfer=1)
               if args.family == "llada" else
               dict(threshold=.95, remasking_strategy="low_confidence_dynamic", mask_id=151669))
original_generate = model.generate
all_results = []
for length in args.contexts:
    assert raw.shape[1] >= length
    assert length + 32 <= model.config.max_position_embeddings
    inputs = torch.cat((raw[:, :length//2], raw[:, -(length-length//2):]), 1).cuda()
    stem = f"{args.family}_{length}_{args.method}"
    warm_seconds = []
    for _ in range(2):
        torch.manual_seed(42)
        torch.cuda.synchronize()
        start = time.perf_counter()
        warm = original_generate(inputs=inputs, **options)
        torch.cuda.synchronize()
        warm_seconds.append(time.perf_counter() - start)
    durations, outputs = [], []
    for repeat in range(3):
        torch.manual_seed(42)
        torch.cuda.synchronize()
        start = time.perf_counter()
        tokens = original_generate(inputs=inputs, **options)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
        outputs.append(tokens.cpu().tolist())
    assert all(x == outputs[0] for x in outputs)
    assert warm.cpu().tolist() == outputs[0]
    timings = []
    for detail in (False, False, False, True):
        probe = Probe(stem, 1, detail)
        handles = module_hooks(model, probe) if detail else []
        driver = inspect.unwrap(original_generate)
        if args.family == "sdar":
            ns = driver.__globals__
            old = ns["block_diffusion_generate"]
            ns["block_diffusion_generate"] = driver_clone(old, probe, args.family)
            call = original_generate
        else:
            call = torch.inference_mode()(driver_clone(driver, probe, args.family))
        try:
            torch.manual_seed(42)
            actual = (call(inputs=inputs, **options) if args.family == "sdar"
                      else call(model, inputs=inputs, **options))
            torch.cuda.synchronize()
            assert probe.done
            assert actual.cpu().tolist() == outputs[0], stem + " instrumentation changed output"
            if not detail:
                timings.append(probe.result)
        finally:
            for h in handles:
                h.remove()
            if args.family == "sdar":
                ns["block_diffusion_generate"] = old
        del handles, call, actual
    source_files = [Path(__file__), Path("src/runtime.py"),
                    Path(method_source_directory(args.method)) / "api.py", Path(f"src/reference/sparse/{args.family}_patch.py"),
                    Path("src/kernels/optimized/small_moe.py")]
    result = dict(family=args.family, method=args.method,
                  source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
                  torch_version=torch.__version__, device=torch.cuda.get_device_name(),
                  prompt_tokens=length, generation_budget=32,
                  output_tokens=len(outputs[0][0]), sample_index=args.index,
                  sample_id=record.get("_id"), dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
                  prompt_sha256=hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
                  warmup_runs=2, warmup_seconds=warm_seconds, end_to_end_seconds=durations,
                  end_to_end_median=statistics.median(durations),
                  options=options, step_zero_based=1, step_description="second actual denoising iteration",
                  phase_runs=timings, instrumentation_tokens_equal=True, repeated_tokens_equal=True,
                  tokens=outputs[0], graphs=False, moe_backend="small_moe")
    (OUT / (stem + ".timing.json")).write_text(json.dumps(result, indent=2))
    all_results.append(result)
    (OUT / (stem + ".matrix.json")).write_text(json.dumps(all_results, indent=2))
    print("MATRIX_DONE", stem, result["end_to_end_median"], flush=True)
