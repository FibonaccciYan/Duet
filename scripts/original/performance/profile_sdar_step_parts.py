"""Single actual denoising iteration; private AST instrumentation, no algorithm edits."""
import ast
import contextlib
import inspect
import json
import os
from pathlib import Path
import textwrap
import time
import torch
from src.reference.losa.generation import load_model_and_tokenizer
from src.runtime import patch_method

OUT = Path("results/versioned_optimization_20260916/step_parts")
OUT.mkdir(exist_ok=True, parents=True)
TIMING_ONLY = os.environ.get("PARTS_TIMING") == "1"


class Probe:
    def __init__(self, name):
        self.name, self.done, self.active = name, False, False

    @contextlib.contextmanager
    def step(self, step):
        if step != 1 or self.done:
            yield
            return
        self.done = True
        torch.cuda.synchronize()
        if TIMING_ONLY:
            self.active = True
            self.marks = []
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            started = time.perf_counter()
            yield
            end.record()
            end.synchronize()
            self.active = False
            result = dict(step_cuda_ms=begin.elapsed_time(end),
                          wall_ms=(time.perf_counter()-started)*1000,
                          phases=[dict(name=n, cuda_ms=a.elapsed_time(b))
                                  for n, a, b in self.marks])
            (OUT / (self.name + ".timing.json")).write_text(json.dumps(result, indent=2))
            print(self.name, result, flush=True)
            return
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            self.active = True
            with torch.profiler.record_function("phase/whole_step"):
                yield
            self.active = False
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(OUT / (self.name + ".trace.json")))
        events = prof.events()
        scopes = []
        for e in events:
            if e.name.startswith(("phase/", "module/")):
                scopes.append(dict(name=e.name, cpu_us=e.cpu_time_total,
                                   gpu_us=e.device_time_total))
        # Exclusive CUDA time on CPU operators, classified by nearest scope.
        buckets = {}
        for e in events:
            if e.device_type != torch.autograd.DeviceType.CPU:
                continue
            us = e.self_device_time_total
            if us <= 0:
                continue
            parent = e
            while parent and not parent.name.startswith(("phase/", "module/")):
                parent = parent.cpu_parent
            label = parent.name if parent else "unattributed"
            buckets[label] = buckets.get(label, 0.) + us
        table = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=100)
        (OUT / (self.name + ".txt")).write_text(table)
        (OUT / (self.name + ".json")).write_text(json.dumps(
            dict(scopes=scopes, exclusive_gpu_us=buckets), indent=2))
        print(self.name, "EXCLUSIVE_GPU_US", buckets, flush=True)

    @contextlib.contextmanager
    def phase(self, name):
        if not self.active:
            yield
        elif TIMING_ONLY:
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            yield
            b.record()
            self.marks.append((name, a, b))
        else:
            with torch.profiler.record_function("phase/" + name):
                yield


def instrument(function, probe):
    f = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fn = tree.body[0]
    fn.decorator_list = []
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == "step"]
    assert len(loops) == 1
    loop = loops[0]
    phases = []
    label = "mask_and_stop"
    for stmt in loop.body:
        text = ast.unparse(stmt)
        if text.startswith("logit_positions ="):
            label = "model_forward"
        elif text.startswith("(x0, x0_p) ="):
            label = "sampling"
        elif text.startswith("if logit_positions is not None"):
            label = "expand_sparse_logits"
        elif text.startswith("confidence ="):
            label = "transfer_selection"
        elif text.startswith("cur_x[transfer_index] ="):
            label = "token_scatter"
        if not phases or phases[-1][0] != label:
            phases.append((label, []))
        phases[-1][1].append(stmt)
    assert {x[0] for x in phases} == {
        "mask_and_stop", "model_forward", "sampling", "expand_sparse_logits",
        "transfer_selection", "token_scatter"}
    body = []
    for label, statements in phases:
        ctx = ast.parse(f"with _probe.phase({label!r}):\n pass").body[0]
        ctx.body = statements
        body.append(ctx)
    outer = ast.parse("with _probe.step(step):\n pass").body[0]
    outer.body = body
    loop.body = [outer]
    ns = {**f.__globals__, "_probe": probe}
    exec(compile(ast.fix_missing_locations(tree), "<step-parts>", "exec"), ns)
    return ns[f.__name__]


model, tokenizer = load_model_and_tokenizer(
    "sdar", model_path="/data0/ysy/models/SDAR-8B-Chat-b32")
data = Path("/data0/gs/losa_accuracy_prep/data/longbench_all21_full/gov_report.jsonl")
record = json.loads(data.read_text().splitlines()[59])
raw = tokenizer(record["prompt"], return_tensors="pt").input_ids
ids = torch.cat((raw[:, :4096], raw[:, -4096:]), 1).cuda()
options = dict(gen_length=32, block_length=32, steps=32, temperature=0.,
               threshold=.95, remasking_strategy="low_confidence_dynamic",
               mask_id=151669, eos_early_stop=False)
for mode in ("dense_optimized", "sparse_optimized"):
    if hasattr(model, "_sparse_original_generate"):
        model.generate = model._sparse_original_generate
    kw = dict(moe_expert_patch=True)
    if mode.startswith("sparse"):
        kw.update(query_sparse=True, prefix_sparse=True, prefix_token_budget=256)
    patch_method(model, mode, model_name="sdar", **kw)
    torch.manual_seed(42)
    reference = model.generate(inputs=ids, **options)
    torch.cuda.synchronize()
    probe = Probe(mode)
    handles, stacks = [], {}
    for name, module in model.named_modules():
        if TIMING_ONLY:
            continue
        last = name.split(".")[-1]
        if last not in {"self_attn", "mlp", "input_layernorm",
                        "post_attention_layernorm", "q_norm", "k_norm",
                        "norm", "lm_head", "o_proj", "gate_proj",
                        "up_proj", "down_proj"}:
            continue
        def pre(mod, args, label=last):
            ctx = (torch.profiler.record_function("module/" + label)
                   if probe.active else contextlib.nullcontext())
            stacks.setdefault(id(mod), []).append(ctx)
            ctx.__enter__()
        def post(mod, args, output):
            stacks[id(mod)].pop().__exit__(None, None, None)
        handles.extend([module.register_forward_pre_hook(pre),
                        module.register_forward_hook(post)])
    ns = inspect.unwrap(model.generate).__globals__
    orig = ns["block_diffusion_generate"]
    ns["block_diffusion_generate"] = instrument(orig, probe)
    try:
        torch.manual_seed(42)
        actual = model.generate(inputs=ids, **options)
        torch.cuda.synchronize()
        assert torch.equal(actual, reference), "instrumentation changed tokens"
        assert probe.done
    finally:
        ns["block_diffusion_generate"] = orig
        for h in handles:
            h.remove()
    print(mode, "TOKENS_UNCHANGED", flush=True)
