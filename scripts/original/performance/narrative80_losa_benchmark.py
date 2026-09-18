"""Exact-input NarrativeQA80 benchmark on physical GPU4; never reserves GPUs."""
import argparse
import ast
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import textwrap
import time
import torch
from src.optimized.losa.api import LoSAOptimizedRuntime
from src.optimized.losa.generation import block_diffusion_generate

DATA = Path("/data0/gs/dataset_preparation/narrativeqa_speed80_20260917/dataset")
MODELS = {"llada21": "/data0/ysy/models/LLaDA2.1-mini",
          "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32"}


class Clock:
    def __init__(self, timing=False):
        self.timing = timing
        self.counts = {}
        self.cached = 0
        self.events = []
        self.block = None

    def wrap_transfer(self, fn):
        def wrapped(*a, **kw):
            result = fn(*a, **kw)
            before = self.counts.get(self.block, 0)
            self.counts[self.block] = before + 1
            self.cached += int(before > 0)
            return result
        return wrapped

    @contextlib.contextmanager
    def iteration(self):
        before = self.counts.get(self.block, 0)
        if self.timing:
            a, b = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            a.record()
        try:
            yield
        finally:
            if self.timing and self.counts.get(self.block, 0) > before:
                b.record()
                self.events.append((self.block, before, a, b))

    def rows(self):
        return [dict(block=block, loop_iteration=index, steady=index >= 1,
                     cuda_ms=a.elapsed_time(b)) for block, index, a, b in self.events]


def clone_losa(generate, clock):
    f = inspect.unwrap(generate)
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fn = tree.body[0]
    fn.decorator_list = []
    blocks = [n for n in ast.walk(fn) if isinstance(n, ast.For)
              and isinstance(n.target, ast.Name) and n.target.id == "block_idx"]
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
    assert len(blocks) == len(loops) == 1
    blocks[0].body.insert(0, ast.parse("_clock.block = block_idx").body[0])
    if clock.timing:
        wrap = ast.parse("with _clock.iteration():\n pass").body[0]
        wrap.body = loops[0].body
        loops[0].body = [wrap]
    ns = {**f.__globals__, "_clock": clock}
    for key in ("transfer_llada", "transfer_sdar"):
        ns[key] = clock.wrap_transfer(ns[key])
    exec(compile(ast.fix_missing_locations(tree), "<narrative80-losa>", "exec"), ns)
    cloned = torch.inference_mode()(ns[f.__name__])
    def call(*a, **kw):
        return cloned(*a, **kw).tokens
    return call


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--method", choices=["losa_optimized"], required=True)
    p.add_argument("--length", type=int, choices=[8192, 16384, 32768], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "4", "Physical GPU4 only"
    args.output.mkdir(parents=True, exist_ok=True)
    family = "sdar" if args.model == "sdar" else "llada"
    tokenizer_family = "sdar" if family == "sdar" else "llada21"
    make_driver = clone_losa
    source = DATA / tokenizer_family / f"{args.length}.jsonl"
    samples = [json.loads(l) for l in source.read_text().splitlines()]
    assert len(samples) == 80 and len({s["source_id"] for s in samples}) == 80
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        assert sha(Path(MODELS[args.model]) / f) == sha(DATA / "tokenizers" / tokenizer_family / f)
    rt = LoSAOptimizedRuntime(family=family, model_path=MODELS[args.model])
    model, tok = rt.load()
    model._losa_optimized_graph_replay = True
    model._losa_optimized_shared_kv = False
    model._losa_optimized_verify_graph = False
    options = dict(gen_length=256, block_length=32, steps=32, temperature=0.,
                   eos_early_stop=False, family=family, use_losa=True,
                   losa_gqa_mode="group_mean", losa_token_budget=256,
                   losa_active_topk=5, losa_page_size=16, losa_fused_state=True)
    options.update(dict(num_to_transfer=1, threshold=.7, editing_threshold=.5)
                   if family == "llada" else
                   dict(threshold=.95, remasking_strategy="low_confidence_dynamic",
                        mask_id=151669))
    prompt_length = min(args.length, model.config.max_position_embeddings - 256)
    assert prompt_length % 32 == 0
    def input_for(s):
        ids = s["input_ids"]
        assert len(ids) == args.length
        if len(ids) > prompt_length:
            ids = ids[:prompt_length//2] + ids[-(prompt_length-prompt_length//2):]
        return torch.tensor([ids], dtype=torch.long, device="cuda")

    manifest = dict(model=args.model, model_path=MODELS[args.model], method=args.method,
                    requested_context=args.length, actual_prompt_tokens=prompt_length,
                    generation=options, sparse_config=dict(losa_gqa_mode="group_mean", losa_token_budget=256,
                        losa_active_topk=5, losa_page_size=16, losa_fused_state=True,
                        graph_replay=True, shared_kv=False),
                    sample_count=min(args.limit,80), repeats=args.repeats, warmups=2,
                    gpu="physical4", gpu_name=torch.cuda.get_device_name(),
                    torch=torch.__version__, source=str(source), source_sha256=sha(source),
                    git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    git_status=subprocess.check_output(["git","status","--short"],text=True),
                    source_hashes={str(f):sha(f) for f in [
                        Path(__file__), *Path("src/optimized/losa").glob("*.py")]},
                    eos_return_trimming=False,
                    steady_definition="all completed steps except first per block (zero-based index >=1); complete loop including initialize, prime, capture, replay and editing; aggregate sum(ms)/sum(count)",
                    e2e_boundary="generate start to CUDA synchronize, excluding tokenize/load/save",
                    step_count_definition="all denoising forwards with transfer including initial block forward; excludes prompt prefill")
    atomic_json(args.output/"manifest.json", manifest)
    first = input_for(samples[0])
    warmups = []
    for i in range(2):
        clock = Clock()
        call = make_driver(block_diffusion_generate, clock)
        torch.manual_seed(42)
        torch.cuda.synchronize()
        t = time.perf_counter()
        output = call(model, inputs=first, **options)
        torch.cuda.synchronize()
        assert output.shape == (1,256)
        warmups.append(time.perf_counter()-t)
        print("WARM",args.model,args.method,args.length,i,warmups[-1],flush=True)
    atomic_json(args.output/"warmup.json",warmups)
    for sample in samples[:args.limit]:
        index = sample["sample_index"]
        dest = args.output/f"sample_{index:03d}.json"
        if dest.exists():
            raise RuntimeError("Refusing to overwrite prior sample: "+str(dest))
        inputs = input_for(sample)
        runs = []
        for repeat in range(args.repeats):
            clock = Clock()
            call = make_driver(block_diffusion_generate, clock)
            torch.manual_seed(42)
            torch.cuda.synchronize()
            t = time.perf_counter()
            output = call(model, inputs=inputs, **options)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-t
            assert output.shape == (1,256)
            tokens = output.cpu().tolist()[0]
            total = sum(clock.counts.values())
            assert total == clock.cached + len(clock.counts)
            runs.append(dict(repeat=repeat,seconds=seconds,step_count=total,
                             cached_forward_count=clock.cached,steps_per_block=clock.counts,
                             output_ids=tokens,
                             generated_text=tok.decode(tokens,skip_special_tokens=False),
                             readable_text=tok.decode(tokens,skip_special_tokens=True)))
            atomic_json(args.output/f"sample_{index:03d}.partial.json",
                        dict(sample_index=index,source_id=sample["source_id"],runs=runs))
        # Event instrumentation is a separate pass, never included in E2E.
        clock = Clock(timing=True)
        call = make_driver(block_diffusion_generate,clock)
        torch.manual_seed(42)
        measured = call(model,inputs=inputs,**options)
        torch.cuda.synchronize()
        records = clock.rows()
        stable = [r["cuda_ms"] for r in records if r["steady"]]
        assert len(stable) == sum(max(0, n-1) for n in clock.counts.values())
        token_match = measured.cpu().tolist()[0] == runs[0]["output_ids"]
        count_match = sum(clock.counts.values()) == runs[0]["step_count"]
        assert token_match and count_match, "Timing instrumentation changed trajectory"
        result = dict(sample_index=index,source_id=sample["source_id"],model=args.model,
                      method=args.method,requested_context=args.length,prompt_tokens=prompt_length,
                      generation_length=256,prompt_sha256=hashlib.sha256(
                          inputs.cpu().numpy().tobytes()).hexdigest(),
                      generation_options=options, runs=runs,
                      e2e_median_seconds=statistics.median(r["seconds"] for r in runs),
                      step_counts=[r["step_count"] for r in runs],
                      output_repeat_equal=all(r["output_ids"]==runs[0]["output_ids"] for r in runs),
                      instrumentation_matches=True,step_records=records,
                      stable_step_count=len(stable),
                      stable_step_median_ms=statistics.median(stable) if stable else None,
                      stable_step_mean_ms=statistics.mean(stable) if stable else None)
        atomic_json(dest,result)
        print("SAMPLE_DONE",args.model,args.method,args.length,index,
              result["e2e_median_seconds"],result["step_counts"],result["stable_step_median_ms"],
              flush=True)
    atomic_json(args.output/"COMPLETE.json",dict(samples=min(args.limit,80),completed=True))


if __name__ == "__main__":
    main()
