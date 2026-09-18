"""optimized FOCUS full dataset benchmark. Import/preflight never runs GPU work."""
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

DATA = Path("/data0/gs/dataset_preparation/narrativeqa_speed80_20260917/dataset")
DENSE = Path("results/narrative80_gpu4_20260917/formal")
MODELS = {"llada21": "/data0/ysy/models/LLaDA2.1-mini",
          "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32"}
LENGTHS = (8192, 16384, 32768)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_ids(sample, length):
    ids = sample["input_ids"]
    if len(ids) != length:
        raise ValueError("dataset input length mismatch")
    size = min(length, 32768 - 256)
    return ids if size == length else ids[:size // 2] + ids[-(size - size // 2):]


def prompt_hash(ids):
    return hashlib.sha256(torch.tensor([ids], dtype=torch.int64).numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def generation_options(model):
    return dict(gen_length=256, block_length=32, steps=32, temperature=0.,
                eos_early_stop=False,
                **(dict(threshold=.7, editing_threshold=.5, num_to_transfer=1)
                   if model == "llada21" else
                   dict(threshold=.95, remasking_strategy="low_confidence_dynamic",
                        mask_id=151669)))


def noninitial_rows(rows):
    # Intentionally do not use historical "steady" labels (which exclude step1).
    return [r for r in rows if r["loop_iteration"] >= 1]


def step_summary(rows):
    selected = noninitial_rows(rows)
    values = [r["cuda_ms"] for r in selected]
    return dict(noninitial_step_count=len(values), noninitial_step_total_ms=sum(values),
                noninitial_step_mean_ms=statistics.mean(values) if values else None)


class Clock:
    def __init__(self, timing=False, event_factory=None):
        self.timing, self.block = timing, None
        self.counts, self.events = {}, []
        self.event_factory = event_factory or (
            lambda: torch.cuda.Event(enable_timing=True))

    def wrap_transfer(self, function):
        def call(*args, **kwargs):
            output = function(*args, **kwargs)
            self.counts[self.block] = self.counts.get(self.block, 0) + 1
            return output
        return call

    @contextlib.contextmanager
    def iteration(self):
        block = self.block
        index = self.counts.get(block, 0)
        if self.timing:
            begin, end = self.event_factory(), self.event_factory()
            begin.record()
        try:
            yield
        finally:
            # Handles LLaDA's final editing pass that breaks after transfer.
            # Excludes loop-exit checks without an executed transfer.
            after = self.counts.get(block, 0)
            if after != index:
                if after != index + 1:
                    raise RuntimeError("more than one transfer in an iteration")
                if self.timing:
                    end.record()
                    self.events.append((block, index, begin, end))

    def rows(self):
        return [dict(block=block, loop_iteration=index, noninitial=index >= 1,
                     cuda_ms=a.elapsed_time(b))
                for block, index, a, b in self.events]


def clone_driver(function, clock):
    f = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
    fn = tree.body[0]
    fn.decorator_list = []
    blocks = [n for n in ast.walk(fn) if isinstance(n, ast.For)
              and isinstance(n.target, ast.Name) and n.target.id == "block_idx"]
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.While)]
    if len(blocks) != 1 or len(loops) != 1:
        raise RuntimeError("FOCUS driver structure changed")
    blocks[0].body.insert(0, ast.parse("_clock.block = block_idx").body[0])
    if clock.timing:
        wrapper = ast.parse("with _clock.iteration():\n pass").body[0]
        wrapper.body = loops[0].body
        loops[0].body = [wrapper]
    ns = {**f.__globals__, "_clock": clock}
    for key in ("_selected_llada_transfer", "_selected_sdar_transfer"):
        ns[key] = clock.wrap_transfer(ns[key])
    exec(compile(ast.fix_missing_locations(tree), "<focus-full-dataset>", "exec"), ns)
    return torch.inference_mode()(ns[f.__name__])


def load_cell(model, length, check_dense=True):
    source = DATA / model / f"{length}.jsonl"
    samples = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(samples) == 80 and len({s["source_id"] for s in samples}) == 80
    assert [s["sample_index"] for s in samples] == list(range(80))
    for file in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        assert sha(Path(MODELS[model]) / file) == sha(DATA / "tokenizers" / model / file)
    ids = [prompt_ids(s, length) for s in samples]
    if check_dense:
        cell = DENSE / f"{model}_{length}_dense_optimized"
        manifest = json.loads((cell / "manifest.json").read_text())
        assert manifest["actual_prompt_tokens"] == len(ids[0])
        assert manifest["sample_count"] == 80 and manifest["repeats"] == 3
        for sample, tokens in zip(samples, ids):
            result = json.loads((cell / f"sample_{sample['sample_index']:03d}.json").read_text())
            assert result["source_id"] == sample["source_id"]
            assert result["prompt_sha256"] == prompt_hash(tokens)
            assert result["generation_length"] == 256
            assert result["instrumentation_matches"]
            for key, value in generation_options(model).items():
                assert result["generation_options"][key] == value, (model, length, key)
            expected = result["runs"][0]["step_count"] - len(
                result["runs"][0]["steps_per_block"])
            assert len(noninitial_rows(result["step_records"])) == expected
    return source, samples


def preflight():
    identity = None
    cells = []
    for model in MODELS:
        for length in LENGTHS:
            source, samples = load_cell(model, length)
            current = [s["source_id"] for s in samples]
            if identity is None:
                identity = current
            assert identity == current
            cells.append(dict(model=model, nominal_length=length,
                              prompt_tokens=len(prompt_ids(samples[0], length)),
                              samples=80, repeats=3, source_sha256=sha(source)))
    return dict(status="PREPARED_NOT_STARTED", method="focus_optimized", cells=cells,
                formal_gpu=4, smoke_gpu=5, generation=256,
                noninitial_definition="per-block zero-based iteration >=1; include editing",
                dataset_mean="sum(noninitial cuda_ms) / sum(noninitial step count)",
                dense="reuse saved results; reaggregate raw step records, never rerun",
                holder_policy="formal GPU4: no reserver calls; smoke GPU5: reserved wrapper")


def validate_mode(args):
    if args.smoke:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "5" or args.limit != 1 or args.repeats != 1:
            raise ValueError("smoke requires physical GPU5, --limit 1 --repeats 1")
    elif os.environ.get("CUDA_VISIBLE_DEVICES") != "4" or args.limit != 80 or args.repeats != 3:
        raise ValueError("formal requires physical GPU4, 80 samples, 3 repeats")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--length", type=int, choices=LENGTHS)
    p.add_argument("--output", type=Path)
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2))
        return
    if not args.model or not args.length or args.output is None:
        p.error("--model, --length and --output required")
    validate_mode(args)
    source, samples = load_cell(args.model, args.length)
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Refusing to overwrite existing output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    from src.optimized.focus.api import FocusOptimizedRuntime
    from src.optimized.focus.generation import focus_optimized_generate
    family = "llada" if args.model == "llada21" else "sdar"
    rt = FocusOptimizedRuntime(family=family, model_path=MODELS[args.model])
    model, tok = rt.load()
    assert model.config.max_position_embeddings == 32768
    options = generation_options(args.model)
    call_options = dict(options, family=family, alpha=rt.alpha,
                        attention_backend=rt.attention_backend)
    hashes = {str(path):sha(path) for path in [
        Path(__file__), *sorted(Path("src/optimized/focus").glob("*.py")),
        Path("src/reference/focus/generation.py"), Path("src/reference/focus/algorithm.py"),
        Path("src/runtime_compat.py"), Path("src/reference/losa/generation.py"),
        Path("src/kernels/optimized/small_moe.py"),
        Path("src/kernels/optimized/sdar_norm.py"),
        Path("src/kernels/optimized/gqa_view.py"),
        Path("src/kernels/optimized/block_scratch_cache.py"),
        Path("src/reference/sparse/triton_kernels.py")]}
    manifest = dict(model=args.model, model_path=MODELS[args.model], method="focus_optimized",
                    requested_context=args.length,
                    actual_prompt_tokens=len(prompt_ids(samples[0], args.length)),
                    generation=options, focus_config=dict(alpha=rt.alpha, backend=rt.attention_backend),
                    warmups=2, repeats=args.repeats, sample_count=args.limit, smoke=args.smoke,
                    physical_gpu=5 if args.smoke else 4, gpu_name=torch.cuda.get_device_name(),
                    torch=torch.__version__, source=str(source), source_sha256=sha(source),
                    source_hashes=hashes,
                    git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    git_status=subprocess.check_output(["git","status","--short"],text=True),
                    step_definition="whole loop with actual transfer; excludes prefill/finalize/empty stop",
                    noninitial_definition="exclude exactly first executed step of each block (index0)",
                    mean_definition="sum(cuda_ms for index>=1) / count(index>=1)",
                    e2e_boundary="counted generate call through CUDA synchronize; exclude load/save/tokenize",
                    timing_pass="separate; not included in E2E; compared to first timed output",
                    dense_reference=str(DENSE/f"{args.model}_{args.length}_dense_optimized"))
    atomic_json(args.output/"manifest.json", manifest)
    def inputs_for(sample):
        return torch.tensor([prompt_ids(sample,args.length)],dtype=torch.long,device="cuda")
    first = inputs_for(samples[0])
    warmups=[]
    for _ in range(2):
        torch.manual_seed(42)
        torch.cuda.synchronize()
        started=time.perf_counter()
        output=focus_optimized_generate(model,inputs=first,**call_options)
        torch.cuda.synchronize()
        warmups.append(time.perf_counter()-started)
        assert output.tokens.shape==(1,256)
    atomic_json(args.output/"warmup.json",warmups)
    warm_reference=output.tokens.cpu().tolist()[0]
    for sample in samples[:args.limit]:
        index=sample["sample_index"]
        inputs=inputs_for(sample)
        runs=[]
        for repeat in range(args.repeats):
            clock=Clock()
            function=clone_driver(focus_optimized_generate,clock)
            torch.manual_seed(42)
            torch.cuda.synchronize()
            started=time.perf_counter()
            output=function(model,inputs=inputs,**call_options)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-started
            assert output.tokens.shape==(1,256)
            tokens=output.tokens.cpu().tolist()[0]
            if index == 0:
                assert tokens == warm_reference, "counting instrumentation changed plain output"
            runs.append(dict(repeat=repeat,seconds=seconds,step_count=sum(clock.counts.values()),
                             steps_per_block=clock.counts,output_ids=tokens,trace=output.trace,
                             generated_text=tok.decode(tokens,skip_special_tokens=False),
                             readable_text=tok.decode(tokens,skip_special_tokens=True)))
            atomic_json(args.output/f"sample_{index:03d}.partial.json",
                        dict(sample_index=index,source_id=sample["source_id"],runs=runs))
        clock=Clock(timing=True)
        function=clone_driver(focus_optimized_generate,clock)
        torch.manual_seed(42)
        measured=function(model,inputs=inputs,**call_options)
        torch.cuda.synchronize()
        records=clock.rows()
        assert measured.tokens.cpu().tolist()[0]==runs[0]["output_ids"]
        assert measured.trace==runs[0]["trace"]
        assert clock.counts==runs[0]["steps_per_block"]
        assert len(records)==runs[0]["step_count"]
        stats=step_summary(records)
        assert stats["noninitial_step_count"]==len(records)-len(clock.counts)
        result=dict(sample_index=index,source_id=sample["source_id"],model=args.model,
                    method="focus_optimized",requested_context=args.length,
                    prompt_tokens=inputs.shape[1],generation_length=256,
                    prompt_sha256=prompt_hash(prompt_ids(sample,args.length)),
                    generation_options=options,runs=runs,
                    e2e_median_seconds=statistics.median(r["seconds"] for r in runs),
                    step_counts=[r["step_count"] for r in runs],
                    output_repeat_equal=all(r["output_ids"]==runs[0]["output_ids"] for r in runs),
                    instrumentation_matches=True,step_records=records,**stats)
        atomic_json(args.output/f"sample_{index:03d}.json",result)
        print("SAMPLE_DONE",args.model,args.length,index,result["e2e_median_seconds"],
              stats["noninitial_step_count"],stats["noninitial_step_mean_ms"],flush=True)
    atomic_json(args.output/"COMPLETE.json",dict(completed=True,samples=args.limit,
                                               smoke=args.smoke,source_hashes=hashes))


if __name__=="__main__":
    main()
