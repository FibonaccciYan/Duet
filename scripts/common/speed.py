"""Configurable full-speed protocol using the preserved timing drivers."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch
from scripts.original.performance import narrative80_llada_benchmark as base
from scripts.original.performance import narrative80_losa_benchmark as losa
from scripts.original.performance import narrative80_focus_benchmark as focus

ROOT = Path(__file__).resolve().parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    temp.replace(path)


def cast(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def run(c):
    out = Path(c["output"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("nonempty output directory")
    out.mkdir(parents=True, exist_ok=True)
    source = Path(c["data_dir"]) / c["model"] / f'{c["length"]}.jsonl'
    stored = [json.loads(l) for l in source.read_text().splitlines()]
    assert len(stored) == 80 and len({s["source_id"] for s in stored}) == 80
    manifest = json.loads((Path(c["data_dir"]) / "manifest.json").read_text())
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        snapshot = Path(c["data_dir"]) / "tokenizers" / c["model"] / name
        assert sha(snapshot) == sha(Path(c["model_path"]) / name), "tokenizer mismatch"
    assert sha(source) == manifest["files"][f'{c["model"]}/{c["length"]}.jsonl']["sha256"]
    family = "llada" if c["model"] == "llada21" else "sdar"
    method = c["method"]
    overrides = {k: cast(v) for k, v in c["overrides"].items()}
    allowed = {
        "dense": {"exact_rope", "moe_expert_patch"},
        "sparse": {"ratio", "top_k", "selection_interval", "selection_layer",
                   "query_sparse", "prefix_sparse", "prefix_token_budget", "prefix_dense_before_query_selection", "prefix_rescreen_full_kv", "prefix_strict_budget", "prefix_selector",
                   "exact_rope", "moe_expert_patch", "query_dense_threshold"},
        "losa": {"losa_gqa_mode", "losa_token_budget", "losa_page_size", "losa_active_topk",
                 "losa_backend", "losa_fused_state", "shared_kv", "graph_replay",
                 "moe_expert_patch", "exact_rope"},
        "focus": {"alpha", "attention_backend", "moe_expert_patch", "exact_rope"},
    }
    if set(overrides) - allowed[method]:
        raise ValueError("unsupported method overrides: " + str(set(overrides) - allowed[method]))
    options = {k: c[k] for k in ("gen_length", "block_length", "steps",
                                 "temperature", "threshold", "eos_early_stop")}
    if family == "llada":
        options.update(editing_threshold=c["editing_threshold"], num_to_transfer=1)
    else:
        options.update(remasking_strategy=c["remasking_strategy"], mask_id=151669)
    if method in ("dense", "sparse"):
        model, tok = base.load_model_and_tokenizer(family, model_path=c["model_path"])
        patch = dict(moe_expert_patch=True)
        if method == "sparse":
            patch.update(query_sparse=True, prefix_sparse=True, prefix_token_budget=256)
        patch.update(overrides)
        base.patch_method(model, method + "_optimized", model_name=family, **patch)
        Clock = base.Clock
        clone = base.clone_sdar if family == "sdar" else base.clone_driver
        def call(clock, ids):
            return clone(model.generate, clock)(model, inputs=ids, **options)
        runtime_config = getattr(model.config, family + "_sparse_config")
    elif method == "losa":
        from src.optimized.losa.api import LoSAOptimizedRuntime
        rt = LoSAOptimizedRuntime(family=family, model_path=c["model_path"], **overrides)
        model, tok = rt.load()
        model._losa_optimized_graph_replay = rt.graph_replay
        model._losa_optimized_shared_kv = rt.shared_kv
        model._losa_optimized_verify_graph = rt.verify_graph
        runtime_config = {k: getattr(rt, k) for k in allowed[method]}
        extra = {k: getattr(rt, k) for k in ("losa_gqa_mode", "losa_token_budget", "losa_page_size",
                                             "losa_active_topk", "losa_backend", "losa_fused_state")}
        Clock = losa.Clock
        def call(clock, ids):
            return losa.clone_losa(losa.block_diffusion_generate, clock)(
                model, inputs=ids, family=family, use_losa=True, **extra, **options)
    else:
        from src.optimized.focus.api import FocusOptimizedRuntime
        from src.optimized.focus.generation import focus_optimized_generate
        rt = FocusOptimizedRuntime(family=family, model_path=c["model_path"], **overrides)
        model, tok = rt.load()
        runtime_config = {k: getattr(rt, k) for k in allowed[method]}
        Clock = focus.Clock
        def call(clock, ids):
            return focus.clone_driver(focus_optimized_generate, clock)(
                model, inputs=ids, family=family, alpha=rt.alpha,
                attention_backend=rt.attention_backend, **options)
    prompt_length = min(c["length"], model.config.max_position_embeddings - c["gen_length"])
    if prompt_length <= 0 or prompt_length % c["block_length"]:
        raise ValueError("prompt must leave a positive block-aligned generation window")
    def inputs(sample):
        ids = sample["input_ids"]
        assert len(ids) == c["length"]
        if len(ids) > prompt_length:
            ids = ids[:prompt_length//2] + ids[-(prompt_length-prompt_length//2):]
        return torch.tensor([ids], dtype=torch.long, device=model.device)
    def tokens(output):
        return output if torch.is_tensor(output) else output.tokens
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src").rglob("*.py")}
    atomic(out / "manifest.json", dict(config=c, generation=options, runtime=runtime_config,
           implementation=method+"_optimized", actual_prompt_tokens=prompt_length,
           source_sha256=sha(source), source_hashes=hashes,
           steady_definition="all complete steps except first per block; total_time/total_count",
           e2e_boundary="generate through cuda synchronize; exclude load/tokenize/save"))
    first = inputs(stored[0])
    warmups = []
    for i in range(c["warmups"]):
        clock = Clock()
        torch.manual_seed(c["seed"])
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = call(clock, first)
        torch.cuda.synchronize()
        warmups.append(time.perf_counter() - start)
        assert tokens(output).shape == (1, c["gen_length"])
        print("WARM", i, warmups[-1], flush=True)
    atomic(out / "warmup.json", warmups)
    results = []
    for sample in stored[:c["samples"]]:
        ids = inputs(sample)
        runs = []
        for repeat in range(c["repeats"]):
            clock = Clock()
            torch.manual_seed(c["seed"])
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = call(clock, ids)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            value = tokens(output)
            assert value.shape == (1, c["gen_length"])
            ids_out = value.cpu().tolist()[0]
            runs.append(dict(repeat=repeat, seconds=seconds, step_count=sum(clock.counts.values()),
                             steps_per_block=clock.counts, output_ids=ids_out,
                             generated_text=tok.decode(ids_out, skip_special_tokens=False),
                             readable_text=tok.decode(ids_out, skip_special_tokens=True)))
            atomic(out / f'sample_{sample["sample_index"]:03d}.partial.json',
                   dict(sample_index=sample["sample_index"], source_id=sample["source_id"], runs=runs))
        clock = Clock(timing=True)
        torch.manual_seed(c["seed"])
        measured = call(clock, ids)
        torch.cuda.synchronize()
        assert tokens(measured).cpu().tolist()[0] == runs[0]["output_ids"]
        assert clock.counts == runs[0]["steps_per_block"]
        assert all(r["output_ids"] == runs[0]["output_ids"] for r in runs)
        records = clock.rows()
        stable = [r["cuda_ms"] for r in records if r["loop_iteration"] >= 1]
        assert len(stable) == runs[0]["step_count"]-len(clock.counts)
        result = dict(sample_index=sample["sample_index"], source_id=sample["source_id"],
                      model=c["model"], method=method+"_optimized", requested_context=c["length"],
                      prompt_tokens=prompt_length, generation_length=c["gen_length"],
                      prompt_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
                      generation_options=options, runs=runs,
                      e2e_median_seconds=statistics.median(r["seconds"] for r in runs),
                      step_counts=[r["step_count"] for r in runs],
                      output_repeat_equal=True, instrumentation_matches=True, step_records=records,
                      noninitial_step_count=len(stable), noninitial_step_total_ms=sum(stable),
                      noninitial_step_mean_ms=statistics.mean(stable) if stable else None)
        atomic(out / f'sample_{sample["sample_index"]:03d}.json', result)
        results.append(result)
        print("SAMPLE_DONE", sample["sample_index"], result["e2e_median_seconds"], flush=True)
    count = sum(r["noninitial_step_count"] for r in results)
    total = sum(r["noninitial_step_total_ms"] for r in results)
    seconds = sum(r["e2e_median_seconds"] for r in results)
    summary = dict(samples=len(results), e2e_mean_seconds=seconds/len(results),
                   throughput_tokens_per_second=c["gen_length"]*len(results)/seconds,
                   noninitial_step_count=count, noninitial_step_total_ms=total,
                   noninitial_step_mean_ms=total/count if count else None)
    atomic(out / "summary.json", summary)
    (out / "REPORT.md").write_text("# Speed cell\n\n```json\n"+json.dumps(summary, indent=2)+"\n```\n")
    atomic(out / "COMPLETE.json", dict(completed=True, samples=len(results), smoke=c["stage"]=="smoke"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    run(json.loads(p.parse_args().config.read_text()))
