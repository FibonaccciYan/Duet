#!/usr/bin/env python3
"""End-to-end long-context benchmark for sparse generation.

Context length means the complete prompt + generated-token window, so the 32K
case stays within checkpoints whose configured maximum is 32768 tokens.
"""

import argparse
import collections
import hashlib
import importlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse import patch_model, resolve_model_family


NARRATIVEQA_PROMPT = (
    "You are given a story, which can be either a novel or a movie script, and a question. "
    "Answer the question asconcisely as you can, using a single phrase if possible. Do not "
    "provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the "
    "story asconcisely as you can, using a single phrase if possible. Do not provide any "
    "explanation.\n\nQuestion: {input}\n\nAnswer:"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llada", "sdar"), required=True)
    parser.add_argument(
        "--model-path",
        default=None,
        help="checkpoint path; defaults to LLADA_MODEL_PATH or SDAR_MODEL_PATH",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "dense",
            "maskless_dense",
            "query",
            "prefix",
            "query_prefix",
            "losa",
            "combined",
            "eval",
        ),
        required=True,
        help="combined enables Query Sparse, Prefix Sparse, and LoSA",
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=(8192, 16384, 32768))
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        help="benchmark exact input lengths instead of prompt+generation windows",
    )
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--losa-active-topk", type=int, default=5)
    parser.add_argument(
        "--losa-score-mode",
        choices=("query", "key_diag"),
        default="query",
    )
    parser.add_argument("--losa-key-samples", type=int, default=32)
    parser.add_argument("--prefix-token-budget", type=int, default=256)
    parser.add_argument("--prefix-rescreen-full-kv", action="store_true")
    parser.add_argument(
        "--prefix-selector",
        choices=("raw_l1", "hadamard_qk", "adamas", "qk"),
        default="raw_l1",
    )
    parser.add_argument("--prefix-min-prefix-length", type=int, default=None)
    parser.add_argument("--prefix-chunk-size", type=int, default=None)
    parser.add_argument(
        "--prefix-share-layer-pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--query-ratio", type=float, default=None)
    parser.add_argument("--query-selection-interval", type=int, default=None)
    parser.add_argument("--query-selection-layer", type=int, default=None)
    parser.add_argument("--query-dense-threshold", type=int, default=None)
    parser.add_argument("--query-min-prefix-length", type=int, default=None)
    parser.add_argument(
        "--deep-only-transfer", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--remasking-strategy",
        choices=(
            "sequential",
            "low_confidence_static",
            "low_confidence_dynamic",
            "entropy_bounded",
        ),
        default="sequential",
    )
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--narrativeqa-data-file", type=Path)
    parser.add_argument("--narrativeqa-index", type=int, default=0)
    parser.add_argument("--narrativeqa-indices", type=int, nargs="+")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="compare the selected sparse mode with dense in one loaded model",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="compare dense, Query-only, Prefix-only, and Query+Prefix",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--phase-profile", action="store_true")
    parser.add_argument("--llada-full-mask", action="store_true")
    return parser.parse_args()


class PhaseProfiler:
    """Low-overhead CUDA-event timings for the existing sparse phases."""

    def __init__(self, model_name):
        self.events = []
        module = importlib.import_module(f"src.sparse.{model_name}_patch")
        names = (
            ("_dense_block_prefill", "dense_prefill"),
            ("_cached_forward", "cached_forward"),
            ("_compact_prefix_cache", "prefix_compaction"),
        ) if model_name == "llada" else (
            ("_sparse_cached_forward", "cached_forward"),
            ("_compact_prefix_cache", "prefix_compaction"),
        )
        for function_name, phase in names:
            original = getattr(module, function_name)

            def timed(*args, _original=original, _phase=phase, **kwargs):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                started = time.perf_counter()
                start.record()
                result = _original(*args, **kwargs)
                end.record()
                metadata = {}
                if _phase != "prefix_compaction":
                    metadata["input_rows"] = int(args[1].shape[1])
                elif result[0]:
                    metadata["selected_rows"] = int(result[0][0][0].shape[-2])
                if _phase == "cached_forward":
                    selected = (
                        result[1]
                        if model_name == "llada"
                        else args[4].get("positions")
                    )
                    metadata["selected_rows"] = (
                        int(selected.numel()) if selected is not None else metadata["input_rows"]
                    )
                self.events.append(
                    (_phase, start, end, (time.perf_counter() - started) * 1000, metadata)
                )
                return result

            setattr(module, function_name, timed)

    def reset(self):
        self.events.clear()

    def summary(self):
        result = {}
        for phase in {event[0] for event in self.events}:
            events = [event for event in self.events if event[0] == phase]
            item = {
                "calls": len(events),
                "cpu_ms": sum(event[3] for event in events),
                "gpu_ms": sum(event[1].elapsed_time(event[2]) for event in events),
            }
            for key in ("input_rows", "selected_rows"):
                counts = collections.Counter(event[4].get(key) for event in events)
                counts.pop(None, None)
                if counts:
                    item[f"{key}_histogram"] = dict(sorted(counts.items()))
            result[phase] = item
        return result


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(args):
    is_sdar = args.model == "sdar"
    dtype = torch.float16 if is_sdar else torch.bfloat16
    model_path = args.model_path or os.environ.get(
        "SDAR_MODEL_PATH" if is_sdar else "LLADA_MODEL_PATH"
    )
    if not model_path:
        raise ValueError(
            "--model-path is required unless the matching "
            "LLADA_MODEL_PATH or SDAR_MODEL_PATH environment variable is set"
        )
    args.model_path = model_path
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=dtype,
    ).eval()
    resolve_model_family(model, args.model)

    combined = args.mode == "combined"
    eval_mode = args.mode == "eval"
    query_sparse = args.mode in {"query", "query_prefix", "combined"} or eval_mode
    prefix_sparse = args.mode in {"prefix", "query_prefix", "combined"} or (
        eval_mode and not is_sdar
    )
    patch_model(
        model,
        model_name=args.model,
        ratio=(
            args.query_ratio
            if args.query_ratio is not None
            else (0.5 if is_sdar else 0.7)
        ),
        top_k=64,
        selection_interval=(
            args.query_selection_interval
            if args.query_selection_interval is not None
            else (1 if is_sdar else 4)
        ),
        query_dense_threshold=(
            args.query_dense_threshold
            if args.query_dense_threshold is not None
            else 4
        ),
        query_min_prefix_length=args.query_min_prefix_length,
        refresh_step=-1 if is_sdar else 2,
        selection_layer=(
            args.query_selection_layer
            if args.query_selection_layer is not None
            else (5 if is_sdar else 1)
        ),
        deep_only_transfer=args.deep_only_transfer,
        query_sparse=query_sparse,
        prefix_sparse=prefix_sparse,
        prefix_min_prefix_length=args.prefix_min_prefix_length,
        prefix_token_budget=args.prefix_token_budget,
        prefix_chunk_size=(args.prefix_chunk_size or 1024),
        prefix_share_layer_pairs=args.prefix_share_layer_pairs,
        prefix_rescreen_full_kv=args.prefix_rescreen_full_kv,
        losa=args.mode in {"losa", "combined"},
        losa_active_topk=args.losa_active_topk,
        losa_score_mode=args.losa_score_mode,
        losa_key_samples=args.losa_key_samples,
        moe_expert_patch=getattr(args, "moe_expert_patch", not is_sdar),
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def exact_prompt(tokenizer, length, device, source_ids=None):
    if source_ids is not None:
        if source_ids.shape[-1] < length:
            raise ValueError(
                f"NarrativeQA prompt has {source_ids.shape[-1]} tokens, needs {length}"
            )
        head = length // 2
        return torch.cat(
            (source_ids[:, :head], source_ids[:, -(length - head) :]), dim=-1
        )
    tail = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Summarize the preceding archive."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    tail = list(tail)[-length:]
    filler = tokenizer.encode(
        "The archive records a sequence of historical events and their consequences. ",
        add_special_tokens=False,
    )
    if not filler:
        raise RuntimeError("tokenizer produced an empty filler sequence")
    prefix_length = length - len(tail)
    prefix = (filler * math.ceil(prefix_length / len(filler)))[:prefix_length]
    return torch.tensor(prefix + tail, dtype=torch.long, device=device).unsqueeze(0)


def load_narrativeqa_prompt(tokenizer, path, index, device):
    if index < 0:
        raise ValueError("--narrativeqa-index must be non-negative")
    with path.open(encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                record = json.loads(line)
                break
        else:
            raise ValueError(f"NarrativeQA index {index} not found in {path}")
    prompt = NARRATIVEQA_PROMPT.format(**record)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    return input_ids.to(device), record.get("_id", index)


def checksum(tensor):
    data = tensor.detach().to(device="cpu", dtype=torch.int64).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def repository_state():
    return {
        "commit": subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ("git", "status", "--porcelain"), cwd=REPO_ROOT, text=True
            ).strip()
        ),
    }


def generation_kwargs(args, tokenizer, input_ids):
    is_sdar = args.model == "sdar"
    kwargs = {
        "inputs": input_ids,
        # Fixed work is required for comparable end-to-end latency. Early EOS
        # makes different sparse selectors execute different numbers of blocks.
        "eos_early_stop": False,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps or args.block_length,
        "threshold": (
            getattr(args, "threshold", None)
            if getattr(args, "threshold", None) is not None
            else (
                0.95
                if is_sdar and args.remasking_strategy == "low_confidence_dynamic"
                else 0.85
                if is_sdar
                else 0.7
            )
        ),
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "mask_id": tokenizer.mask_token_id if is_sdar else 156895,
        "eos_id": None if is_sdar else 156892,
    }
    if is_sdar:
        kwargs.update(remasking_strategy=args.remasking_strategy, eb_threshold=0.35)
    else:
        runtime_mode = getattr(args, "runtime_mode", getattr(args, "mode", "dense"))
        kwargs.update(
            editing_threshold=0.5,
            num_to_transfer=1,
            maskless_attention=(
                runtime_mode != "dense"
                if args.ablation
                else not args.llada_full_mask
            ),
        )
    return kwargs


def run_once(args, model, tokenizer, context_length):
    prompt_length = (
        context_length
        if args.prompt_lengths is not None
        else context_length - args.gen_length
    )
    if prompt_length <= 0:
        raise ValueError("every context must be greater than --gen-length")
    input_ids = exact_prompt(
        tokenizer, prompt_length, model.device, getattr(args, "prompt_ids", None)
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    profiler = getattr(args, "phase_profiler", None)
    if profiler is not None:
        profiler.reset()
    started = time.perf_counter()
    with torch.cuda.nvtx.range(
        f"generate:{args.model}:{args.mode}:{context_length}"
    ):
        sequences = model.generate(**generation_kwargs(args, tokenizer, input_ids))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if sequences.shape[-1] > args.gen_length:
        raise RuntimeError(
            f"expected at most {args.gen_length} generated tokens, got {sequences.shape[-1]}"
        )
    result = {
        "benchmark_tokens": context_length,
        "context_tokens": prompt_length + args.gen_length,
        "prompt_tokens": prompt_length,
        "requested_output_tokens": args.gen_length,
        "generated_tokens": int(sequences.shape[-1]),
        "seconds": elapsed,
        "requested_tokens_per_second": args.gen_length / elapsed,
        "generated_tokens_per_second": int(sequences.shape[-1]) / elapsed,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "input_checksum": checksum(input_ids),
        "output_checksum": checksum(sequences),
    }
    if profiler is not None:
        result["phase_profile"] = profiler.summary()
    return result


def set_runtime_mode(model, model_name, mode):
    prefix = model_name
    setattr(
        model.config,
        f"{prefix}_query_sparse",
        mode in {"query", "query_prefix", "combined", "eval"},
    )
    setattr(
        model.config,
        f"{prefix}_prefix_sparse",
        mode in {"prefix", "query_prefix", "combined"}
        or (mode == "eval" and model_name == "llada"),
    )
    setattr(model.config, f"{prefix}_losa", mode in {"losa", "combined"})


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.ablation and args.mode != "query_prefix":
        raise ValueError("--ablation requires --mode query_prefix")
    set_seed(args.seed)
    if args.prefix_selector != "raw_l1":
        import src.sparse.sparse_ops as sparse

        if args.prefix_selector == "adamas":
            sparse._prefix_indices = sparse._adamas_prefix_indices
        elif args.prefix_selector == "qk":
            sparse._prefix_indices = lambda query, key, budget, *_args, **_kwargs: (
                sparse._qk_prefix_indices(query, key, budget)
            )
        else:
            sparse._prefix_indices = lambda query, key, budget, *_args, **_kwargs: (
                sparse._hadamard_qk_prefix_indices(query, key, budget)
            )
    model, tokenizer = load(args)
    prompts = [(None, None, None)]
    if args.narrativeqa_data_file is not None:
        indices = args.narrativeqa_indices or [args.narrativeqa_index]
        prompts = [
            (
                *load_narrativeqa_prompt(
                    tokenizer, args.narrativeqa_data_file, index, model.device
                ),
                index,
            )
            for index in indices
        ]
    args.prompt_ids, args.prompt_id, _ = prompts[0]
    args.phase_profiler = PhaseProfiler(args.model) if args.phase_profile else None
    compared_modes = (
        ("dense", "maskless_dense", "query", "prefix", "query_prefix")
        if args.ablation and args.model == "llada"
        else ("dense", "query", "prefix", "query_prefix")
        if args.ablation
        else ("dense", args.mode)
        if args.paired
        else (args.mode,)
    )
    lengths = args.prompt_lengths or args.contexts

    # Compile fixed-shape kernels and initialize caches outside the measurement.
    warmup_args = argparse.Namespace(**vars(args))
    warmup_args.gen_length = args.gen_length if args.paired or args.ablation else 32
    if args.paired or args.ablation:
        for context_length in lengths:
            warmup_context = context_length - args.gen_length + warmup_args.gen_length
            for mode in compared_modes:
                set_runtime_mode(model, args.model, mode)
                warmup_args.runtime_mode = mode
                run_once(warmup_args, model, tokenizer, warmup_context)
    else:
        warmup_ids = exact_prompt(tokenizer, 480, model.device, args.prompt_ids)
        model.generate(**generation_kwargs(warmup_args, tokenizer, warmup_ids))
    torch.cuda.synchronize()

    results = []
    print(
        json.dumps(
            {
                "model": args.model,
                "mode": args.mode,
                "losa_score_mode": args.losa_score_mode,
                "prefix_selector": args.prefix_selector,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for repeat in range(args.repeats):
        for sample_position, (prompt_ids, prompt_id, prompt_index) in enumerate(prompts):
            args.prompt_ids, args.prompt_id = prompt_ids, prompt_id
            order = repeat * len(prompts) + sample_position
            contexts = lengths if order % 2 == 0 else reversed(lengths)
            shift = order % len(compared_modes)
            modes = compared_modes[shift:] + compared_modes[:shift]
            for context_length in contexts:
                for mode in modes:
                    set_runtime_mode(model, args.model, mode)
                    args.runtime_mode = mode
                    result = run_once(args, model, tokenizer, context_length)
                    result.update(
                        repeat=repeat,
                        mode=mode,
                        narrativeqa_index=prompt_index,
                        narrativeqa_id=prompt_id,
                    )
                    results.append(result)
                    print(json.dumps(result, sort_keys=True), flush=True)

    if args.paired or args.ablation:
        for context_length in lengths:
            dense = statistics.median(
                row["seconds"] for row in results
                if row["benchmark_tokens"] == context_length and row["mode"] == "dense"
            )
            for mode in compared_modes[1:]:
                sparse = statistics.median(
                    row["seconds"] for row in results
                    if row["benchmark_tokens"] == context_length
                    and row["mode"] == mode
                )
                print(
                    f"SUMMARY {context_length} {mode} {dense} {sparse} "
                    f"{dense / sparse}",
                    flush=True,
                )

    for context_length in lengths:
        for _, _, prompt_index in prompts:
            context_rows = [
                row for row in results
                if row["benchmark_tokens"] == context_length
                and row["narrativeqa_index"] == prompt_index
            ]
            if len({row["input_checksum"] for row in context_rows}) != 1:
                raise RuntimeError(
                    f"input changed across runs at context {context_length}, "
                    f"sample {prompt_index}"
                )
            for mode in compared_modes:
                mode_rows = [row for row in context_rows if row["mode"] == mode]
                if len({row["output_checksum"] for row in mode_rows}) != 1:
                    raise RuntimeError(
                        f"output changed across repeats for {mode} at context "
                        f"{context_length}, sample {prompt_index}"
                    )

    report = {
        **repository_state(),
        "model": args.model,
        "model_path": args.model_path,
        "gpu": torch.cuda.get_device_name(model.device),
        "gpu_uuid": str(torch.cuda.get_device_properties(model.device).uuid),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mode": args.mode,
        "dtype": "float16" if args.model == "sdar" else "bfloat16",
        "gen_length": args.gen_length,
        "length_axis": "prompt" if args.prompt_lengths is not None else "context",
        "block_length": args.block_length,
        "steps": args.steps or args.block_length,
        "losa_active_topk": args.losa_active_topk,
        "losa_score_mode": args.losa_score_mode,
        "losa_key_samples": args.losa_key_samples,
        "prefix_token_budget": args.prefix_token_budget,
        "prefix_rescreen_full_kv": args.prefix_rescreen_full_kv,
        "prefix_selector": args.prefix_selector,
        "prompt_source": "narrativeqa" if prompts[0][0] is not None else "synthetic",
        "narrativeqa_data_file": (
            str(args.narrativeqa_data_file) if args.narrativeqa_data_file else None
        ),
        "narrativeqa_indices": [item[2] for item in prompts if item[0] is not None],
        "narrativeqa_ids": [item[1] for item in prompts if item[0] is not None],
        "original_prompt_tokens": [
            int(item[0].shape[-1]) for item in prompts if item[0] is not None
        ],
        "paired": args.paired,
        "ablation": args.ablation,
        "llada_full_mask": args.llada_full_mask,
        "sparse_config": getattr(model.config, f"{args.model}_sparse_config"),
        "remasking_strategy": args.remasking_strategy,
        "threshold": (
            args.threshold
            if args.threshold is not None
            else (
                0.95
                if args.model == "sdar"
                and args.remasking_strategy == "low_confidence_dynamic"
                else 0.85
                if args.model == "sdar"
                else 0.7
            )
        ),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
