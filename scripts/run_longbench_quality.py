#!/usr/bin/env python3
"""Run integrated-runtime quality checks on LongBench QA JSONL records.

Supports resuming from a previous partial run: pass --resume_from with the
saved JSONL progress file and already-completed (task, index) pairs are
skipped.  The report is rewritten after every record so a killed run keeps
all completed rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runtime import load_runtime


DEFAULT_DATA = Path("/data0/gs/losa_accuracy_prep/data/longbench_budget128_smoke")
TASKS = ("hotpotqa", "triviaqa", "narrativeqa", "qasper", "multifieldqa_en")
GEN_LENGTHS = {"hotpotqa": 32, "triviaqa": 32, "narrativeqa": 128,
               "qasper": 128, "multifieldqa_en": 64, "gov_report": 512}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("llada", "sdar"), required=True)
    p.add_argument("--mode", choices=("dense", "losa", "focus"), default="dense")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--max_context_tokens", type=int, default=32768)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--losa_token_budget", type=int, default=16)
    p.add_argument("--losa_page_size", type=int, default=16)
    p.add_argument("--losa_active_topk", type=int, default=5)
    p.add_argument("--losa_gqa_mode", choices=(
        "per_query_head", "group_mean", "group_max_score"), default="per_query_head")
    p.add_argument("--losa_backend", choices=("torch", "triton", "auto"), default="auto")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--focus_alpha", type=float, default=1.5)
    p.add_argument("--moe_expert_patch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--threshold", type=float, default=None,
                   help="denoise threshold (default: llada 0.7 Q-Mode, sdar 0.95)")
    p.add_argument("--editing_threshold", type=float, default=None,
                   help="LLaDA editing threshold (default: 0.5 Q-Mode)")
    p.add_argument("--remasking_strategy", type=str, default=None,
                   help="SDAR remasking strategy (default: low_confidence_static)")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    p.add_argument("--resume_from", type=Path, default=None,
                   help="JSONL file with completed rows; matching (task, index) pairs are skipped")
    return p.parse_args()


def tokenize(tokenizer, prompt: str) -> torch.Tensor:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    except Exception:
        return tokenizer(prompt, return_tensors="pt").input_ids


def truncate_middle(input_ids: torch.Tensor, budget: int) -> tuple[torch.Tensor, bool]:
    """Keep both ends of an overlong prompt and remove its middle."""
    length = int(input_ids.shape[-1])
    if length <= budget:
        return input_ids, False
    head = budget // 2
    tail = budget - head
    return torch.cat((input_ids[:, :head], input_ids[:, -tail:]), dim=-1), True


def load_completed(resume_from: Path | None) -> dict[tuple[str, int], dict]:
    """Recover completed rows from a previous run's progress log."""
    if resume_from is None:
        return {}
    completed = {}
    with resume_from.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "task" in row and "index" in row:
                completed[(str(row["task"]), int(row["index"]))] = row
    return completed


def main() -> int:
    # The cluster shells often default to an ASCII locale; generated
    # predictions can contain arbitrary Unicode, so force UTF-8 output.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    a = args()
    if not a.tasks:
        raise ValueError("at least one task is required")
    unknown = sorted(set(a.tasks) - set(GEN_LENGTHS))
    if unknown:
        raise ValueError(f"unsupported tasks: {unknown}")
    if a.mode == "losa":
        if a.losa_page_size <= 0:
            raise ValueError("losa_page_size must be positive")
        if a.losa_active_topk <= 0:
            raise ValueError("losa_active_topk must be positive")
        if a.losa_token_budget <= 0:
            raise ValueError("losa_token_budget must be positive")
        if a.model_path is None and a.family == "llada":
            raise ValueError("LLaDA LongBench runs require --model_path to select 2.0 or 2.1")
    runtime_kwargs = {
        "family": a.family,
        "dtype": a.dtype,
    }
    if a.model_path is not None:
        runtime_kwargs["model_path"] = a.model_path
    if a.mode in ("dense", "losa"):
        runtime_kwargs["moe_expert_patch"] = a.moe_expert_patch
    if a.mode == "losa":
        runtime_kwargs.update(
            losa_token_budget=a.losa_token_budget,
            losa_page_size=a.losa_page_size,
            losa_active_topk=a.losa_active_topk,
            losa_gqa_mode=a.losa_gqa_mode,
            losa_backend=a.losa_backend,
        )
    elif a.mode == "focus":
        runtime_kwargs["alpha"] = a.focus_alpha
    runtime = load_runtime(a.mode, **runtime_kwargs)
    model, tokenizer = runtime.load()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed(a.resume_from)
    rows = list(completed.values())
    skipped = 0
    for task in a.tasks:
        path = a.data_dir / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()][:a.limit]
        for index, record in enumerate(records):
            if (task, index) in completed:
                skipped += 1
                continue
            gen_length = GEN_LENGTHS[task]
            ids = tokenize(tokenizer, str(record["prompt"]))
            original = int(ids.shape[-1])
            # Leave room for generation, preserving both prompt boundaries.
            budget = min(a.max_context_tokens - gen_length, int(tokenizer.model_max_length))
            budget = (budget // a.block_length) * a.block_length
            if budget <= 0:
                raise ValueError("max_context_tokens leaves no block-aligned prompt room")
            ids, truncated = truncate_middle(ids, budget)
            ids = ids.to(model.device)
            started = time.perf_counter()
            threshold = a.threshold if a.threshold is not None else (
                0.95 if a.family == "sdar" else 0.7)
            with torch.inference_mode():
                result = runtime.generate(
                    ids, gen_length=gen_length, block_length=a.block_length,
                    steps=a.steps, temperature=0.0, threshold=threshold,
                    mask_id=(tokenizer.mask_token_id or 151669) if a.family == "sdar" else 156895,
                    eos_id=None if a.family == "sdar" else 156892,
                    eos_early_stop=True,
                    **({"remasking_strategy": a.remasking_strategy or "low_confidence_static"}
                       if a.family == "sdar" else
                       {"editing_threshold": a.editing_threshold
                        if a.editing_threshold is not None else 0.5,
                        "num_to_transfer": 1}),
                )
            decoded = tokenizer.decode(result.tokens[0], skip_special_tokens=True)
            row = {
                "task": task, "index": index, "original_input_tokens": original,
                "input_tokens": int(ids.shape[-1]), "truncated": truncated,
                "generated_tokens": int(result.tokens.shape[-1]),
                "elapsed_seconds": time.perf_counter() - started,
                "prediction": decoded, "trace_events": len(result.trace),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            # Incremental report: a killed run keeps every completed row.
            report = {"family": a.family, "mode": a.mode,
                      "max_context_tokens": a.max_context_tokens,
                      "truncation": "drop_middle_keep_head_tail",
                      "model_path": a.model_path,
                      "block_length": a.block_length,
                      "steps": a.steps,
                      "moe_expert_patch": a.moe_expert_patch,
                      "moe_patched_blocks": getattr(
                          getattr(runtime, "moe_patch_report", None), "patched_blocks", None),
                      "losa_token_budget": a.losa_token_budget if a.mode == "losa" else None,
                      "losa_page_size": a.losa_page_size if a.mode == "losa" else None,
                      "losa_active_topk": a.losa_active_topk if a.mode == "losa" else None,
                      "losa_gqa_mode": a.losa_gqa_mode if a.mode == "losa" else None,
                      "losa_backend": a.losa_backend if a.mode == "losa" else None,
                      "threshold": threshold,
                      "data_dir": str(a.data_dir), "resumed": len(completed) > 0,
                      "rows": rows}
            (a.output_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"family": a.family, "mode": a.mode, "max_context_tokens": a.max_context_tokens,
              "truncation": "drop_middle_keep_head_tail",
              "model_path": a.model_path,
              "block_length": a.block_length,
              "steps": a.steps,
              "moe_expert_patch": a.moe_expert_patch,
              "moe_patched_blocks": getattr(
                  getattr(runtime, "moe_patch_report", None), "patched_blocks", None),
              "losa_token_budget": a.losa_token_budget if a.mode == "losa" else None,
              "losa_page_size": a.losa_page_size if a.mode == "losa" else None,
              "losa_active_topk": a.losa_active_topk if a.mode == "losa" else None,
              "losa_gqa_mode": a.losa_gqa_mode if a.mode == "losa" else None,
              "losa_backend": a.losa_backend if a.mode == "losa" else None,
              "threshold": threshold, "editing_threshold": (
                  a.editing_threshold if a.editing_threshold is not None else 0.5)
              if a.family == "llada" else None,
              "remasking_strategy": a.remasking_strategy or "low_confidence_static"
              if a.family == "sdar" else None,
              "data_dir": str(a.data_dir), "resumed": len(completed) > 0,
              "skipped": skipped, "rows": rows}
    (a.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"done": len(rows), "skipped": skipped}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
