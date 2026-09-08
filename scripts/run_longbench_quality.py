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
               "qasper": 128, "multifieldqa_en": 64}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("llada", "sdar"), required=True)
    p.add_argument("--mode", choices=("dense", "losa", "focus"), default="dense")
    p.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--limit", type=int, default=1)
    p.add_argument("--max_context_tokens", type=int, default=32768)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--losa_token_budget", type=int, default=16)
    p.add_argument("--focus_alpha", type=float, default=1.5)
    p.add_argument("--moe_expert_patch", action=argparse.BooleanOptionalAction, default=True)
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
    runtime = load_runtime(
        a.mode, family=a.family, dtype=a.dtype,
        **({"moe_expert_patch": a.moe_expert_patch} if a.mode == "dense" else {}),
        **({"losa_token_budget": a.losa_token_budget} if a.mode == "losa" else {}),
        **({"alpha": a.focus_alpha} if a.mode == "focus" else {}),
    )
    model, tokenizer = runtime.load()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    completed = load_completed(a.resume_from)
    rows = list(completed.values())
    skipped = 0
    for task in TASKS:
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
            with torch.inference_mode():
                result = runtime.generate(
                    ids, gen_length=gen_length, block_length=a.block_length,
                    steps=a.steps, temperature=0.0, threshold=(0.85 if a.family == "sdar" else 0.95),
                    mask_id=(tokenizer.mask_token_id or 151669) if a.family == "sdar" else 156895,
                    eos_id=None if a.family == "sdar" else 156892,
                    eos_early_stop=True,
                    **({"remasking_strategy": "sequential"} if a.family == "sdar" else
                       {"editing_threshold": 0.9, "num_to_transfer": 1}),
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
                      "moe_expert_patch": a.moe_expert_patch,
                      "data_dir": str(a.data_dir), "resumed": len(completed) > 0,
                      "rows": rows}
            (a.output_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {"family": a.family, "mode": a.mode, "max_context_tokens": a.max_context_tokens,
              "truncation": "drop_middle_keep_head_tail", "moe_expert_patch": a.moe_expert_patch,
              "data_dir": str(a.data_dir), "resumed": len(completed) > 0,
              "skipped": skipped, "rows": rows}
    (a.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"done": len(rows), "skipped": skipped}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
