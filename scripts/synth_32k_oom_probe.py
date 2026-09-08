#!/usr/bin/env python3
"""Synthetic 32K-context OOM probe for the dense long-context quality runs.

Builds a prompt longer than the 32K budget out of repeated natural text,
runs it through the same middle-truncation path as run_longbench_quality.py,
generates once with the dense runtime, and reports peak VRAM + wall time.
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
from scripts.run_longbench_quality import truncate_middle

SYNTH_TEXT = (
    "The 32K synthetic probe repeats this natural passage many times so that the "
    "resulting token sequence exercises the same block-aligned KV-cache prefill "
    "and dense attention paths as a real overlong LongBench record. Memory usage "
    "is expected to stay flat during prefill because logits are materialized "
    "one block at a time. "
)


def build_synthetic(tokenizer, target_tokens: int) -> torch.Tensor:
    piece = tokenizer(SYNTH_TEXT, add_special_tokens=False).input_ids
    repeats = target_tokens // max(len(piece), 1) + 2
    ids = (piece * repeats)[:target_tokens]
    return torch.tensor([ids], dtype=torch.long)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("llada", "sdar"), required=True)
    p.add_argument("--gen_length", type=int, default=None)
    p.add_argument("--max_context_tokens", type=int, default=32768)
    p.add_argument("--block_length", type=int, default=32)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--overshoot_tokens", type=int, default=8192)
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    a = p.parse_args()

    gen_length = a.gen_length or (32 if a.family == "llada" else 128)
    runtime = load_runtime("dense", family=a.family, dtype=a.dtype)
    model, tokenizer = runtime.load()

    budget = min(a.max_context_tokens - gen_length, int(tokenizer.model_max_length or 10**12))
    budget = (budget // a.block_length) * a.block_length
    if budget <= 0:
        raise ValueError("max_context_tokens leaves no block-aligned prompt room")

    ids = build_synthetic(tokenizer, budget + a.overshoot_tokens)
    ids, truncated = truncate_middle(ids, budget)
    assert truncated and int(ids.shape[-1]) == budget, (ids.shape, budget)
    ids = ids.to(model.device)

    torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        result = runtime.generate(
            ids, gen_length=gen_length, block_length=a.block_length,
            steps=a.steps, temperature=0.0,
            threshold=(0.85 if a.family == "sdar" else 0.95),
            mask_id=(tokenizer.mask_token_id or 151669) if a.family == "sdar" else 156895,
            eos_id=None if a.family == "sdar" else 156892,
            eos_early_stop=True,
            **({"remasking_strategy": "sequential"} if a.family == "sdar" else
               {"editing_threshold": 0.9, "num_to_transfer": 1}),
        )
    elapsed = time.perf_counter() - started
    report = {
        "family": a.family,
        "mode": "dense",
        "input_tokens": int(ids.shape[-1]),
        "gen_length": gen_length,
        "block_length": a.block_length,
        "steps": a.steps,
        "generated_tokens": int(result.tokens.shape[-1]),
        "elapsed_seconds": round(elapsed, 2),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(model.device) / 1024**2, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(model.device) / 1024**2, 1),
        "device": str(model.device),
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
