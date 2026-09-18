#!/usr/bin/env python3
"""Measure the integrated batch-size-one FOCUS runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.reference.focus import FocusRuntime, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("llada", "sdar"), required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prompt", default="Explain diffusion language models in one sentence.")
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--editing_threshold", type=float, default=0.5)
    parser.add_argument("--max_post_steps", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def tokenize(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    return tokenizer(prompt, return_tensors="pt").input_ids


def main() -> int:
    args = parse_args()
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats positive")
    set_seed(args.seed)
    runtime = FocusRuntime(
        family=args.family,
        model_path=args.model_path,
        alpha=args.alpha,
    )
    model, tokenizer = runtime.load()
    inputs = tokenize(tokenizer, args.prompt).to(model.device)
    mask_id = 156895 if args.family == "llada" else int(tokenizer.mask_token_id or 151669)
    threshold = args.threshold
    if threshold is None:
        threshold = 0.7 if args.family == "llada" else 0.95
    generation_kwargs = dict(
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=threshold,
        mask_id=mask_id,
        eos_early_stop=False,
    )
    if args.family == "llada":
        generation_kwargs.update(
            editing_threshold=args.editing_threshold,
            max_post_steps=args.max_post_steps,
        )
    else:
        generation_kwargs.update(remasking_strategy="low_confidence_dynamic")

    elapsed = []
    output = None
    for repeat in range(args.warmups + args.repeats):
        set_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        output = runtime.generate(inputs, **generation_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        duration = time.perf_counter() - started
        if repeat >= args.warmups:
            elapsed.append(duration)
    assert output is not None
    token_bytes = json.dumps(output.tokens.detach().cpu().tolist(), separators=(",", ":")).encode()
    result = {
        "family": args.family,
        "model_path": args.model_path,
        "alpha": args.alpha,
        "input_tokens": int(inputs.shape[1]),
        "generated_tokens": int(output.tokens.shape[1]),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "elapsed_seconds": elapsed,
        "median_seconds": statistics.median(elapsed),
        "median_tokens_per_second": int(output.tokens.shape[1]) / statistics.median(elapsed),
        "checksum": hashlib.sha256(token_bytes).hexdigest()[:16],
        "focus_trace": output.trace,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
