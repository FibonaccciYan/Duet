#!/usr/bin/env python3
"""Run the four inference methods on the LongBench QA subset."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losa.generation import load_model_and_tokenizer, set_seed
from src.runtime import patch_method
from src.sparse.llada_patch import patch_moe_experts


TASKS = ("hotpotqa", "triviaqa", "narrativeqa", "qasper", "multifieldqa_en")
GEN_LENGTHS = {
    "hotpotqa": 32,
    "triviaqa": 32,
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("llada", "sdar"), required=True)
    parser.add_argument("--method", choices=("sparse", "dense", "focus", "losa"), required=True)
    default_data = (
        Path(os.environ["LONGBENCH_DATA"])
        if "LONGBENCH_DATA" in os.environ
        else ROOT / "data" / "longbench"
    )
    parser.add_argument("--data_dir", type=Path, default=default_data)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_tokens", type=int, default=32768)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--query_dense_threshold", type=int)
    parser.add_argument("--focus_alpha", type=float, default=1.5)
    parser.add_argument("--losa_token_budget", type=int, default=256)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--editing_threshold", type=float)
    parser.add_argument("--remasking_strategy")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--moe_expert_patch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume_from", type=Path, help="previous predictions.jsonl")
    return parser.parse_args()


def tokenize(tokenizer, prompt: str) -> torch.Tensor:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    except Exception:
        return tokenizer(prompt, return_tensors="pt").input_ids


def truncate_middle(input_ids: torch.Tensor, budget: int) -> tuple[torch.Tensor, bool]:
    if input_ids.shape[-1] <= budget:
        return input_ids, False
    head = budget // 2
    return torch.cat((input_ids[:, :head], input_ids[:, -(budget - head):]), dim=-1), True


def normalize_answer(value: object) -> str:
    text = re.sub(r"\b(a|an|the)\b", " ", str(value).lower())
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(text.split())


def answer_scores(prediction: str, answers: list[str]) -> tuple[float, float]:
    prediction_tokens = normalize_answer(prediction).split()
    best_f1 = best_em = 0.0
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        common = sum((collections.Counter(prediction_tokens) & collections.Counter(answer_tokens)).values())
        if not prediction_tokens or not answer_tokens:
            f1 = float(prediction_tokens == answer_tokens)
        elif common:
            precision, recall = common / len(prediction_tokens), common / len(answer_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        best_f1 = max(best_f1, f1)
        best_em = max(best_em, float(normalize_answer(prediction) == normalize_answer(answer)))
    return best_f1, best_em


def load_completed(path: Path | None) -> dict[tuple[str, int], dict]:
    completed = {}
    if path is None:
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                completed[(str(row["task"]), int(row["index"]))] = row
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def summarize(rows: list[dict]) -> dict:
    result = {}
    for task in (*TASKS, "overall"):
        selected = rows if task == "overall" else [row for row in rows if row["task"] == task]
        if selected:
            result[task] = {
                "count": len(selected),
                "f1": sum(row["f1"] for row in selected) / len(selected),
                "exact_match": sum(row["exact_match"] for row in selected) / len(selected),
            }
    return result


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_context_tokens <= 0 or args.block_length <= 0:
        raise ValueError("context and block lengths must be positive")

    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(
        args.family, model_path=args.model_path, dtype=args.dtype
    )
    moe_patch = args.family == "llada" if args.moe_expert_patch is None else args.moe_expert_patch
    options = {"moe_expert_patch": moe_patch} if args.method in {"sparse", "dense"} else {}
    if args.method == "sparse" and args.query_dense_threshold is not None:
        options["query_dense_threshold"] = args.query_dense_threshold
    if args.method == "focus":
        options["alpha"] = args.focus_alpha
    elif args.method == "losa":
        options["token_budget"] = args.losa_token_budget
    patch_method(model, args.method, model_name=args.family, **options)
    if moe_patch and args.family == "llada" and args.method in {"focus", "losa"}:
        patch_moe_experts(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "predictions.jsonl"
    completed = load_completed(args.resume_from)
    rows = list(completed.values())
    progress_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    threshold = args.threshold if args.threshold is not None else (
        0.5 if args.method == "sparse" and args.family == "llada" else
        1.0 if args.method == "sparse" else
        0.95 if args.family == "llada" else 0.85
    )
    editing_threshold = args.editing_threshold if args.editing_threshold is not None else (
        0.0 if args.method == "sparse" else 0.9
    )
    remasking = args.remasking_strategy or (
        "low_confidence_dynamic" if args.method == "focus" else "sequential"
    )

    with progress_path.open("a", encoding="utf-8") as progress:
        for task in TASKS:
            path = args.data_dir / f"{task}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"LongBench task file not found: {path}")
            with path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
            if args.limit is not None:
                records = records[:args.limit]
            for index, record in enumerate(records):
                if (task, index) in completed:
                    continue
                gen_length = GEN_LENGTHS[task]
                input_ids = tokenize(tokenizer, str(record["prompt"]))
                original_length = int(input_ids.shape[-1])
                model_limit = getattr(tokenizer, "model_max_length", args.max_context_tokens)
                if not isinstance(model_limit, int) or model_limit > 1_000_000:
                    model_limit = args.max_context_tokens
                budget = min(args.max_context_tokens - gen_length, model_limit - gen_length)
                budget = (budget // args.block_length) * args.block_length
                if budget <= 0:
                    raise ValueError("max context leaves no block-aligned prompt room")
                input_ids, truncated = truncate_middle(input_ids, budget)
                input_ids = input_ids.to(model.device)
                generation_kwargs = dict(
                    inputs=input_ids,
                    gen_length=gen_length,
                    block_length=args.block_length,
                    steps=args.steps,
                    temperature=0.0,
                    threshold=threshold,
                    mask_id=(tokenizer.mask_token_id or 151669) if args.family == "sdar" else 156895,
                    eos_id=None if args.family == "sdar" else 156892,
                    eos_early_stop=True,
                )
                if args.family == "sdar":
                    generation_kwargs["remasking_strategy"] = remasking
                    if args.method in {"sparse", "dense"}:
                        generation_kwargs["eb_threshold"] = 0.35
                else:
                    generation_kwargs.update(editing_threshold=editing_threshold, num_to_transfer=1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.inference_mode():
                    tokens = model.generate(**generation_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prediction = tokenizer.decode(tokens[0], skip_special_tokens=True)
                answers = record.get("answers", record.get("answer", []))
                answers = [answers] if isinstance(answers, str) else [str(value) for value in answers or []]
                f1, exact_match = answer_scores(prediction, answers)
                row = {
                    "task": task,
                    "index": index,
                    "idx": record.get("idx", index),
                    "answers": answers,
                    "prediction": prediction,
                    "f1": f1,
                    "exact_match": exact_match,
                    "original_input_tokens": original_length,
                    "input_tokens": int(input_ids.shape[-1]),
                    "generated_tokens": int(tokens.shape[-1]),
                    "truncated": truncated,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                rows.append(row)
                progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "family": args.family,
        "method": args.method,
        "data_dir": str(args.data_dir),
        "max_context_tokens": args.max_context_tokens,
        "scores": summarize(rows),
        "rows": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["scores"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
