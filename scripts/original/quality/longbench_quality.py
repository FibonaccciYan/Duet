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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reference.losa.generation import load_model_and_tokenizer, set_seed
from src.runtime import patch_method
from src.reference.sparse.llada_patch import patch_moe_experts


TASKS = ("hotpotqa", "triviaqa", "narrativeqa", "qasper", "multifieldqa_en")
AVAILABLE_TASKS = (*TASKS, "gov_report")
GEN_LENGTHS = {
    "hotpotqa": 32,
    "triviaqa": 32,
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "gov_report": 512,
}
LLADA_THRESHOLD = 0.7
LLADA_EDITING_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("llada", "sdar"), required=True)
    parser.add_argument("--method", choices=("sparse", "sparse_optimized", "dense", "focus", "losa"), required=True)
    default_data = (
        Path(os.environ["LONGBENCH_DATA"])
        if "LONGBENCH_DATA" in os.environ
        else ROOT / "data" / "longbench"
    )
    parser.add_argument("--data_dir", type=Path, default=default_data)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=AVAILABLE_TASKS, default=TASKS)
    parser.add_argument("--model_path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context_tokens", type=int, default=32768)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--query_dense_threshold", type=int, default=4)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--selection_layer", type=int)
    parser.add_argument("--query_sparse", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefix_sparse", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefix_token_budget", type=int)
    parser.add_argument("--prefix_strict_budget", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--collect_prefix_selection_stats", action="store_true")
    parser.add_argument(
        "--prefix_rescreen_full_kv",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--focus_alpha", type=float, default=1.5)
    parser.add_argument("--losa_token_budget", type=int, default=256)
    parser.add_argument("--losa_page_size", type=int, default=16)
    parser.add_argument("--losa_active_topk", type=int, default=5)
    parser.add_argument(
        "--losa_gqa_mode",
        choices=("per_query_head", "group_mean", "group_max_score"),
        default="per_query_head",
    )
    parser.add_argument("--losa_backend", choices=("torch", "triton", "auto"), default="auto")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--editing_threshold", type=float)
    parser.add_argument("--remasking_strategy")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--moe_expert_patch", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--eos_early_stop", action=argparse.BooleanOptionalAction, default=True)
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


def rouge_l_score(prediction: str, answers: list[str]) -> float:
    def lcs(left: list[str], right: list[str]) -> set[str]:
        table = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
        for i, left_word in enumerate(left, 1):
            for j, right_word in enumerate(right, 1):
                table[i][j] = (
                    table[i - 1][j - 1] + 1
                    if left_word == right_word
                    else max(table[i - 1][j], table[i][j - 1])
                )
        words = set()
        i, j = len(left), len(right)
        while i and j:
            if left[i - 1] == right[j - 1]:
                words.add(left[i - 1])
                i, j = i - 1, j - 1
            elif table[i - 1][j] > table[i][j - 1]:
                i -= 1
            else:
                j -= 1
        return words

    def score(answer: str) -> float:
        candidate = [" ".join(part.split()) for part in prediction.split(".") if part]
        reference = [" ".join(part.split()) for part in answer.split(".") if part]
        if not candidate or not reference:
            return 0.0
        candidate_words = set(" ".join(candidate).split(" "))
        reference_words = set(" ".join(reference).split(" "))
        union = set()
        for reference_sentence in reference:
            reference_tokens = reference_sentence.split(" ")
            for candidate_sentence in candidate:
                union.update(lcs(reference_tokens, candidate_sentence.split(" ")))
        precision = len(union) / len(candidate_words)
        recall = len(union) / len(reference_words)
        return 2 * precision * recall / (precision + recall + 1e-8)

    return max((score(answer) for answer in answers), default=0.0)


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


def summarize(rows: list[dict], tasks=TASKS) -> dict:
    result = {}
    for task in tasks:
        selected = [row for row in rows if row["task"] == task]
        if not selected:
            continue
        if task == "gov_report":
            result[task] = {
                "count": len(selected),
                "rouge_l": 100 * sum(row["rouge_l"] for row in selected) / len(selected),
            }
        else:
            result[task] = {
                "count": len(selected),
                "f1": sum(row["f1"] for row in selected) / len(selected),
                "exact_match": sum(row["exact_match"] for row in selected) / len(selected),
            }
    if len(result) == 1:
        result["overall"] = next(iter(result.values())).copy()
    else:
        selected = [row for row in rows if row["task"] != "gov_report"]
        if selected:
            result["overall"] = {
                "count": len(selected),
                "f1": sum(row["f1"] for row in selected) / len(selected),
                "exact_match": sum(row["exact_match"] for row in selected) / len(selected),
            }
    return result


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
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
    options = {"moe_expert_patch": moe_patch} if args.method in {"sparse", "sparse_optimized", "dense"} else {}
    if args.method in {"sparse", "sparse_optimized"}:
        for name in (
            "query_dense_threshold", "ratio", "selection_layer", "query_sparse",
            "prefix_sparse", "prefix_token_budget", "prefix_rescreen_full_kv", "prefix_strict_budget",
        ):
            if getattr(args, name) is not None:
                options[name] = getattr(args, name)
    if args.method == "focus":
        options["alpha"] = args.focus_alpha
    elif args.method == "losa":
        options.update(
            token_budget=args.losa_token_budget,
            page_size=args.losa_page_size,
            active_topk=args.losa_active_topk,
            gqa_mode=args.losa_gqa_mode,
            backend=args.losa_backend,
        )
    patch_method(model, args.method, model_name=args.family, **options)
    if moe_patch and args.family == "llada" and args.method in {"focus", "losa"}:
        patch_moe_experts(model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_enabled = args.collect_prefix_selection_stats and bool(
        getattr(model.config, args.family + "_prefix_sparse", False))
    stats_path = args.output_dir / "prefix_selection.jsonl"
    selection_records = []
    if stats_enabled:
        # Resume only records belonging to already completed predictions.
        done = load_completed(args.resume_from)
        if args.resume_from and stats_path.exists():
            selection_records = [json.loads(line) for line in stats_path.read_text().splitlines()]
            selection_records = [r for r in selection_records
                                 if (r["task"], r["sample_index"]) in done]
        stats_path.write_text("".join(json.dumps(r) + "\n" for r in selection_records))
        model._prefix_selection_stats = []
    progress_path = args.output_dir / "predictions.jsonl"
    completed = load_completed(args.resume_from)
    rows = list(completed.values())
    progress_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    remasking = args.remasking_strategy or (
        "low_confidence_dynamic" if args.method == "focus" else "sequential"
    )
    threshold = args.threshold if args.threshold is not None else (
        LLADA_THRESHOLD if args.family == "llada" else
        0.95 if remasking == "low_confidence_dynamic" else
        0.85
    )
    editing_threshold = args.editing_threshold if args.editing_threshold is not None else (
        LLADA_EDITING_THRESHOLD if args.family == "llada" else
        0.0 if args.method in {"sparse", "sparse_optimized"} else 0.9
    )
    with progress_path.open("a", encoding="utf-8") as progress:
        for task in args.tasks:
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
                    eos_early_stop=args.eos_early_stop,
                )
                if args.family == "sdar":
                    generation_kwargs["remasking_strategy"] = remasking
                    if args.method in {"sparse", "sparse_optimized", "dense"}:
                        generation_kwargs["eb_threshold"] = 0.35
                else:
                    generation_kwargs.update(editing_threshold=editing_threshold, num_to_transfer=1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if stats_enabled:
                    model._prefix_selection_stats.clear()
                started = time.perf_counter()
                with torch.inference_mode():
                    tokens = model.generate(**generation_kwargs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_seconds = time.perf_counter() - started
                if stats_enabled:
                    sample_stats = [dict(r, task=task, sample_index=index,
                                         selection_index=i)
                                    for i, r in enumerate(model._prefix_selection_stats)]
                    with stats_path.open("a") as stat_file:
                        for item in sample_stats:
                            stat_file.write(json.dumps(item) + "\n")
                    selection_records.extend(sample_stats)
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
                    "elapsed_seconds": elapsed_seconds,
                }
                if task == "gov_report":
                    row["rouge_l"] = rouge_l_score(prediction, answers)
                rows.append(row)
                progress.write(json.dumps(row, ensure_ascii=False) + "\n")
                progress.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "family": args.family,
        "method": args.method,
        "sparse_config": getattr(model.config, f"{args.family}_sparse_config", None),
        "generation_config": {
            "block_length": args.block_length,
            "steps": args.steps,
            "threshold": threshold,
            "editing_threshold": editing_threshold,
            "remasking_strategy": remasking if args.family == "sdar" else None,
            "eos_early_stop": args.eos_early_stop,
            "losa": options if args.method == "losa" else None,
        },
        "data_dir": str(args.data_dir),
        "tasks": args.tasks,
        "max_context_tokens": args.max_context_tokens,
        "scores": summarize(rows, args.tasks),
        "rows": rows,
    }
    if stats_enabled:
        from src.reference.sparse.selection_stats import summarize as summarize_prefix
        report["prefix_selection_stats"] = summarize_prefix(selection_records)
        report["prefix_selection_stats"]["timings_are_diagnostic"] = True
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["scores"], ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
