#!/usr/bin/env python3
"""Re-evaluate saved HumanEval generations after normalizing code indentation."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_ROOT))

from lm_eval.tasks.humaneval.sanitize_utils import sanitize  # noqa: E402


def extract_code(response: str) -> str:
    """Extract the first generated code section without trailing explanation."""
    text = response.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4)
    stripped = text.lstrip()
    if stripped.startswith("```"):
        _, separator, remainder = stripped.partition("\n")
        text = remainder if separator else ""
    return text.split("```", 1)[0].strip("\n")


def _normalize_shallow_indentation(code: str, indent_zero: bool) -> str:
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    normalized = []
    for line in lines:
        if not line.strip():
            normalized.append("")
            continue
        spaces = len(line) - len(line.lstrip(" "))
        if spaces < 4 and (indent_zero or spaces > 0):
            line = " " * 4 + line.lstrip(" ")
        normalized.append(line.rstrip())
    return "\n".join(normalized)


def build_prediction(prompt: str, response: str, entrypoint: str) -> str:
    """Build a syntactically normalized program for the requested function."""
    code = extract_code(response)
    definition = re.compile(rf"(?m)^\s*def\s+{re.escape(entrypoint)}\s*\(")

    if definition.search(code):
        full_code = _normalize_shallow_indentation(textwrap.dedent(code), False)
    else:
        body = _normalize_shallow_indentation(code, True)
        full_code = prompt.rstrip() + "\n" + body

    return sanitize(full_code, entrypoint)


def _first_string(value: object) -> str:
    while isinstance(value, list):
        value = value[0]
    if not isinstance(value, str):
        raise TypeError(f"expected a nested string, got {type(value).__name__}")
    return value


def _has_definition(code: str, entrypoint: str) -> bool:
    return re.search(rf"(?m)^def\s+{re.escape(entrypoint)}\s*\(", code) is not None


def load_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"no samples found in {path}")
    return rows


def _execute_candidate(program: str, timeout: float, unsafe_execute) -> None:
    result = []
    unsafe_execute(program, result, timeout)
    os._exit(0 if result and result[0] == "passed" else 1)


def run_candidate(program: str, timeout: float, unsafe_execute) -> dict:
    process = multiprocessing.get_context("fork").Process(
        target=_execute_candidate, args=(program, timeout, unsafe_execute)
    )
    process.start()
    process.join(timeout + 1)
    if process.is_alive():
        process.kill()
        process.join()
        return {"passed": False, "result": "timed out"}
    passed = process.exitcode == 0
    return {"passed": passed, "result": "passed" if passed else "failed"}


def reevaluate(
    samples_path: Path,
    output_path: Path,
    label: str,
    timeout: float,
    cache_dir: Path,
) -> dict:
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise RuntimeError("set HF_ALLOW_CODE_EVAL=1 to execute HumanEval candidates")

    import evaluate

    rows = load_samples(samples_path)
    predictions = []
    references = []
    metadata = []

    for row in rows:
        doc = row["doc"]
        original = _first_string(row["filtered_resps"])
        prediction = build_prediction(
            doc["prompt"], _first_string(row["resps"]), doc["entry_point"]
        )
        predictions.append([prediction])
        references.append(row["target"])
        metadata.append(
            {
                "doc_id": row["doc_id"],
                "task_id": doc["task_id"],
                "prediction_changed": prediction != original,
                "original_has_entrypoint": _has_definition(original, doc["entry_point"]),
                "normalized_has_entrypoint": _has_definition(
                    prediction, doc["entry_point"]
                ),
            }
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    metric = evaluate.load("code_eval", cache_dir=str(cache_dir))
    check_correctness = metric._compute.__globals__["check_correctness"]
    unsafe_execute = check_correctness.__globals__["unsafe_execute"]
    execution_results = [
        run_candidate(prediction[0] + "\n" + reference, timeout, unsafe_execute)
        for prediction, reference in zip(predictions, references)
    ]

    sample_results = []
    for index, item in enumerate(metadata):
        result = execution_results[index]
        sample_results.append(
            {
                **item,
                "passed": bool(result["passed"]),
                "result": result["result"],
            }
        )

    summary = {
        "label": label,
        "created_at": datetime.now().astimezone().isoformat(),
        "input": str(samples_path.resolve()),
        "samples": len(rows),
        "original_passed": sum(row.get("pass@1") == 1 for row in rows),
        "normalized_passed": sum(item["passed"] for item in sample_results),
        "pass@1": sum(item["passed"] for item in sample_results) / len(rows),
        "predictions_changed": sum(item["prediction_changed"] for item in metadata),
        "entrypoints_before": sum(item["original_has_entrypoint"] for item in metadata),
        "entrypoints_after": sum(item["normalized_has_entrypoint"] for item in metadata),
        "runner": "code_eval.unsafe_execute (sandbox-compatible)",
        "timeout_seconds": timeout,
        "results": sample_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def self_test() -> None:
    prompt = 'def answer() -> int:\n    """Return the answer."""\n'
    continuation = " return 42\n```\nExplanation"
    assert build_prediction(prompt, continuation, "answer").endswith("return 42")

    fenced = "```python\ndef answer() -> int:\n return 7\n```"
    assert build_prediction(prompt, fenced, "answer").endswith("return 7")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="?", type=Path, help="lm-eval samples JSONL")
    parser.add_argument("--output", type=Path, help="output summary JSON")
    parser.add_argument("--label", default="humaneval")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/tmp/llada_humaneval_reeval")
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test and args.samples is None:
        parser.error("samples is required unless --self-test is used")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return

    output = args.output or args.samples.with_name(
        f"{args.samples.stem}_reeval_normalized.json"
    )
    summary = reevaluate(
        args.samples,
        output,
        args.label,
        args.timeout,
        args.cache_dir,
    )
    print(
        f"{summary['label']}: {summary['normalized_passed']}/{summary['samples']} "
        f"= {summary['pass@1']:.2%} (was {summary['original_passed']})"
    )
    print(output)


if __name__ == "__main__":
    main()
