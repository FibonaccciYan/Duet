#!/usr/bin/env python3
"""Measure current SparseDLM query selection on the 8K prefix-study samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.original.overlap.llada_layer_overlap import overlap_metrics, selector_mask
from scripts.original.paper.attention_mass.collect_attention_mass import tokenize, truncate_middle
from src.dense.api import DenseRuntime
from src.reference.losa.generation import set_seed
from src.reference.sparse.sparse_ops import _sample_with_confidence
import src.reference.sparse.llada_patch as llada_patch


TASKS = (
    "gov_report",
    "hotpotqa",
    "dureader",
    "passage_retrieval_en",
    "repobench-p",
)
FIELDS = ("task", "sample_idx", "original_index", "layer", "ratio", "calls", "recall")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--source_manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--ratios", type=float, nargs="+", default=(0.5, 0.7, 0.9))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--selection_top_k", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fields=FIELDS) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "task": row["task"],
                    "sample_idx": int(row["sample_idx"]),
                    "original_index": int(row["original_index"]),
                    "layer": int(row["layer"]),
                    "ratio": float(row["ratio"]),
                    "calls": int(row["calls"]),
                    "recall": float(row["recall"]),
                }
            )
    return rows


def summarize_rows(rows: list[dict], tasks: tuple[str, ...] = TASKS):
    dataset = []
    for task in tasks:
        for layer in sorted({row["layer"] for row in rows}):
            for ratio in sorted({row["ratio"] for row in rows}):
                values = [
                    row["recall"]
                    for row in rows
                    if row["task"] == task
                    and row["layer"] == layer
                    and row["ratio"] == ratio
                ]
                if values:
                    dataset.append(
                        dict(task=task, layer=layer, ratio=ratio, samples=len(values),
                             mean=float(np.mean(values)), std=float(np.std(values)))
                    )
    overall = []
    for layer in sorted({row["layer"] for row in rows}):
        for ratio in sorted({row["ratio"] for row in rows}):
            values = [
                row["recall"]
                for row in rows
                if row["layer"] == layer and row["ratio"] == ratio
            ]
            overall.append(
                dict(task="overall", layer=layer, ratio=ratio, samples=len(values),
                     mean=float(np.mean(values)), std=float(np.std(values)))
            )
    return dataset, overall


def plot_summary(rows: list[dict], task: str, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for color, ratio in zip(colors, sorted({row["ratio"] for row in rows})):
        part = [row for row in rows if row["task"] == task and row["ratio"] == ratio]
        layers = np.asarray([row["layer"] for row in part])
        mean = np.asarray([row["mean"] for row in part])
        std = np.asarray([row["std"] for row in part])
        ax.plot(layers, mean, color=color, linewidth=2.2, label=f"Exact {ratio:.0%}")
        ax.fill_between(layers, np.maximum(0, mean - std), np.minimum(1, mean + std),
                        color=color, alpha=0.08, linewidth=0)
    ax.axvline(2, color="0.45", linestyle="--", linewidth=0.9)
    ax.text(2.35, 0.03, "Selection layer", rotation=90,
            color="0.3", fontsize=6.5, va="bottom")
    ax.set(xlabel="Transformer layer", ylabel="Recall of final update positions",
           xlim=(1, 20), ylim=(0, 1.01))
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(True, color="0.82", alpha=0.45, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{task}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


class OverlapCollector:
    def __init__(self, model, ratios, threshold, selection_top_k):
        self.model = model
        self.ratios = tuple(ratios)
        self.threshold = float(threshold)
        self.selection_top_k = int(selection_top_k)
        self.hidden = [None] * len(model.model.layers)
        self.values = defaultdict(list)
        self.handles = [
            layer.register_forward_hook(self._hook(index))
            for index, layer in enumerate(model.model.layers)
        ]

    def _hook(self, index):
        def capture(_module, _inputs, output):
            self.hidden[index] = output[0][:, -32:].detach()
        return capture

    def record(self, active_mask, active_logits, target):
        if not active_mask.any():
            return
        if not target.any():
            raise RuntimeError("final transfer selected no masked position")
        if any(hidden is None for hidden in self.hidden):
            raise RuntimeError("missing decoder-layer output")
        shallow = torch.cat(self.hidden[:-1], dim=0)
        logits = torch.cat((self.model.lm_head(shallow), active_logits), dim=0)
        _, scores = _sample_with_confidence(
            self.model, logits, temperature=0.0, top_p=None,
            top_k=self.selection_top_k,
        )
        mask_count = int(active_mask.sum().item())
        for layer, layer_scores in enumerate(scores, start=1):
            for ratio in self.ratios:
                selected = selector_mask(
                    active_mask,
                    layer_scores.unsqueeze(0),
                    math.ceil(mask_count * ratio),
                    self.threshold,
                )
                self.values[(layer, ratio)].append(
                    overlap_metrics(selected, target)["recall"]
                )

    def close(self):
        for handle in self.handles:
            handle.remove()


def load_record(path: Path, index: int) -> dict:
    with path.open(encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"{path}: no record {index}")


def main() -> int:
    args = parse_args()
    if any(not 0 < ratio <= 1 for ratio in args.ratios) or args.selection_top_k <= 0:
        raise ValueError("ratios must be in (0, 1] and selection_top_k must be positive")
    source = json.loads(args.source_manifest.read_text())
    if tuple(source["tasks"]) != TASKS:
        raise ValueError(f"unexpected source tasks: {source['tasks']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "sample_summary.csv"
    rows = read_rows(samples_path)
    completed = {(row["task"], row["original_index"]) for row in rows}

    set_seed(args.seed)
    runtime = DenseRuntime(
        family="llada", model_path=args.model_path, dtype="bfloat16",
        attn_implementation="sdpa", moe_expert_patch=True,
    )
    model, tokenizer = runtime.load()
    collector = OverlapCollector(model, args.ratios, args.threshold, args.selection_top_k)
    original_transfer = llada_patch._transfer_tokens

    def measured_transfer(*transfer_args, **transfer_kwargs):
        updated, transfer = original_transfer(*transfer_args, **transfer_kwargs)
        active_mask = transfer_args[4]
        active_logits = transfer_args[5]
        collector.record(active_mask, active_logits, transfer & active_mask)
        return updated, transfer

    llada_patch._transfer_tokens = measured_transfer
    try:
        for meta in source["samples"]:
            task = meta["task"]
            original_index = int(meta["original_index"])
            if (task, original_index) in completed:
                continue
            record = load_record(Path(source["data_dir"]) / f"{task}.jsonl", original_index)
            ids = tokenize(tokenizer, str(record["prompt"]))
            ids, _, _ = truncate_middle(ids, int(source["prompt_tokens"]))
            if ids.shape[1] != source["prompt_tokens"]:
                raise RuntimeError(f"{task}/{original_index}: expected 8192 input tokens")
            collector.values.clear()
            print(f"[run] {meta['sample_idx'] + 1}/{len(source['samples'])} "
                  f"task={task} index={original_index}", flush=True)
            model.generate(
                inputs=ids.to(model.device),
                gen_length=int(source["gen_length"]),
                block_length=int(source["block_length"]),
                steps=int(source["steps"]),
                temperature=0.0,
                threshold=args.threshold,
                editing_threshold=float(source["editing_threshold"]),
                max_post_steps=0,
                eos_early_stop=True,
                mask_id=156895,
                eos_id=156892,
                num_to_transfer=1,
            )
            for (layer, ratio), values in sorted(collector.values.items()):
                rows.append(
                    dict(task=task, sample_idx=int(meta["sample_idx"]),
                         original_index=original_index, layer=layer, ratio=ratio,
                         calls=len(values), recall=float(np.mean(values)))
                )
            write_csv(samples_path, rows)
    finally:
        llada_patch._transfer_tokens = original_transfer
        collector.close()

    dataset, overall = summarize_rows(rows)
    summary_fields = ("task", "layer", "ratio", "samples", "mean", "std")
    write_csv(args.output_dir / "dataset_summary.csv", dataset, summary_fields)
    write_csv(args.output_dir / "overall_summary.csv", overall, summary_fields)
    for task in (*TASKS, "overall"):
        plot_summary(dataset if task != "overall" else overall, task, args.output_dir)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "model_path": args.model_path,
        "source_manifest": str(args.source_manifest),
        "tasks": TASKS,
        "samples_per_task": 10,
        "prompt_tokens": source["prompt_tokens"],
        "gen_length": source["gen_length"],
        "block_length": source["block_length"],
        "steps": source["steps"],
        "eos_early_stop": True,
        "ratios": args.ratios,
        "threshold": args.threshold,
        "selection_top_k": args.selection_top_k,
        "shallow_layer_norm": False,
        "aggregation": "mean calls within prompt, then equal-weight prompts",
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
