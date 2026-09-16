#!/usr/bin/env python3
"""Render the paper query and prefix figures from bundled CSV data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finish(fig: plt.Figure, ax: plt.Axes, name: str) -> None:
    ax.grid(True, which="major", color="0.82", alpha=0.45, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(HERE / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    query_rows = read_rows(HERE / "query_position_recall.csv")
    exact_rows = read_rows(HERE / "query_position_recall_exact_no_norm.csv")
    prefix_rows = read_rows(HERE / "hotpotqa_prefix_attention_mass.csv")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    query_fig, query_ax = plt.subplots(figsize=(3.5, 3.0))

    for color, ratio in zip(COLORS, (0.5, 0.7, 0.9)):
        rows = [row for row in query_rows if float(row["keep_ratio"]) == ratio]
        layers = np.asarray([int(row["layer"]) for row in rows])
        recall = np.asarray([float(row["recall"]) for row in rows])
        if not np.array_equal(layers, np.arange(1, 21)):
            raise ValueError(f"Keep {ratio:.0%}: expected layers 1--20")
        query_ax.plot(layers, recall, color=color, linewidth=2.2,
                      label=f"Keep {ratio:.0%}")
    query_ax.axvline(2, color="0.45", linestyle="--", linewidth=0.9)
    query_ax.text(2.35, 0.32, "Selection layer", rotation=90,
                  color="0.3", fontsize=6.5, va="bottom")
    query_ax.set(xlabel="Transformer layer",
                 ylabel="Recall of final update positions",
                 xlim=(1, 20), ylim=(0.3, 1.01))
    query_ax.set_xticks([1, 5, 10, 15, 20])
    query_ax.legend(loc="lower right", frameon=False)
    finish(query_fig, query_ax, "query_position_recall")

    exact_fig, exact_ax = plt.subplots(figsize=(3.5, 3.0))
    for color, ratio in zip(COLORS, (0.5, 0.7, 0.9)):
        rows = [row for row in exact_rows if float(row["keep_ratio"]) == ratio]
        layers = np.asarray([int(row["layer"]) for row in rows])
        recall = np.asarray([float(row["recall"]) for row in rows])
        if not np.array_equal(layers, np.arange(1, 21)):
            raise ValueError(f"Exact {ratio:.0%}: expected layers 1--20")
        exact_ax.plot(layers, recall, color=color, linewidth=2.2,
                      label=f"Exact {ratio:.0%}")
    exact_ax.axvline(2, color="0.45", linestyle="--", linewidth=0.9)
    exact_ax.text(2.35, 0.32, "Selection layer", rotation=90,
                  color="0.3", fontsize=6.5, va="bottom")
    exact_ax.set(xlabel="Transformer layer",
                 ylabel="Recall of final update positions",
                 xlim=(1, 20), ylim=(0.3, 1.01))
    exact_ax.set_xticks([1, 5, 10, 15, 20])
    exact_ax.legend(loc="lower right", frameon=False)
    finish(exact_fig, exact_ax, "query_position_recall_exact_no_norm")

    prefix_fig, prefix_ax = plt.subplots(figsize=(3.5, 3.0))
    for color, budget in zip(COLORS, (128, 256, 512)):
        rows = [row for row in prefix_rows if int(row["budget"]) == budget]
        fraction = np.asarray([float(row["resolved_fraction"]) for row in rows])
        mean = np.asarray([float(row["mean"]) for row in rows])
        std = np.asarray([float(row["std"]) for row in rows])
        if len(rows) != 32 or not np.all(np.diff(fraction) > 0):
            raise ValueError(f"Budget {budget}: expected 32 ordered token positions")
        prefix_ax.plot(fraction, mean, color=color, linewidth=2.2,
                       label=f"Budget {budget}")
        prefix_ax.fill_between(fraction, mean - std, mean + std,
                               color=color, alpha=0.08, linewidth=0)
    prefix_ax.set(
        xlabel="Resolved-token fraction",
        ylabel="Covered prefix-attn mass",
        xlim=(0, 1), ylim=(0.7, 0.91),
    )
    prefix_ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    prefix_ax.legend(loc="lower left", frameon=False)
    finish(prefix_fig, prefix_ax, "prefix_attention_mass")


if __name__ == "__main__":
    main()
