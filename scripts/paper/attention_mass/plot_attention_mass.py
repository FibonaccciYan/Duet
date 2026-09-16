#!/usr/bin/env python3
"""Plot block-start dense-attention KV coverage over one denoising block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=Path, required=True)
    p.add_argument("--parts", nargs="+", default=["run"])
    p.add_argument("--fig_root", type=Path, default=None)
    p.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512])
    return p.parse_args()


def load_samples(base_dir: Path, parts: list[str]):
    samples = []
    for part in parts:
        run_dir = base_dir / part
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["page_size"] != 1:
            raise ValueError("budget curves require token-level collection with page_size=1")
        for sample in manifest["samples"]:
            path = run_dir / f"sample_{sample['sample_idx']:02d}_{sample['task']}.npz"
            with np.load(path) as data:
                mass = data["page_mass"].astype(np.float32)
                layers = data["layers"].astype(int)
                resolved_tokens = data["resolved_tokens"].astype(int)
                block_indices = data["block_indices"].astype(int)
                prefix_lengths = data["prefix_lengths"].astype(int)
            if mass.shape != tuple(sample["page_matrix_shape"]):
                raise RuntimeError(f"{path}: unexpected shape {mass.shape}")
            if mass.shape[1] != len(resolved_tokens):
                raise RuntimeError(f"{path}: resolved-token count mismatch")
            samples.append(
                (sample, mass, layers, resolved_tokens, block_indices, prefix_lengths)
            )
    return samples


def budget_coverage(
    mass: np.ndarray,
    budgets: list[int],
    block_indices: np.ndarray | None = None,
    prefix_lengths: np.ndarray | None = None,
) -> np.ndarray:
    """Select prefix KV at each block start, then hold each set fixed."""
    n_layers, n_observations, _, padded_prefix_length = mass.shape
    block_indices = (
        np.zeros(n_observations, dtype=int) if block_indices is None else block_indices
    )
    prefix_lengths = (
        np.full(n_observations, padded_prefix_length, dtype=int)
        if prefix_lengths is None else prefix_lengths
    )
    curves = np.empty((len(budgets), n_layers, n_observations), dtype=np.float32)
    for block in np.unique(block_indices):
        observations = np.flatnonzero(block_indices == block)
        prefix_length = int(prefix_lengths[observations[0]])
        step0_scores = mass[:, observations[0], :, :prefix_length].mean(axis=1)
        for budget_pos, budget in enumerate(budgets):
            if budget <= 0 or budget > prefix_length:
                raise ValueError(f"budget must be in [1, {prefix_length}], got {budget}")
            for layer in range(n_layers):
                indices = np.argpartition(step0_scores[layer], -budget)[-budget:]
                selected = np.take(mass[layer, observations], indices, axis=-1)
                curves[budget_pos, layer, observations] = selected.sum(axis=-1).mean(axis=-1)
    return curves


def expand_steps_to_tokens(curves, block_indices, resolved_tokens, block_length):
    """Assign each forward's mass to the tokens resolved by that forward."""
    grid = np.arange(block_length)
    block_curves = []
    layer_means = curves.mean(axis=1)
    for block in np.unique(block_indices):
        observations = np.flatnonzero(block_indices == block)
        x = resolved_tokens[observations]
        aligned = np.empty((curves.shape[0], block_length), dtype=np.float32)
        ends = np.append(x[1:], block_length)
        for observation, start, end in zip(observations, x, ends):
            aligned[:, start:end] = layer_means[:, observation, None]
        block_curves.append(aligned)
    block_curves = np.stack(block_curves, axis=1)
    return grid, block_curves


def aggregate_samples(block_curves: list[np.ndarray]) -> np.ndarray:
    """Give every prompt equal weight after averaging its available blocks."""
    return np.stack([curves.mean(axis=1) for curves in block_curves], axis=1)


def render(task, samples, budgets, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("plotting requires matplotlib") from exc
    sample_block_curves = []
    blocks_per_sample = []
    original_indices = []
    for sample, mass, _layers, resolved_tokens, block_indices, prefix_lengths in samples:
        curves = budget_coverage(mass, budgets, block_indices, prefix_lengths)
        progress, block_curves = expand_steps_to_tokens(
            curves, block_indices, resolved_tokens, mass.shape[2]
        )
        sample_block_curves.append(block_curves)
        blocks_per_sample.append(block_curves.shape[1])
        original_indices.append(sample["original_index"])
    sample_curves = aggregate_samples(sample_block_curves)
    mean_curves = sample_curves.mean(axis=1)
    std_curves = sample_curves.std(axis=1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for budget_idx, (budget, values) in enumerate(zip(budgets, mean_curves)):
        ax.plot(progress, values, label=f"budget={budget}")
        ax.fill_between(
            progress,
            values - std_curves[budget_idx],
            values + std_curves[budget_idx],
            alpha=0.15,
        )
    block_length = len(progress)
    ax.set(xlabel="Resolved-token progress within block", ylabel="Prefix attention mass covered", ylim=(0, 1))
    ax.set_xticks(np.arange(0, block_length + 1, 4))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    task_dir = out_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)
    figure_path = task_dir / "attention_mass_by_budget.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    np.savez_compressed(
        task_dir / "attention_mass_by_budget.npz",
        budgets=np.asarray(budgets),
        resolved_token_progress=progress,
        original_indices=np.asarray(original_indices),
        blocks_per_sample=np.asarray(blocks_per_sample),
        sample_curves=sample_curves,
        mean_curves=mean_curves,
        std_curves=std_curves,
    )
    return {
        "task": task,
        "original_indices": original_indices,
        "budgets": budgets,
        "selection_state": "0 resolved tokens",
        "layer_aggregation": "mean",
        "block_aggregation": "mean within each prompt after expanding forwards over transferred tokens",
        "sample_aggregation": "equal-weight mean across prompts; shading is prompt standard deviation",
        "samples": len(samples),
        "blocks_per_sample": blocks_per_sample,
        "total_blocks": sum(blocks_per_sample),
        "figure": str(figure_path),
    }


def main():
    args = parse_args()
    fig_root = args.fig_root or (args.base_dir / "figures_budget")
    fig_root.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.base_dir, args.parts)
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample[0]["task"], []).append(sample)
    records = [render(task, task_samples, args.budgets, fig_root)
               for task, task_samples in grouped.items()]
    payload = {
        "description": "Dense-attention top-K prefix KV selected at block start and held fixed",
        "metric": "mean over layers and queries of prefix-conditional attention mass on selected KV",
        "tasks": records,
    }
    (fig_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {len(samples)} samples across {len(records)} tasks; root={fig_root}", flush=True)


if __name__ == "__main__":
    main()
