#!/usr/bin/env python3
"""Render query-resolved prefix attention-mass images from collector NPZ files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=Path, required=True)
    p.add_argument("--parts", nargs="+", default=["run"])
    p.add_argument("--fig_root", type=Path, default=None)
    p.add_argument("--scale", type=int, default=2, help="pixels per matrix cell")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--compress_level", type=int, default=1)
    return p.parse_args()


def load_samples(base_dir: Path, parts: list[str]):
    samples = []
    for part in parts:
        run_dir = base_dir / part
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        for sample in manifest["samples"]:
            path = run_dir / f"sample_{sample['sample_idx']:02d}_{sample['task']}.npz"
            with np.load(path) as data:
                mass = data["page_mass"].astype(np.float32)
                share = data["prefix_share"].astype(np.float32)
                layers = data["layers"].astype(int)
            expected = tuple(manifest["expected_matrix"])
            if mass.shape != expected:
                raise RuntimeError(f"{path}: shape {mass.shape} != {expected}")
            samples.append((sample, mass, share, layers))
    samples.sort(key=lambda item: item[0]["task"])
    return samples


def mass_to_rgb(mass, vmax, gamma, cmap):
    x = np.clip(mass / max(float(vmax), 1e-12), 0.0, 1.0) ** gamma
    indices = np.clip((x * 255.0).round().astype(np.uint8), 0, 255)
    rgb = cmap(indices)[..., :3]
    return (rgb * 255).astype(np.uint8)


def render_task(sample, mass, layers, out_dir, scale, gamma, cmap, compress_level):
    task = sample["task"]
    task_dir = out_dir / task
    n_layers, n_steps, n_queries, n_pages = mass.shape
    if n_layers != len(layers):
        raise RuntimeError(f"{task}: layer count mismatch")
    vmax = max(float(mass.max()), 1e-12)
    rgb_all = mass_to_rgb(mass, vmax, gamma, cmap)

    # Individual images preserve the query-row/KV-column matrix with uniform cell scaling.
    for layer_pos, layer_idx in enumerate(layers):
        layer_dir = task_dir / f"layer_{layer_idx + 1:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for step in range(n_steps):
            image = Image.fromarray(np.ascontiguousarray(rgb_all[layer_pos, step]), mode="RGB")
            image = image.resize((n_pages * scale, n_queries * scale), Image.Resampling.NEAREST)
            image.save(
                layer_dir / f"step_{step:02d}.png",
                format="PNG",
                compress_level=compress_level,
            )

    # Compact contact sheet: rows are layers, columns are steps. This is only an
    # index/view aid; the authoritative output is the individual long image above.
    thumb_w, thumb_h = 64, 8
    gap = 1
    sheet = Image.new(
        "RGB",
        (n_steps * (thumb_w + gap) + gap, n_layers * (thumb_h + gap) + gap),
        "#101010",
    )
    for layer_pos in range(n_layers):
        for step in range(n_steps):
            thumb = Image.fromarray(np.ascontiguousarray(rgb_all[layer_pos, step]), mode="RGB").resize(
                (thumb_w, thumb_h), Image.Resampling.NEAREST
            )
            sheet.paste(thumb, (gap + step * (thumb_w + gap), gap + layer_pos * (thumb_h + gap)))
    sheet.save(task_dir / "contact_sheet_layers_by_steps.png", format="PNG", compress_level=compress_level)
    return {
        "task": task,
        "original_index": sample["original_index"],
        "input_tokens": sample["input_tokens"],
        "matrix_shape": list(mass.shape),
        "normalization_vmax": vmax,
        "individual_image_size": [n_pages * scale, n_queries * scale],
        "individual_count": int(n_layers * n_steps),
        "output_dir": str(task_dir),
    }


def main():
    args = parse_args()
    fig_root = args.fig_root or (args.base_dir / f"figures_{args.cmap}")
    fig_root.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.base_dir, args.parts)
    print(f"loaded {len(samples)} task samples", flush=True)
    try:
        cmap = matplotlib.colormaps[args.cmap]
    except KeyError as exc:
        raise ValueError(f"unknown colormap: {args.cmap}") from exc
    records = []
    for i, (sample, mass, share, layers) in enumerate(samples):
        print(f"[render] {i + 1}/{len(samples)} {sample['task']} {mass.shape}", flush=True)
        record = render_task(sample, mass, layers, fig_root, args.scale, args.gamma, cmap, args.compress_level)
        record["prefix_share_min"] = float(share.min())
        record["prefix_share_max"] = float(share.max())
        records.append(record)
    payload = {
        "description": "Individual query-row by prefix-column attention-mass images for every layer and step",
        "cell_scale": args.scale,
        "gamma": args.gamma,
        "colormap": args.cmap,
        "normalization": "global per task over all layers and steps; prefix-conditional page mass",
        "tasks": records,
    }
    (fig_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {len(records)} tasks; root={fig_root}", flush=True)


if __name__ == "__main__":
    main()
