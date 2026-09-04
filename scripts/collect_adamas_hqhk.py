#!/usr/bin/env python3
"""Collect the Hq/Hk distribution seen by SparseDLM's Adamas selector."""

import argparse
import json
from pathlib import Path

import torch

import bench_long_context as bench
import src.sparse.sparse_ops as sparse


QUANTILES = (0.158655253931, 0.5, 0.841344746069)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llada", "sdar"), required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_args = argparse.Namespace(
        model=args.model,
        model_path=None,
        mode="prefix",
        prefix_token_budget=1024 if args.model == "llada" else 512,
        losa_active_topk=5,
        losa_score_mode="query",
        losa_key_samples=32,
        seed=42,
        gen_length=args.gen_length,
    )
    bench.set_seed(load_args.seed)
    model, tokenizer = bench.load(load_args)
    bins = torch.linspace(-20, 20, 4001, device=model.device)
    counts = {
        "q": torch.zeros(4002, dtype=torch.int64, device=model.device),
        "k": torch.zeros(4002, dtype=torch.int64, device=model.device),
    }
    sums = {"q": 0.0, "k": 0.0}
    sumsqs = {"q": 0.0, "k": 0.0}
    totals = {"q": 0, "k": 0}
    original = sparse._adamas_prefix_indices

    def collect(query, key, *selector_args, **selector_kwargs):
        for name, tensor in (("q", query), ("k", key)):
            values = sparse._hadamard_transform(tensor).float().flatten()
            counts[name] += torch.bincount(
                torch.bucketize(values, bins), minlength=bins.numel() + 1
            )
            sums[name] += values.sum().item()
            sumsqs[name] += values.square().sum().item()
            totals[name] += values.numel()
        return original(query, key, *selector_args, **selector_kwargs)

    sparse._adamas_prefix_indices = collect
    prompt = bench.exact_prompt(
        tokenizer, args.context - args.gen_length, model.device
    )
    model.generate(**bench.generation_kwargs(load_args, tokenizer, prompt))

    result = {"model": args.model, "context": args.context}
    edges = bins.cpu()
    for name in ("q", "k"):
        cumulative = counts[name].cumsum(0)
        quantile_values = []
        for quantile in QUANTILES:
            index = int(
                torch.searchsorted(
                    cumulative, torch.tensor(quantile * totals[name], device=model.device)
                ).item()
            )
            quantile_values.append(float(edges[min(max(index - 1, 0), len(edges) - 1)]))
        mean = sums[name] / totals[name]
        variance = max(sumsqs[name] / totals[name] - mean * mean, 0.0)
        result[name] = {
            "count": totals[name],
            "mean": mean,
            "std": variance**0.5,
            "bucket_thresholds": quantile_values,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
