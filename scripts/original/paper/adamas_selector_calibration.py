#!/usr/bin/env python3
"""Collect Hq/Hk statistics and calibrate Adamas against exact attention."""

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.original.performance import long_context_benchmark as bench
import src.reference.sparse.sparse_ops as sparse


QUANTILES = (0.158655253931, 0.5, 0.841344746069)


def attention_metrics(query, key, indices, value=None, reference=None):
    """Measure a shared prefix selection against full QK attention."""
    if reference is None:
        groups = query.shape[1] // key.shape[1]
        expanded_key = key.repeat_interleave(groups, dim=1)
        scores = torch.matmul(
            query.float(), expanded_key.float().transpose(-1, -2)
        ) / math.sqrt(query.shape[-1])
        probabilities = scores.softmax(dim=-1)
        reference = {"probabilities": probabilities}
        if value is not None:
            expanded_value = value.repeat_interleave(groups, dim=1).float()
            reference["dense_output"] = probabilities @ expanded_value
    else:
        probabilities = reference["probabilities"]
    selected_count = indices.numel()
    if selected_count and (
        int(indices.min().item()) < 0 or int(indices.max().item()) >= key.shape[-2]
    ):
        raise ValueError(
            f"Adamas index range [{indices.min().item()}, {indices.max().item()}] "
            f"exceeds prefix length {key.shape[-2]}"
        )
    retained_mass = probabilities.index_select(-1, indices).sum(-1).mean()
    token_mass = probabilities.sum(dim=(0, 1, 2))
    oracle = torch.topk(token_mass, selected_count).indices
    recall = torch.isin(indices, oracle).float().mean()
    metrics = {
        "selected_tokens": selected_count,
        "retained_mass": retained_mass.item(),
        "oracle_recall": recall.item(),
    }
    if value is not None:
        groups = query.shape[1] // value.shape[1]
        selected_probabilities = probabilities.index_select(-1, indices)
        selected_values = value.repeat_interleave(groups, dim=1).float().index_select(
            -2, indices
        )
        sparse_output = (
            selected_probabilities
            / selected_probabilities.sum(-1, keepdim=True).clamp_min(1e-20)
        ) @ selected_values
        metrics["attention_output_error"] = (
            (sparse_output - reference["dense_output"]).norm()
            / reference["dense_output"].norm().clamp_min(1e-20)
        ).item()
    return reference, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llada", "sdar"), required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--prefix-token-budget", type=int, default=256)
    parser.add_argument("--max-calibration-calls", type=int, default=0)
    parser.add_argument("--query-edge", type=float, nargs="+")
    parser.add_argument("--key-edge", type=float, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    default_edges = {
        "llada": ((1.1, 1.2, 1.35, 1.5, 1.65), (1.8, 2.0, 2.26, 2.5, 2.7)),
        "sdar": ((1.2, 1.35, 1.5, 1.65, 1.8), (2.3, 2.6, 2.87, 3.1, 3.4)),
    }
    query_edges = args.query_edge or default_edges[args.model][0]
    key_edges = args.key_edge or default_edges[args.model][1]
    threshold_pairs = [
        ((-query_edge, 0.0, query_edge), (-key_edge, 0.0, key_edge))
        for query_edge in query_edges
        for key_edge in key_edges
    ]

    load_args = argparse.Namespace(
        model=args.model,
        model_path=None,
        mode="prefix",
        prefix_token_budget=args.prefix_token_budget,
        prefix_min_prefix_length=0,
        prefix_chunk_size=1024,
        query_ratio=None,
        query_dense_threshold=4,
        query_min_prefix_length=None,
        deep_only_transfer=False,
        remasking_strategy="sequential",
        moe_expert_patch=False,
        block_length=32,
        steps=None,
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
    original_compact = sparse._compact_prefix_cache
    calibration = {}
    calibration_calls = 0
    calibration_values = []
    calibration_value_index = 0

    def compact(*compact_args, **compact_kwargs):
        nonlocal calibration_values, calibration_value_index
        cache = compact_args[1]
        prefix_length = compact_args[2]
        calibration_values = [
            value[:, :, :prefix_length]
            for _, value in cache.to_legacy_cache()
        ]
        calibration_value_index = 0
        return original_compact(*compact_args, **compact_kwargs)

    def collect(query, key, *selector_args, **selector_kwargs):
        nonlocal calibration_calls, calibration_value_index
        for name, tensor in (("q", query), ("k", key)):
            values = sparse._hadamard_transform(tensor).float().flatten()
            counts[name] += torch.bincount(
                torch.bucketize(values, bins), minlength=bins.numel() + 1
            )
            sums[name] += values.sum().item()
            sumsqs[name] += values.square().sum().item()
            totals[name] += values.numel()
        production_indices = original(query, key, *selector_args, **selector_kwargs)
        if not args.max_calibration_calls or calibration_calls < args.max_calibration_calls:
            group_size = 2 if args.model == "sdar" else 1
            value_offset = calibration_value_index * group_size + group_size - 1
            value = calibration_values[value_offset]
            if value.shape[-2] != key.shape[-2]:
                raise ValueError("Attention-output calibration requires a full-prefix value")
            reference, metrics = attention_metrics(
                query, key, production_indices, value=value
            )
            production_thresholds = selector_kwargs.get("bucket_thresholds")
            candidates = [(production_thresholds, production_indices, True)]
            for thresholds in threshold_pairs:
                if thresholds == production_thresholds:
                    continue
                candidate_kwargs = dict(selector_kwargs, bucket_thresholds=thresholds)
                candidates.append(
                    (thresholds, original(query, key, *selector_args, **candidate_kwargs), False)
                )
            token_budget = (
                selector_args[0] if selector_args else selector_kwargs["token_budget"]
            )
            candidates.append(
                (
                    "direct_qk",
                    sparse._qk_prefix_indices(query, key, token_budget),
                    False,
                )
            )
            candidates.append(
                (
                    "hadamard_qk",
                    sparse._hadamard_qk_prefix_indices(query, key, token_budget),
                    False,
                )
            )
            for thresholds, indices, production in candidates:
                if indices is production_indices:
                    candidate_metrics = metrics
                else:
                    _, candidate_metrics = attention_metrics(
                        query, key, indices, value=value, reference=reference
                    )
                label = json.dumps(thresholds)
                direct_method = thresholds in {"direct_qk", "hadamard_qk"}
                entry = calibration.setdefault(
                    label,
                    {
                        "query_thresholds": (
                            None if direct_method else thresholds[0]
                        ),
                        "key_thresholds": (
                            None if direct_method else thresholds[1]
                        ),
                        "method": thresholds if direct_method else "bucketized",
                        "production": production,
                        "calls": 0,
                        "selected_tokens": 0,
                        "retained_mass": 0.0,
                        "oracle_recall": 0.0,
                        "attention_output_error": 0.0,
                    },
                )
                entry["production"] |= production
                entry["calls"] += 1
                for metric, metric_value in candidate_metrics.items():
                    entry[metric] += metric_value
            calibration_calls += 1
        calibration_value_index += 1
        return production_indices

    sparse._compact_prefix_cache = compact
    sparse._prefix_indices = collect
    if args.model == "llada":
        import src.reference.sparse.llada_patch as model_patch
    else:
        import src.reference.sparse.sdar_patch as model_patch
    model_patch._compact_prefix_cache = compact
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
    result["calibration"] = sorted(
        (
            {
                **{key: value for key, value in entry.items() if key not in {
                    "selected_tokens", "retained_mass", "oracle_recall",
                    "attention_output_error"
                }},
                "selected_tokens_mean": entry["selected_tokens"] / entry["calls"],
                "retained_mass_mean": entry["retained_mass"] / entry["calls"],
                "oracle_recall_mean": entry["oracle_recall"] / entry["calls"],
                "attention_output_error_mean": (
                    entry["attention_output_error"] / entry["calls"]
                ),
            }
            for entry in calibration.values()
        ),
        key=lambda entry: (
            entry["attention_output_error_mean"],
            -entry["retained_mass_mean"],
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
