"""Measure how well SDAR intermediate layers predict final transfer positions."""

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse.sdar_patch import (
    _stop_ids,
    block_diffusion_generate,
    entropy_from_logits,
    patch_sdar_model,
    sample_with_temperature_topk_topp,
    select_transfer,
)


def overlap_metrics(predicted, actual):
    overlap = int((predicted & actual).sum().item())
    predicted_count = int(predicted.sum().item())
    actual_count = int(actual.sum().item())
    union = predicted_count + actual_count - overlap
    return {
        "predicted_count": predicted_count,
        "final_count": actual_count,
        "overlap_count": overlap,
        "precision": overlap / max(predicted_count, 1),
        "recall": overlap / max(actual_count, 1),
        "jaccard": overlap / max(union, 1),
    }


def token_prediction_metrics(
    logits, transfer, final_tokens, final_log_probs, final_probs
):
    count = int(final_tokens.numel())
    if not count:
        return {"count": 0, "top1": None, "top5": None, "kl": None}
    selected = logits[transfer].float()
    shallow_log_probs = selected.log_softmax(dim=-1)
    top1 = selected.argmax(dim=-1)
    top5 = selected.topk(min(5, selected.shape[-1]), dim=-1).indices
    return {
        "count": count,
        "top1": float((top1 == final_tokens).float().mean().item()),
        "top5": float((top5 == final_tokens[:, None]).any(dim=-1).float().mean().item()),
        "kl": float((final_probs * (final_log_probs - shallow_log_probs)).sum(-1).mean().item()),
    }


def sample_layer_summary(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        for layer in record["layers"]:
            layer_id = layer["layer"]
            token = layer["token_prediction"]
            for key in ("top1", "top5", "kl"):
                if token[key] is not None:
                    grouped[layer_id][key].append(token[key])
            for method, metrics in layer.items():
                if method.startswith("candidate@"):
                    grouped[layer_id][f"{method}_recall"].append(metrics["recall"])
    return [
        {
            "layer": layer,
            **{
                key: sum(values) / len(values)
                for key, values in metrics.items()
                if values
            },
        }
        for layer, metrics in sorted(grouped.items())
    ]


def summarize(values):
    keys = (
        "predicted_count",
        "final_count",
        "overlap_count",
        "precision",
        "recall",
        "jaccard",
    )
    return {
        "steps": len(values),
        **{
            f"{key}_mean": sum(value[key] for value in values) / len(values)
            for key in keys
        },
    }


def positions(mask, offset):
    return (torch.where(mask[0])[0] + offset).detach().cpu().tolist()


def prediction_scores(logits, args):
    _, scores = sample_with_temperature_topk_topp(
        logits,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    entropy = (
        entropy_from_logits(logits, args.temperature, args.top_k, args.top_p)
        if args.strategy == "entropy_bounded"
        else None
    )
    return scores, entropy


def layer_logits(model, outputs, layer, num_layers):
    hidden_states = outputs.hidden_states[layer]
    if layer < num_layers:
        hidden_states = model.model.norm(hidden_states)
    return model.lm_head(hidden_states)


def candidate_mask(mask, scores, ratio):
    count = min(math.ceil(mask.sum().item() * ratio), int(mask.sum().item()))
    selected = torch.zeros_like(mask)
    if count:
        ranked = torch.where(mask, scores, -torch.inf).topk(count, dim=1).indices
        selected.scatter_(1, ranked, True)
    return selected


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run dense SDAR generation and compare each layer's predicted "
            "positions with the final layer's transfer positions."
        )
    )
    parser.add_argument(
        "--model_path", default="/data0/ysy/models/SDAR-8B-Chat-b32"
    )
    parser.add_argument(
        "--prompt", default="Write a Python function that sorts a list of integers."
    )
    parser.add_argument("--gen_length", type=int, default=16)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--mask_id", type=int, default=151669)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--strategy",
        choices=(
            "low_confidence_dynamic",
            "low_confidence_static",
            "sequential",
            "entropy_bounded",
        ),
        default="sequential",
    )
    parser.add_argument("--eb_threshold", type=float, default=0.35)
    parser.add_argument(
        "--layer_candidate_ratio", type=float, nargs="+", default=(0.5,)
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="1-based decoder layers to test; defaults to all layers.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples_jsonl")
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eos_early_stop", action="store_true")
    parser.add_argument("--output_dir", default="experiments/layer_overlap_results/sdar")
    parser.add_argument("--plot_overlap_stats", action="store_true")
    parser.add_argument("--no_exact", action="store_true")
    return parser.parse_args()


def plot_results(summary, step_records, output_dir):
    import matplotlib.pyplot as plt

    methods = sorted({row["method"] for row in summary})
    fig, ax = plt.subplots(figsize=(11, 5))
    for method in methods:
        rows = [row for row in summary if row["method"] == method]
        ax.plot(
            [row["layer"] for row in rows],
            [row["recall_mean"] for row in rows],
            marker="o",
            label=method,
        )
    ax.set(xlabel="Layer", ylabel="Mean recall", title="Layer-wise position overlap")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "overlap_summary.png", dpi=200)
    plt.close(fig)

    layers = sorted({row["layer"] for row in summary})
    fig, axes = plt.subplots(
        len(methods), 1, figsize=(12, max(4, 3 * len(methods))), squeeze=False
    )
    for ax, method in zip(axes[:, 0], methods):
        heatmap = [
            [
                next(
                    layer[method]["recall"]
                    for layer in step["layers"]
                    if layer["layer"] == layer_id
                )
                for step in step_records
            ]
            for layer_id in layers
        ]
        image = ax.imshow(heatmap, aspect="auto", origin="lower", vmin=0, vmax=1)
        ax.set(title=method, xlabel="Denoising call", ylabel="Layer")
        ax.set_yticks(range(len(layers)), layers)
        fig.colorbar(image, ax=ax, label="Recall")
    fig.tight_layout()
    fig.savefig(output_dir / "overlap_heatmaps.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    if args.temperature > 0:
        raise ValueError("This diagnostic requires greedy temperature=0 sampling")
    if args.gen_length <= 0:
        raise ValueError("gen_length must be positive")
    if not 1 <= args.steps <= args.block_length:
        raise ValueError("steps must satisfy 1 <= steps <= block_length")
    if any(not 0 < ratio <= 1 for ratio in args.layer_candidate_ratio):
        raise ValueError("layer_candidate_ratio values must be in (0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device).eval()
    patch_sdar_model(model, query_sparse=False, prefix_sparse=False)
    num_layers = len(model.model.layers)
    layers = args.layers or list(range(1, num_layers + 1))
    if any(layer < 1 or layer > num_layers for layer in layers):
        raise ValueError(f"layers must be between 1 and {num_layers}")
    layers = sorted(set(layers))
    samples = [(0, None, args.prompt, False)]
    if args.samples_jsonl:
        samples = []
        with Path(args.samples_jsonl).open() as handle:
            for line in handle:
                row = json.loads(line)
                generation = next(iter(row["arguments"].values()))
                samples.append(
                    (row["doc_id"], row["doc"].get("task_id"), generation["arg_0"], True)
                )
        samples = samples[args.sample_start :]
        if args.limit is not None:
            samples = samples[: args.limit]
    if not samples:
        raise ValueError("No prompts selected")

    step_records = []
    aggregate = defaultdict(list)
    sample_summaries = []
    generated_samples = []
    current_sample = None

    @torch.no_grad()
    def measure_step(
        model,
        block_tokens,
        attention_mask,
        position_ids,
        past_key_values,
        block_start,
        step,
        minimum,
        **kwargs,
    ):
        outputs = model(
            block_tokens,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            store_kv=False,
            output_hidden_states=True,
        )
        mask = block_tokens == args.mask_id
        final_scores, final_entropy = prediction_scores(outputs.logits, args)
        final_transfer = select_transfer(
            mask,
            torch.where(mask, final_scores, -torch.inf),
            minimum,
            args.strategy,
            args.threshold,
            entropy=final_entropy,
            entropy_budget=args.eb_threshold,
        )
        final_selected = outputs.logits[final_transfer].float()
        final_log_probs = final_selected.log_softmax(dim=-1)
        final_probs = final_log_probs.exp()
        final_tokens = final_selected.argmax(dim=-1)
        record = {
            "sample": current_sample[0],
            "task_id": current_sample[1],
            "call": len(step_records),
            "block": block_start // args.block_length,
            "step": step,
            "mask_count": int(mask.sum().item()),
            "minimum_transfer": minimum,
            "final_positions": positions(final_transfer, block_start),
            "layers": [],
        }

        for layer in layers:
            logits = layer_logits(model, outputs, layer, num_layers)
            layer_scores, layer_entropy = prediction_scores(logits, args)
            layer_record = {
                "layer": layer,
                "token_prediction": token_prediction_metrics(
                    logits,
                    final_transfer,
                    final_tokens,
                    final_log_probs,
                    final_probs,
                ),
            }
            if not args.no_exact:
                layer_transfer = select_transfer(
                    mask,
                    torch.where(mask, layer_scores, -torch.inf),
                    minimum,
                    args.strategy,
                    args.threshold,
                    entropy=layer_entropy,
                    entropy_budget=args.eb_threshold,
                )
                exact = overlap_metrics(layer_transfer, final_transfer)
                exact["positions"] = positions(layer_transfer, block_start)
                aggregate[(layer, "exact")].append(exact)
                layer_record["exact"] = exact

            for ratio in args.layer_candidate_ratio:
                method = f"candidate@{ratio:g}"
                ranking_scores = (
                    -layer_entropy
                    if args.strategy == "entropy_bounded"
                    else layer_scores
                )
                selected = candidate_mask(mask, ranking_scores, ratio)
                metrics = overlap_metrics(selected, final_transfer)
                metrics["positions"] = positions(selected, block_start)
                aggregate[(layer, method)].append(metrics)
                layer_record[method] = metrics
            record["layers"].append(layer_record)

        step_records.append(record)
        return outputs.logits, None

    prompt_tokens = []
    for current_sample in samples:
        sample_start = len(step_records)
        if current_sample[3]:
            text = current_sample[2]
        else:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": current_sample[2]}],
                add_generation_prompt=True,
                tokenize=False,
            )
        input_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(model.device)
        prompt_tokens.append(int(input_ids.shape[1]))
        output = block_diffusion_generate(
            model,
            prompt={"input_ids": input_ids},
            mask_id=args.mask_id,
            gen_length=args.gen_length,
            block_length=args.block_length,
            denoising_steps=args.steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            remasking_strategy=args.strategy,
            confidence_threshold=args.threshold,
            eb_threshold=args.eb_threshold,
            stopping_criteria_idx=(
                _stop_ids(model, None) if args.eos_early_stop else None
            ),
            denoise_fn=measure_step,
        )
        generated = output[:, input_ids.shape[1] : input_ids.shape[1] + args.gen_length]
        rows = sample_layer_summary(step_records[sample_start:])
        for row in rows:
            row.update(sample=current_sample[0], task_id=current_sample[1])
            sample_summaries.append(row)
        generated_samples.append(
            {
                "sample": current_sample[0],
                "task_id": current_sample[1],
                "text": tokenizer.decode(generated[0], skip_special_tokens=True),
            }
        )
    summary = []
    for (layer, method), values in sorted(aggregate.items()):
        summary.append({"layer": layer, "method": method, **summarize(values)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(args),
        "prompt_tokens": prompt_tokens,
        "generated_text": generated_samples[0]["text"],
        "generated_samples": generated_samples,
        "summary": summary,
        "sample_summary": sample_summaries,
        "steps": step_records,
    }
    (output_dir / "overlap_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    with (output_dir / "overlap_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "sample_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sample_summaries[0].keys())
        writer.writeheader()
        writer.writerows(sample_summaries)
    if args.plot_overlap_stats:
        plot_results(summary, step_records, output_dir)

    print(f"Recorded {len(step_records)} denoising calls in {output_dir}")
    print(f"{'layer':>5} {'method':>16} {'recall':>9} {'precision':>10} {'jaccard':>9}")
    for row in summary:
        print(
            f"{row['layer']:5d} {row['method']:>16} "
            f"{row['recall_mean']:9.4f} {row['precision_mean']:10.4f} "
            f"{row['jaccard_mean']:9.4f}"
        )


if __name__ == "__main__":
    main()
