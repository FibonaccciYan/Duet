"""Measure how well LLaDA intermediate layers predict final transfer positions."""

import argparse
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

from scripts.test_sdar_layer_overlap import (
    candidate_mask,
    overlap_metrics,
    plot_results,
    positions,
    summarize,
)
from src.sparse.llada_patch import _transfer_tokens
from src.sparse.sparse_ops import _sample_with_confidence


def block_attention_mask(total_length, block_length, dtype, device):
    blocks = total_length // block_length
    allowed = torch.tril(torch.ones(blocks, blocks, dtype=torch.bool, device=device))
    allowed = allowed.repeat_interleave(block_length, 0).repeat_interleave(
        block_length, 1
    )
    return torch.zeros((1, 1, total_length, total_length), dtype=dtype, device=device).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )


def prediction_scores(model, logits, args):
    _, confidence = _sample_with_confidence(
        model, logits, args.temperature, args.top_p, args.top_k
    )
    return confidence


def layer_logits(model, outputs, layer, num_layers):
    hidden_states = outputs.hidden_states[layer]
    # if layer < num_layers:
    #     hidden_states = model.model.norm(hidden_states)
    return model.lm_head(hidden_states).float()


def selector_mask(mask, scores, minimum, threshold):
    confidence = torch.where(mask, scores, -torch.inf)
    high_confidence = (confidence[0] > threshold) & mask[0]
    transfer = torch.zeros_like(mask)
    minimum = min(int(minimum), int(mask.sum().item()))
    if int(high_confidence.sum().item()) >= minimum:
        transfer[0] = high_confidence
    elif minimum:
        indices = torch.topk(confidence[0], k=minimum).indices
        transfer[0, indices] = True
    return transfer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LLaDA intermediate-layer transfer positions with final logits."
    )
    parser.add_argument("--model_path", default="/data0/ysy/models/LLaDA2.1-mini")
    parser.add_argument(
        "--prompt", default="Write a Python function that sorts a list of integers."
    )
    parser.add_argument("--gen_length", type=int, default=32)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--mask_id", type=int, default=156895)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--editing_threshold", type=float, default=0.0)
    parser.add_argument("--num_to_transfer", type=int, default=1)
    parser.add_argument("--max_post_steps", type=int, default=16)
    parser.add_argument("--query_ratio", type=float, default=0.5)
    parser.add_argument("--layer_candidate_ratio", type=float, nargs="+", default=(0.5,))
    parser.add_argument("--layers", type=int, nargs="+", help="1-based layers; defaults to all.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="layer_overlap_results/llada_block32")
    parser.add_argument("--plot_overlap_stats", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.temperature != 0:
        raise ValueError("This diagnostic requires greedy temperature=0 sampling")
    if args.gen_length <= 0 or args.block_length <= 0 or args.steps <= 0:
        raise ValueError("gen_length, block_length, and steps must be positive")
    if args.num_to_transfer <= 0 or args.max_post_steps < 0:
        raise ValueError("num_to_transfer must be positive and max_post_steps non-negative")
    if not 0 < args.query_ratio <= 1:
        raise ValueError("query_ratio must be in (0, 1]")
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
    num_layers = len(model.model.layers)
    layers = sorted(set(args.layers or range(1, num_layers + 1)))
    if any(layer < 1 or layer > num_layers for layer in layers):
        raise ValueError(f"layers must be between 1 and {num_layers}")

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)
    prompt_length = input_ids.shape[1]
    blocks = (prompt_length + args.gen_length + args.block_length - 1) // args.block_length
    total_length = blocks * args.block_length
    attention_mask = block_attention_mask(total_length, args.block_length, dtype, model.device)
    position_ids = torch.arange(total_length, device=model.device).unsqueeze(0)
    tokens = torch.full(
        (1, total_length), args.mask_id, dtype=torch.long, device=model.device
    )
    tokens[:, :prompt_length] = input_ids

    records = []
    aggregate = defaultdict(list)
    max_iterations = max(args.steps, args.block_length) + args.max_post_steps

    with torch.no_grad():
        for block in range(prompt_length // args.block_length, blocks):
            block_start = block * args.block_length
            block_end = block_start + args.block_length
            prompt_mask = torch.zeros(
                args.block_length, dtype=torch.bool, device=model.device
            )
            if block_start < prompt_length:
                prompt_mask[: prompt_length - block_start] = True
            post_steps = 0

            for step in range(max_iterations):
                current = tokens[:, :block_end]
                mask = current[:, -args.block_length:] == args.mask_id
                outputs = model(
                    current,
                    attention_mask=attention_mask[:, :, :block_end, :block_end],
                    position_ids=position_ids[:, :block_end],
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                final_logits = outputs.logits[:, -args.block_length:].float()
                final_scores = prediction_scores(model, final_logits, args)

                if mask.any():
                    final_transfer = selector_mask(
                        mask,
                        final_scores,
                        args.num_to_transfer,
                        args.threshold,
                    )
                    query_minimum = math.ceil(
                        int(mask.sum().item()) * args.query_ratio
                    )
                    record = {
                        "call": len(records),
                        "block": block,
                        "step": step,
                        "mask_count": int(mask.sum().item()),
                        "minimum_transfer": args.num_to_transfer,
                        "minimum_query": query_minimum,
                        "final_positions": positions(final_transfer, block_start),
                        "layers": [],
                    }
                    for layer in layers:
                        logits = layer_logits(model, outputs, layer, num_layers)
                        logits = logits[:, -args.block_length:]
                        scores = prediction_scores(model, logits, args)
                        predicted = selector_mask(
                            mask,
                            scores,
                            query_minimum,
                            args.threshold,
                        )
                        exact = overlap_metrics(predicted, final_transfer)
                        exact["positions"] = positions(predicted, block_start)
                        aggregate[(layer, "exact")].append(exact)
                        layer_record = {"layer": layer, "exact": exact}
                        for ratio in args.layer_candidate_ratio:
                            method = f"candidate@{ratio:g}"
                            candidate = candidate_mask(mask, scores, ratio)
                            metrics = overlap_metrics(candidate, final_transfer)
                            metrics["positions"] = positions(candidate, block_start)
                            aggregate[(layer, method)].append(metrics)
                            layer_record[method] = metrics
                        record["layers"].append(layer_record)
                    records.append(record)

                old_block = current[:, -args.block_length:].clone()
                updated, transfer = _transfer_tokens(
                    model,
                    current[:, -args.block_length:],
                    old_block,
                    prompt_mask,
                    mask,
                    final_logits,
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    args.threshold,
                    args.editing_threshold,
                    args.num_to_transfer,
                )
                tokens[:, block_start:block_end] = updated
                if not mask.any():
                    post_steps += 1
                    if post_steps > args.max_post_steps:
                        break
                elif not transfer.any():
                    raise RuntimeError("Masked block made no transfer progress")

            if (tokens[:, block_start:block_end] == args.mask_id).any():
                raise RuntimeError(f"Block {block} still contains mask tokens")

    summary = [
        {"layer": layer, "method": method, **summarize(values)}
        for (layer, method), values in sorted(aggregate.items())
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": vars(args),
        "prompt_tokens": int(prompt_length),
        "generated_text": tokenizer.decode(
            tokens[0, prompt_length : prompt_length + args.gen_length], skip_special_tokens=True
        ),
        "summary": summary,
        "steps": records,
    }
    (output_dir / "overlap_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    with (output_dir / "overlap_summary.csv").open("w", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    if args.plot_overlap_stats:
        plot_results(summary, records, output_dir)

    print(f"Recorded {len(records)} denoising calls in {output_dir}")


if __name__ == "__main__":
    main()
