import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from local_demo_sparse_patch import patch_model as patch_llada_demo_sparse


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark LLaDA vanilla vs Sparse-KV throughput.")
    parser.add_argument("--model_path", type=str, default="/data0/ysy/models/LLaDA2.1-mini")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Write a short story about history.")
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--editing_threshold", type=float, default=0.0)
    parser.add_argument("--num_to_transfer", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--mask_id", type=int, default=156895)
    parser.add_argument("--sparse_ratio", type=float, default=0.5)
    parser.add_argument("--dense_fallback_mask_count", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager",
        dtype=torch.bfloat16,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def run_generate(model, input_ids, args):
    return model.generate(
        inputs=input_ids,
        eos_early_stop=False,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=args.threshold,
        editing_threshold=args.editing_threshold,
        num_to_transfer=args.num_to_transfer,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_id=args.mask_id,
    )


def benchmark_mode(mode, args):
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    if mode == "sparse_kv":
        patch_llada_demo_sparse(
            model,
            ratio=args.sparse_ratio,
            block_length=args.block_length,
            mask_id=args.mask_id,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            sparse_mode="kv",
            threshold=args.threshold,
            editing_threshold=args.editing_threshold,
            num_to_transfer=args.num_to_transfer,
            dense_fallback_mask_count=args.dense_fallback_mask_count,
        )

    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )

    for _ in range(args.warmup):
        with torch.no_grad():
            _ = run_generate(model, input_ids, args)
        synchronize()

    times = []
    actual_output_tokens = []
    samples = []
    for repeat_idx in range(args.repeats):
        synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            sequences = run_generate(model, input_ids, args)
        synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        actual_output_tokens.append(int(sequences.shape[-1]))
        if repeat_idx == 0:
            samples.append(tokenizer.decode(sequences[0], skip_special_tokens=True))

    result = {
        "mode": mode,
        "times_s": times,
        "mean_time_s": statistics.mean(times),
        "median_time_s": statistics.median(times),
        "min_time_s": min(times),
        "requested_gen_length": args.gen_length,
        "requested_tokens_per_s_mean": args.gen_length / statistics.mean(times),
        "requested_tokens_per_s_median": args.gen_length / statistics.median(times),
        "actual_output_tokens": actual_output_tokens,
        "actual_tokens_per_s_mean": statistics.mean(actual_output_tokens) / statistics.mean(times),
        "sample_output": samples[0] if samples else "",
    }

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def write_report(output_dir, args, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "results": results,
    }
    (output_dir / "throughput_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )

    by_mode = {item["mode"]: item for item in results}
    vanilla = by_mode["vanilla"]
    sparse = by_mode["sparse_kv"]
    speedup = (
        vanilla["median_time_s"] / sparse["median_time_s"]
        if sparse["median_time_s"] > 0
        else float("nan")
    )

    lines = [
        "# LLaDA Sparse-KV 0.5 Throughput Benchmark",
        "",
        "## Config",
        "",
        f"- model_path: `{args.model_path}`",
        f"- gen_length: `{args.gen_length}`",
        f"- block_length: `{args.block_length}`",
        f"- steps: `{args.steps}`",
        f"- sparse mode: `kv`",
        f"- sparse ratio: `{args.sparse_ratio}`",
        f"- dense_fallback_mask_count: `{args.dense_fallback_mask_count}`",
        f"- repeats: `{args.repeats}`",
        f"- warmup: `{args.warmup}`",
        f"- CUDA_VISIBLE_DEVICES: `{os.environ.get('CUDA_VISIBLE_DEVICES')}`",
        "",
        "## Results",
        "",
        "| mode | median time s | mean time s | requested tok/s median | requested tok/s mean | actual output tokens |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in results:
        lines.append(
            "| "
            f"{item['mode']} | "
            f"{item['median_time_s']:.4f} | "
            f"{item['mean_time_s']:.4f} | "
            f"{item['requested_tokens_per_s_median']:.2f} | "
            f"{item['requested_tokens_per_s_mean']:.2f} | "
            f"{item['actual_output_tokens']} |"
        )
    lines.extend(
        [
            "",
            f"Sparse-KV median-time speedup vs vanilla: `{speedup:.4f}x`",
            "",
            "## Sample Outputs",
            "",
            "### vanilla",
            "",
            vanilla["sample_output"],
            "",
            "### sparse_kv",
            "",
            sparse["sample_output"],
            "",
        ]
    )
    (output_dir / "throughput_report.md").write_text("\n".join(lines))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing benchmark artifacts to: {output_dir}")

    results = []
    for mode in ("vanilla", "sparse_kv"):
        print(f"Running mode={mode}")
        result = benchmark_mode(mode, args)
        results.append(result)
        print(
            f"mode={mode} median_time_s={result['median_time_s']:.4f} "
            f"requested_tok_s={result['requested_tokens_per_s_median']:.2f}"
        )

    write_report(output_dir, args, results)
    print(f"Saved: {output_dir / 'throughput_results.json'}")
    print(f"Saved: {output_dir / 'throughput_report.md'}")


if __name__ == "__main__":
    main()
