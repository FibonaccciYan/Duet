import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse import patch_model, resolve_model_family


DEFAULT_MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}
DEFAULT_PROMPT = "Write a short story about history."


def parse_bool(value):
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA/SDAR block-diffusion inference")
    parser.add_argument("--model", choices=("llada", "sdar"), default="llada")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument(
        "--pattern",
        choices=("patch", "default"),
        default="patch",
    )
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--remasking_strategy",
        choices=(
            "low_confidence_dynamic",
            "low_confidence_static",
            "sequential",
            "entropy_bounded",
        ),
        default="sequential",
        help="SDAR transfer/query strategy; ignored by LLaDA",
    )
    parser.add_argument(
        "--eb_threshold",
        type=float,
        default=0.35,
        help="SDAR entropy budget; ignored by LLaDA",
    )
    parser.add_argument("--editing_threshold", type=float, default=0.5)
    parser.add_argument("--num_to_transfer", type=int, default=1)
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--eos_id", type=int, default=None)
    parser.add_argument("--sparse_dlm_ratio", type=float, default=None)
    parser.add_argument("--sparse_dlm_top_k", type=int, default=None)
    parser.add_argument("--sparse_dlm_selection_interval", type=int, default=None)
    parser.add_argument("--query_dense_threshold", type=int, default=None)
    parser.add_argument("--query_min_prefix_length", type=int, default=None)
    parser.add_argument("--sparse_dlm_refresh_step", type=int, default=None)
    parser.add_argument(
        "--sparse_dlm_selection_layer",
        type=int,
        default=None,
        help="zero-based decoder layer after which Query positions are selected",
    )
    parser.add_argument("--sparse_dlm_deep_only_transfer", type=parse_bool, default=None)
    parser.add_argument("--query_sparse", type=parse_bool, default=True)
    parser.add_argument("--query_losa_union", type=parse_bool, default=False)
    parser.add_argument("--prefix_sparse", type=parse_bool, default=None)
    parser.add_argument("--prefix_min_prefix_length", type=int, default=None)
    parser.add_argument("--prefix_token_budget", type=int, default=256)
    parser.add_argument("--prefix_chunk_size", type=int, default=None)
    parser.add_argument("--prefix_share_layer_pairs", type=parse_bool, default=False)
    parser.add_argument("--losa", type=parse_bool, default=False)
    parser.add_argument("--losa_active_topk", type=int, default=5)
    parser.add_argument(
        "--losa_score_mode",
        choices=("query", "key_diag"),
        default="query",
    )
    parser.add_argument("--losa_key_samples", type=int, default=32)
    parser.add_argument("--moe_expert_patch", type=parse_bool, default=None)
    parser.add_argument("--profile_output", default=None)
    parser.add_argument("--profile_trace", type=parse_bool, default=False)
    parser.add_argument("--warmup_runs", type=int, default=0)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args):
    args.model_path = args.model_path or DEFAULT_MODEL_PATHS[args.model]
    if args.model == "sdar" and args.pattern == "default":
        raise ValueError(
            "SDAR requires --pattern patch; use --query_sparse false for its dense path"
        )
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
    ).eval()
    resolve_model_family(model, args.model)
    is_sdar = args.model == "sdar"
    args.sparse_dlm_refresh_step = (
        args.sparse_dlm_refresh_step
        if args.sparse_dlm_refresh_step is not None
        else (-1 if is_sdar else 2)
    )
    args.moe_expert_patch = (
        args.moe_expert_patch if args.moe_expert_patch is not None else not is_sdar
    )
    args.sparse_dlm_selection_interval = (
        args.sparse_dlm_selection_interval or (1 if is_sdar else 4)
    )
    if args.query_dense_threshold is None:
        args.query_dense_threshold = 4
    args.prefix_sparse = (
        not is_sdar if args.prefix_sparse is None else args.prefix_sparse
    )
    if args.pattern != "default":
        patch_model(
            model,
            model_name=args.model,
            ratio=args.sparse_dlm_ratio,
            top_k=args.sparse_dlm_top_k,
            selection_interval=args.sparse_dlm_selection_interval,
            query_dense_threshold=args.query_dense_threshold,
            query_min_prefix_length=args.query_min_prefix_length,
            refresh_step=args.sparse_dlm_refresh_step,
            selection_layer=args.sparse_dlm_selection_layer,
            deep_only_transfer=args.sparse_dlm_deep_only_transfer,
            query_sparse=args.query_sparse,
            prefix_sparse=args.prefix_sparse,
            prefix_min_prefix_length=args.prefix_min_prefix_length,
            prefix_token_budget=args.prefix_token_budget,
            prefix_chunk_size=args.prefix_chunk_size,
            prefix_share_layer_pairs=args.prefix_share_layer_pairs,
            losa=args.losa,
            losa_active_topk=args.losa_active_topk,
            losa_score_mode=args.losa_score_mode,
            losa_key_samples=args.losa_key_samples,
            query_losa_union=args.query_losa_union,
            moe_expert_patch=args.moe_expert_patch,
        )
        args.pattern = "patch"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    args.block_length = args.block_length or 32
    args.steps = args.steps or 32
    args.threshold = args.threshold if args.threshold is not None else (
        0.95
        if is_sdar and args.remasking_strategy == "low_confidence_dynamic"
        else 0.85
        if is_sdar
        else 0.7
    )
    args.mask_id = args.mask_id if args.mask_id is not None else (
        tokenizer.mask_token_id if is_sdar else 156895
    )
    if args.eos_id is None and not is_sdar:
        eos_id = getattr(model.generation_config, "eos_token_id", None)
        args.eos_id = int(eos_id[0] if isinstance(eos_id, list) else eos_id or 156892)
    return model, tokenizer


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup_runs must be non-negative")
    set_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"Model: {args.model}; pattern: {args.pattern}")
    print(f"Attention implementation: {args.attn_implementation}")
    if args.pattern == "patch":
        print(
            f"Query sparse: {args.query_sparse}; prefix sparse: {args.prefix_sparse}; "
            f"prefix token budget: {args.prefix_token_budget}; "
            f"LoSA: {args.losa}; MoE expert patch: {args.moe_expert_patch}"
        )
    generation_kwargs = {
        "inputs": input_ids,
        "eos_early_stop": True,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "steps": args.steps,
        "threshold": args.threshold,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "mask_id": args.mask_id,
        "eos_id": args.eos_id,
    }
    if args.model == "sdar":
        generation_kwargs.update(
            remasking_strategy=args.remasking_strategy,
            eb_threshold=args.eb_threshold,
        )
    else:
        generation_kwargs.update(
            editing_threshold=args.editing_threshold,
            num_to_transfer=args.num_to_transfer,
        )

    for _ in range(args.warmup_runs):
        model.generate(**generation_kwargs)
    synchronize()

    profiler = None
    if args.profile_output:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
        )

    start = time.perf_counter()
    if profiler is None:
        sequences = model.generate(**generation_kwargs)
    else:
        with profiler:
            sequences = model.generate(**generation_kwargs)
    synchronize()
    elapsed = time.perf_counter() - start
    generated_tokens = int(sequences.shape[-1])

    print(tokenizer.decode(sequences[0], skip_special_tokens=True))
    print(f"Time taken: {elapsed:.4f} seconds")
    print(f"Generated token num: {generated_tokens}")
    print(f"Generated token num per second: {generated_tokens / elapsed:.4f}")
    if profiler is not None:
        output_path = Path(args.profile_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        averages = profiler.key_averages()
        shaped_averages = profiler.key_averages(group_by_input_shape=True)
        report = "CUDA time (aggregate)\n" + averages.table(
            sort_by="self_cuda_time_total", row_limit=100
        )
        report += "\n\nCPU time (aggregate)\n" + averages.table(
            sort_by="self_cpu_time_total", row_limit=100
        )
        report += "\n\nCUDA time by input shape\n" + shaped_averages.table(
            sort_by="self_cuda_time_total", row_limit=100
        )
        output_path.write_text(report)
        if args.profile_trace:
            profiler.export_chrome_trace(str(output_path.with_suffix(".json")))
        print(f"Profiler report: {output_path}")


if __name__ == "__main__":
    main()
