#!/usr/bin/env python3
"""End-to-end A/B benchmark for long-context sparse generation.

Run this script in two processes with SPARSE_DLM_TRITON=false/true.  Context
length means the complete prompt + generated-token window, so the 32K case
stays within checkpoints whose configured maximum is 32768 tokens.
"""

import argparse
import hashlib
import json
import math
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


MODEL_PATHS = {
    "llada": "/data0/ysy/models/LLaDA2.1-mini",
    "sdar": "/data0/ysy/models/SDAR-8B-Chat-b32",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llada", "sdar"), required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--mode",
        choices=(
            "dense",
            "query",
            "prefix",
            "query_prefix",
            "losa",
            "combined",
            "eval",
        ),
        required=True,
        help="combined enables Query Sparse, Prefix Sparse, and LoSA",
    )
    parser.add_argument("--contexts", type=int, nargs="+", default=(8192, 16384, 32768))
    parser.add_argument("--gen-length", type=int, default=64)
    parser.add_argument("--losa-active-topk", type=int, default=5)
    parser.add_argument(
        "--losa-score-mode",
        choices=("query", "key_diag"),
        default="query",
    )
    parser.add_argument("--losa-key-samples", type=int, default=32)
    parser.add_argument("--prefix-token-budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load(args):
    is_sdar = args.model == "sdar"
    dtype = torch.float16 if is_sdar else torch.bfloat16
    model_path = args.model_path or MODEL_PATHS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="sdpa",
        torch_dtype=dtype,
    ).eval()
    resolve_model_family(model, args.model)

    combined = args.mode == "combined"
    eval_mode = args.mode == "eval"
    query_sparse = args.mode in {"query", "query_prefix", "combined"} or eval_mode
    prefix_sparse = args.mode in {"prefix", "query_prefix", "combined"} or (
        eval_mode and not is_sdar
    )
    patch_model(
        model,
        model_name=args.model,
        ratio=0.5 if is_sdar else 0.7,
        top_k=64,
        selection_interval=1 if is_sdar else 4,
        query_dense_threshold=0 if is_sdar else 4,
        refresh_step=-1 if is_sdar else 2,
        selection_layer=5 if is_sdar else 1,
        deep_only_transfer=False,
        query_sparse=query_sparse,
        prefix_sparse=prefix_sparse,
        prefix_token_budget=args.prefix_token_budget,
        prefix_chunk_size=256,
        losa=args.mode in {"losa", "combined"},
        losa_active_topk=args.losa_active_topk,
        losa_score_mode=args.losa_score_mode,
        losa_key_samples=args.losa_key_samples,
        moe_expert_patch=not is_sdar,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def exact_prompt(tokenizer, length, device):
    tail = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Summarize the preceding archive."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    tail = list(tail)[-length:]
    filler = tokenizer.encode(
        "The archive records a sequence of historical events and their consequences. ",
        add_special_tokens=False,
    )
    if not filler:
        raise RuntimeError("tokenizer produced an empty filler sequence")
    prefix_length = length - len(tail)
    prefix = (filler * math.ceil(prefix_length / len(filler)))[:prefix_length]
    return torch.tensor(prefix + tail, dtype=torch.long, device=device).unsqueeze(0)


def checksum(tensor):
    data = tensor.detach().to(device="cpu", dtype=torch.int64).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:16]


def generation_kwargs(args, tokenizer, input_ids):
    is_sdar = args.model == "sdar"
    kwargs = {
        "inputs": input_ids,
        # Fixed work is required for comparable end-to-end latency. Early EOS
        # makes different sparse selectors execute different numbers of blocks.
        "eos_early_stop": False,
        "gen_length": args.gen_length,
        "block_length": 32,
        "steps": 32,
        "threshold": 1.0 if is_sdar else 0.5,
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "mask_id": tokenizer.mask_token_id if is_sdar else 156895,
        "eos_id": None if is_sdar else 156892,
    }
    if is_sdar:
        kwargs.update(remasking_strategy="sequential", eb_threshold=0.35)
    else:
        kwargs.update(editing_threshold=0.0, num_to_transfer=1)
    return kwargs


def run_once(args, model, tokenizer, context_length):
    prompt_length = context_length - args.gen_length
    if prompt_length <= 0:
        raise ValueError("every context must be greater than --gen-length")
    input_ids = exact_prompt(tokenizer, prompt_length, model.device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    sequences = model.generate(**generation_kwargs(args, tokenizer, input_ids))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "context_tokens": context_length,
        "prompt_tokens": prompt_length,
        "requested_output_tokens": args.gen_length,
        "generated_tokens": int(sequences.shape[-1]),
        "seconds": elapsed,
        "requested_tokens_per_second": args.gen_length / elapsed,
        "generated_tokens_per_second": int(sequences.shape[-1]) / elapsed,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "input_checksum": checksum(input_ids),
        "output_checksum": checksum(sequences),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    set_seed(args.seed)
    model, tokenizer = load(args)

    # Compile fixed-shape Triton kernels and initialize remote-code caches outside
    # the measured region. The A/B runs perform the same warmup.
    warmup_args = argparse.Namespace(**vars(args))
    warmup_args.gen_length = 32
    warmup_ids = exact_prompt(tokenizer, 480, model.device)
    model.generate(**generation_kwargs(warmup_args, tokenizer, warmup_ids))
    torch.cuda.synchronize()

    triton_env = os.getenv("SPARSE_DLM_TRITON", "auto")
    results = []
    print(
        json.dumps(
            {
                "model": args.model,
                "mode": args.mode,
                "triton": triton_env,
                "losa_score_mode": args.losa_score_mode,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for context_length in args.contexts:
        result = run_once(args, model, tokenizer, context_length)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    report = {
        "model": args.model,
        "mode": args.mode,
        "triton": triton_env,
        "dtype": "float16" if args.model == "sdar" else "bfloat16",
        "gen_length": args.gen_length,
        "losa_active_topk": args.losa_active_topk,
        "losa_score_mode": args.losa_score_mode,
        "losa_key_samples": args.losa_key_samples,
        "prefix_token_budget": args.prefix_token_budget,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
