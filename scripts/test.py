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
    "sdar": "/data0/ysy/models/SDAR-8B-Chat",
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
        choices=("patch", "default", "block_cache_sparse_dlm", "sdar_block_diffusion"),
        default="patch",
    )
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--editing_threshold", type=float, default=0.0)
    parser.add_argument("--num_to_transfer", type=int, default=1)
    parser.add_argument("--mask_id", type=int, default=None)
    parser.add_argument("--eos_id", type=int, default=None)
    parser.add_argument("--sparse_dlm_ratio", type=float, default=0.5)
    parser.add_argument("--sparse_dlm_top_k", type=int, default=64)
    parser.add_argument("--sparse_dlm_selection_interval", type=int, default=None)
    parser.add_argument("--sparse_dlm_dense_fallback_mask_count", type=int, default=None)
    parser.add_argument("--query_sparse", type=parse_bool, default=True)
    parser.add_argument("--prefix_sparse", type=parse_bool, default=None)
    parser.add_argument("--prefix_token_budget", type=int, default=256)
    parser.add_argument("--prefix_chunk_size", type=int, default=256)
    parser.add_argument("--losa", type=parse_bool, default=False)
    parser.add_argument("--losa_active_topk", type=int, default=5)
    parser.add_argument("--moe_expert_patch", type=parse_bool, default=True)
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
    args.sparse_dlm_selection_interval = (
        args.sparse_dlm_selection_interval or (1 if is_sdar else 4)
    )
    if args.sparse_dlm_dense_fallback_mask_count is None:
        args.sparse_dlm_dense_fallback_mask_count = 0 if is_sdar else 4
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
            dense_fallback_mask_count=args.sparse_dlm_dense_fallback_mask_count,
            query_sparse=args.query_sparse,
            prefix_sparse=args.prefix_sparse,
            prefix_token_budget=args.prefix_token_budget,
            prefix_chunk_size=args.prefix_chunk_size,
            losa=args.losa,
            losa_active_topk=args.losa_active_topk,
            moe_expert_patch=args.moe_expert_patch,
        )
        args.pattern = "patch"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    args.block_length = args.block_length or (4 if is_sdar else 32)
    args.steps = args.steps or (4 if is_sdar else 32)
    args.threshold = args.threshold if args.threshold is not None else (0.85 if is_sdar else 0.5)
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
    synchronize()
    start = time.perf_counter()
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
    if args.model == "llada":
        generation_kwargs.update(
            editing_threshold=args.editing_threshold,
            num_to_transfer=args.num_to_transfer,
        )
    sequences = model.generate(**generation_kwargs)
    synchronize()
    elapsed = time.perf_counter() - start
    generated_tokens = int(sequences.shape[-1])

    print(tokenizer.decode(sequences[0], skip_special_tokens=True))
    print(f"Time taken: {elapsed:.4f} seconds")
    print(f"Generated token num: {generated_tokens}")
    print(f"Generated token num per second: {generated_tokens / elapsed:.4f}")


if __name__ == "__main__":
    main()
