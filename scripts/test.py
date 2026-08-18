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

from src.sparse.block_cache_sparse_dlm_patch import patch_model


DEFAULT_MODEL_PATH = "/data0/ysy/models/LLaDA2.1-mini"
DEFAULT_PROMPT = "Write a short story about history."


def parse_args():
    parser = argparse.ArgumentParser(description="LLaDA block-cache SparseDLM inference")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument(
        "--pattern",
        choices=("block_cache_sparse_dlm", "default"),
        default="block_cache_sparse_dlm",
    )
    parser.add_argument("--gen_length", type=int, default=512)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--editing_threshold", type=float, default=0.0)
    parser.add_argument("--num_to_transfer", type=int, default=1)
    parser.add_argument("--mask_id", type=int, default=156895)
    parser.add_argument("--eos_id", type=int, default=156892)
    parser.add_argument("--sparse_dlm_ratio", type=float, default=0.5)
    parser.add_argument("--sparse_dlm_top_k", type=int, default=64)
    parser.add_argument("--sparse_dlm_selection_interval", type=int, default=4)
    parser.add_argument("--sparse_dlm_dense_fallback_mask_count", type=int, default=4)
    return parser.parse_args()


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(args):
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        dtype=dtype,
    ).eval()
    if args.pattern == "block_cache_sparse_dlm":
        patch_model(
            model,
            ratio=args.sparse_dlm_ratio,
            top_k=args.sparse_dlm_top_k,
            selection_interval=args.sparse_dlm_selection_interval,
            dense_fallback_mask_count=args.sparse_dlm_dense_fallback_mask_count,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
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

    print(f"Pattern: {args.pattern}")
    print(f"Attention implementation: {args.attn_implementation}")
    synchronize()
    start = time.perf_counter()
    sequences = model.generate(
        inputs=input_ids,
        eos_early_stop=True,
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
        eos_id=args.eos_id,
    )
    synchronize()
    elapsed = time.perf_counter() - start
    generated_tokens = int(sequences.shape[-1])

    print(tokenizer.decode(sequences[0], skip_special_tokens=True))
    print(f"Time taken: {elapsed:.4f} seconds")
    print(f"Generated token num: {generated_tokens}")
    print(f"Generated token num per second: {generated_tokens / elapsed:.4f}")


if __name__ == "__main__":
    main()
