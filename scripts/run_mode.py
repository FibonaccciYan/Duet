#!/usr/bin/env python3
"""Run one dense or paper-faithful LoSA generation path for LLaDA/SDAR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_RUNTIME_PREFIX = "/data0/ysy/sparse"
DEFAULT_PROMPT = (
    "Give one concise sentence about diffusion language models. "
    "This prompt is intentionally long enough to cross the first thirty two "
    "token block in the smoke test, so prefix attention and LoSA cache reuse "
    "are actually exercised."
)

os.environ.setdefault("HF_HOME", str(WORKSPACE_ROOT / ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", str(WORKSPACE_ROOT / ".hf_cache" / "datasets"))
os.environ.setdefault("HF_MODULES_CACHE", str(WORKSPACE_ROOT / ".hf_cache" / "modules"))

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.runtime import load_runtime
from src.losa import set_seed


def assert_no_forbidden_sparse_runtime(stage: str) -> None:
    bad_paths = [
        item
        for item in sys.path
        if isinstance(item, str) and item.startswith(FORBIDDEN_RUNTIME_PREFIX)
        and not item.startswith(str(WORKSPACE_ROOT))
    ]
    bad_modules = []
    for name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if (
            isinstance(module_file, str)
            and module_file.startswith(FORBIDDEN_RUNTIME_PREFIX)
            and not module_file.startswith(str(WORKSPACE_ROOT))
        ):
            bad_modules.append((name, module_file))
    if bad_paths or bad_modules:
        raise RuntimeError(
            "Forbidden ysy sparse runtime dependency detected:\n"
            + json.dumps(
                {
                    "stage": stage,
                    "forbidden_prefix": FORBIDDEN_RUNTIME_PREFIX,
                    "sys_path": bad_paths,
                    "modules": bad_modules[:20],
                },
                indent=2,
                ensure_ascii=False,
            )
        )


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("llada", "sdar"), required=True)
    parser.add_argument(
        "--mode",
        choices=("dense", "losa", "losa_v2", "focus", "focus_v2"),
        required=True,
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--attn_implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--gen_length", type=int, default=32)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--editing_threshold", type=float, default=0.9)
    parser.add_argument("--max_post_steps", type=int, default=16)
    parser.add_argument("--minimal_topk", type=int, default=1)
    parser.add_argument("--eos_early_stop", type=parse_bool, default=True)
    parser.add_argument("--losa_page_size", type=int, default=16)
    parser.add_argument("--losa_token_budget", type=int, default=256)
    parser.add_argument("--losa_active_topk", type=int, default=5)
    parser.add_argument(
        "--losa_gqa_mode",
        choices=("per_query_head", "group_mean", "group_max_score"),
        default="per_query_head",
    )
    parser.add_argument(
        "--losa_backend",
        choices=("torch", "triton", "auto"),
        default="auto",
    )
    parser.add_argument("--losa_trace_detail", type=parse_bool, default=False)
    parser.add_argument("--focus_alpha", type=float, default=1.5)
    parser.add_argument("--moe_expert_patch", type=parse_bool, default=False)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def tokenize_chat(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    encoded = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        except Exception:
            encoded = None
    if encoded is None:
        encoded = tokenizer(prompt, return_tensors="pt")
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if not torch.is_tensor(encoded):
        encoded = torch.tensor(encoded)
    return encoded

def token_checksum(tokens: torch.Tensor) -> str:
    data = json.dumps(tokens.detach().cpu().tolist(), separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()[:16]


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    assert_no_forbidden_sparse_runtime("startup")
    set_seed(args.seed)

    runtime_kwargs = dict(
        family=args.family,
        model_path=args.model_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    if args.mode in {"dense", "losa", "losa_v2", "focus_v2"}:
        runtime_kwargs["moe_expert_patch"] = args.moe_expert_patch
    if args.mode in {"losa", "losa_v2"}:
        runtime_kwargs.update(
            losa_page_size=args.losa_page_size,
            losa_token_budget=args.losa_token_budget,
            losa_active_topk=args.losa_active_topk,
            losa_gqa_mode=args.losa_gqa_mode,
            losa_backend=args.losa_backend,
            losa_trace_detail=args.losa_trace_detail,
        )
    elif args.mode in {"focus", "focus_v2"}:
        runtime_kwargs.update(alpha=args.focus_alpha)
    runtime = load_runtime(args.mode, **runtime_kwargs)
    model, tokenizer = runtime.load()
    assert_no_forbidden_sparse_runtime("after_model_load")

    input_ids = tokenize_chat(tokenizer, args.prompt).to(model.device)
    threshold = args.threshold
    if threshold is None:
        threshold = 0.85 if args.family == "sdar" else 0.95
    mask_id = (tokenizer.mask_token_id or 151669) if args.family == "sdar" else 156895
    eos_id = None
    if args.family == "llada":
        eos = getattr(model.generation_config, "eos_token_id", None)
        eos_id = int(eos[0] if isinstance(eos, list) else eos or 156892)

    generation_kwargs = dict(
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=threshold,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mask_id=mask_id,
        eos_id=eos_id,
        eos_early_stop=args.eos_early_stop,
    )
    if args.family == "sdar":
        generation_kwargs.update(
            remasking_strategy=(
                "low_confidence_dynamic" if args.mode in {"focus", "focus_v2"} else "sequential"
            )
        )
    else:
        generation_kwargs.update(
            editing_threshold=args.editing_threshold,
            max_post_steps=args.max_post_steps,
            minimal_topk=args.minimal_topk,
            num_to_transfer=1,
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        generation = runtime.generate(input_ids, **generation_kwargs)
    assert_no_forbidden_sparse_runtime("after_generation")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = generation.tokens
    decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else None
    )

    result = {
        "family": args.family,
        "mode": args.mode,
        "model_path": args.model_path,
        "runtime": type(runtime).__name__,
        "generation": {
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "steps": args.steps,
            "threshold": threshold,
            "editing_threshold": args.editing_threshold,
            "max_post_steps": args.max_post_steps,
            "minimal_topk": args.minimal_topk,
            "temperature": args.temperature,
            "eos_early_stop": bool(args.eos_early_stop),
        },
        "losa": {
            "enabled": args.mode in {"losa", "losa_v2"},
            "variant": args.mode,
            "page_size": args.losa_page_size,
            "token_budget": args.losa_token_budget,
            "active_topk": args.losa_active_topk,
            "gqa_mode": args.losa_gqa_mode,
            "requested_backend": args.losa_backend,
            "trace_detail": bool(args.losa_trace_detail),
            "trace_events": len(generation.trace),
        },
        "focus": {
            "enabled": args.mode in {"focus", "focus_v2"},
            "variant": args.mode,
            "alpha": args.focus_alpha,
            "trace_events": len(generation.trace) if args.mode in {"focus", "focus_v2"} else 0,
        },
        "moe_expert_patch": bool(args.moe_expert_patch),
        "moe_patch_report": (
            dict(getattr(runtime, "moe_patch_report", None))
            if isinstance(getattr(runtime, "moe_patch_report", None), dict)
            else getattr(getattr(runtime, "moe_patch_report", None), "__dict__", None)
        ),
        "input_tokens": int(input_ids.shape[-1]),
        "generated_shape": list(generated.shape),
        "checksum": token_checksum(generated),
        "elapsed_seconds": elapsed,
        "peak_memory_gib": peak_memory,
        "decoded": decoded,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
