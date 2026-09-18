"""Preserve the existing QA protocol while selecting an explicit implementation."""
import argparse
import sys


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--implementation", choices=["reference", "optimized"], default="reference")
    args, remaining = pre.parse_known_args()
    from scripts.original.quality import longbench_quality as original
    from src.runtime import patch_method
    sys.argv = [sys.argv[0], *remaining]
    options = original.parse_args()
    old_parse = original.parse_args
    original.parse_args = lambda: options
    old = original.patch_method
    old_moe = original.patch_moe_experts
    def patch(model, method, model_name="auto", **kwargs):
        target = method if args.implementation == "reference" else method.removesuffix("_optimized") + "_optimized"
        if args.implementation == "optimized":
            kwargs.setdefault("moe_expert_patch", options.family == "llada"
                              if options.moe_expert_patch is None else options.moe_expert_patch)
        print("UNIFIED_RUNTIME", target, flush=True)
        return patch_method(model, target, model_name=model_name, **kwargs)
    original.patch_method = patch
    if args.implementation == "optimized":
        # Optimized runtime installation already chooses its shared MoE backend.
        original.patch_moe_experts = lambda model: None
    try:
        original.main()
    finally:
        original.patch_method = old
        original.patch_moe_experts = old_moe
        original.parse_args = old_parse


if __name__ == "__main__":
    main()
