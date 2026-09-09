import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.sparse.sparse_ops import (
    _adamas_prefix_indices,
    _attention_output_lse,
    _hadamard_qk_prefix_indices,
    _losa_active_indices,
    _merge_attention_states,
    _new_losa_state,
    _qk_prefix_indices,
)


def benchmark(name, function, warmup=3, iterations=10):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    print(f"{name}: {torch.tensor(samples).median().item():.3f} ms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("llada", "sdar"), default="sdar")
    parser.add_argument("--prefix-length", type=int, default=4096)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--active-topk", type=int, default=5)
    parser.add_argument("--profile-adamas", action="store_true")
    parser.add_argument("--show-adamas-indices", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    heads, kv_heads = (16, 4) if args.model == "llada" else (32, 8)
    dtype = torch.bfloat16 if args.model == "llada" else torch.float16
    device = "cuda"
    query = torch.randn(
        1, heads, args.query_length, 128, device=device, dtype=dtype
    )
    key = torch.randn(
        1, kv_heads, args.prefix_length, 128, device=device, dtype=dtype
    )
    value = torch.randn_like(key)
    mask = torch.zeros(
        1, 1, args.query_length, args.prefix_length, device=device, dtype=dtype
    )

    benchmark(
        "attention_output_lse",
        lambda: _attention_output_lse(query, key, value, mask, heads // kv_heads),
    )
    prefix_output, prefix_lse = _attention_output_lse(
        query, key, value, mask, heads // kv_heads
    )
    block_output = torch.randn_like(prefix_output)
    block_lse = torch.randn_like(prefix_lse)
    benchmark(
        "merge_attention_states",
        lambda: _merge_attention_states(
            prefix_output, prefix_lse, block_output, block_lse
        ),
    )

    state = _new_losa_state(query, args.query_length)
    state["previous_query"].normal_()
    state["valid"].fill_(True)
    state["fully_valid"] = True
    positions = torch.arange(args.query_length, device=device)
    benchmark(
        "losa_active_indices",
        lambda: _losa_active_indices(
            state, query, positions, args.active_topk
        ),
    )
    benchmark(
        "adamas_prefix_indices",
        lambda: _adamas_prefix_indices(
            query, key, token_budget=256, chunk_size=256
        ),
        warmup=1,
        iterations=3,
    )
    benchmark(
        "qk_prefix_indices",
        lambda: _qk_prefix_indices(query, key, token_budget=256),
        warmup=1,
        iterations=3,
    )
    benchmark(
        "hadamard_qk_prefix_indices",
        lambda: _hadamard_qk_prefix_indices(query, key, token_budget=256),
        warmup=1,
        iterations=3,
    )
    if args.profile_adamas:
        with torch.profiler.profile(
            activities=(
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ),
            record_shapes=True,
        ) as profiler:
            _adamas_prefix_indices(
                query, key, token_budget=256, chunk_size=256
            )
        print(
            profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=20
            )
        )
    if args.show_adamas_indices:
        indices = _adamas_prefix_indices(
            query, key, token_budget=256, chunk_size=256
        )
        weights = torch.arange(1, indices.numel() + 1, device=indices.device)
        print(
            f"adamas_indices: count={indices.numel()} sum={indices.sum().item()} "
            f"weighted_sum={(indices * weights).sum().item()}"
        )


if __name__ == "__main__":
    main()
