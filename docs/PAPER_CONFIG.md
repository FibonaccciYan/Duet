# Paper Sparse configuration

Branch `clean` is based on online master
`fee342cedaa4ae5e88e1eeef42e817f231db7e7f`. This page was prepared by static
review only; no tests, model runs, or benchmarks were executed.

| Paper row | Query settings |
|---|---|
| LLaDA2.1-mini | ratio 0.7, zero-based selection layer 1, selection interval 1, query dense threshold 0, block 32, configured steps 32 |
| SDAR-8B | ratio 0.5, zero-based selection layer 5, selection interval 1, query dense threshold 0, sequential remasking, block 32, steps 16, deep-only transfer |

Paper prefix settings use nominal budgets 256, 512, and 1024 with:

- `query_sparse=true`
- `prefix_sparse=true`
- `prefix_strict_budget=true`
- `prefix_rescreen_full_kv=true` (full refresh/rescreen)
- `prefix_dense_before_query_selection=true` (dense shallow layers)
- `prefix_selector=qk_tc`

All three formal Sparse benchmark entries enforce `prefix_strict_budget=true`.
The LongBench `strict-budget` profile selects the optimized 256/512/1024 matrix.
Every suffix layer retains exactly `min(budget, prefix_length)` prefix positions;
the shallow layers retain the full prefix. Selection is repeated over the full
historical cache once per block, and query selection is repeated each step.

These strict-budget settings do not automatically reproduce earlier
soft-budget tables. Historical numerical results may use different checkpoints,
wrappers, backends, or settings and must not be presented as reproduced.
Source runtime defaults and backends remain unchanged by this documentation
cleanup. Generation quality enables EOS early stopping; speed disables it.
For SDAR, generating 256 tokens in eight blocks uses 128 denoising steps,
including the first step of every block. Stable-step latency excludes each
block's first denoising step, not just the first step of the entire sample.
