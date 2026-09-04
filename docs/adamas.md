# Adamas optimization recovery notes

Branch: `query_losa_correlation`

Baseline commit: `0624110 perf: share prefix selection across layers`

## Completed in the current worktree

- Made `faster_hadamard_transform` the required Hadamard implementation.
- Added `scripts/collect_adamas_hqhk.py`, following `qk_distribute`'s GPU
  histogram/empirical-quantile method.
- Collected 8K, 16K, and 32K Hq/Hk statistics under `results/hqhk_*.json`.
- Collected model-specific empirical bucket thresholds. SDAR uses its measured
  Hq `[-1.50, 0, 1.49]`, Hk `[-2.87, 0, 2.86]`; LLaDA keeps the v1.2 Hq
  `[-1.35, 0, 1.35]`, Hk `[-2.26, 0, 2.26]` because full HumanEval A/B at
  Prefix-256 reached 71/164 official and 128/164 normalized, versus 64/164 and
  127/164 with the empirical thresholds.
- Changed Triton Adamas distance output from int64 to int32. The maximum
  distance is `3 * head_dim = 384` for both checkpoints.

Microbenchmark at 32K before/after Faster Hadamard:

- LLaDA: 12.22ms -> 10.90ms; with int32 output: 10.71ms.
- SDAR: 16.55ms -> 14.23ms; with int32 output: 13.06ms.

32K steady E2E with calibrated thresholds + Faster Hadamard + int32:

- LLaDA Prefix-1024: 25.253s (previous shared-prefix path 25.426s).
- SDAR Prefix-512: 34.536s (previous shared-prefix path 34.600s).

The output checksums changed, so HumanEval is used to validate the new defaults.

## Validation status

- LLaDA HumanEval 32: 9/32 official, 30/32 normalized (unchanged).
- SDAR HumanEval 32: 32/32 official and normalized (unchanged).
- LLaDA full HumanEval: 74/164 official and 137/164 normalized (unchanged),
  output `/data0/ysy/sparse/llada_exp/calibrated_fht_full`.
- SDAR full HumanEval: 129/164 official and 129/164 normalized (unchanged),
  output `/data0/ysy/sparse/sdar_exp/calibrated_fht_full`.
- Unit tests: 5 GPU Triton and 36 shared CPU tests pass.

## Recovery state and possible follow-ups

1. Repeated 8K/16K/32K E2E is complete. Steady sparse repeats were LLaDA
   6.244s/12.290s/25.427s and SDAR 6.228s/12.209s/34.669s.
   Matched dense repeats were LLaDA 6.725s/12.440s/26.584s and SDAR
   6.456s/12.489s/35.621s.
2. Faster Hadamard is registered in both runtime environments using the
   existing import hook. A clean environment must build/install the extension.

3. Distance lookup-table experiment: rejected for the current unpacked Triton
   representation. Adamas' CUDA code contains a 256-entry
   unpack LUT, but it is commented out in favor of packed XOR/popcount. A
   16-entry LUT would add a lookup to integer subtract/abs. Revisit only as
   part of a packed 2-bit distance kernel.
4. GQA experiment: representative-first and group-mean queries were tested.
   At 32K LLaDA they took 25.406s and 25.691s versus exact 25.253s, so both
   were removed. SDAR group-mean reached 34.184s in one run versus exact
   34.536s, but the possible ~1% gain is close to run variance and changes
   selection. Revisit only with selected-index overlap and quality validation;
   no GQA approximation remains in the production code.

## Reproduction commands

Distribution collection:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data0/ysy/Adamas/kernels/3rdparty/applied-ai/kernels/cuda/inference/hadamard_transform \
  /home/ysy/anaconda3/envs/llada/bin/python scripts/collect_adamas_hqhk.py \
  --model llada --context 16384 --output results/hqhk_llada_16k.json
```

Selector microbenchmark:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/data0/ysy/Adamas/kernels/3rdparty/applied-ai/kernels/cuda/inference/hadamard_transform \
  /home/ysy/anaconda3/envs/llada/bin/python scripts/bench_sparse_ops.py \
  --model llada --prefix-length 32704 --query-length 32 --show-adamas-indices
```

Before resuming, run `git status --short` and do not discard the uncommitted
worktree changes described above.
