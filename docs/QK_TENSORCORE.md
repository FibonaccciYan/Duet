# Optional Tensor Core QK selector

`prefix_selector="raw_l1"` remains the default. `qk` retains the explicit FP32
pairwise-tree oracle and its original optimized implementation.
`prefix_selector="qk_tc"` opts into FP16/BF16 Tensor Core dot with FP32 accumulation.
No extra normalization, scaling, group-mean query, TF32, quantization or
approximate candidate prefilter is used.

## Implementation

`src/optimized/sparse/qk_tc_prefix.py` groups work by physical KV head. Each
16-query / 64-key tile computes negative dot scores over D=128 using `tl.dot`.
Each tile emits up to local_budget candidates per query plus token-score minima.
The existing QK lexicographic ranking, per-query union, fill, strict trimming
inside the union, and final sorted candidate-local indices remain unchanged.
The tile top-k loop is statically unrolled because Triton3.1 crashed compiling
the dynamic loop with LOCAL=4; the failing implementation was not accepted.

Supported optimized shapes: CUDA sm80+, matching FP16/BF16 Q/K, D=128,
candidate_length<=32768, local_budget<=8. Other inputs fall back to QK reference,
never Raw L1. Full score matrices are generated only by the explicit diagnostic
`return_scores=True` API, not ordinary selection.

Reference dispatch of `qk_tc` intentionally runs the FP32 QK oracle.
Stats report actual `execution_backend` and `score_definition`; fallback reports
pairwise-tree arithmetic, optimized execution reports Tensor Core arithmetic.
All-retained compactor bypass reports `not_computed`.

Tensor Core accumulation order can differ from the tree oracle. Identical scores
prefer smaller token indices, but near-boundary rounding may change which scores
are tied or ranked. This is an explicit numerical variant, not a bitwise
equivalence guarantee. Inputs are assumed finite model activations.

## Verification on n33 GPU5 H800 (2026-09-20)

- 35 new tests passed: FP16/BF16, GQA, per-KV-head tile tails, noncontiguous strides,
  lengths17..32768, local budgets1/4, soft/strict selection, exact and near ties,
  CPU/non128/FP32/local-budget>8 fallback.
- Score comparison to FP32 tree oracle uses atol3e-4, rtol3e-5. Selection processing
  is independently checked exactly against lexicographic selection on actual TC
  scores; this avoids hiding selection errors behind a score tolerance.
- Generated PTX includes MMA instructions, verified in tests.
- Compute Sanitizer memcheck and initcheck:35 tests each,0 errors.
- Original QK/Raw L1/adapter/quality regression:79 tests passed.
- Real NarrativeQA speed80 sample0, SDAR and LLaDA2.1 Q, prompts8192/32704,
  generation64, budget256, soft budget, refresh off, interval4, EOS stop disabled:
  three alternating E2E runs after warmup; separate on-trajectory selector audit.
  All audited TC indices equal QK reference and outputs equal QK in these four
  cases. This does not establish full-dataset quality equivalence.

|Model|Prompt|Raw L1 E2E s|FP32 QK E2E s|TC QK E2E s|
|---|---:|---:|---:|---:|
|SDAR|8192|1.1464|2.6146|1.1803|
|SDAR|32704|5.2427|11.0436|5.2471|
|LLaDA2.1 Q|8192|0.6931|1.0821|0.6944|
|LLaDA2.1 Q|32704|2.2030|3.8579|2.2993|

Medians of three complete generation runs; no profiler/audit in E2E. Raw L1 and
QK are different algorithms; timings may include different decoding trajectories.
Small single-sample differences are not statistical proof of a speed regression.
Full LongBench quality has not been run for qk_tc alone. The completed
2026-09-21 full run combined `qk_tc` with the separately documented
`prefix_dense_before_query_selection` feature and must not be relabeled as a
selector-only result.

## Usage

```python
from src.optimized.sparse import patch_model
patch_model(model, model_name="sdar", query_sparse=True, prefix_sparse=True,
            prefix_token_budget=256, prefix_selector="qk_tc")
```

Unified quality CLI: `--set prefix_selector=qk_tc`.
Direct QA CLI: `--prefix_selector qk_tc`.
lm-eval: `prefix_selector=qk_tc`; eval.sh: `PREFIX_SELECTOR=qk_tc`.

Only the selector execution changes. Query selection parameters, confidence,
transfer, EOS, RoPE/normalization, attention and KV update algorithms are unchanged.
Dense/LoSA/FOCUS and original Raw L1 kernels are not modified by this TC change.
