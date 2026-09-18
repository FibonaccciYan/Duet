# Incremental strict prefix budget

Implemented on 2026-09-18 in existing `src/reference/sparse` and `src/optimized/sparse`.
No new runtime version. No Dense, LoSA or FOCUS algorithm edits.

## Semantics

`prefix_strict_budget=False` is the default for both families.
The original per-query candidate construction, union, under-budget fill and
sorted return order remain unchanged when disabled.

When enabled, an overflowing candidate union is ranked by each token's global
minimum Raw L1 query distance, restricted to union members, keeping the best
`min(token_budget, candidate_length)` tokens. This is not a top-k over all prefix
tokens. Current-block tokens are additional to the prefix budget.

Ties use the existing PyTorch topk behavior; no cross-device tie determinism is
promised. Per-block candidate reuse remains unchanged. Strict mode may change
generation outputs and requires separate quality evaluation.

## Usage

```python
patch_method(model, "sparse_optimized", model_name="sdar",
             query_sparse=True, prefix_sparse=True,
             prefix_token_budget=256, prefix_strict_budget=True)
```

The same option is supported by `src.reference.sparse.api.patch_model`.

LongBench:
`--method sparse_optimized --prefix_strict_budget --collect_prefix_selection_stats`
(also supports `--method sparse` and `--no-prefix_strict_budget`).

lm-eval adapter: `prefix_strict_budget=true`.
eval.sh: `PREFIX_STRICT_BUDGET=true`.
Statistics can also be collected by explicitly assigning
`model._prefix_selection_stats = []`; remove the attribute or set it to None
to disable collection.

## Statistics

Optional `prefix_selection.jsonl` and `report.json.prefix_selection_stats`.
Records include task/sample index and selection index, representative layer,
full prefix length, candidate length, effective budget, local budget,
union size before fill/trim, selected size, strict flag and bypass flag.

Full-retention/empty-budget selector returns have complete fields with
`bypassed=True`; union_size is a sentinel equal to returned size on these paths,
not a computed query union. Bypassed records are excluded from overflow and
shortfall rates. Empty cache produces no layer record.

Overflow/shortfall measure the original union. `strict_final_violations` measures
actual final strict-budget overflow. Collection is based on effective model
configuration, not solely the explicit CLI prefix flag. Sample stats are written
after timed generation; timings with collection are still marked diagnostic.
Resume statistics cover only retained raw records from the same output directory.

## Minimal v3 changes

Triton kernels are unchanged. The Python selector wrapper propagates options to
fallback, clips only overflowing unions when enabled and fills optional stats.
A safety dispatch guard now sends non-128 head dimensions to the Raw L1 reference.
The pre-existing fused kernel produced invalid indices on a head_dim32 synthetic
case, including when run from the pre-change backup. Both deployed model families
use head_dim128; their fused dispatch remains unchanged. A kernel-level fix for
other dimensions was deliberately not mixed into this feature.

## Verification

- CPU regression: 77 passed, 9 skipped (including CUDA-only cases).
- GPU selector unit suite: 10 passed before three additional CPU tests were added.
- 108 production-shape FP16/BF16 checks: strict=False indices equal pre-change
  original/v3 wrappers; strict=True count, uniqueness and bounds checks passed.
  Covers candidate lengths64/1024/2064 and budgets8/32/256.
- Both model checkpoints, both sparse/original and sparse_optimized, strict off/on:
  512-token prompt with64-token generation and budget8. SDAR maximum soft selection
  120, LLaDA62; strict selections capped at8 across all recorded calls.
- Real LongBench narrativeqa runner sample, v3, prefix enabled via default (no
  explicit prefix flag):20 layer records,11 overflow,8 shortfall,0 final violations;
  all selected_size8.

These are functional and compatibility tests, not long-context quality or speed
acceptance. No claim that strict mode improves throughput or accuracy.
The pre-change source snapshot is `prefix_strict_before_20260918.tar.gz`.
