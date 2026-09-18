# Runtime contract

Canonical packages are split between `src.reference` and `src.optimized`.
See [PACKAGE_LAYOUT.md](PACKAGE_LAYOUT.md) for the migration mapping.

```python
from src.runtime import load_runtime, patch_method

# Construct without loading weights until .load() or .generate().
baseline = load_runtime("dense_optimized", family="sdar")
losa = load_runtime("losa_optimized", family="sdar")
focus = load_runtime("focus_optimized", family="llada")

# Reference algorithms remain available.
reference_losa = load_runtime("losa", family="sdar")
reference_focus = load_runtime("focus", family="llada")

# Sparse patches an already loaded model, preserving model.generate's contract.
patch_method(model, "sparse_optimized", model_name="sdar",
             query_sparse=True, prefix_sparse=True, prefix_token_budget=256)
```

Optimized dense uses Efficient Attention. `dense` is a compatibility alias for
that same runtime. Reference Sparse uses method `sparse`; reference LoSA and
FOCUS use methods `losa` and `focus`. Do not use a bare name when intending to
benchmark an optimized implementation.

Runtime wrappers return objects with `.tokens`. Patched Sparse's
`model.generate` returns a tensor. Preserve each API's EOS semantics; the
full-dataset benchmark's private output-trimming controls are benchmark-specific.

Example CLI (run at checkout root, on an explicitly selected idle GPU):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m scripts.original.smoke.runtime_mode \
  --family llada --mode focus_optimized \
  --threshold 0.7 --editing_threshold 0.5 --gen_length 64
```

This command does not reserve GPU0. For the user's reserved GPU5, run through
`scripts/original/performance/run_reserved_gpu5.sh` and verify restoration on exit.

Reference implementations and optimized ones may share selectors and numerical
helpers. Package separation is not a claim of independent upstream provenance.
Historical integration instructions are archived under `docs/history/` and must
not be used as current import paths.
