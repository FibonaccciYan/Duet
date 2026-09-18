# Source layout and migration

This is a naming and packaging migration of the tested, uncommitted
`codex/integrate-all-20260916` checkout, not an algorithm change.

| Before | Canonical package | Runtime / method |
|---|---|---|
| `src/sparse` | `src.reference.sparse` | `sparse` (patch API) |
| `src/losa` | `src.reference.losa` | `losa` |
| `src/focus` | `src.reference.focus` | `focus` |
| `src/sparse_v3` | `src.optimized.sparse` | `sparse_optimized` (patch API) |
| `src/dense_v3` | `src.optimized.dense` | `dense_optimized`, alias `dense` |
| `src/losa_v6` | `src.optimized.losa` | `losa_optimized` |
| `src/focus_v8` | `src.optimized.focus` | `focus_optimized` |
| `src/kernels/versioned` | `src.kernels.optimized` | shared execution helpers |

The original dense implementation had already been removed in the frozen
checkout. `src.dense` remains a lightweight compatibility export of optimized
efficient dense; it is not presented as a recovered reference implementation.

Optimized implementations intentionally reuse reference selectors and drivers.
Moving them into separate namespaces does not duplicate or change these rules.
Reference package means the retained reference from this checkout, not a
bitwise archive of an external upstream paper repository.

Public optimized classes are `DenseOptimizedRuntime`, `LoSAOptimizedRuntime`,
and `FocusOptimizedRuntime`. Optimized FOCUS functions use `focus_optimized_*`.
The LoSA operator uses `losa_optimized_attention_step`. Runtime-internal
attributes were renamed together with every caller.

## Benchmarks

The NarrativeQA80 scripts use the same input IDs, warmup, generation, EOS and
timing logic. Output method labels now use `*_optimized`; source manifests
point to canonical package paths. Historical JSON/CSV reports retain their old
labels and paths: do not interpret them as executable import instructions.
Use fresh output directories for new runs; do not overwrite completed studies.

Shell launchers resolve the checkout relative to their own location. Model,
dataset, interpreter and GPU-reserver paths retain the documented n33 setup.
GPU4 jobs do not reserve GPU4 on completion. GPU5 smoke work must use the
existing `run_reserved_gpu5.sh` wrapper, including on failures.

Some older Sparse benchmark fields called `steady` exclude two initial steps.
For cross-method reporting, reaggregate raw records using `loop_iteration >= 1`
and `sum(cuda_ms) / sum(count)`. This structural migration does not silently
change historical benchmark measurement definitions.

## Validation

Run from the repository root with the CUDA-enabled project environment:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python -m scripts.original.performance.narrative80_focus_benchmark --help
PYTHONPATH=. python -m scripts.original.smoke.runtime_mode --help
```

`tests/test_package_layout.py` checks package boundaries, import origins and
runtime identities. GPU migration verification compares old and new checkouts
on the same real input and checks output token IDs, not timing equivalence.
No full benchmark or quality claim is implied by smoke coverage.
