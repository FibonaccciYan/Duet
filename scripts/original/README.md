# Script entry points

The maintained entry points are grouped by purpose:

- `quality/`: formal task-quality evaluation and prediction generation.
- `performance/`: latency, throughput, kernel, and memory benchmarks.
- `overlap/`: LLaDA and SDAR layer/selection overlap analysis.
- `paper/`: paper-specific attention-mass, query-recall, and selector studies.
- `smoke/`: single-prompt runtime and generation checks.
- `utils/`: operational helpers that do not run experiments themselves.
- `legacy/`: historical correlation and one-off launch scripts. These are not
  maintained as current experiment entry points.

General tools accept model, data, output, Python, and device selection through
CLI arguments or environment variables. The NarrativeQA80 launchers deliberately
pin GPU4 for formal jobs and GPU5 for reserved smoke jobs. Do not bypass
their device checks. Shell launchers resolve the checkout from their own path,
not from a hard-coded remote repository directory.

Canonical runtime names are `dense_optimized`, `sparse_optimized`,
`losa_optimized`, and `focus_optimized`. Bare `sparse`, `losa`, and `focus`
retain reference semantics; `dense` is an alias for efficient optimized dense.
See `../docs/PACKAGE_LAYOUT.md` for migrated source locations.

Primary entry points:

```text
quality/longbench_quality.py
quality/instruct_dense_quality.sh
performance/long_context_benchmark.py
performance/sparse_ops_benchmark.py
overlap/run_llada_layer_overlap.sh
overlap/run_sdar_layer_overlap_all_strategies.sh
paper/attention_mass/collect_attention_mass.py
paper/attention_mass/plot_attention_mass.py
paper/llada_query_recall_longbench.py
smoke/runtime_mode.py
```
