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

Formal scripts accept model, data, output, Python, and device selection through
CLI arguments or environment variables. Set `CUDA_VISIBLE_DEVICES` externally;
the maintained wrappers do not select a fixed GPU.

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
