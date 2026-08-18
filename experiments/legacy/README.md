# Legacy Experiments

These files predate the block-cache SparseDLM implementation and are retained
for reproducibility rather than used by the main inference path.

- `trace_generation.py`: layer recall, transfer, attention, and plotting diagnostics.
- `local_demo_sparse_patch.py`: earlier sparse-Q/sparse-KV attention experiment.
- `bench_sparse_kv_throughput.py`: earlier sparse-KV throughput comparison.
- `run_ratio_sweep.sh`: layer candidate-ratio sweep.
- `attn_plots/`, `llada_early_predict_analysis_20260521/`, and
  `throughput_sparse_kv_0p5_gen512/`: saved results from those experiments.

Run an old trace with:

```bash
bash experiments/legacy/run_trace.sh
```
