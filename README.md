# LLaDA Block-Cache SparseDLM

This repository centers on the block-cache SparseDLM inference path for
`LLaDA2.1-mini`.

Current evaluation and throughput results are summarized in [`RESULTS.md`](RESULTS.md).

## Layout

- `src/sparse/block_cache_sparse_dlm_patch.py`: production patch.
- `scripts/test.py`: single-prompt inference entry point, following Dream's test script structure.
- `scripts/test.sh`: environment-variable wrapper for the main inference path.
- `eval_instruct/`: Dream-compatible lm-eval harness and LLaDA adapter.
- `tests/`: small CPU regression tests built from the LLaDA remote-code class.
- `experiments/legacy/`: earlier sparse-attention, trace, ratio-sweep, plots, and throughput work.

## Run

```bash
bash scripts/test.sh
```

Common overrides:

```bash
GEN_LENGTH=128 BLOCK_LENGTH=32 STEPS=32 \
SPARSE_DLM_RATIO=0.5 SPARSE_DLM_SELECTION_INTERVAL=4 \
bash scripts/test.sh
```

Use `PATTERN=default` for the native LLaDA baseline.

## Instruct Evaluation

```bash
LIMIT=1 bash eval_instruct/eval.sh
```

See `eval_instruct/README.md` for dependency setup, benchmark selection, and
native-baseline commands.

## Test

```bash
/home/ysy/anaconda3/envs/llada/bin/python tests/test_block_cache_sparse_dlm.py
```
