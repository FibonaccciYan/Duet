# LLaDA Block-Cache SparseDLM

This repository centers on the block-cache SparseDLM inference path for
`LLaDA2.1-mini`.

Current evaluation and throughput results are summarized in [`RESULTS.md`](RESULTS.md).
New `eval_instruct` results default to `../llada_exp/` in the parent `sparse/`
directory; set `OUTPUT_ROOT` to override it.

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

The current-block query optimization and Adamas prefix selection are
independent. Prefix selection uses a per-layer prefix-only budget of 256
tokens by default:

```bash
QUERY_SPARSE=false PREFIX_SPARSE=true PREFIX_TOKEN_BUDGET=256 \
bash scripts/test.sh
```

Set `PREFIX_SPARSE=false` to retain the full prefix cache. `PREFIX_CHUNK_SIZE`
controls peak memory used by the Python Adamas selector and defaults to 256.

The experimental PyTorch LoSA reference path is disabled by default. It caches
prefix attention output/LSE between refinement steps and merges it with the
fresh current-block attention using online-softmax state:

```bash
LOSA=true LOSA_ACTIVE_TOPK=5 QUERY_SPARSE=false PREFIX_SPARSE=false \
bash scripts/test.sh
```

This is a correctness/reference path, not a Triton kernel. Keep query and
prefix sparse disabled for the first numerical comparison.

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
