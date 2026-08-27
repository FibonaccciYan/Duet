# Sparse Diffusion Model Patches

This repository provides one sparse-feature entry point for LLaDA and SDAR.
Model checkpoint files are never modified.

Current evaluation and throughput results are summarized in [`RESULTS.md`](RESULTS.md).
New `eval_instruct` results default to `../llada_exp/` or `../sdar_exp/` in the
parent `sparse/` directory; set `OUTPUT_ROOT` to override it.

## Layout

- `src/sparse/core.py`: public `patch_model(...)` entry point. It validates the
  model family and enables query sparse, prefix sparse, LoSA, and the MoE patch.
- `src/sparse/block_cache_sparse_dlm_patch.py`: LLaDA generation/attention patch.
- `src/sparse/sdar_block_diffusion_patch.py`: SDAR generation/KV-cache patch.
- `src/sparse/moe_expert_patch.py`: MoE kernel implementation used by the core.
  It packs
  routed expert weights and dispatches the SiLU MLP through Triton grouped
  kernels without modifying the model directory. It is a no-op when CUDA or
  Triton is unavailable.
- `scripts/test.py`: single-prompt inference entry point, following Dream's test script structure.
- `scripts/test.sh`: environment-variable wrapper for the main inference path.
- `eval_instruct/`: lm-eval harness with separate `llada` and `sdar` adapters.
- `tests/`: small CPU regression tests built from the LLaDA remote-code class.
- `experiments/legacy/`: earlier sparse-attention, trace, ratio-sweep, plots, and throughput work.

## Run

```bash
bash scripts/test.sh
```

Select the model explicitly with `MODEL_TYPE`; the checkpoint path is optional
when using the repository defaults:

```bash
MODEL_TYPE=llada bash scripts/test.sh
MODEL_TYPE=sdar bash scripts/test.sh
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

Use `PATTERN=default` for the native LLaDA baseline. `PATTERN=patch` is the
default and routes every feature through `src.sparse.patch_model`.

Run SDAR with its checkpoint defaults (`block_length=4`, `steps=4`, mask ID
151669). Query sparsity is enabled by default; use `QUERY_SPARSE=false` for the
official dense-block path:

```bash
MODEL_TYPE=sdar MODEL_PATH=/data0/ysy/models/SDAR-8B-Chat \
PROMPT="Explain diffusion models briefly." bash scripts/test.sh
```

The SDAR adapter applies query selection after layer 2 and keeps all decoded
positions plus the configured mask ratio for later layers. The current feature
support is:

| Model | query sparse | prefix sparse | LoSA | MoE expert patch |
| --- | --- | --- | --- | --- |
| LLaDA | yes | yes | yes | yes |
| SDAR | yes | no | no | no-op on its dense experts |

## Instruct Evaluation

```bash
LIMIT=1 bash eval_instruct/eval.sh
```

For SDAR, explicitly select the `sdar` adapter. Results then default to
`../sdar_exp/`:

```bash
MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat \
LIMIT=1 BENCHMARK=gsm8k \
bash eval_instruct/eval.sh
```

See `eval_instruct/README.md` for dependency setup, benchmark selection, and
native-baseline commands.

## Test

```bash
/home/ysy/anaconda3/envs/llada/bin/python tests/test_block_cache_sparse_dlm.py
/home/ysy/anaconda3/envs/llada/bin/python -m unittest discover -s tests -v
```
