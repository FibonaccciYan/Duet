# Dense, LoSA, and FOCUS Runtime Contract

The project preserves `src.sparse.patch_model` as the compatibility reference.
Dense and LoSA expose matching patch entry points:

```python
from src.dense import patch_model as patch_dense
from src.focus import patch_model as patch_focus
from src.losa import patch_model as patch_losa

patch_dense(model, model_name="llada")
patch_losa(model, model_name="sdar", token_budget=256)
patch_focus(model, model_name="llada", alpha=1.5)
```

`patch_dense` uses the shared block-cache decoder with query, prefix, and LoSA
sparsity disabled. This preserves dense attention while avoiding the
checkpoint decoder's long-context quadratic mask and prompt logits.
`patch_losa` changes the attention partition used after a prefix has been
established. The integrated LoSA operators are self-contained in `src/losa`.

For command-line runs, `scripts/run_mode.py` defaults to checkpoint-faithful
LLaDA values. The speed profile must be selected explicitly:

```bash
python scripts/run_mode.py --family llada --mode losa \
  --threshold 0.5 --editing_threshold 0
```

FOCUS exposes both `FocusRuntime` and `src.focus.patch_model`. The runtime
executes the FOCUS row-retention algorithm directly on the loaded HF model for
LLaDA 2.1 and SDAR. LLaDA 2.1's `editing_threshold=0` is passed through as the
official permissive T2T threshold; it does not disable editing.

GPU verification must run on a node with a live CUDA driver, with
`CUDA_VISIBLE_DEVICES` containing at most two idle cards.

## FOCUS v2

`src.focus_v2` is the speed-test implementation of FOCUS.  It preserves the
reference selection semantics from `src.focus`, but projects Q/K/V only once in
each executed decoder layer and reuses the post-RoPE tensors for layer-0/layer-1
importance.  It also exposes the packed routed-MoE backend used by the MoE speed
paths (`moe_expert_patch=True` by default).

Use mode `focus_v2` for benchmarking:

```bash
python scripts/run_mode.py --family llada --mode focus_v2 \
  --moe_expert_patch true \
  --threshold 0.5 --editing_threshold 0
```

The dedicated throughput wrapper is:

```bash
python scripts/run_focus_v2_throughput.py --family llada
```

Correctness gates are in `tests/test_focus_v2.py`.  The no-eviction gate
compares FOCUS v2 against the portable FOCUS reference; the selected gate
compares retained positions on the first selection step.

### FOCUS v2 model matrix

- LLaDA2.0-mini: `--family llada --model_path /data0/gs/models/LLaDA2.0-mini`.
  Use `--editing_threshold 1.0` to keep LLaDA2.0 mask-to-token-only behavior.
- LLaDA2.1-mini: `--family llada --model_path /data0/ysy/models/LLaDA2.1-mini`.
  Speed profile uses `--threshold 0.5 --editing_threshold 0`.
- SDAR-8B-Chat-b32: `--family sdar`. The loader fills the checkpoint-missing
  `pad_token_id` with its EOS id when required.

All three have smoke coverage through `scripts/run_mode.py --mode focus_v2`.
