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

`patch_dense` enables the self-contained, chunked dense block-cache decoder and
the packed MoE expert path by default. This is required
because the canonical decoder performs both mask-to-token and token-to-token
editing. `patch_losa` keeps that decoder schedule while changing only the
attention partition used after a prefix has been established. The integrated
LoSA operators are self-contained in `src/losa`.

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
