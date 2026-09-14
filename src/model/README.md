# Version-adapted model code

This package contains project-managed copies of checkpoint model code. The
sparse loaders use these classes instead of importing `modeling_*.py` files
from a checkpoint. This makes transformers-version compatibility explicit and
keeps model-weight directories free of runtime edits.

| Directory | Checkpoint | Notes |
|---|---|---|
| `llada2_0/` | LLaDA 2.0 mini | Original forward/generation behavior |
| `llada2_1/` | LLaDA 2.1 mini | Includes fallbacks for newer masking/RoPE APIs |
| `sdar/` | SDAR 8B Chat b32 | Includes kwargs compatibility for newer transformers |

`registry.resolve_model_code` selects the implementation. Set
`SPARSEDLM_MODEL_CODE=remote` to temporarily restore the old
`trust_remote_code=True` checkpoint behavior.
