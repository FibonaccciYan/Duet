# SparseDLM: LLaDA / SDAR

Reference and performance implementations of Sparse, LoSA and FOCUS for
LLaDA2.1-mini and SDAR-8B-Chat-b32. Algorithms and optimized execution paths
are separated explicitly; directory names are not experiment version numbers.

## Project structure

```text
src/
  reference/             # Retained reference algorithms
    sparse/
    losa/
    focus/
  optimized/             # Frozen performance implementations
    dense/               # Efficient Attention baseline
    sparse/
    losa/
    focus/
  kernels/optimized/     # Shared kernels, caches and graph experiments
  model/                 # Repository model implementations
  runtime.py             # Runtime / patch registry
  runtime_compat.py      # Checkpoint loading compatibility
  dense/                 # Compatibility export of optimized dense only
scripts/
  unified/               # 12 method/task shell entry points
  common/                # Launching, protocol adapters and reporting
  configs/               # Explicit, auditable task defaults
  original/              # Preserved pre-unification scripts
data/narrativeqa_speed80/ # Committed exact-ID speed dataset and tokenizer snapshots
tests/
eval_instruct/           # Instruction-following evaluation adapters
```

See [package migration](docs/PACKAGE_LAYOUT.md) for the exact old-to-new mapping.
Historical measurements and configurations remain in `docs/` and `paper/`;
they do not override current runtime defaults. The previous README is preserved
in [HISTORICAL_README.md](docs/history/HISTORICAL_README.md).

## Environment

The tested n33 environment uses Python 3.10, PyTorch 2.5.1+cu124,
Triton 3.1.0, Transformers 4.57.1 and FlashAttention 2.6.3.
Dependency pins are in `requirements.txt`. The supplied FlashAttention wheel
is environment-specific; verify CUDA/PyTorch ABI compatibility before installing
it on a different machine. No installation is needed in the validated n33 env.

Run commands from the repository root with `PYTHONPATH=.`. Runtime constructors
accept `model_path`; built-in defaults refer to the user's n33 model directories.

## Runtime entry points

```python
from src.runtime import load_runtime

runtime = load_runtime("losa_optimized", family="llada",
                       model_path="/path/to/LLaDA2.1-mini")
model, tokenizer = runtime.load()
# inputs is a batch-size-one token tensor.
output = runtime.generate(inputs, gen_length=256, block_length=32, steps=32,
                          threshold=0.7, editing_threshold=0.5,
                          temperature=0.0, eos_early_stop=False)
tokens = output.tokens
```

| Entry | Implementation |
|---|---|
| `load_runtime("dense_optimized", ...)` | Frozen Efficient Attention dense |
| `load_runtime("losa_optimized", ...)` | LoSA: group_mean, fused state, graph replay |
| `load_runtime("focus_optimized", ...)` | FOCUS: LLaDA native Flash; SDAR SDPA |
| `load_runtime("losa", ...)` | Reference LoSA |
| `load_runtime("focus", ...)` | Reference FOCUS |

Sparse uses the model-patching API:

```python
from src.reference.losa.generation import load_model_and_tokenizer
from src.runtime import patch_method

model, tokenizer = load_model_and_tokenizer("llada", model_path="/path/to/model")
patch_method(model, "sparse_optimized", model_name="llada",
             query_sparse=True, prefix_sparse=True, prefix_token_budget=256)
tokens = model.generate(inputs=inputs, gen_length=256, block_length=32, steps=32,
                        threshold=0.7, editing_threshold=0.5,
                        temperature=0.0, eos_early_stop=False)
```

Use `patch_method(..., "sparse", ...)` for reference Sparse. No implicit fallback
to a different algorithm was added by this reorganization.

## Reproducible speed tests

```bash
bash scripts/unified/sparse_speed_full.sh \
  --model llada21 --gpu 0 --contexts 8192 \
  --samples 80 --repeats 3 --output results/new_run/llada_sparse
```

Unified entry points require explicit GPU selection and refuse busy devices.
GPU0/GPU2 are released normally; no holder is started on those cards. GPU5 can
opt into its existing holder protocol with `--gpu 5 --restore-holder`.
Preserved original launchers retain their older fixed-GPU checks.
Do not bypass occupancy checks or overwrite an existing output directory.

- NarrativeQA80 stored input IDs; no repeated filling or re-tokenization.
- 8K/16K inputs are 8192/16384 tokens; 32K is 32512 input plus 256 generation.
- LLaDA default thresholds: 0.7/0.5.
- SDAR speed tests: low_confidence_dynamic, threshold 0.95.
- Unified short benchmarks use optimized runtimes; SDAR threshold defaults to
  0.95 and MMLU defaults to five-shot for every method.
- E2E includes prefill, cache construction, generation and finalization.
- CUDA-event step timing is a separate run; validate tokens and step counts.
- Noninitial-step means exclude only each block's first step; include editing
  and LoSA prime/capture/replay. Reaggregate raw records, not old medians.
- Shared MoE and other common optimizations must be matched in dense comparisons.
- Efficient dense versus sparse Flash gains include backend effects; they are
  not automatically pure sparsity gains or evidence of quality equivalence.

S-mode-specific entry points have been removed. See [scripts/README.md](scripts/README.md)
for the unified launchers, defaults and explicit parameter overrides.

## Verification and submission

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python -m pytest tests -q
PYTHONPATH=. python -m scripts.original.smoke.runtime_mode --help
git diff --check
```

This migration changes package paths and identifiers, not algorithm rules.
Historical run outputs are not rewritten. Review `git status` before staging:
generated results, environments and model weights do not belong in the source
commit. No automatic commit or push is performed.
