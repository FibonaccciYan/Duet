# Duet (SparseDLM)

Paper inference and evaluation code for LLaDA2.1-mini and SDAR-8B-Chat-b32.
Branch `clean` starts from online master
`fee342cedaa4ae5e88e1eeef42e817f231db7e7f`.

## Setup

Use Python 3.10 and a compatible CUDA environment:

```bash
PYTHON=python3 bash install.sh
```

Provide model paths through `--model-path`, `LLADA_MODEL_PATH`, or
`SDAR_MODEL_PATH`. Data paths and GPU IDs are supplied by the caller.

## Paper Settings

| Model | Decoding | Query ratio | Selection layer |
|---|---|---:|---:|
| LLaDA2.1-mini | Q mode, configured steps 32 | 0.7 | 1 |
| SDAR-8B | Sequential, steps 16 | 0.5 | 5 |

All formal Sparse entries use `qk_tc`, full-prefix rescreening, shallow dense
layers, and strict budgets: each suffix layer retains exactly
`min(budget, prefix_length)` prefix positions. Budgets are 256/512/1024.
Query selection runs every step. Quality enables EOS early stopping; speed
disables it and generates 256 tokens.

## Evaluation

Use `python -m scripts.common.run` with methods `dense`, `sparse`, `losa`, or
`focus`, and tasks `longbench`, `short`, or `speed`. Optimized implementations
are the defaults; reference code remains available.

```bash
python -m scripts.common.run \
  --method sparse --task longbench --model llada21 \
  --model-path /path/to/LLaDA2.1-mini --data-dir /path/to/longbench \
  --gpu "$GPU_ID" --profile strict-budget --output results/longbench
```

LongBench uses the five paper QA tasks. ShortBench uses GSM8K, MMLU,
HumanEval, and MATH-500. Speed uses the bundled NarrativeQA80 token IDs at
8K/16K/32K, two warmups, and three repeats.

GSM8K extraction v3.1, MATH-500 scoring v2, MMLU five-shot, and the final
HumanEval indentation-normalized pass@1 are preserved. HumanEval executes
generated code; use an isolated environment.

See `scripts/configs/defaults.json` and `docs/PAPER_CONFIG.md` for configuration.
This cleanup was reviewed statically without running tests or experiments;
earlier soft-budget results are not claimed to be reproduced. Model code,
attention backends, benchmark token IDs, and scoring implementations are
unchanged. Retain applicable upstream licenses and copyright notices.
