# LLaDA/SDAR Instruct Evaluation

This directory vendors the same lm-evaluation-harness 0.4.8 fork used by
Dream's `eval_instruct`, with generation-only adapters for LLaDA2.1 and SDAR.

## Install

Install the harness and the task extras into the LLaDA environment:

```bash
cd eval_instruct
/home/ysy/anaconda3/envs/llada/bin/python -m pip install -e ".[ifeval,math]"
```

## Run

The default is the LLaDA block-cache sparse path on MMLU. `MODEL_TYPE` is the
single model-adapter selector and accepts `llada` or `sdar`:

```bash
bash eval_instruct/eval.sh
MODEL_TYPE=sdar bash eval_instruct/eval.sh
```

Select another Dream benchmark suite with `BENCHMARK`:

```bash
BENCHMARK=gsm8k bash eval_instruct/eval.sh
BENCHMARK=math bash eval_instruct/eval.sh
BENCHMARK=humaneval bash eval_instruct/eval.sh
BENCHMARK=all bash eval_instruct/eval.sh
```

Outputs default to `../llada_exp/default` (that is,
`/data0/ysy/sparse/llada_exp/default`). Set a directory name relative to
`llada_exp` with `OUTPUT_PATH`:

```bash
OUTPUT_PATH=A BENCHMARK=gsm8k bash eval_instruct/eval.sh
```

This writes to `../llada_exp/A`. `OUTPUT_ROOT` remains available when a full
custom path is needed.

Run the native LLaDA baseline with the same generation settings:

```bash
SPARSE_DLM=false bash eval_instruct/eval.sh
```

Use the shared SDAR defaults:

```bash
MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat-b32 \
LIMIT=1 BENCHMARK=gsm8k \
bash eval_instruct/eval.sh
```

The wrapper passes `--model sdar`, uses block length/steps `32/32`, ratio 0.5,
zero-based selection layer 5, selection interval 1, refresh `-1`, sequential
remasking, and no MoE expert patch. It selects the compatible Python
environment and writes to `../sdar_exp/`; it does not infer the adapter from
the checkpoint path.
`SPARSE_DLM=false` disables every sparse feature. `QUERY_SPARSE=false` selects
SDAR's official dense-query path while still allowing Adamas or LoSA to be
enabled explicitly. Both are disabled by default for SDAR:

```bash
MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat-b32 \
BLOCK_LENGTH=32 STEPS=32 QUERY_SPARSE=false PREFIX_SPARSE=true \
PREFIX_TOKEN_BUDGET=256 LIMIT=1 BENCHMARK=humaneval \
bash eval_instruct/eval.sh

MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat-b32 \
BLOCK_LENGTH=32 STEPS=32 QUERY_SPARSE=false LOSA=true \
LOSA_ACTIVE_TOPK=5 LIMIT=1 BENCHMARK=humaneval \
bash eval_instruct/eval.sh
```

Useful overrides include `MODEL_TYPE`, `MODEL`, `PYTHON`, `GEN_LENGTH`, `BLOCK_LENGTH`,
`STEPS`, `SPARSE_DLM_RATIO`, `SPARSE_DLM_TOP_K`,
`SPARSE_DLM_SELECTION_INTERVAL`, `QUERY_SPARSE`, `PREFIX_SPARSE`,
`PREFIX_TOKEN_BUDGET`, `PREFIX_CHUNK_SIZE`, `LOSA`, `LOSA_ACTIVE_TOPK`,
`MOE_EXPERT_PATCH`,
`NUM_FEWSHOT`, `LIMIT`, and `OUTPUT_PATH` or `OUTPUT_ROOT`.
`LIMIT=1` is useful for a smoke test. Evaluation currently requires
`--batch_size 1`, matching the block-cache implementation.

The experimental LoSA reference path is disabled by default. For a first
correctness comparison, keep both existing sparse selectors disabled:

```bash
LOSA=true QUERY_SPARSE=false PREFIX_SPARSE=false LIMIT=1 \
bash eval_instruct/eval.sh
```

Run the SDAR LoSA matrix on only GPUs 4 and 5. GPU 4 owns `topk=5`, GPU 5
owns `topk=32`; each lane runs dense, prefix, query, and prefix+query in order.
The SDAR matrix defaults to the b32 checkpoint with block/steps `32/32`:

```bash
MODEL_TYPE=sdar bash eval_instruct/run_gsm8k_matrix.sh
MODEL_TYPE=sdar bash eval_instruct/run_humaneval_matrix.sh
```

Use `DRY_RUN=true` to print the eight jobs without launching evaluation.

HumanEval runs automatically normalize and re-evaluate the saved samples after
generation. The summary is saved beside the matching `samples_*.jsonl` as
`reeval_indent_normalized.json`. Every aggregated `results_*.json` also records
`generated_tokens`, `generation_time_seconds`, and
`generation_tokens_per_second` in its `config` section.

The MMLU, GSM8K, Minerva Math, and HumanEval configurations follow Dream's
evaluation script. They are all generative tasks; likelihood-based tasks are
not supported by the `llada` adapter.

The first run may download benchmark datasets. Evaluation outputs are written
under `../llada_exp/` by default.
