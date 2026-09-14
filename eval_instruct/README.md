# LLaDA/SDAR Instruct Evaluation

This directory vendors the same lm-evaluation-harness 0.4.8 fork used by
Dream's `eval_instruct`, with generation-only adapters for LLaDA2.1 and SDAR.

## Install

Install the repository environment, then install the harness in editable mode:

```bash
cd eval_instruct
python -m pip install -e ".[ifeval,math]"
```

## Run

`METHOD` selects one of the four paper methods: `sparse`, `dense`, `focus`, or
`losa`. The default remains `sparse`. `MODEL_TYPE` accepts `llada` or `sdar`:

```bash
bash eval_instruct/eval.sh
METHOD=dense BENCHMARK=gsm8k bash eval_instruct/eval.sh
METHOD=focus BENCHMARK=humaneval bash eval_instruct/eval.sh
METHOD=losa BENCHMARK=math bash eval_instruct/eval.sh
MODEL_TYPE=sdar bash eval_instruct/eval.sh
```

Select another Dream benchmark suite with `BENCHMARK`:

```bash
BENCHMARK=gsm8k bash eval_instruct/eval.sh
BENCHMARK=math bash eval_instruct/eval.sh
BENCHMARK=humaneval bash eval_instruct/eval.sh
BENCHMARK=mmlu bash eval_instruct/eval.sh
BENCHMARK=all bash eval_instruct/eval.sh
```

Run the complete four-method, five-benchmark matrix sequentially:

```bash
bash eval_instruct/run_matrix.sh
MODEL_TYPE=sdar bash eval_instruct/run_matrix.sh
```

The fifth benchmark is a five-task LongBench QA subset (`hotpotqa`, `triviaqa`,
`narrativeqa`, `qasper`, and `multifieldqa_en`). Point `LONGBENCH_DATA` at a
directory containing those JSONL files; each record needs `prompt` and
`answers` (or `answer`). Its `predictions.jsonl` supports `--resume_from`, and
`report.json` contains per-task and overall token-F1/exact-match scores.

`METHODS` and `BENCHMARKS` accept space-separated subsets. `DRY_RUN=true`
prints the selected jobs. A `.done` marker is written only after each job and
HumanEval post-processing both succeed. For example:

```bash
LONGBENCH_DATA=/path/to/longbench BENCHMARKS=longbench LIMIT=1 \
  bash eval_instruct/run_matrix.sh
```

Outputs default to `../llada_exp/default` (that is,
`/data0/ysy/sparse/llada_exp/default`). Set a directory name relative to
`llada_exp` with `OUTPUT_PATH`:

```bash
OUTPUT_PATH=A BENCHMARK=gsm8k bash eval_instruct/eval.sh
```

This writes to `../llada_exp/A`. `OUTPUT_ROOT` remains available when a full
custom path is needed.

The old dense selector remains supported when `METHOD` is unset:

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
The remaining `SPARSE_DLM_*`, `QUERY_SPARSE`, `PREFIX_SPARSE`, and `LOSA`
switches are internal ablations of `METHOD=sparse`; they do not select the
standalone paper LoSA baseline. Both prefix sparsity and integrated LoSA are
disabled by default for SDAR:

```bash
METHOD=sparse MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat-b32 \
BLOCK_LENGTH=32 STEPS=32 QUERY_SPARSE=false PREFIX_SPARSE=true \
PREFIX_TOKEN_BUDGET=256 LIMIT=1 BENCHMARK=humaneval \
bash eval_instruct/eval.sh

METHOD=sparse MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat-b32 \
BLOCK_LENGTH=32 STEPS=32 QUERY_SPARSE=false LOSA=true \
LOSA_ACTIVE_TOPK=5 LIMIT=1 BENCHMARK=humaneval \
bash eval_instruct/eval.sh
```

Useful overrides include `METHOD`, `MODEL_TYPE`, `MODEL`, `PYTHON`, `GEN_LENGTH`,
`BLOCK_LENGTH`, `STEPS`, `FOCUS_ALPHA`, `PAPER_LOSA_PAGE_SIZE`,
`PAPER_LOSA_TOKEN_BUDGET`, `PAPER_LOSA_ACTIVE_TOPK`, `SPARSE_DLM_RATIO`, `SPARSE_DLM_TOP_K`,
`SPARSE_DLM_SELECTION_INTERVAL`, `QUERY_SPARSE`, `PREFIX_SPARSE`,
`PREFIX_TOKEN_BUDGET`, `PREFIX_CHUNK_SIZE`, `LOSA`, `LOSA_ACTIVE_TOPK`,
`MOE_EXPERT_PATCH`,
`NUM_FEWSHOT`, `LIMIT`, and `OUTPUT_PATH` or `OUTPUT_ROOT`.
`LIMIT=1` is useful for a smoke test. Evaluation currently requires
`--batch_size 1`, matching the block-cache implementation.

The integrated sparse LoSA experiment is disabled by default. For a first
correctness comparison, keep both other sparse selectors disabled:

```bash
METHOD=sparse LOSA=true QUERY_SPARSE=false PREFIX_SPARSE=false LIMIT=1 \
bash eval_instruct/eval.sh
```

The older scripts below remain sparse-feature ablation matrices; they are not
the four-method paper matrix:

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
