# LLaDA-Instruct Evaluation

This directory vendors the same lm-evaluation-harness 0.4.8 fork used by
Dream's `eval_instruct`, with a generation-only adapter for LLaDA2.1.

## Install

Install the harness and the task extras into the LLaDA environment:

```bash
cd eval_instruct
/home/ysy/anaconda3/envs/llada/bin/python -m pip install -e ".[ifeval,math]"
```

## Run

The default is the block-cache SparseDLM path on MMLU:

```bash
bash eval_instruct/eval.sh
```

Select another Dream benchmark suite with `BENCHMARK`:

```bash
BENCHMARK=gsm8k bash eval_instruct/eval.sh
BENCHMARK=math bash eval_instruct/eval.sh
BENCHMARK=humaneval bash eval_instruct/eval.sh
BENCHMARK=all bash eval_instruct/eval.sh
```

Run the native LLaDA baseline with the same generation settings:

```bash
SPARSE_DLM=false bash eval_instruct/eval.sh
```

Useful overrides include `MODEL`, `PYTHON`, `GEN_LENGTH`, `BLOCK_LENGTH`,
`STEPS`, `SPARSE_DLM_RATIO`, `SPARSE_DLM_TOP_K`,
`SPARSE_DLM_SELECTION_INTERVAL`, `NUM_FEWSHOT`, `LIMIT`, and `OUTPUT_ROOT`.
`LIMIT=1` is useful for a smoke test. Evaluation currently requires
`--batch_size 1`, matching the block-cache implementation.

The MMLU, GSM8K, Minerva Math, and HumanEval configurations follow Dream's
evaluation script. They are all generative tasks; likelihood-based tasks are
not supported by the `llada` adapter.

The first run may download benchmark datasets. Evaluation outputs are written
under `eval_instruct/output_reproduce/` and ignored by Git.
