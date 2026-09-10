#!/usr/bin/env bash
# Sequential dense quality driver for eval_instruct/eval.sh.
# Usage: driver_eval_dense.sh MODEL_TYPE OUTPUT_NAME GPU_INDEX PORT
set -u

# Cluster egress proxy (from ~/.bashrc); datasets download needs it
export http_proxy=http://114.212.80.7:21087
export https_proxy=http://114.212.80.7:21087
HF_HUB_DOWNLOAD_TIMEOUT=60; export HF_HUB_DOWNLOAD_TIMEOUT
model_type="$1"
out_name="$2"
gpu="$3"
port="$4"

export CUDA_VISIBLE_DEVICES="$gpu"
export MODEL_TYPE="$model_type"
export RUNTIME_MODE=dense
export MAIN_PROCESS_PORT="$port"
export OUTPUT_ROOT="/data0/gs/eval_results/${out_name}"

case "$model_type" in
  llada)
    # Official Q Mode (README best practices / tech report); overridable
    # so LLaDA 2.0 runs can pass its own decode config (threshold=0.95,
    # editing disabled via 1.0).
    export THRESHOLD="${THRESHOLD:-0.7}"
    export EDITING_THRESHOLD="${EDITING_THRESHOLD:-0.5}"
    # Official recommended output length (README: 16384 for most scenarios).
    export GEN_LENGTH="${GEN_LENGTH:-16384}"
    ;;
  sdar)
    # Official eval config (eval_sdar_hf.py): thr 0.95, low_confidence remasking,
    # b32 checkpoint keeps block=32/steps=32
    export THRESHOLD=0.95
    export REMASKING_STRATEGY=low_confidence_static
    export BLOCK_LENGTH=32
    export STEPS=32
    # Official eval_sdar_hf.py uses gen_length=4096 for every task.
    export GEN_LENGTH="${GEN_LENGTH:-4096}"
    ;;
  *)
    echo "unknown model type: $model_type" >&2
    exit 2
    ;;
esac

# Isolated datasets cache per pipeline: the llada and dream envs ship
# incompatible `datasets` versions (5.0.1 vs 3.6.0) and must not share one.
# Respect a caller-provided (warm) cache; default to an isolated one.
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data0/gs/.hf_cache_${out_name}}"
mkdir -p "$OUTPUT_ROOT" "$HF_DATASETS_CACHE"
cd /data0/gs/SparseDLM_LLaDA_SDAR/eval_instruct || exit 1

started=$([[ -n "${START_BENCH:-}" ]] && echo 0 || echo 1)
for b in ${BENCHMARKS:-gsm8k humaneval mmlu math}; do
  if [[ "$started" == 0 ]]; then
    if [[ "$b" == "${START_BENCH:-}" ]]; then
      started=1
    else
      echo "skip $b (resuming from ${START_BENCH})"
      continue
    fi
  fi
  if [[ "$b" == "mmlu" ]]; then
    # Official protocol: MMLU 5-shot (SDAR opencompass uses 5 fix examples).
    export NUM_FEWSHOT="${MMLU_FEWSHOT:-5}"
  else
    export NUM_FEWSHOT=0
  fi

  if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
    echo "=== $(date -u +%H:%M:%SZ) smoke $b ==="
    if ! LIMIT="${SMOKE_LIMIT:-2}" BENCHMARK="$b" bash ./eval.sh; then
      echo "SMOKE FAILED for $b; aborting" >&2
      exit 1
    fi
  else
    echo "=== $(date -u +%H:%M:%SZ) skip smoke, straight to full $b ==="
  fi
  echo "=== $(date -u +%H:%M:%SZ) full $b ==="
  if ! BENCHMARK="$b" bash ./eval.sh; then
    echo "FULL FAILED for $b; aborting" >&2
    exit 1
  fi
  echo "=== $(date -u +%H:%M:%SZ) done $b ==="
done
echo "ALL DONE"
