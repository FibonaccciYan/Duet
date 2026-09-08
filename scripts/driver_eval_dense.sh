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
    # Official Q Mode (README best practices / tech report)
    export THRESHOLD=0.7
    export EDITING_THRESHOLD=0.5
    ;;
  sdar)
    # Official eval config (eval_sdar_hf.py): thr 0.95, low_confidence remasking,
    # b32 checkpoint keeps block=32/steps=32
    export THRESHOLD=0.95
    export REMASKING_STRATEGY=low_confidence_static
    export BLOCK_LENGTH=32
    export STEPS=32
    ;;
  *)
    echo "unknown model type: $model_type" >&2
    exit 2
    ;;
esac

mkdir -p "$OUTPUT_ROOT"
cd /data0/gs/SparseDLM_LLaDA_SDAR/eval_instruct || exit 1

for b in gsm8k humaneval mmlu math; do
  echo "=== $(date -u +%H:%M:%SZ) smoke $b ==="
  if ! LIMIT=8 BENCHMARK="$b" bash ./eval.sh; then
    echo "SMOKE FAILED for $b; aborting" >&2
    exit 1
  fi
  echo "=== $(date -u +%H:%M:%SZ) full $b ==="
  if ! BENCHMARK="$b" bash ./eval.sh; then
    echo "FULL FAILED for $b; aborting" >&2
    exit 1
  fi
  echo "=== $(date -u +%H:%M:%SZ) done $b ==="
done
echo "ALL DONE"
