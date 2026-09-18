#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
ROOT=results/narrative80_losa_gpu4_20260917/formal
mkdir -p "$ROOT"
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/EXIT_CODE"; nvidia-smi -i 4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; exit "$code"' EXIT
git rev-parse HEAD > "$ROOT/BASE_COMMIT"
used=$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 256 ]; then echo "GPU4 busy: $used MiB"; exit 91; fi
for model in sdar llada21; do
 for length in 8192 16384 32768; do
  cell=${model}_${length}_losa_optimized
  echo "START $cell $(date -Is)"
  /data0/ysy/anaconda3/envs/sparse/bin/python -B scripts/original/performance/narrative80_losa_benchmark.py --model "$model" --length "$length" --method losa_optimized --limit 80 --repeats 3 --output "$ROOT/$cell" > "$ROOT/$cell.log" 2>&1 || { tail -50 "$ROOT/$cell.log"; exit 1; }
  echo "DONE $cell $(date -Is)"
 done
done
echo "ALL_DONE $(date -Is)"
