#!/usr/bin/env bash
# Explicitly never calls gpu_reserver: GPU5 stays reserved, GPU4 is released.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
ROOT=results/narrative80_gpu4_20260917/formal
mkdir -p "$ROOT"
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/EXIT_CODE"; nvidia-smi -i 4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; exit "$code"' EXIT
git rev-parse HEAD > "$ROOT/BASE_COMMIT"
# A competing GPU4 process is a blocker, not permission to share.
used=$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$used" -gt 256 ]; then echo "GPU4 busy: $used MiB"; exit 91; fi
for model in sdar llada21; do
    for length in 8192 16384 32768; do
        for method in dense_optimized sparse_optimized; do
            cell=${model}_${length}_${method}
            echo "START $cell $(date -Is)"
            /data0/ysy/anaconda3/envs/sparse/bin/python -B \
                scripts/original/performance/narrative80_llada_benchmark.py \
                --model "$model" --length "$length" --method "$method" \
                --limit 80 --repeats 3 --output "$ROOT/$cell" \
                > "$ROOT/$cell.log" 2>&1 || {
                    tail -50 "$ROOT/$cell.log"
                    exit 1
                }
            echo "DONE $cell $(date -Is)"
        done
    done
done
echo "ALL_DONE $(date -Is)"
