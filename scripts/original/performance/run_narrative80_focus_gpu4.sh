#!/usr/bin/env bash
# No reservation/holder commands: GPU4 is left free after completion or failure.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
PY=/data0/ysy/anaconda3/envs/sparse/bin/python
SCRIPT=scripts/original/performance/narrative80_focus_benchmark.py
if [[ "${1:---plan}" == "--plan" ]]; then
    CUDA_VISIBLE_DEVICES="" "$PY" -B "$SCRIPT" --preflight
    exit 0
fi
if [[ "${1:-}" != "--run" || "$#" -ne 1 ]]; then
    echo "Usage: bash $0 --plan | --run" >&2
    exit 2
fi
CUDA_VISIBLE_DEVICES="" "$PY" -B "$SCRIPT" --preflight
export CUDA_VISIBLE_DEVICES=4
ROOT=results/narrative80_focus_gpu4_20260917/formal
if [[ -e "$ROOT" ]]; then
    echo "Refusing to overwrite existing formal directory: $ROOT" >&2
    exit 90
fi
used=$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits)
if [[ "$used" -gt 256 ]]; then
    echo "GPU4 busy: $used MiB; no process will be stopped." >&2
    exit 91
fi
mkdir -p "$ROOT"
trap 'code=$?; printf "%s\n" "$code" > "$ROOT/EXIT_CODE"; nvidia-smi -i 4 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true; exit "$code"' EXIT
git rev-parse HEAD > "$ROOT/BASE_COMMIT"
for model in sdar llada21; do
    for length in 8192 16384 32768; do
        cell=${model}_${length}_focus_optimized
        echo "START $cell $(date -Is)"
        "$PY" -B "$SCRIPT" --model "$model" --length "$length" \
            --limit 80 --repeats 3 --output "$ROOT/$cell" \
            > "$ROOT/$cell.log" 2>&1 || {
                tail -60 "$ROOT/$cell.log"
                exit 1
            }
        echo "DONE $cell $(date -Is)"
    done
done
"$PY" -B scripts/original/performance/summarize_narrative80_focus.py \
    --focus-root "$ROOT" --output "$ROOT/summary.json"
echo "ALL_DONE $(date -Is)"
