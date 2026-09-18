#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT=results/longbench_strict_quality_20260918
PY=/data0/ysy/anaconda3/envs/sparse/bin/python
stage=${1:?smoke or full}
[[ "$stage" == smoke || "$stage" == full ]]
mkdir -p "$ROOT/$stage"
if [[ "$stage" == full ]]; then
    [[ -f "$ROOT/SMOKE_PASSED" ]]
    date -Is > "$ROOT/FULL_STARTED"
fi
git rev-parse HEAD > "$ROOT/base_commit.txt"
for family in llada sdar; do
    for budget in 256 512 1024; do
        cell=${family}_b${budget}
        output="$ROOT/$stage/$cell"
        mkdir -p "$output"
        extra=()
        if [[ "$stage" == smoke ]]; then
            extra+=(--limit 1 --collect_prefix_selection_stats)
        fi
        if [[ "$family" == llada ]]; then
            model=/data0/ysy/models/LLaDA2.1-mini
            extra+=(--threshold 0.7 --editing_threshold 0.5)
        else
            model=/data0/ysy/models/SDAR-8B-Chat-b32
            extra+=(--threshold 0.95 --remasking_strategy low_confidence_dynamic)
        fi
        echo "START $stage $cell $(date -Is)"
        "$PY" -B scripts/original/quality/longbench_quality.py \
            --family "$family" --method sparse_optimized --model_path "$model" \
            --data_dir /data0/gs/losa_accuracy_prep/data/longbench_all21_full \
            --tasks hotpotqa triviaqa narrativeqa qasper multifieldqa_en \
            --max_context_tokens 32768 --block_length 32 --steps 32 \
            --query_sparse --prefix_sparse --prefix_strict_budget \
            --prefix_token_budget "$budget" --eos_early_stop --seed 42 \
            --output_dir "$output" "${extra[@]}" > "$output/run.log" 2>&1 || {
                tail -60 "$output/run.log"
                exit 1
            }
        "$PY" -B scripts/original/performance/validate_strict_quality.py "$output" "$stage" "$budget"
        echo "DONE $stage $cell $(date -Is)"
    done
done
if [[ "$stage" == smoke ]]; then date -Is > "$ROOT/SMOKE_PASSED"; fi
if [[ "$stage" == full ]]; then date -Is > "$ROOT/ALL_DONE"; fi
