#!/usr/bin/env bash
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ROOT=results/longbench_strict_quality_20260918
# Wrapper restores GPU5 before this launcher publishes terminal status.
bash scripts/original/performance/run_reserved_gpu5.sh \
    bash scripts/original/performance/run_quality_matrix.sh full
rc=$?
printf '%s\n' "$rc" > "$ROOT/EXIT_CODE"
date -Is > "$ROOT/FINISHED_AT"
exit "$rc"
