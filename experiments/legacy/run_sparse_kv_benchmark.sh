#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ysy/anaconda3/envs/llada/bin/python}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/throughput_sparse_kv_0p5_gen512}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/run_${TIMESTAMP}}"

mkdir -p "${OUTPUT_DIR}"

COMMAND=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/bench_sparse_kv_throughput.py"
  --output_dir "${OUTPUT_DIR}"
  --gen_length 512
  --block_length "${BLOCK_LENGTH:-32}"
  --steps "${STEPS:-32}"
  --sparse_ratio 0.5
  --dense_fallback_mask_count "${DENSE_FALLBACK_MASK_COUNT:-4}"
  --warmup "${WARMUP:-1}"
  --repeats "${REPEATS:-3}"
)

printf '%q ' "${COMMAND[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"

"${COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/benchmark.log"

echo "Benchmark artifacts saved to: ${OUTPUT_DIR}"
