#!/usr/bin/env bash
set -euo pipefail
method=$1
task=$2
shift 2
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -B -m scripts.common.run --method "$method" --task "$task" "$@"
