#!/usr/bin/env bash
# Install pinned dependencies into the current Python environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
TORCH_VERSION="${SPARSEDLM_TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${SPARSEDLM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

printf '%s\n' "[install] Python: $PYTHON"
printf '%s\n' "[install] Upgrading build tools"
"$PYTHON" -m pip install --upgrade pip setuptools wheel

printf '%s\n' "[install] Installing PyTorch $TORCH_VERSION + CUDA 12.4 wheels"
"$PYTHON" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

printf '%s\n' "[install] Installing pinned dependencies"
"$PYTHON" -m pip install -r requirements.txt

printf '%s\n' "[install] Complete. Run from the repository root with PYTHONPATH=."
