#!/usr/bin/env bash
# One-command environment setup for the A800 inference stack.
#   bash install.sh
#   conda activate sparse
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${SPARSEDLM_ENV_NAME:-sparse}"
PYTHON_VERSION="${SPARSEDLM_PYTHON_VERSION:-3.10}"
TORCH_VERSION="${SPARSEDLM_TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${SPARSEDLM_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
CONDA_CHANNEL="${SPARSEDLM_CONDA_CHANNEL:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"

log() { printf '\n\033[1;34m[install]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[install error]\033[0m %s\n' "$*" >&2; exit 1; }

# --- Locate or bootstrap conda ---------------------------------------------
find_conda() {
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    echo "$CONDA_EXE"
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return
  fi
  local candidate
  for candidate in "$HOME/miniforge3" "$HOME/miniconda3" "/root/miniforge3" "/root/miniconda3" "/opt/conda"; do
    if [[ -x "$candidate/bin/conda" ]]; then
      echo "$candidate/bin/conda"
      return
    fi
  done
  return 1
}

CONDA_EXE="$(find_conda || true)"
if [[ -z "$CONDA_EXE" ]]; then
  log "Installing Miniforge to $HOME/miniforge3"
  case "$(uname -m)" in
    x86_64) MINIFORGE_ARCH="x86_64" ;;
    aarch64|arm64) MINIFORGE_ARCH="aarch64" ;;
    *) die "unsupported architecture: $(uname -m)" ;;
  esac
  installer="/tmp/Miniforge3-${MINIFORGE_ARCH}.sh"
  url="${MINIFORGE_URL:-https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${MINIFORGE_ARCH}.sh}"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress "$url" -O "$installer"
  elif command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$installer"
  else
    die "need wget or curl to bootstrap Miniforge"
  fi
  "$installer" -b -p "$HOME/miniforge3"
  CONDA_EXE="$HOME/miniforge3/bin/conda"
fi
CONDA_BASE="$("$CONDA_EXE" info --base)"
export PATH="$CONDA_BASE/bin:$PATH"
log "Using conda: $CONDA_EXE"

# --- Create environment ------------------------------------------------------
if ! "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  log "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
  conda_channel_args=()
  if [[ -n "$CONDA_CHANNEL" ]]; then
    conda_channel_args=(--override-channels --channel "$CONDA_CHANNEL")
  fi
  "$CONDA_EXE" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y "${conda_channel_args[@]}"
else
  log "Reusing conda environment: $ENV_NAME"
fi
CONDA_PREFIX="$("$CONDA_EXE" info --base)/envs/$ENV_NAME"
PYTHON="$CONDA_PREFIX/bin/python"
export PIP_INDEX_URL="$PIP_INDEX_URL"
export PIP_DEFAULT_INDEX_URL="$PIP_INDEX_URL"

if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-12.6 /usr/local/cuda-12.4; do
    if [[ -x "$candidate/bin/nvcc" ]]; then
      export CUDA_HOME="$candidate"
      break
    fi
  done
fi
[[ -x "${CUDA_HOME:-}/bin/nvcc" ]] || die "CUDA nvcc not found; set CUDA_HOME to a CUDA 12.x toolkit"
export PATH="$CUDA_HOME/bin:$PATH"

log "Python: $($PYTHON --version)"
log "CUDA: $($CUDA_HOME/bin/nvcc --version | tail -1)"

# --- Python dependencies ------------------------------------------------------
log "Installing PyTorch $TORCH_VERSION + CUDA 12.4 wheels"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

log "Installing pinned Python dependencies"
"$PYTHON" -m pip install -r requirements.txt

# --- Verify -------------------------------------------------------------------
log "Verifying core runtime"
"$PYTHON" - <<'PY'
import sys
import torch
import triton
import transformers
import accelerate
import flash_attn
from flash_attn import flash_attn_func

print("python", sys.version.split()[0])
print("torch", torch.__version__, torch.version.cuda)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
print("gpu", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("triton", triton.__version__)
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("flash_attn", flash_attn.__version__)
q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
out = flash_attn_func(q, q, q, causal=True)
assert out.shape == q.shape and torch.isfinite(out).all()
print("flash_attention_check ok")
PY

cat <<EOF

Environment setup complete.
Activate it with:
    conda activate $ENV_NAME

Run from the repository root:
    PYTHONPATH=. python scripts/quality/longbench_quality.py --help
EOF
