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
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
APPLIED_AI_REPO="${SPARSEDLM_APPLIED_AI_REPO:-https://github.com/meta-pytorch/applied-ai.git}"
APPLIED_AI_COMMIT="${SPARSEDLM_APPLIED_AI_COMMIT:-2391954b19988bd76cf3c2ea84c1ce74b68d568b}"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
THIRD_PARTY_DIR="${SPARSEDLM_THIRD_PARTY_DIR:-$REPO_ROOT/third_party/applied-ai}"
MAX_JOBS="${MAX_JOBS:-8}"

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
  "$CONDA_EXE" create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
else
  log "Reusing conda environment: $ENV_NAME"
fi
CONDA_PREFIX="$("$CONDA_EXE" info --base)/envs/$ENV_NAME"
PYTHON="$CONDA_PREFIX/bin/python"
export PIP_INDEX_URL="$PIP_INDEX_URL"
export PIP_DEFAULT_INDEX_URL="$PIP_INDEX_URL"

# --- Optional system build compiler -----------------------------------------
# CUDA 12.6 rejects the default GCC 13.3 on Ubuntu 24.04. GCC 12 is tested.
if ! command -v gcc-12 >/dev/null 2>&1 || ! command -v g++-12 >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing GCC 12 for the CUDA extension build"
    if [[ "${EUID}" -eq 0 ]]; then
      DEBIAN_FRONTEND=noninteractive apt-get update -qq
      DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc-12 g++-12
    elif command -v sudo >/dev/null 2>&1; then
      sudo -n DEBIAN_FRONTEND=noninteractive apt-get update -qq || sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
      sudo -n DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc-12 g++-12 || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gcc-12 g++-12
    else
      die "gcc-12/g++-12 are required; install them or rerun as root"
    fi
  else
    die "gcc-12/g++-12 are required; install them manually"
  fi
fi

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
export CC="${CC:-/usr/bin/gcc-12}"
export CXX="${CXX:-/usr/bin/g++-12}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-12}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS

log "Python: $($PYTHON --version)"
log "CUDA: $($CUDA_HOME/bin/nvcc --version | tail -1)"

# --- Python dependencies ------------------------------------------------------
log "Installing PyTorch $TORCH_VERSION + CUDA 12.4 wheels"
"$PYTHON" -m pip install --upgrade pip setuptools wheel
"$PYTHON" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"

log "Installing pinned Python dependencies"
"$PYTHON" -m pip install -r requirements/requirements.txt

log "Installing FlashAttention 2.6.3 (sm_80/sm_90 wheel)"
"$PYTHON" -m pip install --no-cache-dir "$FLASH_ATTN_WHEEL"

# --- faster_hadamard_transform ------------------------------------------------
log "Preparing applied-ai at $APPLIED_AI_COMMIT"
if [[ ! -d "$THIRD_PARTY_DIR/.git" ]]; then
  mkdir -p "$(dirname "$THIRD_PARTY_DIR")"
  git clone --depth 1 --filter=blob:none --sparse "$APPLIED_AI_REPO" "$THIRD_PARTY_DIR"
fi
git -C "$THIRD_PARTY_DIR" sparse-checkout set kernels/cuda/inference/hadamard_transform
current_commit="$(git -C "$THIRD_PARTY_DIR" rev-parse HEAD)"
if [[ "$current_commit" != "$APPLIED_AI_COMMIT" ]]; then
  git -C "$THIRD_PARTY_DIR" fetch --depth 1 origin "$APPLIED_AI_COMMIT"
  git -C "$THIRD_PARTY_DIR" checkout "$APPLIED_AI_COMMIT"
fi
log "Building faster_hadamard_transform"
"$PYTHON" -m pip install --no-build-isolation \
  "$THIRD_PARTY_DIR/kernels/cuda/inference/hadamard_transform"

# --- Verify -------------------------------------------------------------------
log "Verifying core runtime"
"$PYTHON" - <<'PY'
import sys
import torch
import triton
import transformers
import accelerate
import flash_attn
import faster_hadamard_transform as hadamard
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
x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
y = hadamard.hadamard_transform(x)
assert out.shape == q.shape and torch.isfinite(out).all()
assert y.shape == x.shape and torch.isfinite(y).all()
print("flash_attention_check ok")
print("hadamard_check ok")
PY

cat <<EOF

Environment setup complete.
Activate it with:
    conda activate $ENV_NAME

Run from the repository root:
    PYTHONPATH=. python scripts/run_longbench_quality.py --help
EOF
