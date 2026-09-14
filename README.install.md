# One-command environment setup

On the A800 machine, clone this repository and run:

```bash
bash install.sh
conda activate sparse
```

The script creates a conda environment named `sparse` with the validated stack:

- Python 3.10
- PyTorch 2.5.1+cu124
- Triton 3.1.0
- Transformers 4.57.1
- FlashAttention 2.6.3
- `faster_hadamard_transform` built from `meta-pytorch/applied-ai`

It also installs GCC 12 when needed, because CUDA 12.6 rejects Ubuntu 24.04's
default GCC 13.3 while building the Hadamard extension.

Useful overrides:

```bash
SPARSEDLM_ENV_NAME=sparse       # conda environment name
SPARSEDLM_PYTHON_VERSION=3.10   # Python version
SPARSEDLM_TORCH_INDEX_URL=...   # PyTorch wheel index
PIP_INDEX_URL=...               # PyPI mirror
MAX_JOBS=8                      # parallel C++/CUDA compile jobs
```

The Hadamard kernel is built for `TORCH_CUDA_ARCH_LIST=8.0` by default. Set
another architecture list explicitly for non-A800 GPUs.
