# Installation

`install.sh` installs pinned dependencies into the current Python environment.
It does not create a personal Conda environment, discover GPUs, run verification
commands, smoke jobs, tests, models, or benchmarks.

Use Python 3.10 with a CUDA 12.4-compatible toolkit. From the repository root:

```bash
PYTHON=${PYTHON:-python3} bash install.sh
```

The script installs PyTorch 2.5.1 from the official CUDA 12.4 wheel index and
then installs pinned packages from `requirements.txt`. The FlashAttention wheel
must match the active Python, PyTorch, and CUDA ABI.
