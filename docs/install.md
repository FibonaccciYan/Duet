# 环境安装

推荐环境为 Python 3.10、CUDA 12.4；当前依赖组合已在 A800（sm_80）上验证。

## 手动安装

在仓库根目录执行：

```bash
conda create -n sparse python=3.10 -y
conda activate sparse
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## 一键安装

```bash
bash install.sh
conda activate sparse
```

可通过环境变量调整环境名、Python 版本、PyTorch wheel 源和 PyPI 镜像：

```bash
SPARSEDLM_ENV_NAME=sparse
SPARSEDLM_PYTHON_VERSION=3.10
SPARSEDLM_CONDA_CHANNEL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
SPARSEDLM_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
PIP_INDEX_URL=https://pypi.org/simple
```

默认 Raw L1 不依赖 `faster_hadamard_transform`，基础环境不安装该扩展。
