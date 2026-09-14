# SparseDLM：LLaDA 与 SDAR 稀疏推理运行时

本仓库为 LLaDA2.1-mini 与 SDAR-8B-Chat-b32 提供统一的扩散语言模型推理运行时，
包含查询稀疏、前缀稀疏、Dense、LoSA、FOCUS 及实验性 v2 runtime。所有优化均以
运行时补丁生效，不修改模型权重。

当前 SparseDLM 默认使用 Raw L1 前缀选择器、关闭 cost gate，并让 SDAR 每层独立
筛选前缀索引。SDAR 默认采用顺序重掩码策略；动态 confidence 策略因质量回退不作为
默认配置。

## 当前默认配置

配置来源为 `src/sparse/config.py`，精度评测入口为 `eval_instruct/eval.sh`。

| 设置 | LLaDA | SDAR |
| --- | ---: | ---: |
| 数据类型 | BF16 | FP16 |
| 分块长度 / 去噪步数 | 32 / 32 | 32 / 32 |
| 重掩码策略 | confidence | sequential |
| 查询保留比例 | 0.7 | 0.5 |
| 查询选择层 | 1 | 5 |
| 选择间隔 | 4 | 1 |
| 查询转稠密阈值 | 0 | 0 |
| 前缀预算 | 256 | 256 |
| 前缀选择器 | Raw L1 | Raw L1 |
| Prefix 层索引共享 | 关闭 | 关闭 |
| 前缀稀疏默认值 | 开启 | 关闭 |
| LoSA | 关闭 | 关闭 |
| 混合专家补丁 | 开启 | 关闭 |

## 质量结果

HumanEval 同时保留官方判分和缩进归一化诊断；后者不能替代官方成绩。质量门槛为：
官方成绩相对同配置稠密基线回退不超过 2/164；归一化成绩回退不超过 2/164，或绝对
成绩达到 130/164。

| 模型与配置 | 官方成绩 | 缩进归一化 | 状态 |
| --- | ---: | ---: | --- |
| LLaDA 同优化稠密基线 | 74/164 | 137/164 | 基线 |
| LLaDA Raw L1 Prefix-256 | 68/164 | 128/164 | 未通过 |
| LLaDA Raw L1 Query+Prefix-256 | **83/164** | **134/164** | 通过 |
| SDAR sequential 稠密基线 | 129/164 | 130/164 | 基线 |
| SDAR Raw L1 Prefix-256，每层独立 | 128/164 | 128/164 | 通过 |
| SDAR Raw L1 Query+Prefix-256，每层独立 | 128/164 | 129/164 | 通过 |
| SDAR Raw L1 Prefix-512，每层独立 | 130/164 | 131/164 | 通过 |
| SDAR Raw L1 Query+Prefix-512，每层独立 | 130/164 | 132/164 | 通过 |
| SDAR dynamic 稠密基线 | 121/164 | 125/164 | 辅助基线 |
| SDAR dynamic Query-0.5 | 115/164 | 117/164 | 未通过 |
| SDAR dynamic Query+Prefix-512 | 113/164 | 115/164 | 未通过 |

更完整的实验条件与辅助结果见 [`paper/`](paper/)；不要把其中明确标记为历史配置的
结果与当前默认实现混用。

## 环境配置

推荐 Python 3.10 与 CUDA 12.4：

```bash
conda create -n sparse python=3.10 -y
conda activate sparse
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

也可使用一键安装脚本：

```bash
bash install.sh
conda activate sparse
```

默认 Raw L1 不依赖 `faster_hadamard_transform`。浮点 Hadamard L1 和 Adamas 仅作为
可选对照入口保留，运行这些入口时需自行安装该扩展。详细说明见
[`docs/install.md`](docs/install.md)。

## 快速使用

SparseDLM 生成：

```bash
conda activate sparse
MODEL_TYPE=llada PYTHON="$CONDA_PREFIX/bin/python" bash scripts/generate.sh
MODEL_TYPE=sdar PYTHON="$CONDA_PREFIX/bin/python" bash scripts/generate.sh
```

Python 接口：

```python
from src.sparse import patch_model

patch_model(
    model,
    model_name="llada",  # 或 "sdar"
    query_sparse=True,
    prefix_sparse=True,
    prefix_token_budget=256,
)
```

Dense、LoSA 与 FOCUS runtime：

```bash
python scripts/run_mode.py --family llada --mode dense
python scripts/run_mode.py --family llada --mode losa
python scripts/run_mode.py --family llada --mode focus
python scripts/run_mode.py --family llada --mode losa_v2
python scripts/run_mode.py --family llada --mode focus_v2
```

HumanEval 或 GSM8K 四模式矩阵：

```bash
MODEL_TYPE=llada CUDA_VISIBLE_DEVICES=3,4,5 bash eval_instruct/run_humaneval_matrix.sh
MODEL_TYPE=sdar CUDA_VISIBLE_DEVICES=3,4,5 bash eval_instruct/run_humaneval_matrix.sh
MODEL_TYPE=llada CUDA_VISIBLE_DEVICES=3,4,5 bash eval_instruct/run_gsm8k_matrix.sh
```

## 公平评测协议

主基线是同优化稠密基线：模型、数据类型、生成实现与公共 Triton/MoE 优化相同，只
关闭查询稀疏、前缀稀疏和 LoSA。

1. 稠密与稀疏在同一显卡、同一模型进程中配对执行。
2. 输入、随机种子、分块、步数和生成预算保持一致。
3. 每个形状和模式先预热，执行顺序轮换。
4. 探索至少重复 3 次，最终数据至少重复 4 次并报告中位数、显存和校验和。
5. 不把首次编译、模型加载、不同输出预算或不同显卡计入稀疏收益。
6. 输出变化后必须执行完整质量评测。

长上下文四模式配对入口：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/bench_long_context.py \
  --model llada --mode query_prefix --ablation \
  --contexts 8192 16384 32768 --gen-length 256 \
  --prefix-token-budget 256 --prefix-selector raw_l1 --repeats 4 \
  --output /path/to/llada_gen256.json
```

## 实现结构

```text
src/sparse/       SparseDLM 查询/前缀稀疏与共享 CUDA 内核
src/dense/        同生成框架的稠密基线
src/losa/         Paper-LoSA baseline
src/focus/        FOCUS baseline
src/losa_v2/      实验性 LoSA v2 runtime
src/focus_v2/     实验性 FOCUS v2 runtime
src/model/        自包含模型适配代码
eval_instruct/    lm-eval 任务与质量评测入口
scripts/          生成、分析与性能评测工具
paper/            当前研究说明与辅助实验结果
docs/             安装与 runtime 接口文档
```

最低回归检查：

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s tests -p 'test_*.py' -q
python -m pip check
git diff --check
```

`results/`、`experiments/` 和评测输出是本地生成产物，已由 `.gitignore` 排除，不应
作为可从 Git checkout 复现的证据。发布结果时需同时保存提交、完整配置、显卡和软件
版本元数据。
