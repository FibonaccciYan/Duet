# LLaDA 与 SDAR 的 SparseDLM 稀疏推理运行时

本仓库为 LLaDA2.1-mini 与 SDAR-8B-Chat-b32 提供统一的稀疏扩散推理运行时，
包含查询稀疏、前缀稀疏（默认 Raw L1）、LoSA 和 LLaDA 混合专家 Triton 优化；
浮点 Hadamard L1 与 Adamas 作为对照选择器保留。
所有优化都通过运行时补丁生效，不修改模型权重。

当前默认已改为无 cost gate，且 SDAR 每层独立筛选。新测试表明 SDAR 顺序策略在
budget 256 时 Prefix 与 Query+Prefix 均通过 HumanEval 门槛；动态 confidence 策略
虽然端到端更快，但其 Query 与 Query+Prefix 均有明显质量回退，因此不作为默认。

## 当前结果

### 端到端速度

#### 自适应生产配置（Raw L1）

下列数字来自同一模型进程内的四模式配对消融，报告相对同优化稠密基线的
中位数加速比。所有点均在同一提交上独占运行 4 次。计时包含完整生成过程，
不包含模型加载，并固定生成计算预算；前缀选择器为学习后 Q/K 基底上的精确 L1。

| LLaDA 自适应查询+前缀稀疏-256 / 稠密 | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| 生成长度 256 | **1.225 倍** | **1.321 倍** | **1.817 倍** |
| 生成长度 768 | **1.095 倍** | **1.113 倍** | **1.883 倍** |

| SDAR 顺序生成查询+前缀稀疏-256 / 稠密 | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| 生成长度 256 | **1.401 倍** | **1.307 倍** | **1.364 倍** |
| 生成长度 768 | **1.446 倍** | **1.383 倍** | **1.541 倍** |

上表为旧的两层索引共享结果，只保留作历史速度参考。当前每层独立、无门控配置已完成
动态 confidence、budget 512 的重测：

| SDAR 动态 confidence 查询+前缀稀疏-512 / 稠密 | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| 生成长度 256 | **1.314 倍** | **1.269 倍** | **1.298 倍** |
| 生成长度 768 | **1.336 倍** | **1.319 倍** | **1.475 倍** |

该配置六点均加速，但不能作为质量保持候选：对应 HumanEval 为 113/115，而动态
稠密为 121/125。Prefix-only 在 budget 512 下质量与动态稠密相同，但 8K/16K
均为 0.984 倍，只有 32K 加速（1.103/1.145 倍）。

#### LLaDA Query 全零掩码移除组件验证（历史 Adamas）

为隔离 Query cached attention 中全零 mask 的影响，下表强制
`query_min_prefix_length=0` 和 `prefix_min_prefix_length=0`，固定
`query_dense_threshold=4`，在同一 H800 上对四种模式运行 3 次并报告中位数。
单元格为“秒 / 相对 Dense 加速比”；这是组件实验，不替代上述含自适应
门控的最终候选结果。

| 生成长度 | 上下文 | Dense / 秒 | Query | Prefix | Query+Prefix |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 8K | 1.224 | 1.829 / 0.669 倍 | 1.181 / 1.036 倍 | 1.328 / 0.921 倍 |
| 256 | 16K | 1.793 | 1.559 / 1.151 倍 | 1.499 / 1.196 倍 | 1.632 / 1.099 倍 |
| 256 | 24K | 2.685 | 2.364 / 1.136 倍 | 1.998 / 1.344 倍 | 2.123 / 1.265 倍 |
| 256 | 32K | 5.457 | 3.172 / 1.720 倍 | 3.495 / 1.561 倍 | 2.822 / **1.933 倍** |
| 512 | 8K | 2.153 | 1.899 / 1.134 倍 | 1.857 / 1.160 倍 | 2.285 / 0.942 倍 |
| 512 | 16K | 3.671 | 3.112 / 1.180 倍 | 2.374 / 1.546 倍 | 2.607 / 1.408 倍 |
| 512 | 24K | 4.174 | 3.716 / 1.123 倍 | 2.841 / 1.469 倍 | 3.038 / 1.374 倍 |
| 512 | 32K | 9.207 | 4.765 / 1.932 倍 | 4.665 / 1.974 倍 | 4.214 / **2.185 倍** |
| 768 | 8K | 3.442 | 3.210 / 1.072 倍 | 3.365 / 1.023 倍 | 3.677 / 0.936 倍 |
| 768 | 16K | 3.954 | 3.614 / 1.094 倍 | 3.278 / 1.206 倍 | 3.926 / 1.007 倍 |
| 768 | 24K | 5.083 | 4.300 / 1.182 倍 | 3.919 / 1.297 倍 | 4.103 / 1.239 倍 |
| 768 | 32K | 9.505 | 5.304 / 1.792 倍 | 4.741 / 2.005 倍 | 4.795 / **1.982 倍** |

最佳点为 32K/生成 512 的 Query+Prefix：9.207 秒降至 4.214 秒，
即 2.185 倍加速、耗时减少 54.24%。相对保留全零 mask 的对照，32K 的
Query-only 在生成 256/512/768 时分别再快 34.2%/28.0%/35.8%；
Query+Prefix 只再快 0.5%--1.1%，因为 Prefix 已经压缩了 K/V。

#### LLaDA 8K/16K Query+Prefix 修复（历史 Adamas）

阶段拆解定位到的根因不是组合实现本身变慢，而是收益与生成轨迹的耦合。8K/生成
512 的旧强制组合配置中，Prefix 已把 K/V 压到 256 token；此时 Query 行裁剪只让
单次 cached forward 从 23.84 ms 降到 23.55 ms，却把调用次数从 39 次改成 56 次，
所以 Prefix 的 1.840 秒变成 Query+Prefix 的 2.254 秒。16K 同样从 39 次变成
49 次。比例、稠密阈值、选择层和选择间隔搜索均不能稳定抵消这种轨迹变化。

最终修复只调整自适应门控：名义 8K 的固定前缀不足 8192 token，跳过无收益的
Prefix 压缩；Query 请求仍省略全零 attention mask，但 24K 以下不做行裁剪和双
cache。下表是提交 `00ee7bf` 在独占 H800 上的五次中位数，单元格为“秒 / 相对
Dense 加速比”。8K 的 Query+Prefix 输出校验和与 Query-only 在每次重复中一致。

| 生成长度 | 上下文 | Dense / 秒 | Query | Prefix | Query+Prefix |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 8K | 1.227 | 1.003 / 1.223 倍 | 1.227 / 1.000 倍 | 1.002 / **1.225 倍** |
| 256 | 16K | 1.796 | 1.484 / 1.210 倍 | 1.498 / 1.199 倍 | 1.484 / **1.210 倍** |
| 512 | 8K | 2.121 | 1.740 / 1.219 倍 | 2.122 / 0.999 倍 | 1.738 / **1.220 倍** |
| 512 | 16K | 3.632 | 2.373 / 1.531 倍 | 2.330 / 1.559 倍 | 2.295 / **1.583 倍** |
| 768 | 8K | 3.394 | 3.032 / 1.120 倍 | 3.396 / 1.000 倍 | 3.031 / **1.120 倍** |
| 768 | 16K | 3.924 | 2.956 / 1.328 倍 | 3.229 / 1.215 倍 | 3.228 / **1.216 倍** |

LLaDA 的 Query 行裁剪只在长前缀上划算，因此固定前缀不足 24K 时自动关闭；
maskless attention 仍作为 Query 方法的一部分保留。名义 8K 的 Query+Prefix
退化为 maskless Query，16K 启用 Prefix 与 maskless Query，32K 才再叠加 Query
行裁剪。SDAR 的主要收益来自查询稀疏；前缀稀疏在 24K 以下自动关闭，所以 8K、
16K 的查询+前缀配置实际退化为查询稀疏，32K 才叠加前缀稀疏。
两种稀疏存在交互，不能把单项加速比直接相乘。
非单调曲线的阶段拆解见 [SparseDLM 加速比与上下文长度](docs/context_scaling_analysis.md)。

### HumanEval 精度

评测同时报告官方判分和缩进归一化判分。缩进归一化只用于识别代码缩进格式问题，
不能替代官方成绩。

| 模型与配置 | 官方成绩 | 缩进归一化 | 状态 |
| --- | ---: | ---: | --- |
| LLaDA 同优化稠密基线 | 74/164 | 137/164 | 基线 |
| LLaDA Raw L1 前缀稀疏-256，强制启用 | 68/164 | 128/164 | 未通过 |
| LLaDA Raw L1 查询+前缀稀疏-256，强制启用 | **83/164** | **134/164** | 速度候选，通过 |
| LLaDA Query-only 原生 GQA，最短前缀 0 | **85/164** | **134/164** | 全零 mask 组件验证，通过 |
| SDAR 顺序生成同优化稠密基线 | 129/164 | 130/164 | 基线 |
| SDAR Raw L1 前缀稀疏-256，每层独立 | 128/164 | 128/164 | 通过 |
| SDAR Raw L1 查询+前缀稀疏-256，每层独立 | 128/164 | 129/164 | 通过 |
| SDAR Raw L1 前缀稀疏-512，每层独立 | 130/164 | 131/164 | 通过 |
| SDAR Raw L1 查询+前缀稀疏-512，每层独立 | 130/164 | 132/164 | 通过 |
| SDAR 动态生成稠密基线 | 121/164 | 125/164 | 动态策略自身已低于顺序策略 |
| SDAR 动态生成查询稀疏，比例 0.5 | 115/164 | 117/164 | 拒绝 |
| SDAR 动态 Raw L1 前缀稀疏-512 | 121/164 | 125/164 | 相对动态稠密通过 |
| SDAR 动态 Raw L1 查询+前缀稀疏-512 | 113/164 | 115/164 | 拒绝 |

质量门槛为：官方成绩相对同优化稠密基线回退不超过 2/164；缩进归一化成绩回退
不超过 2/164，或绝对成绩达到 130/164。输出发生变化后必须完整重跑 164 题。
Raw L1 的选择器隔离实验把前缀门控设为 0。LLaDA Prefix-only 未通过，
Query+Prefix 达到绝对归一化门槛。SDAR 每层独立筛选修复了旧两层共享配置在
budget 256 下的主要回退；但动态策略的 Query 路径仍未通过。官方成绩同时保留，
不把格式敏感分数隐藏。

### NarrativeQA 真实 32K 端到端对比

下表使用同一张 H800、同一批 91 个真实 NarrativeQA 样本：原文至少 32640 token，
提示截断到 32640，最多生成 128 token，并允许自然 EOS。加速比为稠密平均耗时
2.889 秒除以对应稀疏平均耗时；“实生”是实际生成 token 的均值。

| 预算 | 选择器 | 模式 | F1 | 端到端 / 秒 | 加速比 | 实生 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| - | Dense | Dense | 0.2525 | 2.889 | 1.000 倍 | 20.81 |
| 256 | Raw L1 | Prefix | 0.2132 | 2.024 | 1.427 倍 | 10.19 |
| 256 | Raw L1 | Query+Prefix | 0.2226 | 2.096 | 1.379 倍 | 10.88 |
| 256 | 浮点 Hadamard L1 | Prefix | 0.1246 | 1.924 | 1.502 倍 | 4.80 |
| 256 | 浮点 Hadamard L1 | Query+Prefix | 0.1385 | 2.012 | 1.436 倍 | 7.03 |
| 256 | Adamas | Prefix | 0.1665 | 2.013 | 1.436 倍 | 8.38 |
| 256 | Adamas | Query+Prefix | 0.1975 | 2.046 | 1.413 倍 | 9.10 |
| 1024 | Raw L1 | Prefix | **0.2575** | 2.033 | 1.422 倍 | 12.56 |
| 1024 | Raw L1 | Query+Prefix | 0.2448 | 2.074 | 1.393 倍 | 12.40 |
| 1024 | 浮点 Hadamard L1 | Prefix | 0.2261 | 1.942 | 1.488 倍 | 7.57 |
| 1024 | 浮点 Hadamard L1 | Query+Prefix | 0.2336 | 2.023 | 1.428 倍 | 9.68 |
| 1024 | Adamas | Prefix | 0.2451 | 2.033 | 1.421 倍 | 11.48 |
| 1024 | Adamas | Query+Prefix | 0.2417 | 2.085 | 1.386 倍 | 11.60 |

Raw L1 在两个预算、两种稀疏模式中的 F1 均高于浮点 Hadamard L1 和 Adamas，
其中 1024 预算的 Query+Prefix 与 Adamas 接近。因为自然 EOS 导致各方法生成
长度不同，这里的端到端加速同时包含生成轨迹差异，不能解释为等工作量的内核加速。

## 默认与候选配置

`eval_instruct/eval.sh` 是精度评测的配置入口，`src/sparse/config.py` 是 Python
接口默认值的来源。两种模型使用独立配置，不强行共享模型相关参数；SDAR 默认关闭
前缀稀疏，显式开启 Raw L1 时须注意上述 HumanEval 质量回退。

| 设置 | LLaDA | SDAR |
| --- | ---: | ---: |
| 数据类型 | BF16 | FP16 |
| 分块长度 / 去噪步数 | 32 / 32 | 32 / 32 |
| 重掩码策略 | LLaDA confidence 策略 | 顺序策略；dynamic confidence 已测试并拒绝作为统一默认 |
| 查询保留比例 | 0.7 | 0.5 |
| 查询选择层 | 1 | 5 |
| 选择间隔 | 4 | 1 |
| 查询转稠密阈值 | 0 | 0 |
| 查询最短前缀 | 0 | 0 |
| 前缀预算 | 256 | 256 |
| 前缀分块 | 1024 | 1024 |
| 前缀选择器 | Raw L1 | Raw L1 |
| 前缀最短长度 | 0 | 0 |
| Prefix 层索引共享 | 不共享 | 默认不共享；可显式恢复两层共享 |
| 前缀稀疏默认值 | 开启 | 关闭；论文查询+前缀实验显式开启 |
| LoSA | 关闭 | 关闭 |
| 混合专家补丁 | 开启 | 关闭 |

## 公平对比协议

主基线是**同优化稠密基线**：使用相同模型、数据类型、生成实现以及公共
Triton/混合专家优化，只关闭查询稀疏、前缀稀疏和 LoSA。原始官方稠密实现可
作为辅助数据，但不能用于计算主加速比。

1. 稠密与稀疏必须在同一显卡、同一模型进程中配对执行。
2. 输入、随机种子、分块、步数和生成预算必须一致；长上下文关闭终止符提前停止。
3. 每个形状和模式先完整预热；执行顺序轮换，上下文顺序交替。
4. 探索至少运行 3 次，最终数据至少运行 4 次，报告中位数、延迟、峰值显存和校验和。
5. 不把首次编译、模型加载、不同生成预算或不同显卡的差异算作稀疏收益。
6. 最终提交必须重跑稠密、仅前缀、仅查询、查询+前缀四种模式；六个长上下文点
   全部快于稠密后，才能宣称全范围加速。

唯一认可的长上下文计时入口如下：

```bash
CUDA_VISIBLE_DEVICES=0 /home/ysy/anaconda3/envs/llada/bin/python \
  scripts/bench_long_context.py \
  --model llada --mode query_prefix --ablation \
  --contexts 8192 16384 32768 --gen-length 256 \
  --prefix-token-budget 256 --prefix-selector raw_l1 --repeats 4 \
  --output /path/to/llada_gen256.json

CUDA_VISIBLE_DEVICES=0 /home/ysy/anaconda3/envs/dream/bin/python \
  scripts/bench_long_context.py \
  --model sdar --mode query_prefix --ablation \
  --contexts 8192 16384 32768 --gen-length 256 \
  --prefix-token-budget 256 --prefix-selector raw_l1 --repeats 4 \
  --output /path/to/sdar_gen256.json
```

脚本会记录提交、工作树状态、显卡标识、PyTorch/CUDA 版本以及输入输出校验和，
并在同一模式的重复输出不一致或实际生成长度错误时立即失败。

## 已保留的优化

### 公平评测与可选 cost gate

- 配对四模式消融、逐形状预热、轮换顺序、固定输出长度和校验和检查，消除了早期
  稠密独自承担首次编译所产生的约 2--4 倍虚高结果。
- 当前默认不启用长度或小 Query cost gate；相关阈值仍保留为显式配置，可用于部署
  时按实测成本开启。

### 前缀稀疏与 Raw L1

- 默认直接在模型学习得到的 Q/K 基底上计算精确 L1 距离，不做 Hadamard 变换或
  四桶量化。直接 QK、浮点 Hadamard L1 与 Adamas 仍可通过
  `PREFIX_SELECTOR=qk|hadamard_qk|adamas` 做对照。
- 滚动候选复用上一生成块的候选，减少重复扫描完整前缀。
- 固定提示词的键值缓存只编码一次；新生成块只刷新后缀，并为每层保存紧凑键值。
- Triton 分块因果预填充使用隐式分块元数据，不构造平方规模的注意力掩码；查询块
  只扫描可见的键值块。
- 历史 Adamas 32K 选择器微基准：LLaDA 从 12.22 毫秒降至 10.71 毫秒，SDAR 从
  16.55 毫秒降至 13.06 毫秒；LLaDA 32K 峰值显存从约 64 GiB 降至
  34--36 GiB。

### LLaDA 查询稀疏与混合专家

- 第 1 层置信度选择器和间隔 4 保留；当前默认从最短前缀开始启用 Prefix 与 Query
  行裁剪，不再隐式改变不同上下文长度实际运行的组件。
- Query 和 Query+Prefix 的 cached attention 省略全零 mask，使 PyTorch SDPA
  使用原生 GQA；Dense、Prefix-only 和 LoSA 保留 mask。完整 HumanEval 为
  85/164（缩进归一化 134/164），通过质量门。
- 混合专家使用融合路由、32x128x64 分块、下投影 8 个线程束、只计算一次
  SwiGLU、按路由顺序直接写回，以及按真实专家数量紧凑发射计算块。
- 紧凑混合专家计算块的阶段测量中，32K/生成 256 的稠密耗时从 7.676 秒降至
  5.463 秒，稀疏耗时从 5.035 秒降至 2.834 秒；nsys 中下投影和门控上投影
  分别减少约 84% 和 75%，校验和不变。

### SDAR 查询稀疏

- 查询路径复用一次投影得到的 Q/K/V，并跳过未使用的缓存 logits 与同步。
- RMSNorm 与 SwiGLU 使用模型原生实现；实验性 Triton 替换会改变数值，因此不再
  作为默认补丁安装。
- 顺序策略使用连续切片，并跳过只为置信度选择器服务的最终归一化。
- 删除顺序选择器归一化后，8K/生成 256 的稀疏耗时从 5.823 秒降至
  5.723 秒；Raw L1 最终六点查询+前缀加速比为 1.307--1.541 倍。

### LoSA

LoSA 将历史前缀和当前生成块的注意力拆开，缓存前缀输出与对数归一化项，并用
在线 softmax 合并。实现已通过全激活正确性对照，但当前没有可靠的端到端收益，
因此默认关闭，不进入论文最终配置。

## 已拒绝的优化方向

下列结论已有实测证据，在底层条件没有变化时不应重复搜索。

| 方向 | 结果 | 重访条件 |
| --- | --- | --- |
| LLaDA 查询间隔 8/16、比例 0.5/0.6 | 小规模混合专家效率抵消 token 减少，不能稳定改善 8K/16K | 选择器或混合专家成本结构改变 |
| 小专家 `BLOCK_M=16`、直接收集原 token | 无端到端收益或因非连续读取变慢 | 新数据布局或分组矩阵乘 |
| 注意力 `BLOCK_N=128` | 改变归约顺序，逐元素一致性失败 | 明确接受精度预算 |
| BF16 混合专家下投影 | HumanEval 为 72/129，且 8K/生成 768 仅 0.996 倍 | 不再重访 |
| LoSA 键采样与频繁刷新 | 额外估算未换来可靠端到端收益 | 前缀注意力再次成为主热点 |
| SDAR 强制 cuDNN SDPA | 8K/生成 256 从 5.84 秒变为 6.16 秒，首次编译约 62 秒 | 后端实质更新 |
| SDAR 串行无掩码 Triton 注意力 | 5.84 秒变为 5.96 秒 | 有高效分片归约与合并内核 |
| SDAR 外部 FlashAttention GQA | 查询 8--32 与 SDPA 约同速，查询 4 更慢 | 新版内核有明确收益 |
| SDAR SwiGLU 分块与 K 调参 | 最好内核收益约 1.5%，折算端到端小于 0.3% | SwiGLU 占比显著上升 |
| SDAR 查询转稠密阈值 2/4/8 | 8K/生成 256 从 5.823 秒退化至 5.887/5.980/6.370 秒 | 分块或选择器成本变化 |
| LLaDA block/steps 64/64 | Query 单次收益缩小且调用次数增加；32K/生成 256 从 1.081 倍降至 0.780 倍 | 生成与调度算法改变 |
| 直接 QK 前缀选择 | HumanEval（官方/归一化）：LLaDA Prefix/Combined 为 66/127、71/132，SDAR 为 125/126、123/127，因此只保留实验入口 | 专用融合选择器在最终提交上同时通过质量与速度门 |

### Adamas 阈值校准结论（历史对照）

校准工具测量选中 token 数、保留的 softmax 质量、精确等数量 top-k 召回率，以及
真实 V 上的归一化注意力输出误差。LLaDA 在单一重复合成提示词上的离线最优阈值
`Hq=[0,0,0]、Hk=[-2.5,0,2.5]`，将跨长度平均输出误差从约 0.573 降至
0.383、保留质量从 0.664 提至 0.754；但完整 HumanEval 只有 66/125，低于旧
生产阈值的 71/128。

较保守的 `Hq=[-0.1,0,0.1]、Hk=[-2.26,0,2.26]` 为 69/132，也因官方成绩
明显落后同优化稠密基线而拒绝。结论是：单一合成提示词的局部注意力重建指标不能
预测逐步代码生成质量。以下旧阈值仅供复现实验，不再是默认配置：LLaDA
`Hq=[-1.35,0,1.35]、Hk=[-2.26,0,2.26]`，SDAR
`Hq=[-1.50,0,1.49]、Hk=[-2.87,0,2.86]`。只有建立多样化真实代码提示词、
逐层且覆盖多个生成步的校准集后才重访。

### SDAR 动态策略结论

固定 b32、比例 0.5、选择层 5、间隔 1、阈值 1.0 时，完整 HumanEval 重测确认
动态稠密本身只有 121/125，低于顺序稠密 129/130。动态 Query 为 115/117；
Query+Prefix 在 budget 256/512/1024 分别为 115/117、113/115、114/117，均未
通过门槛。动态 Prefix 在 budget 512 和 1024 均为 121/125，可保持动态基线质量，
但 budget 512 的六点速度中有四点慢于稠密。因此不能把两个模型统一为当前这套
confidence-based Query 策略，SDAR 默认继续使用顺序策略。

| 不改比例的语义候选 | HumanEval 结果 | 结论 |
| --- | ---: | --- |
| 默认 32 题 | 29/29 | 筛选基线 |
| 只允许深层选中位置执行迁移 | 28/28 | 浅层迁移不是主因 |
| 固定 50% 预算内轮换陈旧掩码 | 27/27 | 挤掉关键高置信候选 |
| 刷新首个稀疏深层的全量键值 | 28/28 | 单层键值陈旧不是主因 |
| 首次稀疏步复用前一稠密步置信度 | 28/29 | 陈旧排序不能替代当前选择器 |
| 第 2 步刷新 | 30/30；历史完整 114/119 | 有改善但仍未通过质量门槛 |
| 刷新步直接使用完整最终 logits | 30/30；完整 114/118 | 未超过原第 2 步刷新，回退 |

这些结果说明动态策略的误差来自多层状态、候选排序与逐步迁移的耦合，不能靠单层
修补或局部“更精确”恢复。生产配置继续使用顺序策略。

## 一键环境安装

在 A800 机器上 clone 本仓库后执行：

```bash
bash install.sh
conda activate sparse
```

详见 [`README.install.md`](README.install.md)。

## 使用方法

仓库预期使用已有 CUDA、PyTorch、Triton 环境。默认 Raw L1 不需要 Hadamard
扩展；仅运行浮点 Hadamard L1 或 Adamas 对照时需要
`faster_hadamard_transform`。当前机器上的扩展来源为：

```text
/data0/ysy/Adamas/kernels/3rdparty/applied-ai/kernels/cuda/inference/hadamard_transform
```

快速生成：

```bash
MODEL_TYPE=llada bash scripts/generate.sh
MODEL_TYPE=sdar bash scripts/generate.sh
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
    losa=False,
)
```

HumanEval 与 GSM8K 四模式矩阵：

```bash
MODEL_TYPE=llada CUDA_VISIBLE_DEVICES=3,4,5 \
  bash eval_instruct/run_humaneval_matrix.sh
MODEL_TYPE=sdar CUDA_VISIBLE_DEVICES=3,4,5 \
  bash eval_instruct/run_humaneval_matrix.sh

MODEL_TYPE=llada CUDA_VISIBLE_DEVICES=3,4,5 \
  bash eval_instruct/run_gsm8k_matrix.sh
```

最低回归检查：

```bash
CUDA_VISIBLE_DEVICES=0 /home/ysy/anaconda3/envs/llada/bin/python \
  -m unittest discover -s tests -q
git diff --check
```

## 代码结构

```text
src/sparse/
├── __init__.py         公共导出
├── api.py              模型识别与 patch_model(...)
├── config.py           模型族专属默认配置
├── sparse_ops.py       公共缓存、Raw L1/对照选择器、LoSA、RoPE、在线合并
├── triton_kernels.py   唯一 CUDA 内核实现
├── llada_patch.py      LLaDA 生成、注意力、查询稀疏与混合专家
└── sdar_patch.py       SDAR 生成、注意力与查询稀疏

src/dense/               Dense baseline adapter
src/losa/                Paper-LoSA operators, generation, and model patch
src/focus/               Integrated FOCUS importance, eviction, and generation
```

依赖方向保持单向：

```text
api -> llada_patch / sdar_patch -> sparse_ops -> triton_kernels
```

辅助工具：

- `scripts/bench_sparse_ops.py`：稀疏算子微基准。
- `scripts/collect_adamas_hqhk.py`：Hq/Hk 与注意力感知阈值校准。
- `scripts/analyze_llada_layer_overlap.py`：LLaDA 选择层重合度分析。
- `scripts/analyze_sdar_layer_overlap.py`：SDAR 选择层重合度分析。
- `scripts/analyze_llada_query_losa_correlation.py`：查询稀疏与 LoSA 相关性分析。

`results/` 与 `experiments/` 是本地生成产物，已由 `.gitignore` 排除。长期实验结果
应保存在仓库外，并记录提交、显卡和软件版本元数据。旧阶段数据见
[`RESULTS.md`](RESULTS.md)，Adamas 早期分析见 [`docs/adamas.md`](docs/adamas.md)。

## 下一步

1. 在最终论文提交上统一重跑两模型六点四模式配对基准，并保存绝对延迟、峰值显存、
   校验和与环境元数据。
2. 在同一提交重跑稠密、仅前缀、仅查询、查询+前缀 HumanEval 完整矩阵。
3. LLaDA 只继续分析预填充/解码注意力和前缀缓存拼接；不为历史 Adamas 路径
   新增专用融合，除非真实任务重新显示明确收益。
4. SDAR 只在具备高效分片合并或新版后端时重访注意力；动态策略在新的多层状态
   方法出现前停止局部启发式调整。
5. 每个候选只改变一个因素：单测与校验和 → 六点端到端 → 输出变化时完整精度。
