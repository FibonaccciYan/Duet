# SparseDLM runtime for LLaDA and SDAR

This repository provides one GPU runtime for Query Sparse, Prefix Sparse,
LoSA, and the LLaDA MoE optimization. Checkpoint files are never modified.

## 当前状态与会话接手指南

本节是后续开发与论文实验的入口，状态快照为
`query_losa_correlation@bd468ac`。后文保留了历史实验，但除非在本节明确
列为当前结果，否则不能直接作为论文最终数字。

### 目标与验收准则

目标是在尽量保持生成质量的同时，使 LLaDA 和 SDAR 的
Query+Prefix Sparse 相对 dense 获得公平、可复现的端到端加速。论文重点测试
8K/16K/32K context 与 `gen_length=256/768`，而不是只比较 kernel 或短任务吞吐。

论文主 baseline 是 **matched optimized dense**：使用相同模型、dtype、生成
实现及所有公共 Triton/MoE 优化，只关闭 Query Sparse、Prefix Sparse 和 LoSA。
原始官方 dense 实现可以作为辅助 baseline 单独报告，但不能与优化后的 sparse
直接计算主加速比。所有结论遵守以下规则：

1. 每组 dense/sparse A/B 必须在同一 GPU、同一模型进程中 paired 执行；不同
   context 或模型可以分配到不同 GPU 并行。
2. 输入、seed、生成长度和去噪配置必须相同；长上下文测试关闭 EOS early stop，
   保证实际生成长度一致。
3. 每个 mode 都先用相同 shape 完整 warmup，执行顺序轮换、context 顺序交替；
   当前探索至少 3 次，论文最终数据至少 4 次并报告 median、绝对延迟、speedup、
   peak memory 和 output checksum。
4. 不得把首次编译、模型加载、不同实际输出长度或不同 GPU 的绝对时间计为
   sparse 收益。`scripts/bench_long_context.py --paired/--ablation` 是唯一认可的
   长上下文计时入口。
5. HumanEval 同时报告 official 与 indentation-normalized；后者不能替代前者。
   一旦输出发生变化，必须完整重跑。official 相对 matched dense 回退不得超过
   **2/164 题**；indentation-normalized 回退不超过 **2/164 题**或绝对成绩达到
   **130/164** 均可接受。
6. 论文最终配置还必须在同一最终提交上重跑 dense、Prefix-only、Query-only、
   Query+Prefix；只有六个长上下文点的 median 都快于 dense，才宣称全范围加速。

### 当前生产配置与可信结果

`eval_instruct/eval.sh` 是精度配置的 source of truth。两种模型拥有独立 config，
不要为了统一接口而强行共享数值参数。当前论文速度候选使用 Prefix budget 256、
LoSA off。LLaDA 在 prefix 小于 4K 时关闭 Prefix Sparse、小于 24K 时关闭
Query Sparse；SDAR 在 prefix 小于 24K 时关闭无收益的 Prefix Sparse。

当前 LLaDA 四模式、四次 paired 消融结果如下；这些结果已经排除首次 shape
编译偏差：

| LLaDA Query+Prefix-256 / dense | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| `gen_length=256` | 1.036x | 1.193x | 1.928x |
| `gen_length=768` | 1.017x | 1.208x | 1.941x |

对应 HumanEval 为 74/164 official、132/164 indentation-normalized。另有
Prefix-1024 calibrated-quality 配置达到 74/164 official、137/164 normalized；
两者不是同一论文配置，不得混用精度与速度数字。最新精确 kernel 优化保持了
六点 synthetic output checksum；现有证据未显示精度变化，但论文最终提交仍需
完整重跑上述 HumanEval 矩阵。

SDAR 当前四模式、四次 paired 消融确认 Query 是主要收益来源；HumanEval 为
127/164 official、129/164 normalized。SDAR-b4 在现有配置下慢于 b32，暂不替代
b32 主实验。24K Prefix gate 与 sequential selector 跳过无用 norm 后结果为：

| SDAR Query+Prefix-256 / dense | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| `gen_length=256` | 1.410x | 1.338x | 1.368x |
| `gen_length=768` | 1.426x | 1.402x | 1.533x |

8K/16K 的 Query+Prefix 与 Query-only checksum 一致；六点输出均与删除 selector
norm 前一致。

`low_confidence_dynamic` 在相同 b32、ratio 0.5、threshold 1.0 配置下仍快于
matched dense，但慢于 sequential；三次 paired 中位数如下：

| SDAR dynamic Query+Prefix-256 / dense | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| `gen_length=256` | 1.325x | 1.262x | 1.288x |
| `gen_length=768` | 1.338x | 1.309x | 1.446x |

完整 HumanEval 中 dynamic dense 为 121/164 official、124/164
indentation-normalized；dynamic Query Sparse 仅为 115/164、117/164，相对其
matched dense 再低 6/7 题，也分别比 sequential Query Sparse 低 12 题。两者均
未通过质量门槛，因此生产配置继续使用 sequential。官方 README 推荐的是 b4、
dynamic threshold 0.9，不能作为本项目 b32/threshold 1.0 配置的精度证据。
提高 dynamic Query ratio 没有得到可验收配置：ratio 0.75 提升至 118/164
official、121/164 normalized，但相对 dynamic dense 仍低 3/3；ratio 0.875
反而降至 113/164、120/164，说明该精度不是随计算量单调恢复。停止继续调整
ratio，且不为未通过质量门槛的候选补跑六点速度。

后续精度研究只保留两条路线。Adamas 不再用 pooled Hq/Hk 边缘分布的经验
分位点直接决定阈值，而应在固定 budget 下优化真实 attention top-k recall、保留的
softmax mass 或 attention-output error，并分别隔离量化阈值与 rolling-candidate
截断造成的误差；先搜索每模型一个对称 Hq/Hk 阈值对，只有全局阈值确实不足时
才增加逐层参数。SDAR dynamic 固定 ratio 0.5、layer 5、interval 1 和 threshold
1.0，优先检查深层 KV/state 一致性。当前树的 32 题 matched 筛选为：默认
`deep_only_transfer=false` 29/29 official/normalized，只允许深层位置 transfer
反而为 28/28；单次 `refresh_step=2` 为 30/30，并复现了历史完整 164 题中
111/114→114/119 的改善方向。因此停止 deep-only 路线，下一候选必须直接解决
未选 mask 的陈旧深层 KV，而不是继续调整 ratio。

第一轮 Adamas 阈值校准已经完成。校准工具现在除 Hq/Hk 分布外，还在固定
Prefix budget 256 下测量选中 token 数、保留 softmax mass、exact top-count oracle
recall，以及使用真实 V 的 normalized attention-output error。8K/16K/32K 的离线
最优全局阈值 `Hq=[0,0,0]、Hk=[-2.5,0,2.5]` 将平均 output error 从旧阈值的
约 0.573 降到 0.383、retained mass 从 0.664 提高到 0.754，但完整 HumanEval
仅为 66/164 official、125/164 normalized，低于旧生产阈值的 71/128。较保守的
`Hq=[-0.1,0,0.1]、Hk=[-2.26,0,2.26]` 也只有 69/132。说明单个 synthetic
repeated prompt 上的局部 attention 重建指标不能可靠预测逐步生成质量；生产阈值
保持 `Hq=[-1.35,0,1.35]、Hk=[-2.26,0,2.26]`。停止继续扫全局阈值；若重访，
先使用多样化真实代码 prompt 做逐层、多生成步校准，再以完整 HumanEval 验收。
离线明细保存在 `results/adamas_attention_output_llada_{8192,16384,32768}.json`；
完整生成结果位于仓库外 `../llada_exp/adamas_attention_{q0_k25,q01_k226}_full_20260907`。

### 已保留的优化

| 子系统 | 当前实现 | 已确认收益或作用 |
| --- | --- | --- |
| 公平评测 | paired/ablation、逐 shape warmup、固定输出长度、轮换顺序、commit/GPU/version 元数据与 checksum fail-fast | 消除了旧报告中 dense 独自承担首次编译、虚高至约 2--4x 的错误加速 |
| Prefix cost gate | LLaDA <4K、SDAR <24K 自动关闭 Prefix Sparse | 短上下文保持精确；SDAR 8K/16K Query+Prefix 退化为更快的 Query-only |
| Prefix/Adamas | Faster Hadamard、模型专属 bucket、int32 distance、rolling candidates、chunk 1024 | 32K selector microbenchmark：LLaDA 12.22ms→10.71ms，SDAR 16.55ms→13.06ms |
| Prefix cache | 固定 prompt KV 只编码一次，新 block 只刷新 suffix；每层 compact KV | 避免重复完整 prefill；LLaDA 32K peak memory 由约 64 GiB 降至 34--36 GiB |
| Triton prefill | 隐式 block metadata，不构造二次方 mask；query tile 只扫描可见 KV | profile 中 attention 1.007s→0.828s；32K/gen256 sparse 5.483s→5.281s |
| LLaDA Query | layer 1 confidence selector、interval 4、prefix 24K cost gate | Query 在 8K/16K 无收益时退化为 Prefix-only，32K 保留 Query 收益 |
| LLaDA MoE | fused routing、32x128x64 tiles、down 使用 8 warps、SwiGLU 只计算一次、down 直接写 route 顺序、按 expert count 紧凑发射 tile | 紧凑 tile 使 32K/gen256 dense 7.676s→5.463s、sparse 5.035s→2.834s；nsys 中 down/gate-up 分别减少 84%/75%，checksum 不变 |
| SDAR Query | fused QKV/RMSNorm/SwiGLU、跳过无用 logits/同步与 sequential selector norm、contiguous slice | 删除 selector norm 使 8K/gen256 sparse 5.823s→5.723s；最终 Query+Prefix 六点为 1.34--1.53x |
| LoSA correctness | prefix/current online-softmax merge、query-driven refresh、full-active control | `active_topk >= block_length` 仍走 LoSA 且接近 dense；当前无可靠 E2E 优势，默认关闭 |

### 已尝试但未保留

以下方向已有实测结论，下一会话不要无条件重复：

| 尝试 | 结论 | 仅在何时重访 |
| --- | --- | --- |
| LLaDA Query interval 8/16、ratio 0.5/0.6 | 不能稳定改善 8K/16K；小 M MoE 效率抵消少算的 token | selector 或 MoE 成本结构发生实质变化 |
| 小 expert 动态 `BLOCK_M=16` | 无端到端收益 | 新 GPU 架构或路由分布变化 |
| compact final norm/logits、direct query capture/去 hooks | 无可测收益 | profile 再次显示对应算子成为热点 |
| LLaDA empirical Hq/Hk threshold | HumanEval 64/164 official、127/164 normalized，劣于 v1.2 threshold 的 71/164、128/164 | 新模型或重新校准完整质量矩阵 |
| unpacked Adamas lookup table | 0--3 的 16-entry LUT 比整数 subtract/abs 多一次 lookup | 与 packed 2-bit XOR/popcount kernel 一起实现 |
| GQA representative/group-mean | LLaDA 32K 均慢于 exact；SDAR 单次约 1% 收益接近噪声且改变选择 | 先验证 index overlap，再跑完整质量 A/B |
| attention `BLOCK_N=128` | 改变归约顺序，未通过逐元素一致性测试 | 有明确精度预算并重新跑完整 eval |
| BF16 MoE down 近似 | 32K/gen256 很快，但 HumanEval 降至 72/164、129/164，8K/gen768 仅 0.996x | 不重访；同时违反精度门槛和六点加速目标 |
| LoSA key samples/频繁 query refresh | 额外估算与小 attention 开销尚未换来可靠 E2E 加速 | profile 证明 Prefix attention 再次主导且六点测试可获益 |
| LLaDA 跳过 prefill 无用 logits | 32K/gen256 sparse 5.065s→5.058s，仅 0.13%，属于噪声 | profile 显示 lm_head 成为热点 |
| SDAR 强制 cuDNN SDPA | 8K/gen256 Query 5.84s→6.16s，首次 shape 编译约 62s | PyTorch/cuDNN backend 有实质更新 |
| SDAR 无 mask Triton decode attention | 消除占 GPU 时间 17.8% 的 semaphore fill，但串行 KV 扫描使 5.84s→5.96s | 有可验证的 split-K/merge kernel |
| LLaDA gate-up 直接 gather 原 token | 取消约 94ms index-select，但非连续读取使 32K/gen256 sparse 5.011s→5.102s | 数据布局或 grouped GEMM 能保持连续读取 |
| SDAR 外部 FlashAttention GQA | 真实 stride 下 q=8--32 与 PyTorch SDPA 同为约 0.05ms，q=4 更慢且输出有微小差异 | 新版 FlashAttention 有明确 kernel 收益 |
| SDAR SwiGLU tile/warp/K 调参 | 当前 32x64x64、4 warps 最快；K=128 仅约 1.5% kernel 收益，折算 E2E <0.3% | profile 中 SwiGLU 占比进一步升高 |
| SDAR Query dense threshold 2/4/8 | 8K/gen256 sparse 从 threshold 0 的 5.823s 退化为 5.887/5.980/6.370s | block size 或 selector 成本结构变化 |
| SDAR dynamic threshold>=1 跳过无效高置信筛选 | 32K/gen256 同卡中位数 10.1731s→10.1614s，仅 0.11%；dynamic 又未通过质量门槛，不增加生产分支 | dynamic 成为通过质量门槛的生产候选 |
| SDAR dynamic greedy confidence 改为 `exp(max-logsumexp)` | 32K/gen256 同卡中位数 10.1731s→10.2865s，慢 1.1% | backend 能融合词表归约且 profile 显示 softmax 成为热点 |
| SDAR dynamic Query ratio 0.75/0.875 | 0.75 为 118/121，距 matched dense 仍差 3/3；0.875 非单调退化至 113/120 | 不继续调 ratio；只有 selector/KV 语义修正后重访 |
| SDAR dynamic 仅允许 deep-selected mask transfer | 当前树 32 题从默认的 29/29 降至 28/28；浅层位置可 transfer 不是主要精度根因 | 不重访；优先诊断未选 mask 的陈旧深层 KV/state |
| LLaDA Adamas 全局阈值 `Hq=0、Hk=±2.5` | 离线 attention-output error 最优，但完整 HumanEval 仅 66/125，低于旧阈值 71/128 | 有多样化真实代码 prompt 的逐层、多步校准集 |
| LLaDA Adamas 全局阈值 `Hq=±0.1、Hk=±2.26` | 完整 HumanEval 69/132；normalized 提升但 official 回退 2 题，距 matched dense 低 5 题 | 同上 |

### 下一会话执行顺序

1. 先检查 `git status --short`、当前 commit、GPU 占用和两套 config；不要修改
   benchmark 公平性逻辑。
2. 当前 dirty tree 已完成 LLaDA、SDAR 的 8K/16K/32K × gen 256/768 四模式、
   四次 paired ablation；冻结最终提交后再统一复核一次并持久化 JSON。
3. 在该最终提交重跑两模型 HumanEval dense/Prefix/Query/Query+Prefix matrix；
   official 与 normalized 都写入最终表。
4. SDAR 下一热点是 Flash split-K attention 的 semaphore fill 与 SwiGLU；只有
   可验证的高效 split-K merge 或新版 backend 才重访。LLaDA 下一热点是
   prefill/decode attention 与 Prefix cache concat；Adamas 当前仅占约 0.3%。当前
   机器的 NCU hardware counter 不可用，管理员开放前使用 nsys 定位。
5. 每个候选只改一个因素：单测与 checksum → 六点端到端 → 输出变化时完整精度
   matrix。失败实验立即回退，不给生产路径增加 fallback 或永久实验开关。

每轮实现后的最低检查为：

```bash
CUDA_VISIBLE_DEVICES=0 /home/ysy/anaconda3/envs/llada/bin/python \
  -m unittest discover -s tests -q
git diff --check
```

当前可读关键 profile 位于 `/tmp/llada_direct_scatter_cuda.nsys-rep`、
`/tmp/llada_compact_moe_tiles_cuda.nsys-rep` 和
`/tmp/sdar_current_query_8k_cuda.nsys-rep`；`/tmp` 文件不保证跨机器或重启存在，
因此论文数据必须另存到仓库外的持久实验目录并在 README 记录路径。

## Code layout

```text
src/sparse/
├── __init__.py         public exports
├── api.py              model detection and patch_model(...)
├── sparse_ops.py       shared cache, Adamas, LoSA, RoPE, and online merge
├── triton_kernels.py   canonical CUDA implementations
├── llada_patch.py      LLaDA attention, generation, Query Sparse, and MoE patch
└── sdar_patch.py       SDAR attention, generation, and Query Sparse patch
```

The dependency direction is intentionally one-way:

```text
api -> llada_patch / sdar_patch -> sparse_ops -> triton_kernels
```

The runtime requires CUDA, Triton, and `faster_hadamard_transform`. It does not
silently switch to CPU or PyTorch implementations. The Faster Hadamard extension
used in the current environments comes from:

```text
/data0/ysy/Adamas/kernels/3rdparty/applied-ai/kernels/cuda/inference/hadamard_transform
```

## Inference

```bash
MODEL_TYPE=llada bash scripts/generate.sh
MODEL_TYPE=sdar bash scripts/generate.sh
```

Direct invocation:

```bash
CUDA_VISIBLE_DEVICES=5 /home/ysy/anaconda3/envs/llada/bin/python \
  scripts/generate.py --model llada --pattern patch \
  --query_sparse true --prefix_sparse true --losa false

CUDA_VISIBLE_DEVICES=5 /home/ysy/anaconda3/envs/dream/bin/python \
  scripts/generate.py --model sdar --pattern patch \
  --query_sparse false --prefix_sparse true --losa false
```

The public Python entry point is:

```python
from src.sparse import patch_model

patch_model(
    model,
    model_name="llada",  # or "sdar"
    query_sparse=False,
    prefix_sparse=True,
    prefix_token_budget=1024,
    losa=False,
)
```

## Sparse features

### Query Sparse

Query Sparse selects mask positions at an intermediate decoder layer, then
runs only those positions through the remaining layers. Decoded positions stay
in the selected set so their hidden states and KV entries remain available.

`query_dense_threshold` is an algorithm threshold: Query Sparse is skipped when
the current mask count is at or below it. LLaDA defaults to 4 and SDAR defaults
to 0.

LLaDA also uses `query_min_prefix_length=24576`: below that prefix length the
runtime keeps Prefix Sparse but skips Query Sparse because its fixed selection
cost exceeds the saved short-prefix attention work. Set it to 0 to force Query
Sparse at every context length.

LLaDA uses its native confidence selector. SDAR uses the same configured
`remasking_strategy` for shallow query selection and final token transfer.

### Prefix Sparse

Prefix Sparse runs one dense refresh at the beginning of a block and captures
every layer's real RoPE query. LLaDA selects historical positions independently
for every layer; SDAR shares one Adamas selection within each adjacent layer
pair. Every layer gathers its own K/V values, and the complete current block is
always retained. The production policy keeps short prefixes dense because the
approximation has no useful latency payoff there: below 4K for LLaDA and below
24K for SDAR.

Validated deployment budgets are:

- LLaDA: `prefix_token_budget=1024`
- SDAR: `prefix_token_budget=512`

The v1.2-compatible speed profile uses `prefix_token_budget=256`. With Prefix
only, it reaches 128/164 normalized HumanEval for LLaDA and 129/164 for SDAR.

The runtime four-bin boundaries are:

| Model | Hq boundaries | Hk boundaries |
| --- | --- | --- |
| LLaDA | `[-1.35, 0, 1.35]` | `[-2.26, 0, 2.26]` |
| SDAR | `[-1.50, 0, 1.49]` | `[-2.87, 0, 2.86]` |

### LoSA

LoSA splits attention into historical prefix and current-block parts. Prefix
output/LSE state is reused between denoising steps, selected query positions
refresh that state, and the two parts are combined with online softmax.

When `losa_active_topk >= block_length`, LoSA still executes the split
prefix/current calculation and refreshes every position. This is the exact
full-active control and should remain close to dense attention.

The current block uses an explicit short-block PyTorch attention calculation;
historical prefix attention, query-delta scoring, Adamas distance, KV writes,
and LLaDA MoE execution use the canonical Triton kernels.

## Evaluation defaults

`eval_instruct/eval.sh` is the source of truth for evaluation configuration.
The runtime also persists the complete family-specific snapshot as
`model.config.llada_sparse_config` or `model.config.sdar_sparse_config`.
These are independent configurations; a parameter omitted by the caller uses
the default for the detected model family.

| Setting | LLaDA | SDAR |
| --- | ---: | ---: |
| dtype | BF16 | FP16 |
| block length / steps | 32 / 32 | 32 / 32 |
| Query Sparse ratio | 0.7 | 0.5 |
| selection interval | 4 | 1 |
| query dense threshold | 4 | 0 |
| selection layer | 1 | 5 |
| Prefix chunk size | 1024 | 1024 |
| Prefix minimum length | 4096 | 24576 |
| Prefix Sparse default | enabled | disabled |
| LoSA default | disabled | disabled |
| MoE patch | enabled | disabled |

HumanEval matrix:

```bash
CUDA_VISIBLE_DEVICES=3,4,5 bash eval_instruct/run_humaneval_matrix.sh
```

GSM8K matrix:

```bash
CUDA_VISIBLE_DEVICES=3,4,5 bash eval_instruct/run_gsm8k_matrix.sh
```

## Benchmark history and reproduction

以下表格是不同阶段的历史记录，用于追踪优化来源，不是同一提交、配置或测速
协议下的论文主表。当前可引用状态与最终重跑要求以上面的接手指南为准。

```bash
CUDA_VISIBLE_DEVICES=5 /home/ysy/anaconda3/envs/llada/bin/python \
  scripts/bench_long_context.py --model llada --mode prefix \
  --contexts 8192 16384 32768 --gen-length 64 --prefix-token-budget 1024

CUDA_VISIBLE_DEVICES=5 /home/ysy/anaconda3/envs/dream/bin/python \
  scripts/bench_long_context.py --model sdar --mode prefix \
  --contexts 8192 16384 32768 --gen-length 64 --prefix-token-budget 512
```

Matched steady measurements from the calibrated Prefix configurations:

| Model/config | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| LLaDA dense | 6.725s | 12.440s | 26.584s |
| LLaDA Prefix-1024 | 6.244s (1.08x) | 12.290s (1.01x) | 25.427s (1.05x) |
| SDAR dense | 6.456s | 12.489s | 35.621s |
| SDAR Prefix-512 | 6.228s (1.04x) | 12.209s (1.02x) | 34.669s (1.03x) |

Matched GPU 2 measurements for the v1.2-compatible Prefix-256 profile:

| Model/config | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| LLaDA dense | 9.033s | 14.839s | 28.676s |
| LLaDA Prefix-256 | 7.869s (1.15x) | 13.981s (1.06x) | 26.689s (1.07x) |
| SDAR dense | 6.361s | 12.719s | 35.763s |
| SDAR Prefix-256 | 6.408s (0.99x) | 12.559s (1.01x) | 34.735s (1.03x) |

Full HumanEval validation for the calibrated quality configurations:

- LLaDA: 74/164 official, 137/164 indentation-normalized, matching dense.
- SDAR: 129/164 official and 129/164 normalized; dense is 129/164 and 130/164.

Full HumanEval for the Prefix-256 speed profile:

- LLaDA: 71/164 official and 128/164 indentation-normalized.
- SDAR: 127/164 official and 129/164 indentation-normalized.

Latest SDAR prefill/Adamas optimization (GPU 2/3, generation length 64):

| Configuration | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| Dense, Triton prefill | 3.77s | 5.70s | 8.79s |
| Query+Prefix-256, Adamas chunk 1024 | — | — | 7.93s |

The original SDAR dense prefill baseline at 32K was 35.76s; the current dense
Triton path is approximately 4.1x faster.

For fair end-to-end ablation, use `--ablation`. Dense, Query-only, Prefix-only,
and Query+Prefix are run on the same loaded model. Every mode is first warmed
up with the same context and generation length as the measured run; the four
measured orders rotate, context order alternates, output length is fixed with
`eos_early_stop=False`, and the table reports four-run medians. This avoids
charging a new generation shape's compilation cost to dense.

```bash
scripts/bench_long_context.py --model llada --mode query_prefix \
  --contexts 8192 16384 32768 --gen-length 256 \
  --prefix-token-budget 256 --repeats 4 --ablation
```

Times and speedups relative to dense:

| Model / generation / mode | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| LLaDA 256 dense | 2.354s | 3.669s | 8.481s |
| LLaDA 256 Query, forced | 3.013s (0.78x) | 3.757s (0.98x) | 7.795s (1.09x) |
| LLaDA 256 Prefix-256 | 2.320s (1.01x) | 3.410s (1.08x) | 6.565s (1.29x) |
| LLaDA 256 adaptive Query+Prefix-256 | 2.334s (1.01x) | 3.416s (1.07x) | 5.938s (1.43x) |
| LLaDA 768 dense | 5.405s | 6.484s | 12.902s |
| LLaDA 768 Query, forced | 5.459s (0.99x) | 7.131s (0.91x) | 11.573s (1.11x) |
| LLaDA 768 Prefix-256 | 5.193s (1.04x) | 5.738s (1.13x) | 8.142s (1.58x) |
| LLaDA 768 adaptive Query+Prefix-256 | 5.228s (1.03x) | 5.761s (1.13x) | 8.362s (1.55x) |
| SDAR-b32 256 dense | 8.288s | 9.213s | 13.783s |
| SDAR-b32 256 Query | 5.939s (1.40x) | 6.926s (1.33x) | 10.593s (1.30x) |
| SDAR-b32 256 Prefix-256 | 8.374s (0.99x) | 9.327s (0.99x) | 12.595s (1.09x) |
| SDAR-b32 256 Query+Prefix-256 | 5.967s (1.39x) | 6.903s (1.33x) | 10.164s (1.36x) |
| SDAR-b32 768 dense | 23.506s | 24.465s | 31.863s |
| SDAR-b32 768 Query | 16.519s (1.42x) | 17.632s (1.39x) | 22.365s (1.42x) |
| SDAR-b32 768 Prefix-256 | 23.738s (0.99x) | 24.721s (0.99x) | 28.179s (1.13x) |
| SDAR-b32 768 Query+Prefix-256 | 16.622s (1.41x) | 17.511s (1.40x) | 20.983s (1.52x) |

LLaDA now defaults to Adamas chunk 1024, block-causal prefill stops each query
tile at its last visible KV block, and routed MoE computes SwiGLU once before
the down projection while launching only the real per-expert tiles. Direct
four-mode paired ablation gives `1.036x / 1.193x / 1.928x` for generation
length 256 and `1.017x / 1.208x / 1.941x` for generation length 768 at
8K/16K/32K. At 32K/gen768 Prefix-only reaches 2.007x and is slightly faster
than Query+Prefix. Output checksums were stable across repeats and matched the
former kernels and chunk-256 runs.

Prefix is the main LLaDA accelerator: it saves 1--8% at 8K/16K and 23--37%
at 32K. Forced LLaDA Query is only beneficial at 32K, so the default adaptive
policy skips it below 24K and makes Query+Prefix faster than dense at every
tested length. Query is the main SDAR accelerator, saving 23--30% by
itself; Prefix is neutral below 32K and adds a further 3--4% of dense latency
savings after Query at 32K. Thus the two selectors have a measurable
interaction and their standalone speedups should not be multiplied.

Adaptive LLaDA Query+Prefix reaches 74/164 official and 132/164
indentation-normalized on HumanEval; forced Query reached 79/164 and 130/164.
SDAR reaches 127/164 official and 129/164 normalized. SDAR-b4 remains slower
than b32 for this configuration.

The runtime stores block-causal structure as implicit metadata, caches the
fixed prompt KV once, and only refreshes the generated suffix for each new
block. Peak LLaDA memory is 33--37 GiB; the former quadratic-mask path used
about 64 GiB at 32K. The routed-MoE kernels use a 32-row tile because a 64-row
tile exceeds the GPU shared-memory limit.

## Supporting tools

- `scripts/bench_sparse_ops.py`: kernel-level sparse operation benchmark.
- `scripts/collect_adamas_hqhk.py`: collect real Hq/Hk distributions.
- `scripts/analyze_llada_layer_overlap.py`: LLaDA query-layer overlap.
- `scripts/analyze_sdar_layer_overlap.py`: SDAR query-layer overlap.
- `scripts/analyze_llada_query_losa_correlation.py`: Query/LoSA correlation.

Historical results are in [`RESULTS.md`](RESULTS.md). Adamas distribution,
lookup-table, and GQA findings are in [`docs/adamas.md`](docs/adamas.md).
