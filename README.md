# Sparse Diffusion Model Patches

This repository provides one sparse-feature entry point for LLaDA and SDAR.
Model checkpoint files are never modified.

Current evaluation and throughput results are summarized in [`RESULTS.md`](RESULTS.md).
New `eval_instruct` results default to `../llada_exp/` or `../sdar_exp/` in the
parent `sparse/` directory; set `OUTPUT_ROOT` to override it.

## Layout

- `src/sparse/core.py`: public `patch_model(...)` entry point. It validates the
  model family and enables query sparse, prefix sparse, LoSA, and the MoE patch.
- `src/sparse/block_cache_sparse_dlm_patch.py`: LLaDA generation/attention patch.
- `src/sparse/sdar_block_diffusion_patch.py`: SDAR generation/KV-cache patch.
- `src/sparse/llada_moe_expert_patch.py`: LLaDA MoE kernel used by the core.
  It packs
  routed expert weights and dispatches the SiLU MLP through Triton grouped
  kernels without modifying the model directory. It is a no-op when CUDA or
  Triton is unavailable.
- `scripts/test.py`: single-prompt inference entry point, following Dream's test script structure.
- `scripts/test.sh`: environment-variable wrapper for the main inference path.
- `eval_instruct/`: lm-eval harness with separate `llada` and `sdar` adapters.
- `tests/`: small CPU regression tests built from the LLaDA remote-code class.
- `experiments/legacy/`: earlier sparse-attention, trace, ratio-sweep, plots, and throughput work.

## Run

```bash
bash scripts/test.sh
```

Select the model explicitly with `MODEL_TYPE`; the checkpoint path is optional
when using the repository defaults:

```bash
MODEL_TYPE=llada bash scripts/test.sh
MODEL_TYPE=sdar bash scripts/test.sh
```

Common overrides:

```bash
GEN_LENGTH=128 BLOCK_LENGTH=32 STEPS=32 \
SPARSE_DLM_RATIO=0.5 SPARSE_DLM_SELECTION_INTERVAL=4 \
bash scripts/test.sh
```

The current-block query optimization and Adamas prefix selection are
independent. Prefix selection uses a per-layer prefix-only budget of 256
tokens by default:

Query Sparse 的核心不变量是：在浅层提前预测 Query positions 时，必须复用最终
输出 selector 的同一套排序、置信度阈值判定和 top-k 保底规则。LLaDA 最终输出不
使用 strategy，因此其浅层 Query selector 也固定使用原生 confidence selector；
两者只因用途不同而传入不同的最低数量：最终输出使用 `num_to_transfer`，浅层使用
`ceil(mask_count * sparse_dlm_ratio)`。SDAR 则在浅层与最终输出两处使用同一个
`remasking_strategy`。

LLaDA 的浅层/最终 selector overlap 可用同一实现直接检查：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_llada_layer_overlap.sh
```

`SPARSE_DLM_SELECTION_LAYER` is zero-based and controls the LLaDA and SDAR
Query selectors. The default `5` means selecting after `layers[5]` and running
`layers[6:]` sparsely; it must be smaller than the checkpoint's final decoder
layer index.

```bash
QUERY_SPARSE=false PREFIX_SPARSE=true PREFIX_TOKEN_BUDGET=256 \
bash scripts/test.sh
```

Set `PREFIX_SPARSE=false` to retain the full prefix cache. `PREFIX_CHUNK_SIZE`
controls peak memory used by the Python Adamas selector and defaults to 256.

The experimental PyTorch LoSA reference path is disabled by default. It caches
prefix attention output/LSE between refinement steps and merges it with the
fresh current-block attention using online-softmax state:

```bash
LOSA=true LOSA_ACTIVE_TOPK=5 QUERY_SPARSE=false PREFIX_SPARSE=false \
bash scripts/test.sh
```

This is a correctness/reference path, not a Triton kernel. Keep query and
prefix sparse disabled for the first numerical comparison.

Use `PATTERN=default` for the native LLaDA baseline. `PATTERN=patch` is the
default and routes every feature through `src.sparse.patch_model`.

Run SDAR with the shared defaults: the `SDAR-8B-Chat-b32` checkpoint,
`block_length=32`, `steps=32`, ratio 0.5, zero-based selection layer 5,
selection interval 1, refresh disabled (`-1`), sequential remasking, and no
MoE expert patch. Query sparsity is enabled by default; use
`QUERY_SPARSE=false` for the official dense-block path:

```bash
MODEL_TYPE=sdar MODEL_PATH=/data0/ysy/models/SDAR-8B-Chat-b32 \
PROMPT="Explain diffusion models briefly." bash scripts/test.sh
```

SDAR keeps Adamas and LoSA disabled by default. Enable them independently with
the same switches as LLaDA; for the block-32 checkpoint, for example:

```bash
MODEL_TYPE=sdar MODEL_PATH=/data0/ysy/models/SDAR-8B-Chat-b32 \
BLOCK_LENGTH=32 STEPS=32 QUERY_SPARSE=false \
PREFIX_SPARSE=true PREFIX_TOKEN_BUDGET=256 bash scripts/test.sh

MODEL_TYPE=sdar MODEL_PATH=/data0/ysy/models/SDAR-8B-Chat-b32 \
BLOCK_LENGTH=32 STEPS=32 QUERY_SPARSE=false PREFIX_SPARSE=false \
LOSA=true LOSA_ACTIVE_TOPK=5 bash scripts/test.sh
```

Adamas selects each layer's historical prefix KV after the first dense denoise
step and always keeps the complete current block. LoSA reuses prefix attention
output/LSE while recomputing current-block attention every step. Setting
`PREFIX_TOKEN_BUDGET` at least as large as the prefix, or
`LOSA_ACTIVE_TOPK >= BLOCK_LENGTH`, falls back to the corresponding dense path.
Both features can be combined with each other and with query sparse, although
the current Python reference kernels are intended for quality validation before
long-context performance tuning.

The SDAR adapter applies query selection after `layers[5]` (the former
1-based layer 6) and keeps all decoded
positions plus a strategy-selected mask set for later layers. The configured
mask ratio is the minimum Query budget; dynamic and entropy strategies may
select more positions. It uses the official
`sdar_generate.py` block/transfer loop for both dense and sparse inference.
Query sparse is calibrated on the matching SDAR checkpoints for
`block_length=4` and `block_length=32`. Large blocks remain approximate because
errors in current-block KV compound across more denoising rounds; the current
b32 `ratio=0.5` HumanEval result is no longer catastrophic, but is still 4.88 pp
below the dense baseline. A `ratio=0.75` quality upper bound recovers the dense
aggregate score, but does not yet provide acceleration. The current feature
support is:

| Model | query sparse | prefix sparse | LoSA | MoE expert patch |
| --- | --- | --- | --- | --- |
| LLaDA | yes | yes | yes | yes |
| SDAR | yes | yes | yes | no-op on its dense experts |

## SDAR 状态与后续接手说明（2026-08-29）

本节是 SDAR 工作的单一交接记录。历史结果位于
`/data0/ysy/sparse/sdar_exp/`，层重合度结果位于
`layer_overlap_results/sdar_block32/`。这些结果来自不同 checkpoint、block
配置和代码版本，只有表中明确配对的行可以直接比较。模型目录文件始终未修改。

### 当前实现

- SDAR 的 Adamas 只压缩已经完成 block 的历史 prefix KV；每层根据该 block 首次
  dense denoise 的真实 RoPE query 独立选位，因此支持每层不同的 compact prefix
  长度，current-block KV 始终完整保留。
- SDAR 的 LoSA 在 refinement step 间缓存 prefix attention output/LSE，仅对
  `LOSA_ACTIVE_TOPK` 个 query 刷新 prefix 状态；current-block attention 每步重算，
  再用 online softmax 合并。prefill、首个 dense denoise 和最终 block KV 写入仍走
  SDAR 原生路径。该实现目前是 PyTorch correctness/reference path，默认关闭。
- GPU 5 的 `SDAR-8B-Chat-b32`、block/steps=32、64-token smoke 已覆盖 Adamas、
  LoSA 及与 Query Sparse 的三者组合。当前 prefix 只有 32 token 时，Adamas
  full-budget 和 LoSA full-active fallback 的生成文本都与 dense 一致；实际压缩/
  复用配置也均能完成生成。该 smoke 不替代 HumanEval/GSM8K 质量验证或长 prefix
  吞吐测试。

- `src/sparse/sdar_generate.py` 保存并复用官方 block diffusion 的 prefill、
  current-block denoise、完成 block 后写入 KV cache 的主循环。Dense 与 Query
  Sparse 经过同一个 transfer 循环，Sparse 只通过 `denoise_fn` 替换单步 forward，
  避免两套生成逻辑继续漂移。
- `src/sparse/sdar_block_diffusion_patch.py` 在运行时 patch `model.generate`。
  每个 block 的第一个 denoise step 保持 dense，用于建立完整 current-block KV；
  后续 step 默认在 `layers[5]` 后执行 `norm -> lm_head -> sampling` 选位，
  从 `layers[6]` 开始只计算所有已解码位置和配置比例的 mask 位置。选位层已暴露为
  `SPARSE_DLM_SELECTION_LAYER`，不再需要修改源码做 layer ablation。
- 未选择 mask 的 layer-7 以后 hidden/KV 不会重算；最终 transfer 仍会看到所有
  mask，其中未选择位置使用 layer-6 hidden 经过 `norm/lm_head` 的近似 logits，
  已选择位置使用完整后续层输出。`_BlockDualCache` 只覆盖本轮选择位置的 KV。
- 对 block length >= 16，每个 block 默认在 sparse-call index 2（第三次 sparse
  forward，即 denoise step 3）将 layer-7 以后临时扩为全 block 一次，刷新所有
  current-block late-layer KV；同一步仍只把原 Query
  选中位置的深层 hidden 写回 logits，后续 step 恢复稀疏。可用
  `SPARSE_DLM_REFRESH_STEP=-1` 关闭，或用 `-2` 执行每个 sparse step 都刷新的诊断
  上界；小于 16 的 block 不执行该刷新。
- `SPARSE_DLM_DEEP_ONLY_TRANSFER=true` 是 P1 诊断开关：最终 transfer 只能看到本轮
  Query 选中位置的深层 logits，不能使用未选中位置的 layer-6 近似 logits。默认
  保持 `false`，以兼容现有行为。
- Query 候选的最低数量为 `ceil(mask_count * sparse_dlm_ratio)`，且不会小于该
  denoise step 的最低 transfer quota；实际选位复用当前 transfer strategy：
  static 取最高 confidence，dynamic 可纳入所有超过 threshold 的位置，entropy
  按低 entropy 和累计 budget 扩展，sequential 按位置顺序选择。默认 ratio 为
  0.5、selection interval 为 1、dense fallback mask count 为 0。
- `SPARSE_DLM_TOP_K` 的语义是 Query 预测所用的采样词表截断，当前已从 patch
  配置传入选位采样路径；它不是 Query 位置数。默认 `temperature=0` 时为 greedy
  argmax，top-k 不改变 token，因此 top-k=64 只有在随机采样时才实际改变分布。
- Transfer 支持 `low_confidence_dynamic`、`low_confidence_static`、`sequential`
  和 `entropy_bounded`。Entropy 使用每个位置在过滤后词表分布上的
  `-sum(p * log(p))`，而不是把已采样 token 的单个概率误当作 entropy。
  `entropy_bounded` 按低 entropy 排序，在累计 budget 内 transfer，同时保证官方
  per-step 最低 transfer 数。
- `eval_instruct` 有独立的 `sdar` adapter 和 `gsm8k_sdar` task。后者复用了论文
  仓库的答案抽取/等价判断，并恢复了 SDAR GSM8K 所需的长生成上限；此前直接用
  `gsm8k`/`gsm8k_cot` 的 prompt、短生成和普通抽取只得到约 13%，并非模型真实
  GSM8K 能力。
- `eval_instruct/eval.sh` 和 `scripts/test.sh` 当前都默认使用
  `/data0/ysy/models/SDAR-8B-Chat-b32`、block/steps 32/32。
- SDAR remote code 需要 `/home/ysy/anaconda3/envs/dream/bin/python`
  （Transformers 4.53.3）。`llada` 环境的 Transformers 4.57.1 在本机缺少
  `transformers.utils.LossKwargs`，会在加载 `modeling_sdar.py` 时失败。SDAR
  shell wrapper 已默认选择 `dream` 环境。

### 已完成的修复和验证

1. 将 SDAR 适配移入公共 `src.sparse.patch_model(...)` 分发器，不修改 checkpoint
   内的 `modeling_sdar.py` 或 generate 文件。
2. 对照官方 `sdar_generate.py` 重构 dense/sparse 生成循环，修正 block attention、
   prefix/current-block KV 写入、transfer quota、EOS 和 generation 参数传递。
3. 修正 lm-eval 的 SDAR chat template、GSM8K prompt/评分、生成长度、FP16 和
   block 参数链路。官方 block-4 GSM8K dense 已恢复至 90.67%，接近论文的 91%。
4. 让 early-layer 预测与最终输出共用同一 `norm -> lm_head -> sampling` 后处理；
   删除不一致的强制 `.float()`，只有 entropy 内部为数值稳定性转为 FP32。
5. Sparse forward 返回所有 mask 的 logits，避免 transfer 只能从本轮 Query
   子集选 token；同时保证可用候选数至少覆盖本轮最低 transfer 数。
6. 修正 `entropy_bounded` 为逐位置、沿词表维求 entropy，并把策略和
   `EB_THRESHOLD` 暴露到直接推理、lm-eval 和 layer-overlap 脚本。
7. 修正 `SPARSE_DLM_TOP_K=64` 只写 config、不进入 SDAR Query 采样路径的问题。
8. 增加 `scripts/test_sdar_layer_overlap.py` 和
   `scripts/run_sdar_layer_overlap_all_strategies.sh`，可比较所有 decoder layer 和
   四种 transfer strategy。相关 CPU 单测以及 dense/query GPU 冒烟测试已通过。
9. 隔离 LLaDA 与 SDAR 的 token transfer：SDAR 保留四种
   `remasking_strategy` 和 entropy budget；LLaDA 恢复 checkpoint 原生的置信度
   阈值/最低 top-k transfer。LLaDA Query Sparse 的浅层 selector 同步使用相同的
   confidence 规则，不再接收 strategy；稀疏路径仅使用更大的 Query 最低预算。
10. 修正 layer-overlap 诊断缺少 final RMSNorm、候选数向下取整的问题。按生产路径
    重测后发现 layer 4 的 `candidate@0.5` recall 只有 26.67%，因此默认选位层改为
    layer 6；5 个 block-32 prompt 的 dense token 一致率均值从 60.63% 提升到
    96.88%。
11. 在对应的 `SDAR-8B-Chat-b32` checkpoint 上完成 164 题 HumanEval 全量重跑：
    layer-6 Query Sparse 为 67.68%（111/164），缩进归一化复评为 69.51%
    （114/164），23.20 tok/s。相对历史同配置 dense 74.39%，差距缩到 6.71 pp；
    相对旧 Query 19.51%--20.12%，提升 47.56--48.17 pp。
12. 量化未选中 mask 的 stale KV：相对 fresh dense KV 的 K-cache RMS 偏差约为
    15%--21%。每步全刷新上界把前 16 题从 15/16 恢复到 16/16，但吞吐仅
    11.37 tok/s；定位后改为每 block 只在 sparse-call index 2 刷新一次 late-layer KV。
    b32 HumanEval 全量达到 69.51%（114/164），缩进归一化 72.56%（119/164），
    23.55 tok/s；相对不刷新分别提升 1.83/3.05 pp，吞吐未下降。逐题官方口径为
    修复 8 题、回归 5 题（净增 3），说明该刷新位置是全量净收益而非单调改进。
13. 正确重跑 layer-5 ablation。HumanEval 官方/缩进归一化为 107/164、114/164，
    GSM8K 为 86.88%；对应 layer 6 为 114/164、119/164 和 88.48%。layer 5 在两项
    任务上都更差，因此默认和后续实验继续使用 layer 6。此前配置错误的 layer-5
    HumanEval 结果已明确排除，未参与比较。
14. 完成 P0：`ratio=1.0` 的 patched Query 路径与 dense baseline 均为 HumanEval
    122/164（归一化 126/164），164 条 response 逐条相同，samples 文件 SHA256
    也相同。这排除了 SDAR 生成循环、基础 cache 路径或 patch 本身在不裁剪时发生
    漂移。
15. 完成 P1：`ratio=0.5` 的 deep-only transfer 与当前 all-mask transfer 均为
    114/164（归一化 119/164），164 条 response 逐条相同，说明当前
    `threshold=1.0 + low_confidence_dynamic` 配置下，未选中位置的 layer-6 logits
    并没有改变 transfer。每步刷新 late-layer KV 只提升到 116/164（归一化
    123/164）；把 Query 候选比例提高到 0.75 则达到 122/164（归一化 127/164）。
    因而剩余误差的主因是 `ratio=0.5` 候选集合的召回/预算，KV 陈旧是次要因素。

### 关键结果

速度是结果 JSON 中的纯 generation tokens/s。以下历史实验中的“layer N”沿用当时
的 1-based 记法，因此 layer 6 对应当前 0-based `layers[5]`。标为“旧”的 Query 来自修复前代码；
b32 layer-6 行的 dense 数值沿用同 checkpoint、同 block 配置的历史全量基线，Query
是当前代码的全量重跑。

| checkpoint / 配置 | benchmark | Dense | Query Sparse | Dense tok/s | Query tok/s | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `SDAR-8B-Chat`, block=4, steps=4 | GSM8K | 90.67% | 89.46% | 23.39 | 22.57 | 精度仅降 1.21 pp，但未加速 |
| `SDAR-8B-Chat`, block=4, steps=4 | HumanEval | 81.10% | 73.78%--75.00% | 23.18 | 21.95--22.24 | 小 block 仍掉点且更慢 |
| `SDAR-8B-Chat`, block=16, steps=16 | HumanEval | 71.95% | 10.37%（v0.5），7.93%（v0.6） | 26.56 | 24.39--25.26 | Query Sparse 严重失真 |
| `SDAR-8B-Chat-b32`, block=32, steps=32，旧 Query | HumanEval | 74.39% | 19.51%--20.12% | 25.50 | 24.37--24.43 | 修复前严重失真 |
| `SDAR-8B-Chat-b32`, block=32, steps=32，layer 6、无刷新 | HumanEval | 74.39% | 67.68%（归一化 69.51%） | 25.50 | 23.20 | 主要崩溃已修复，仍差 6.71 pp |
| `SDAR-8B-Chat-b32`, block=32, steps=32，layer 6、refresh index 2 | HumanEval | 74.39% | **69.51%**（归一化 72.56%） | 25.50 | 23.55 | stale-KV 误差得到缓解，官方差距缩至 4.88 pp |

最新 P0/P1 使用同一个 `SDAR-8B-Chat-b32`、block/steps=32、FP16、layer 6 和完整
164 题 HumanEval。除表中变量外，其余参数保持一致：

| 诊断配置 | 官方 pass@1 | 缩进归一化 | tok/s | 相对 dense 的含义 |
| --- | ---: | ---: | ---: | --- |
| Dense baseline | 122/164（74.39%） | 126/164（76.83%） | 25.57 | 对照 |
| P0: Query `ratio=1.0` | 122/164（74.39%） | 126/164（76.83%） | 25.57 | 与 dense 逐 response 完全相同 |
| 当前：`ratio=0.5`、refresh index 2 | 114/164（69.51%） | 119/164（72.56%） | 23.93 | 当前质量/速度基准 |
| P1: 上一行 + deep-only transfer | 114/164（69.51%） | 119/164（72.56%） | 23.64 | 与当前逐 response 完全相同 |
| P1: `ratio=0.5`、每步 refresh | 116/164（70.73%） | 123/164（75.00%） | 24.20 | stale KV 有影响，但不能解释主要差距 |
| P1: `ratio=0.75`、refresh index 2 | 122/164（74.39%） | 127/164（77.44%） | 23.48 | 聚合精度恢复，指向候选召回不足 |

这些效应不是可加的，不能把每个净增题数机械分摊为独立误差比例。`ratio=0.75`
也只有 61/164 条 raw response 与 dense 完全相同（归一化口径相对 dense 修复 3 题、
回归 2 题），所以它是质量上界/工作点，而不是已实现的行为等价或加速方案。

### GPU profile（2026-08-29，GPU 5）

使用 `SDAR-8B-Chat-b32`、FP16、block/steps=32、gen=128、threshold=1.0、固定
128 token 输出，在 GPU 5 上先 warmup 一次，再分别采集普通计时和 PyTorch
CPU/CUDA operator profile。普通计时不包含模型加载和 profiler 开销：
注意直接推理脚本的 SDAR 默认 threshold 是 0.85，而 `eval_instruct/eval.sh` 默认
为 1.0；复现本表时必须显式设置 `THRESHOLD=1.0`，否则两边可能执行不同数量的
dynamic transfer/denoise forward。

| 指标 | Dense | Query 优化前 | Query 完成优化 1/2 后 |
| --- | ---: | ---: | ---: |
| wall time | 5.060 s | 5.478 s | 5.316 s |
| output tok/s | 25.30 | 23.37 | 24.08 |
| profiler self CUDA total | 1.997 s | 2.139 s | 2.068 s |
| profiler self CPU total | 6.252 s | 6.748 s | 6.377 s |
| `aten::mm` calls / CUDA | 37,950 / 1.516 s | 38,090 / 1.606 s | 38,090 / 1.607 s |
| CUDA kernel launches | 125,452 | 150,130 | 121,350 |
| stream synchronizations | 6,281 | 7,401 | 7,401 |
| `index_select` calls / CUDA | 150 / 0.624 ms | 12,580 / 40.267 ms | 850 / 3.129 ms |
| `index_copy_` calls / CUDA | 可忽略 | 8,380 / 30.919 ms | 280 / 0.741 ms |
| fused K/V write calls / CUDA | 0 | 0 | 4,050 / 9.135 ms |
| `torch.ones` calls | 1 | 5,041 | 141 |

结论不是“理论 FLOPs 没有下降”，而是当前 shape 下 FLOPs 下降不能转化为 kernel
时间下降：

1. Query 仍执行与 dense 相同的 5,400 次 attention/layer 调用。它只把 layer 7--36
   的 token 维从 32 缩为约 17--31；所有 decoded position 都必须保留，加上一次
   late-KV refresh 后，本次 prompt 的总 token-layer 工作量实际只下降约 17%，并非
   `ratio=0.5` 字面上的 50%。
2. H800 的这些 GEMM 沿 token 维都没有超过一个主要 tile。以 MLP up projection 为
   例，`m=17--31` 约 60.2--60.8 us，dense `m=32` 为 60.8 us；down projection
   约 66--69 us，dense 为 66.1 us。矩阵行数减少，但 kernel latency 几乎不变。
3. 每个 sparse step 为选位额外执行一次 layer-6 `lm_head`。这正好增加 140 次
   `aten::mm`；完整词表 projection 单次约 0.65 ms，是 `mm` CUDA 时间净增约
   90 ms 的主要来源。`SPARSE_DLM_TOP_K` 在 projection 之后才生效，不能降低这次
   full-vocab GEMM 的成本。
4. 优化前 `_sparse_cached_forward` 在每个 late layer 重复 gather 相同的
   `position_ids`/RoPE embeddings，并分别 `index_copy_` K、V，还逐层创建全 1
   attention mask。优化 1 把这些 forward 内不变量只构造一次；优化 2 用一个 Triton
   kernel 同时写回 K/V，并为无 Triton/CPU 路径保留原生回退。`index_select` CUDA
   时间下降 92.2%，`index_copy_` 下降 97.6%，kernel launch 下降 19.2%。
5. 两项优化使 Query wall time 相对优化前下降 2.95%，self CUDA/CPU 分别下降
   3.3%/5.5%，且生成文本保持一致。但 Query 仍比 dense 慢 5.1%；同步次数完全未变，
   selector 增加的 140 次 full-vocab `lm_head`/`aten::mm` 也未变，已成为下一步最有
   价值的目标。当前瓶颈不是 FlashAttention 算力：三组均调用 5,400 次。

后续已验证两类 `lm_head` 优化，但均不适合设为默认：

- 完全复用上一轮 final confidence 可把 `aten::mm` 从 38,090 降到与 dense 相同的
  37,950 次，self CUDA 从 2.068 s 降到 1.968 s；但完整 HumanEval 只有
  67/164（归一化 69/164），24.005 tok/s。相对当前 114/119 严重回归，说明一轮
  陈旧的 confidence 无法替代当前 layer-6 ranking。
- 每两步重算一次 layer-6、其余步使用 previous confidence 的 32 题筛选也只有
  23/32（归一化 25/32）；同 32 题当前基准为 30/32、30/32，且吞吐从 23.93 降到
  23.53 tok/s。旧 `selection_interval=2/4` 的位置复用同样已经失败。
- H800 上 selector 专用 FP8 projection 的孤立 GEMM 比 FP16 快 1.74--1.78 倍，
  但接入固定 128-token 生成后三次 wall time 中位数为 5.317 s，与 FP16 的 5.316 s
  相同，同时生成文本已经改变。其理论收益太小，不值得承担额外 0.62 GB 权重和
  ranking 风险。

这些实验代码已撤回，默认仍为每步当前 layer-6 FP16 selector。若继续优化该项，
需要训练或蒸馏一个能保持 candidate recall 的轻量 ranker，并以完整 HumanEval
paired quality 为准；单纯缓存旧分数、固定 interval 或量化 full-vocab head 都不能
安全解决当前 wall-time 差距。

`scripts/test.py` 现在支持默认关闭的 `--profile_output`、`--profile_trace` 和
`--warmup_runs`。聚合报告在 `/tmp/sdar_profile/dense_t1.txt` 与
`/tmp/sdar_profile/query_t1.txt`、`/tmp/sdar_profile/query_opt_t1.txt`；Chrome trace 仅在显式设置
`--profile_trace true` 时导出，避免默认产生约 0.7 GB/次的文件。

对应的主要目录为 `sdar_dense_v0.4`/`sdar_query_sparse_v0.4`（GSM8K）、
`sdar_dense_v0.5`/`sdar_query_sparse_v0.5`/`v0.6`（block 16）、
`sdar_dense_v0.8`/`sdar_query_sparse_v0.10`/`v0.15`（b32 block 32）以及
`sdar_dense_v0.7`/`sdar_query_sparse_v0.14`/`v0.16`（block 4）。不要根据版本号
假设代码单调演进，应读取每个 results JSON 的 `model_args`。

### Layer-overlap 诊断

`layer_overlap_results/sdar_block32/` 中的旧结果使用一个代码 prompt、block=32、
steps=32、ratio=0.5 测量 early-layer 候选集合覆盖最终 transfer 位置的程度。
旧脚本在 intermediate hidden 上直接执行 `lm_head`，遗漏了生产路径的 final
RMSNorm，因此下面的旧表不能用于选择 runtime selection layer：

| lm_head 所在层 | layer 2 | layer 4 | layer 5 | layer 6 |
| --- | ---: | ---: | ---: | ---: |
| candidate recall | 26.67% | 80.00% | 84.44% | 88.89% |
| exact transfer recall | 17.78% | 35.56% | 44.44% | 48.89% |

修正为 `norm -> lm_head` 并使用与 runtime 相同的 `ceil` 候选数后，同一类
block-32 单 prompt 对照中，layer 4 的 `candidate@0.5`/exact recall 为
26.67%/17.78%，layer 6 为 84.44%/42.22%。在 sparse 自身轨迹上的逐 step 对照也
显示，layer 4 的 dense transfer 候选命中/最终位置一致为 16/43、14/43，layer 6
提升到 36/43、34/43。因此当前默认使用 layer 6。对应 b32 HumanEval 全量结果为
67.68%，证明 layer 4 过早裁剪是历史大幅崩溃的主要原因，但不能消除全部近似误差。

`sequential` 的 exact overlap 恒为 1 是策略本身只按位置顺序选择的结果，不能
证明 early-layer logits 正确。历史 layer-overlap 脚本曾按 strategy 排 Query，
但当时 runtime 一律使用 token confidence；当前 runtime 已对齐为同一套 strategy，
因此旧图只能作为历史诊断证据，仍需用当前版本重跑后再与质量结果关联。

### 尝试后撤回或尚未奏效的方案

- 曾加入 Dream 风格的额外 quota transfer 规则，实测生成更慢，已撤回。当前只
  保留 SDAR 官方每步最低 transfer 数以及各 strategy 自身的动态扩展规则。
- 曾尝试每 2/4 步 dense refresh current-block KV。block-16 smoke 的 HumanEval
  仍只有约 25%（小样本），完整 block-16 结果约 14.63%，同时增加 dense 计算；
  `SPARSE_DLM_DENSE_REFRESH_INTERVAL` 已删除。该旧实验使用修复前选层并周期性执行
  额外完整 forward，不等同于当前只在 refresh index 2 内联扩展 late layers 一次
  的方案。
- 曾只给本轮选择位置返回 logits，导致未选择 mask 永远不能 transfer；改为
  all-mask logits 后 block-4 恢复较好，但 block-16/32 仍严重掉点，说明它不是
  唯一原因。
- `block_length=16/32`、减少 steps、使用 b32 checkpoint、调整 refresh interval
  以及重构到官方生成循环都未单独消除 Query Sparse 精度崩溃。
- 将默认选位继续后移到 layer 10 并不会单调改善质量：同一 b32 HumanEval 前 16 题
  layer 6 为 15/16，layer 10 仅 11/16（缩进归一化 12/16），且速度从 22.51 降到
  19.16 tok/s，因此保留 layer 6。
- 当前没有为 SDAR Query Sparse 写专用 fused kernel。此时先写 kernel 风险较高，
  因为算法仍未在 block-16/32 保住精度，并且现有瓶颈包含许多小 Tensor 操作和
  cache 更新，而不只是单个可替换算子。

### 最可能的失败原因

1. **`ratio=0.5` 的 early-layer 候选召回不足。** P0 证明不裁剪时 patched 路径与
   dense 完全一致；P1 中仅把比例提高到 0.75 就恢复 dense 聚合精度，而每步刷新
   KV 仍有差距。layer 6 的单 prompt `candidate@0.5` recall 也只有 84.44%。这是
   当前剩余精度损失的首要解释。
2. **未选择位置的状态陈旧。** 每轮只更新 Query 子集的 layer-7 以后 hidden/KV；
   其他 mask 保留旧 KV。随着相邻 mask 被解码，attention 上下文发生变化，但旧
   mask 的 late-layer KV 不随之更新。上界实验和全量结果已确认它会造成任务错误；
   refresh-index-2 单次刷新把 b32 官方差距从 6.71 pp 缩到 4.88 pp，但没有消除
   全部误差。
3. **旧诊断与 runtime 曾不对齐，导致过早裁剪。** 旧 overlap 漏掉 final
   RMSNorm，把 layer 4 的候选召回估得过高。该问题已修复且默认选位层已改为 6；
   正确的 layer-5 全量实验也确认不应把默认层前移。
4. **浅层 logits 参与 transfer 不是当前配置的误差源。** deep-only 与 all-mask
   的 164 条输出完全相同。这个结论仅覆盖当前 `threshold=1.0`、
   `low_confidence_dynamic` 和 ratio=0.5；更低 threshold 或其他 strategy 仍需
   单独验证。
5. **历史 Query 选择与 transfer strategy 未对齐。** 该实现错误现已修复，但表中
   block-16/32 与旧 overlap 结果都包含它的影响，需要重跑才能判断修复幅度。
6. **小 block 没有足够算术收益。** block=4、ratio=0.5 时每轮最多省约两个 mask
   的后续层计算，但仍要执行前 6 层 dense、额外 lm_head、topk、index_select、
   cache `index_copy_` 和 Python 调度，H800 上这些开销超过节省，因此 block-4
   Query 比 dense 慢约 3%--5%。
7. **稀疏形状没有高效 kernel。** 当前大量 compact gather/scatter、DynamicCache
   转换和小尺寸 attention/MLP kernel 无法充分利用 GPU；即使理论 FLOPs 减少，
   wall time 也不一定下降。
8. **历史实验配置混杂。** `SDAR-8B-Chat` 与 `SDAR-8B-Chat-b32`、BF16 与 FP16、
   block 4/16/32、普通 GSM8K 与 `gsm8k_sdar` 都出现过。错误的跨配置比较曾把
   评测链路问题误判为模型或 Sparse 算法问题。

### 后续建议顺序

1. 优先改善固定 `ratio=0.5` 下的 selector recall，而不是增加 refresh 次数。先在
   HumanEval 实际 sparse states 上记录“layer-6 候选是否覆盖 dense teacher 最终
   transfer”，再测试最小改动：从一半预算中保留少量 age/exploration 槽位，或仅
   在 layer-6 排名 margin 很小时动态扩容。`ratio=0.75` 作为质量上界。
2. 对新的 selector 做 paired ablation：保持同一 checkpoint/seed/refresh step，
   同时报告官方、缩进归一化、逐 response 一致率和 tok/s，避免只看聚合分数掩盖
   trajectory churn。
3. 暂不把每步 late-KV refresh 作为默认修复：它只带来 +1.22 pp 官方/+2.44 pp
   归一化提升，仍未达到 dense。可保留 `SPARSE_DLM_REFRESH_STEP=-2` 作为诊断上界。
4. 用当前已对齐 strategy 的实现重跑 overlap 和小规模质量测试，分别验证
   confidence、entropy 与 sequential 的 Query/transfer 集合是否一致。
5. 若需要覆盖小 block，再用当前代码在 `SDAR-8B-Chat` 上重跑 block-4
   dense/query HumanEval 和 GSM8K；结果使用新的 OUTPUT_PATH，不能覆盖历史目录。
6. 算法质量稳定后再 profile。优先融合 selected KV gather/write 和减少 Python
   cache 操作；只有 profiler 显示单一热点后再写专用 kernel。

常用命令：

```bash
# Dense / Query Sparse，务必显式指定同一个 checkpoint 和输出目录
MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat-b32 \
QUERY_SPARSE=false BENCHMARK=humaneval OUTPUT_PATH=<dense-name> \
bash eval_instruct/eval.sh

MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat-b32 \
QUERY_SPARSE=true SPARSE_DLM_REFRESH_STEP=2 \
BENCHMARK=humaneval OUTPUT_PATH=<query-name> \
bash eval_instruct/eval.sh

# 所有 transfer strategy 的 layer overlap
CUDA_VISIBLE_DEVICES=5 bash scripts/run_sdar_layer_overlap_all_strategies.sh

# CPU 回归
PYTHONPATH=. /home/ysy/anaconda3/envs/llada/bin/python \
  -m unittest tests.test_sdar_block_diffusion_patch tests.test_sdar_layer_overlap -v
```

## Instruct Evaluation

```bash
LIMIT=1 bash eval_instruct/eval.sh
```

For SDAR, explicitly select the `sdar` adapter. Results then default to
`../sdar_exp/`:

```bash
MODEL_TYPE=sdar MODEL=/data0/ysy/models/SDAR-8B-Chat \
LIMIT=1 BENCHMARK=gsm8k \
bash eval_instruct/eval.sh
```

See `eval_instruct/README.md` for dependency setup, benchmark selection, and
native-baseline commands.

## Test

```bash
/home/ysy/anaconda3/envs/llada/bin/python tests/test_block_cache_sparse_dlm.py
/home/ysy/anaconda3/envs/llada/bin/python -m unittest discover -s tests -v
```
