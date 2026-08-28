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
- `src/sparse/moe_expert_patch.py`: MoE kernel implementation used by the core.
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

Run SDAR with its checkpoint defaults (`block_length=4`, `steps=4`, mask ID
151669). Query sparsity is enabled by default; use `QUERY_SPARSE=false` for the
official dense-block path:

```bash
MODEL_TYPE=sdar MODEL_PATH=/data0/ysy/models/SDAR-8B-Chat \
PROMPT="Explain diffusion models briefly." bash scripts/test.sh
```

The SDAR adapter applies query selection after layer 6 and keeps all decoded
positions plus a strategy-selected mask set for later layers. The configured
mask ratio is the minimum Query budget; dynamic and entropy strategies may
select more positions. It uses the official
`sdar_generate.py` block/transfer loop for both dense and sparse inference.
Query sparse is calibrated on the matching SDAR checkpoints for
`block_length=4` and `block_length=32`. Large blocks remain approximate because
errors in current-block KV compound across more denoising rounds; the current
b32 HumanEval result is no longer catastrophic, but is still 4.88 pp below the
dense baseline. The current feature support is:

| Model | query sparse | prefix sparse | LoSA | MoE expert patch |
| --- | --- | --- | --- | --- |
| LLaDA | yes | yes | yes | yes |
| SDAR | yes | no | no | no-op on its dense experts |

## SDAR 状态与后续接手说明（2026-08-28）

本节是 SDAR 工作的单一交接记录。历史结果位于
`/data0/ysy/sparse/sdar_exp/`，层重合度结果位于
`layer_overlap_results/sdar_block32/`。这些结果来自不同 checkpoint、block
配置和代码版本，只有表中明确配对的行可以直接比较。模型目录文件始终未修改。

### 当前实现

- `src/sparse/sdar_generate.py` 保存并复用官方 block diffusion 的 prefill、
  current-block denoise、完成 block 后写入 KV cache 的主循环。Dense 与 Query
  Sparse 经过同一个 transfer 循环，Sparse 只通过 `denoise_fn` 替换单步 forward，
  避免两套生成逻辑继续漂移。
- `src/sparse/sdar_block_diffusion_patch.py` 在运行时 patch `model.generate`。
  每个 block 的第一个 denoise step 保持 dense，用于建立完整 current-block KV；
  后续 step 在第 6 个 decoder layer 后执行 `norm -> lm_head -> sampling` 选位，
  从 layer 7 开始只计算所有已解码位置和配置比例的 mask 位置。
- 未选择 mask 的 layer-7 以后 hidden/KV 不会重算；最终 transfer 仍会看到所有
  mask，其中未选择位置使用 layer-6 hidden 经过 `norm/lm_head` 的近似 logits，
  已选择位置使用完整后续层输出。`_BlockDualCache` 只覆盖本轮选择位置的 KV。
- 对 block length >= 16，每个 block 默认在 sparse step 2 将 layer-7 以后临时扩为
  全 block 一次，刷新所有 current-block late-layer KV；同一步仍只把原 Query
  选中位置的深层 hidden 写回 logits，后续 step 恢复稀疏。可用
  `SPARSE_DLM_REFRESH_STEP=-1` 关闭，小于 16 的 block 不执行该刷新。
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
- 当前 `eval_instruct/eval.sh` 默认 checkpoint 是
  `/data0/ysy/models/SDAR-8B-Chat-b32`，而 `scripts/test.sh` 默认仍是
  `/data0/ysy/models/SDAR-8B-Chat`。对比实验必须显式传 `MODEL`/`MODEL_PATH`，
  不要把两个 checkpoint 的结果混在一起。
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
9. 统一 LLaDA layer-2 与 SDAR layer-6 的 Query 预测规则：浅层选位和最终
   transfer 现在接收同一个 `remasking_strategy`、threshold 与 entropy budget；
   LLaDA 的直接推理和 lm-eval 参数链路也已补齐四种 strategy。
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
    11.37 tok/s；定位后改为每 block 只在 sparse step 2 刷新一次 late-layer KV。
    b32 HumanEval 全量达到 69.51%（114/164），缩进归一化 72.56%（119/164），
    23.55 tok/s；相对不刷新分别提升 1.83/3.05 pp，吞吐未下降。逐题官方口径为
    修复 8 题、回归 5 题（净增 3），说明 step 2 是全量净收益而非单调改进。

### 关键结果

速度是结果 JSON 中的纯 generation tokens/s。标为“旧”的 Query 来自修复前代码；
b32 layer-6 行的 dense 数值沿用同 checkpoint、同 block 配置的历史全量基线，Query
是当前代码的全量重跑。

| checkpoint / 配置 | benchmark | Dense | Query Sparse | Dense tok/s | Query tok/s | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `SDAR-8B-Chat`, block=4, steps=4 | GSM8K | 90.67% | 89.46% | 23.39 | 22.57 | 精度仅降 1.21 pp，但未加速 |
| `SDAR-8B-Chat`, block=4, steps=4 | HumanEval | 81.10% | 73.78%--75.00% | 23.18 | 21.95--22.24 | 小 block 仍掉点且更慢 |
| `SDAR-8B-Chat`, block=16, steps=16 | HumanEval | 71.95% | 10.37%（v0.5），7.93%（v0.6） | 26.56 | 24.39--25.26 | Query Sparse 严重失真 |
| `SDAR-8B-Chat-b32`, block=32, steps=32，旧 Query | HumanEval | 74.39% | 19.51%--20.12% | 25.50 | 24.37--24.43 | 修复前严重失真 |
| `SDAR-8B-Chat-b32`, block=32, steps=32，layer 6、无刷新 | HumanEval | 74.39% | 67.68%（归一化 69.51%） | 25.50 | 23.20 | 主要崩溃已修复，仍差 6.71 pp |
| `SDAR-8B-Chat-b32`, block=32, steps=32，layer 6、step-2 刷新 | HumanEval | 74.39% | **69.51%**（归一化 72.56%） | 25.50 | 23.55 | stale-KV 误差得到缓解，官方差距缩至 4.88 pp |

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
  额外完整 forward，不等同于当前只在 step 2 内联扩展 late layers 一次的方案。
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

1. **旧诊断与 runtime 不对齐，导致过早裁剪。** 旧 overlap 漏掉 final RMSNorm，
   把 layer 4 的候选召回估得过高；真实 runtime 在首个 sparse step 就可能选错
   transfer 候选。当前已把诊断对齐并将默认 selection layer 改为 6。
2. **未选择位置的状态陈旧。** 每轮只更新 Query 子集的 layer-7 以后 hidden/KV；
   其他 mask 保留旧 KV。随着相邻 mask 被解码，attention 上下文发生变化，但旧
   mask 的 late-layer KV 不随之更新。上界实验和全量结果已确认它会造成任务错误；
   step-2 单次刷新把 b32 官方差距从 6.71 pp 缩到 4.88 pp，但没有消除全部误差。
3. **early-layer 排序仍不完美。** 修正后的 layer 6 单 prompt
   `candidate@0.5` recall 为 84.44%，仍有最终应 transfer 的位置不会进入完整
   后续层；这些位置只能用 layer-6 logits 做决定。
4. **历史 Query 选择与 transfer strategy 未对齐。** 该实现错误现已修复，但表中
   block-16/32 与旧 overlap 结果都包含它的影响，需要重跑才能判断修复幅度。
5. **小 block 没有足够算术收益。** block=4、ratio=0.5 时每轮最多省约两个 mask
   的后续层计算，但仍要执行前 6 层 dense、额外 lm_head、topk、index_select、
   cache `index_copy_` 和 Python 调度，H800 上这些开销超过节省，因此 block-4
   Query 比 dense 慢约 3%--5%。
6. **稀疏形状没有高效 kernel。** 当前大量 compact gather/scatter、DynamicCache
   转换和小尺寸 attention/MLP kernel 无法充分利用 GPU；即使理论 FLOPs 减少，
   wall time 也不一定下降。
7. **历史实验配置混杂。** `SDAR-8B-Chat` 与 `SDAR-8B-Chat-b32`、BF16 与 FP16、
   block 4/16/32、普通 GSM8K 与 `gsm8k_sdar` 都出现过。错误的跨配置比较曾把
   评测链路问题误判为模型或 Sparse 算法问题。

### 后续建议顺序

1. 在完整 HumanEval/GSM8K 上比较 refresh step 1--4；单样本上这些 step 均能修复
   已知错误，但当前只完整验证了 step 2，不能从单样本推断最优 step。
2. 继续记录 Query/transfer overlap 和 early/final token 一致率，定位剩余 4.88 pp
   是 early selector 漏选还是未选中位置的浅层 logits 参与 transfer 所致。
3. 用当前已对齐 strategy 的实现重跑 overlap 和小规模质量测试，分别验证
   confidence、entropy 与 sequential 的 Query/transfer 集合是否一致。
4. 若需要覆盖小 block，再用当前代码在 `SDAR-8B-Chat` 上重跑 block-4
   dense/query HumanEval 和 GSM8K；结果使用新的 OUTPUT_PATH，不能覆盖历史目录。
5. 算法质量稳定后再 profile。优先融合 selected KV gather/write 和减少 Python
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
