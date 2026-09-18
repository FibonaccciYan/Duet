# FOCUS v8 NarrativeQA80 全量测试准备（未启动）

日期：2026-09-17。目标仓库分支 codex/integrate-all-20260916。
参考 Sparse 全量测试，复用已有 dense_v3 全量结果，不运行 dense。

## 测试矩阵

- 模型：SDAR-8B-Chat-b32、LLaDA2.1-mini（Q mode）。
- 方法仅 focus_v8。LLaDA 默认 native FlashAttention；SDAR 默认 maskless SDPA，保留 early-stop 修复。
- 数据：/data0/gs/dataset_preparation/narrativeqa_speed80_20260917/dataset。
- 每个模型每档80条，三个档位；共480个样本条件、1440次正式生成。
- 每个模型/长度组2次完整warmup；每条3次E2E重复，加1次独立Event计时，不将计时pass纳入E2E。
- 名义8K/16K/32K实际输入8192/16384/32512，输出固定256。32K总窗口32768，与已完成dense测试对齐；不是严格32768输入。
- 原始input_ids不重新tokenize；全部480个输入哈希已与保存的dense逐条校验。tokenizer文件哈希一致。
- block_length=32，steps=32，temperature=0，seed=42，关闭EOS提前结束。
- LLaDA threshold=0.7，editing_threshold=0.5，num_to_transfer=1。
- SDAR low_confidence_dynamic，threshold=0.95，mask_id=151669。
- 使用runtime已有公共MoE/RMSNorm优化；不启用CUDA Graph，不修改算法。

## 单步指标：只排除每个block的第一步

完整step从denoising循环体进入到该轮退出，含forward、选择/采样、token写回和控制开销。
计入所有实际执行transfer的迭代，包含LLaDA末尾的editing确认步，即使随后break。
不计prefill、finalize以及没有执行transfer的退出检查。

- 每block第一步是零基index0，且仅这一条排除。
- index1（第二步）及以后全部计入，包括editing步。
- 每条样本mean = 所有非首步cuda_ms之和 / 非首步数量。
- 每个模型/长度的全量mean = 80条样本所有非首步cuda_ms之和 / 所有非首步数量。
- 不是中位数，也不是80个样本mean的简单平均。后者仅作为明确命名的诊断值。
- 若某样本所有block都只走一步，其非首步均值为null，不伪造0。

已有dense结果旧steady标记采用index>=2。新的汇总器不使用该标记，而是从原始step_records按index>=1重新聚合；已离线重算全部6组dense，无GPU推理。
比较时核对sample_index、source_id、prompt_sha256、prompt_tokens和generation_length，缺样本或不一致会报错，不输出不完整的全量比较。

## 正式运行入口（仅准备，尚未执行）

```bash
cd /data0/gs/SparseDLM_LLaDA_SDAR_integration_20260916
bash scripts/performance/run_narrative80_focus_gpu4.sh --plan
# 后续明确启动时才执行：
bash scripts/performance/run_narrative80_focus_gpu4.sh --run
```

- 不带参数默认只运行CPU预检；必须显式--run才启动正式任务。
- 固定CUDA_VISIBLE_DEVICES=4，按SDAR、LLaDA21及8K/16K/32K顺序运行6组FOCUS。
- GPU4忙时拒绝启动，不停止任何已有进程。
- 脚本没有占卡程序调用；完成、失败退出后不重新占用GPU4。
- 正式目录已存在则拒绝覆盖，不自动删数据或重跑已完成样本。
- 正式输出：results/narrative80_focus_gpu4_20260917/formal。
- 每条保存三次E2E、step_counts、每block步数、输出token/text、trace、全部step_records、非首步数量/总毫秒/均值。
- 每组保存manifest（数据/源码哈希、GPU、配置）、warmup、partial、COMPLETE；运行结束保存EXIT_CODE。
- 汇总器自动读取旧dense结果：results/narrative80_gpu4_20260917/formal。
- E2E全量加速比 = dense各样本E2E中位数之和 / FOCUS对应值之和。
- step加速比使用双方重新聚合的非首步加权平均时间。

## 已完成验证

- CPU测试11 passed，含index1不排除、按步数加权、无样本返回null、LLaDA最终break计数、finalize不计入step以及GPU角色限制。
- 六组数据80条全部输入/配置与旧dense结果匹配，plan状态PREPARED_NOT_STARTED。
- GPU5仅smoke：两个模型各1条最大长度样本，limit1/repeats1，另有2次warmup和独立计时。
- SDAR：49总步、41非首步；LLaDA：53总步、45非首步。计时输出/trace/步数一致。
- GPU5 smoke通过run_reserved_gpu5.sh自动恢复占用；最终PID623246，约72992MiB。
- smoke结果位于results/narrative80_focus_preparation_20260917，与正式目录隔离。
- 全量尚未启动；正式目录未创建。
