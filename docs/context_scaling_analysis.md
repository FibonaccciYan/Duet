# SparseDLM 加速比与上下文长度

## 结论

上下文长度本身不是加速比曲线的唯一变量。

- LLaDA 使用置信度阈值提前转移 token。稀疏近似会改变 token 置信度和输出，进而
  改变去噪迭代次数。Query 每次前向的节省较小，因此调用次数变化主导了非单调曲线。
- Prefix 将历史 KV 压缩为固定 256 token，使缓存前向耗时基本不再随上下文增长；
  但不可压缩的稠密 prompt prefill 仍随上下文增长，所以总加速不会线性增长。
- SDAR 的顺序策略在本实验中始终执行 62 次缓存前向。Query 带来的绝对节省近似
  固定，而 prompt prefill 随上下文增长，因此 Query 相对加速比随长度下降。
- SDAR Prefix 的压缩开销随上下文增长，单独使用到 32K 才刚刚摊平；与 Query
  组合后，32K 的前缀收益使曲线重新上升。

因此不能只用 FLOP 或上下文长度解释端到端曲线；至少要同时报告 prompt prefill、
缓存前向调用次数、每次实际选中行数和前缀压缩成本。

## 实证

以下是 H800、生成 64 token、block/steps=32 的单次诊断运行。它用于拆分阶段，
不是替代正式的多次中位数测速。`cached ms` 和 `prefill ms` 是 CUDA event 累计值。

### LLaDA

| 上下文 | 模式 | 总耗时 / 秒 | cached 次数 | cached ms | prefill ms | 输出 token |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8K | dense / query / prefix / combined | 1.066 / 0.975 / 1.133 / 1.048 | 26 / 22 / 32 / 27 | 733 / 639 / 767 / 684 | 316 / 319 / 322 / 322 | 54 / 46 / 56 / 52 |
| 16K | dense / query / prefix / combined | 1.286 / 1.493 / 1.111 / 1.131 | 14 / 19 / 14 / 14 | 559 / 760 / 337 / 355 | 711 / 710 / 723 / 724 | 33 / 33 / 32 / 33 |
| 24K | dense / query / prefix / combined | 3.548 / 2.622 / 1.641 / 1.669 | 45 / 27 / 14 / 14 | 2277 / 1354 / 335 / 355 | 1237 / 1235 / 1240 / 1249 | 54 / 33 / 31 / 31 |
| 32K | dense / query / prefix / combined | 3.872 / 4.619 / 2.327 / 3.392 | 31 / 44 / 16 / 55 | 1934 / 2694 / 382 / 1388 | 1903 / 1874 / 1864 / 1907 | 26 / 33 / 34 / 61 |

Query 每次保留 23--32 行，单次 cached-forward 与 dense 很接近。例如 24K 时
dense 与 Query 平均每次约为 50.6 与 50.2 ms；1.35 倍端到端加速主要来自
27 次对 45 次调用。32K 时情况反转为 44 次对 31 次，因此 Query 变为 0.84 倍。
Prefix/combined 的单次 cached-forward 则稳定在约 24--25 ms。

后续门控搜索将 LLaDA 的 `query_dense_threshold` 从 4 提至 20：32K/生成 256
的三次中位数从 2.880 秒降至 2.764 秒（1.042 倍），生成 768 从 4.897 秒
降至 4.798 秒（1.021 倍）。阶段计时显示 cached-forward 从 26 次、645 ms
降至 21 次、514 ms；dense prefill 与 Prefix compaction 不变。LongBench 五任务
50 题中有 6 题触发长上下文 Query，整体 F1 从 0.5243 升至 0.5348，EM 均为
0.32，因此默认值更新为 20。

### SDAR

| 上下文 | dense / query / prefix / combined 秒 | Query 加速比 | Prefix 加速比 | Combined 加速比 |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 2.432 / 1.848 / 2.465 / 1.855 | 1.316 | 0.987 | 1.311 |
| 16K | 3.428 / 2.863 / 3.530 / 2.873 | 1.197 | 0.971 | 1.193 |
| 24K | 4.943 / 4.505 / 5.125 / 4.357 | 1.097 | 0.964 | 1.134 |
| 32K | 7.202 / 6.572 / 7.085 / 6.275 | 1.096 | 1.017 | 1.148 |

四个长度的 Query/Prefix/combined 都执行 62 次 cached-forward。Query 的该阶段
约为 1.18 秒，Prefix 约为 1.78 秒，均几乎不随上下文变化；Query 每轮依次保留
17--32 行。Prefix compaction 从 8K 的 29 ms 增至 32K 的 70 ms。

## 复现

`--phase-profile` 只应用于诊断；CUDA events 会扰动端到端计时，正式速度数据仍应
去掉该选项并至少运行三次。

```bash
CUDA_VISIBLE_DEVICES=0 /home/ysy/anaconda3/envs/llada/bin/python \
  scripts/bench_long_context.py --model llada --mode query_prefix --ablation \
  --contexts 8192 16384 24576 32768 --gen-length 64 --repeats 1 \
  --query-min-prefix-length 0 --prefix-min-prefix-length 0 --phase-profile \
  --output results/sparsedlm_phase_profile_llada.json

CUDA_VISIBLE_DEVICES=0 /home/ysy/anaconda3/envs/dream/bin/python \
  scripts/bench_long_context.py --model sdar --mode query_prefix --ablation \
  --contexts 8192 16384 24576 32768 --gen-length 64 --repeats 1 \
  --query-min-prefix-length 0 --prefix-min-prefix-length 0 --phase-profile \
  --output results/sparsedlm_phase_profile_sdar.json
```
