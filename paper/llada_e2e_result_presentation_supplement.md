# LLaDA 端到端结果展示方案（补充文档）

本文档规定 10-sample NarrativeQA 端到端实验在论文中的展示方式。目标是突出
方法在目标长上下文场景中的实测优势，同时保持基线、统计口径和负结果透明。

详细实验数据见 `paper/llada_narrativeqa_10sample_e2e_report.md`，原始记录见：

- `results/llada_narrativeqa_10samples_8k_32k_g256_five_modes.json`
- `results/llada_narrativeqa_10samples_8k_32k_g512_five_modes.json`
- `results/llada_narrativeqa_10samples_8k_32k_g768_five_modes.json`

## 1. 主结果的选择

主结果聚焦 32K 输入，因为 Prefix 稀疏针对长前缀开销，当前实验也显示其收益主要
出现在 24K--32K 区间。主算法基线使用 **Maskless Dense**，而不是 Full-mask
Dense；这样不会把 maskless attention 的工程收益计入稀疏算法收益。

主指标采用 **逐样本配对加速比的中位数**：先对每条 NarrativeQA 样本计算
`Maskless Dense 耗时 / 稀疏模式耗时`，再对 10 个比值取中位数。同时报告
“快于基线的样本数”，避免仅靠单个聚合值掩盖跨样本波动。

### 建议主表

| 请求输出长度 | Prefix vs Maskless Dense | Query+Prefix vs Maskless Dense |
|---:|---:|---:|
| 256 | 1.127x；9/10 样本更快 | **1.177x；8/10 样本更快** |
| 512 | 1.127x；9/10 样本更快 | **1.163x；9/10 样本更快** |
| 768 | 1.105x；10/10 样本更快 | **1.139x；9/10 样本更快** |

建议正文使用以下严格限定的表述：

> 在 32K NarrativeQA 输入的 10 样本配对测试中，Prefix 获得
> 1.105--1.127x 中位加速，Query+Prefix 获得 1.139--1.177x；两者分别在
> 9--10/10 和 8--9/10 个样本上快于采用相同 generation policy 的
> Maskless Dense。

这段表述只描述当前实验直接测得的范围，不外推到未测试的模型、上下文长度、生成
长度或数据分布。

## 2. 建议主图

使用三面板折线图：

- 三个面板分别对应请求输出长度 256、512、768；
- 横轴为输入长度 8K、16K、24K、32K；
- 纵轴为相对 Maskless Dense 的逐样本配对加速比中位数；
- 每个面板绘制 Query、Prefix、Query+Prefix 三条线；
- 添加 1.0x 水平参考线；
- 图注注明每个点包含 10 个配对样本，并在点旁或图注中给出快于基线的样本数。

主图必须保留全部四个输入长度，不能只绘制 32K。完整曲线能够同时展示当前配置在
8K/16K 的额外开销、24K 附近的转折以及 32K 的实测收益区间。

## 3. 工程收益与算法收益分解

五种模式的实际开关如下：

| 模式 | Full attention mask | Query sparse | Prefix sparse |
|---|---:|---:|---:|
| Dense | 是 | 否 | 否 |
| Maskless Dense | 否 | 否 | 否 |
| Query | 否 | 是 | 否 |
| Prefix | 否 | 否 | 是 |
| Query+Prefix | 否 | 是 | 是 |

不同对比的含义必须分开：

| 对比 | 所包含的变化 | 推荐用途 |
|---|---|---|
| Maskless Dense / Dense | 移除全零 attention mask | 工程组件消融 |
| Query / Maskless Dense | 仅增加 Query 稀疏 | Query 算法净收益 |
| Prefix / Maskless Dense | 仅增加 Prefix 稀疏 | Prefix 算法净收益 |
| Query+Prefix / Maskless Dense | 增加 Query 与 Prefix 稀疏 | 联合算法净收益 |
| Query+Prefix / Dense | maskless 与联合稀疏全部开启 | 完整系统端到端收益 |

Query+Prefix 相对 Full-mask Dense 的 32K 中位延迟比为：

| 请求输出长度 | Query+Prefix / Full-mask Dense |
|---:|---:|
| 256 | 1.405x |
| 512 | 1.405x |
| 768 | 1.385x |

这组数字只能标为“完整系统端到端加速”，不能作为纯稀疏算法加速，因为其中包含
maskless attention 的收益。

## 4. 显存结果

32K 输入下的最大实测 peak allocation：

| 请求输出长度 | Maskless Dense | Prefix | Query+Prefix |
|---:|---:|---:|---:|
| 256 | 35.452 GiB | 34.101 GiB | 34.112 GiB |
| 512 | 35.556 GiB | 34.111 GiB | 34.122 GiB |
| 768 | 35.660 GiB | 34.120 GiB | 34.131 GiB |

相对 Maskless Dense，Prefix 和 Query+Prefix 的最大峰值分配下降约
3.8%--4.3%。该数据可作为主表的辅助列或独立小表。Query-only 在相同 32K
形状下记录到约 36.7--36.9 GiB，因此不用于强调显存优势，但必须保留在完整消融表
中。

## 5. 统计口径

推荐按以下顺序报告：

1. 主指标：逐样本配对加速比中位数；
2. 同时报告：快于基线的样本数；
3. 延迟分布：中位数及 `[Q1, Q3]`；
4. 附录：平均值、最小值、最大值和全部逐样本记录；
5. 单列长尾样本，不从统计中删除。

以下三种统计量回答的问题不同，不能混用：

- `median(baseline_i / method_i)`：典型样本上的配对相对收益；
- `median(baseline_i) / median(method_i)`：两个独立中位延迟之比；
- `mean(baseline_i) / mean(method_i)`：包含长尾贡献的平均成本比。

本实验中，样本 18 的 32K 记录具有明显长尾。其原始耗时如下，必须保留：

| 请求输出长度 | Dense | Maskless Dense | Query | Prefix | Query+Prefix |
|---:|---:|---:|---:|---:|---:|
| 256 | 18.795s | 4.000s | 17.838s | 2.782s | 2.988s |
| 512 | 35.271s | 4.830s | 19.291s | 3.544s | 3.727s |
| 768 | 51.994s | 5.796s | 20.308s | 4.463s | 4.636s |

不应通过删除该样本来提高均值，也不应只给平均值而隐藏中位数和逐样本分布。

## 6. 主文与附录的内容分配

### 主文

- 32K 主结果表；
- 8K--32K 上下文扩展曲线；
- Maskless Dense 作为稀疏算法主基线；
- Prefix、Query+Prefix 的配对中位加速比和快于基线的样本数；
- 32K 峰值显存；
- 一句话说明 8K/16K 下尚未形成稳定稀疏收益。

### 附录

- 五模式完整 4 x 3 网格；
- 中位数、IQR、平均值、min/max；
- 全部 120 行逐样本延迟；
- Full-mask Dense 到 Maskless Dense 的工程组件消融；
- 输出 checksum 一致数量；
- 样本 18 的长尾记录；
- GPU UUID、样本 ID、原始 prompt token 数和完整运行参数。

## 7. 质量证据边界

该端到端实验测量速度、显存和输出 checksum，没有计算 NarrativeQA 答案质量。
因此不能仅凭本实验将某一速度配置描述为“质量保持”。

当前既有 HumanEval 证据中，LLaDA Raw-L1 budget=256 的 Query+Prefix 通过仓库所述
质量门槛，而 Prefix-only 未通过。由于本次五模式测试还改变了 maskless 组合方式，
最终论文若要将 Prefix 或 Query+Prefix 写为质量保持配置，应补充与本次运行参数完全
匹配的质量评测。质量结果应与速度结果并列，而不能用输出 checksum 相同率替代。

## 8. 不采用的展示方式

- 不只选择最快的单个 NarrativeQA 样本；
- 不只展示 32K 而完全隐藏 8K/16K/24K；
- 不把相对 Full-mask Dense 的完整系统加速写成稀疏算法净收益；
- 不在同一张表中无标注地混用均值比、中位数比和配对比值中位数；
- 不删除样本 18 或其他长尾记录；
- 不根据当前速度结果宣称尚未测量的质量、模型或数据集结论。
