# LLaDA Test Results

汇总时间：2026-08-14。以下结果来自仓库中现存的 JSON/报告；HumanEval
使用现存生成 JSONL 修正代码抽取后重新执行测试，没有重新生成模型输出。

## Instruct 评测参数列

共同配置：`LLaDA2.1-mini`、BF16、SDPA、batch size 1、temperature 0、
block length 32、steps 32、`num_to_transfer=1`。GSM8K/Math/HumanEval 为
0-shot，MMLU 为 4-shot。

| 参数 | Native Speed Mode | Native Quality Mode | Sparse Speed Mode | Sparse Quality Mode |
|---|---:|---:|---:|---:|
| block-cache SparseDLM | 否 | 否 | 是 | 是 |
| sparse ratio | N/A | N/A | 0.5 | 0.5 |
| selection top-k | N/A | N/A | 64 | 64 |
| selection interval | N/A | N/A | 4 | 4 |
| dense fallback mask count | N/A | N/A | 4 | 4 |
| threshold | 0.5 | 0.7 | 0.5 | 0.7 |
| editing threshold | 0.0 | 0.5 | 0.0 | 0.5 |

同一路径下的 Speed/Quality 结果拥有不同时间戳，且 threshold 和 editing
threshold 不同，因此按独立列展示。

## Instruct 主结果

| 测试 | 指标 | 样本数 | 生成上限 | Native Speed | Native Quality | Sparse Speed | Sparse Quality |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K CoT | flexible exact match | 1,319 | 256 | 52.08% | 52.08% | 48.37% | 43.29% |
| HumanEval Instruct（修正抽取后） | pass@1 | 164 | 1024 | 56.10% | 53.66% | 75.00% | 51.83% |
| Minerva Math（加权总分） | math_verify | 5,000 | 512 | 37.88% | 38.06% | 39.92% | 39.16% |
| MMLU Generative | exact match | 14,042 | 128 | 20.92% | 21.03% | 44.20% | 43.08% |

说明：HumanEval 的 model args 记录 `gen_length=768`，但任务配置传入
`max_gen_toks=1024`；适配器优先使用任务参数，因此表中有效生成上限为 1024。
GSM8K 的 strict-match 四列均为 0，主表采用 harness 同时报告的
`flexible-extract` 指标。

## HumanEval 代码抽取修正

原 `humaneval_instruct` 后处理会把生成续写直接拼到函数 docstring 后。大量
续写的首行只有 1 个空格，导致 Python 缩进错误，随后被 sanitizer 丢弃。
`eval_instruct/reevaluate_humaneval.py` 现在会：

1. 截取第一个生成代码段并去掉后续 Markdown 说明。
2. 对函数续写补足最小 4 空格缩进；若响应已包含目标函数，则保留顶层定义。
3. 复用 `code_eval` 的 `unsafe_execute`、临时目录、可靠性保护和 3 秒超时执行测试。

| 模式 | 原 pass@1 | 修正后 pass@1 | 通过数 | 新增通过 | 目标函数保留数（前→后） |
|---|---:|---:|---:|---:|---:|
| Native Speed Mode | 20.73% | 56.10% | 92/164 | +58 | 56→131 |
| Native Quality Mode | 25.00% | 53.66% | 88/164 | +47 | 68→129 |
| Sparse Speed Mode | 37.20% | 75.00% | 123/164 | +62 | 83→160 |
| Sparse Quality Mode | 21.34% | 51.83% | 85/164 | +50 | 58→124 |

四组重评均直接使用原始 `resps`，没有重新运行模型；原先通过的样本均未回退。
由于沙箱禁止原 `code_eval` 的 Manager 本地套接字，重评脚本以逐题子进程退出码
代替 Manager 回传结果，候选代码、测试、可靠性保护和超时逻辑保持不变。

复现单个 JSONL：

```bash
HF_ALLOW_CODE_EVAL=1 /home/ysy/anaconda3/envs/llada/bin/python \
  eval_instruct/reevaluate_humaneval.py <samples.jsonl> \
  --label <name> --output <result.json>
```

## Minerva Math 子集

| 子集 | 样本数 | Native Speed | Native Quality | Sparse Speed | Sparse Quality |
|---|---:|---:|---:|---:|---:|
| Algebra | 1,187 | 51.31% | 51.47% | 58.89% | 57.20% |
| Counting & Probability | 474 | 47.26% | 48.31% | 45.99% | 43.04% |
| Geometry | 479 | 21.29% | 20.88% | 24.22% | 24.01% |
| Intermediate Algebra | 903 | 12.74% | 13.62% | 13.84% | 14.51% |
| Number Theory | 540 | 38.52% | 39.81% | 34.81% | 35.00% |
| Prealgebra | 871 | 65.56% | 64.41% | 65.90% | 64.75% |
| Precalculus | 546 | 11.90% | 11.72% | 13.92% | 13.92% |
| **加权总分** | **5,000** | **37.88%** | **38.06%** | **39.92%** | **39.16%** |

Quality Mode 相对同路径 Speed Mode：Native 提升 `0.18 pp`，Sparse 下降
`0.76 pp`；Sparse Quality 仍比 Native Quality 高 `1.10 pp`。

## MMLU 分类结果

| 分类 | Native Speed | Native Quality | Sparse Speed | Sparse Quality |
|---|---:|---:|---:|---:|
| STEM | 11.23% | 11.29% | 35.62% | 34.51% |
| Humanities | 11.52% | 12.22% | 33.62% | 32.54% |
| Social Sciences | 24.86% | 24.02% | 57.46% | 55.22% |
| Other | 41.10% | 41.29% | 55.78% | 55.71% |
| **总分** | **20.92%** | **21.03%** | **44.20%** | **43.08%** |

Native Speed Mode 与 Sparse Speed Mode 均为 4-shot、57 个子任务、14,042 个样本，且使用
相同的 Speed Mode 去噪参数；两者仅 `sparse_dlm` 及其稀疏执行路径不同。
Sparse Quality Mode 相对 Sparse Speed Mode 总分下降 `1.12 pp`。
Native Quality Mode 相对 Native Speed Mode 总分提升 `0.11 pp`。

## Instruct 整次评测耗时

| 测试 | Native Speed | Native Quality | Sparse Speed | Sparse Quality |
|---|---:|---:|---:|---:|
| GSM8K | 03:50:14.95 | 04:48:49.24 | 03:19:10.18 | 04:54:25.47 |
| HumanEval 原始评测 | 03:30:10.77 | 04:37:11.45 | 01:11:01.59 | 01:31:51.66 |
| Minerva Math | 31:47:55.09 | 39:54:39.78 | 20:27:48.34 | 29:30:17.54 |
| MMLU | 61:43:55.55 | 78:11:14.05 | 27:23:41.00 | 32:22:44.94 |

耗时来自各 `results_*.json` 的 `total_evaluation_time_seconds`，包含模型加载、
数据准备、生成、评分和结果写入，不是纯 decode 时间；HumanEval 后处理重评耗时
不包含在表中。

## 历史吞吐实验

这部分来自 `experiments/legacy` 的早期 `sparse_kv` 实现，不是当前
block-cache SparseDLM，因此不能与上面的准确率列直接归为同一方法。

| 参数组合 | 模式 | 重复次数 | median time | requested tok/s | actual output tokens | 相对加速 |
|---|---|---:|---:|---:|---:|---:|
| gen=512, block=32, steps=32, ratio=0.5, fallback=4 | Vanilla | 3 | 100.5305 s | 5.09 | 512 | 1.0000x |
| gen=512, block=32, steps=32, ratio=0.5, fallback=4 | Sparse KV | 3 | 73.6573 s | 6.95 | 332 | 1.3648x |
| gen=4, block=4, steps=4, ratio=0.5, fallback=0 | Vanilla | 1 | 1.3897 s | 2.88 | 4 | 1.0000x |
| gen=4, block=4, steps=4, ratio=0.5, fallback=0 | Sparse KV | 1 | 1.1197 s | 3.57 | 4 | 1.2411x |

gen=512 实验中 Sparse KV 实际只输出 332 tokens，而吞吐按请求的 512 tokens
计算；因此 6.95 requested tok/s 和 1.3648x 只能作为延迟口径，不能视作等量
实际输出吞吐。其实际输出吞吐均值为 4.48 tok/s，低于 Vanilla 的 5.10 tok/s。

## 历史 Early-Layer 选位实验

单 prompt 配置为 gen=128、block=32、steps=32、threshold=0.5、
editing threshold=0.0。下表为候选层对最终修改位置的 recall，不同 ratio 独立分列。

| predict layer | exact decode recall | recall@0.3 | recall@0.5 | recall@0.7 | recall@0.9 | recall@1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47.3% | 52.1% | 71.8% | 83.5% | 93.1% | 99.5% |
| 2 | 52.7% | 45.2% | 71.3% | 87.2% | 96.8% | 99.5% |

该结果仅来自一个 prompt，用于验证选位方向，不作为通用质量结论。

## Prefix/Query Sparse v0.1 消融

共同配置：block-cache、BF16、SDPA、block length 32、steps 32、
`sparse_dlm_ratio=0.5`、`threshold=0.5`、`editing_threshold=0.0`；
prefix 稀疏配置使用每层 `token_budget=256`。四组 HumanEval 均对保存的
生成 JSONL 运行相同的代码抽取和缩进归一化重评，未重新生成模型输出。

| 配置 | query sparse | prefix sparse | 原始 pass@1 | 重评 pass@1 | 通过数 | 原始评测时间 |
|---|---:|---:|---:|---:|---:|---:|
| Dense block cache | 否 | 否 | 46.95% | **80.49%** | 132/164 | 00:34:41.78 |
| Prefix only | 否 | 是 | 43.90% | 79.27% | 130/164 | 00:39:11.77 |
| Query only | 是 | 否 | 37.20% | 75.00% | 123/164 | 01:02:03.09 |
| Prefix + Query | 是 | 是 | 31.10% | 68.29% | 112/164 | 01:07:21.73 |

相对 Dense block cache，Prefix only 的重评准确率下降 1.22 pp，Query only
下降 5.49 pp，组合下降 12.20 pp；对应总评测耗时分别增加 13.0%、78.8% 和
94.1%。这些时间来自 harness 的原始评测，未包含重评阶段的代码执行时间。

## 回归测试

`tests/test_block_cache_sparse_dlm.py` 当前包含并通过 8 项 CPU 测试：

1. query confidence selection 使用真实 decode confidence。
2. dual cache 只覆盖选中的 current-block KV。
3. Python Hadamard/Adamas prefix selector 遵守预算。
4. ratio=1.0 的 cached forward logits 与完整 dense forward 对齐。
5. LoSA online-softmax merge 与拼接后的完整 attention 对齐。
6. LoSA 首次 cached forward 与 dense forward 对齐。
7. compact prefix cache 与 dense masked prefix 对齐。
8. ratio=0.5、selection interval=3 的多 block sparse generation 能完成且无 mask 残留。

## 结果文件

- Instruct JSON：`eval_instruct/output_reproduce/`
- HumanEval 重评脚本：`eval_instruct/reevaluate_humaneval.py`
- Native Speed Mode 重评：`eval_instruct/output_reproduce/default/humaneval/__data0__ysy__models__LLaDA2.1-mini/reeval_indent_normalized_native.json`
- Native Quality Mode 重评：`eval_instruct/output_reproduce/default/humaneval/__data0__ysy__models__LLaDA2.1-mini/reeval_indent_normalized_native_quality.json`
- Sparse Speed Mode 重评：`eval_instruct/output_reproduce/block_cache_sparse_dlm_block32_ratio0.5/humaneval/__data0__ysy__models__LLaDA2.1-mini/reeval_indent_normalized_speed.json`
- Sparse Quality Mode 重评：`eval_instruct/output_reproduce/block_cache_sparse_dlm_block32_ratio0.5/humaneval/__data0__ysy__models__LLaDA2.1-mini/reeval_indent_normalized_quality.json`
- MMLU Native Speed Mode：`eval_instruct/output_reproduce/default/mmlu/__data0__ysy__models__LLaDA2.1-mini/results_2026-08-10T13-32-11.337696.json`
- MMLU Native Quality Mode：`eval_instruct/output_reproduce/default/mmlu/__data0__ysy__models__LLaDA2.1-mini/results_2026-08-14T01-39-46.628410.json`
- MMLU Sparse Speed Mode：`eval_instruct/output_reproduce/block_cache_sparse_dlm_block32_ratio0.5/mmlu/__data0__ysy__models__LLaDA2.1-mini/results_2026-08-07T23-34-56.742146.json`
- Native Quality Mode HumanEval 原始结果：`eval_instruct/output_reproduce/default/humaneval/__data0__ysy__models__LLaDA2.1-mini/results_2026-08-10T18-44-07.406230.json`
- Native Quality Mode GSM8K：`eval_instruct/output_reproduce/default/gsm8k/__data0__ysy__models__LLaDA2.1-mini/results_2026-08-10T18-55-32.611222.json`
- Native Quality Mode Math：`eval_instruct/output_reproduce/default/math/__data0__ysy__models__LLaDA2.1-mini/results_2026-08-12T11-23-52.736027.json`
- Sparse Quality Mode Math：`eval_instruct/output_reproduce/block_cache_sparse_dlm_block32_ratio0.5/math/__data0__ysy__models__LLaDA2.1-mini/results_2026-08-11T17-10-43.313049.json`
- Sparse Quality Mode MMLU：`eval_instruct/output_reproduce/block_cache_sparse_dlm_block32_ratio0.5/mmlu/__data0__ysy__models__LLaDA2.1-mini/results_2026-08-11T22-12-19.842411.json`
- 历史吞吐：`experiments/legacy/throughput_sparse_kv_0p5_gen512/`
- Early-layer ratio sweep：`experiments/legacy/llada_early_predict_analysis_20260521/`
