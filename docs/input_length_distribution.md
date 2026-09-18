# 当前短任务输入长度分布

统计日期：2026-09-18

## 协议

- tokenizer：`/data0/ysy/models/LLaDA2.1-mini`、`/data0/ysy/models/SDAR-8B-Chat-b32`
- 任务：GSM8K、HumanEval、MATH-500 为 0-shot；MMLU 为 5-shot
- 输入由当前 `eval_instruct` 的 lm-eval task 配置构造
- 启用各 tokenizer 自带的 chat template，`fewshot_as_multiturn=false`
- 对最终输入使用 `add_special_tokens=false` 计数，与模型 adapter 一致
- MMLU few-shot seed：`1234`
- 运行环境：`CUDA_VISIBLE_DEVICES=3`；分词实际在 CPU 执行

## 分布

| Tokenizer | Task | N | Min | Mean | P50 | P90 | P95 | P99 | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaDA 2.1 | GSM8K | 1,319 | 46 | 83.72 | 80 | 114 | 126.1 | 149.82 | 211 |
| LLaDA 2.1 | HumanEval | 164 | 135 | 338.35 | 301 | 531.8 | 635.8 | 731.96 | 871 |
| LLaDA 2.1 | MATH-500 | 500 | 43 | 104.92 | 84 | 171.1 | 245.15 | 395.06 | 862 |
| LLaDA 2.1 | MMLU 5-shot | 14,042 | 426 | 898.44 | 725 | 1,805 | 1,909 | 2,914.59 | 3,337 |
| SDAR | GSM8K | 1,319 | 46 | 84.31 | 80 | 115 | 128 | 150.64 | 211 |
| SDAR | HumanEval | 164 | 113 | 305.32 | 273 | 489.8 | 584.4 | 679.74 | 819 |
| SDAR | MATH-500 | 500 | 32 | 94.16 | 74 | 164.1 | 224.1 | 365.06 | 814 |
| SDAR | MMLU 5-shot | 14,042 | 414 | 871.61 | 701 | 1,777 | 1,883 | 2,856.59 | 3,283 |

## 长度阈值覆盖率

| Tokenizer | Task | <=256 | <=512 | <=1024 | >2048 |
|---|---|---:|---:|---:|---:|
| LLaDA 2.1 | GSM8K | 100.00% | 100.00% | 100.00% | 0.00% |
| LLaDA 2.1 | HumanEval | 29.27% | 87.20% | 100.00% | 0.00% |
| LLaDA 2.1 | MATH-500 | 96.00% | 99.20% | 100.00% | 0.00% |
| LLaDA 2.1 | MMLU 5-shot | 0.00% | 14.26% | 79.23% | 3.08% |
| SDAR | GSM8K | 100.00% | 100.00% | 100.00% | 0.00% |
| SDAR | HumanEval | 41.46% | 90.85% | 100.00% | 0.00% |
| SDAR | MATH-500 | 96.80% | 99.40% | 100.00% | 0.00% |
| SDAR | MMLU 5-shot | 0.00% | 16.09% | 80.10% | 2.95% |

## 结论

- GSM8K 输入最短，两套 tokenizer 的最大长度均为 211。
- HumanEval 的长尾比 MATH-500 更明显，但全部输入仍小于 1,024 token。
- 5-shot MMLU 是唯一超过 2,048 token 的任务：LLaDA 2.1 为 3.08%，SDAR 为 2.95%。
- SDAR 在 HumanEval、MATH-500 和 MMLU 上整体略短；GSM8K 基本一致。
