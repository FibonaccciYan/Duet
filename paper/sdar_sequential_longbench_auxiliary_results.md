# SDAR Sequential LongBench Auxiliary Results

Date: 2026-09-14. All runs use SDAR-8B-Chat-b32 on physical GPU 4 with
sequential remasking, threshold 0.95, layer 5, ratio 0.7, 32-token blocks,
32 denoising steps, independent per-layer prefix selection, and recursive
Raw-L1 prefix screening. Prefix budget is 512. Each QA configuration contains
950 examples; `gov_report` contains 200 examples.

## Five LongBench QA tasks

The overall columns are sample-weighted across HotpotQA, TriviaQA,
NarrativeQA, Qasper, and MultiFieldQA-en.

| Configuration | Overall F1 | Exact match |
|---|---:|---:|
| Dense | 54.0965 | 37.4737 |
| Query | 54.0448 | 37.4737 |
| Prefix | 51.7289 | 35.3684 |
| Query + Prefix | 51.5483 | 35.3684 |

| Task | Dense | Query | Prefix | Query + Prefix |
|---|---:|---:|---:|---:|
| HotpotQA | 56.9918 | 56.9918 | 55.7984 | 55.5922 |
| TriviaQA | 90.2397 | 90.2397 | 89.8373 | 89.3423 |
| NarrativeQA | 30.2163 | 30.2163 | 28.1588 | 28.1276 |
| Qasper | 37.7676 | 37.7276 | 33.3388 | 33.2944 |
| MultiFieldQA-en | 55.6573 | 55.3831 | 51.4385 | 51.3310 |

Relative to sequential dense, Query changes overall F1 by -0.0517 points,
Prefix by -2.3676, and Query + Prefix by -2.5481.

## `gov_report`

| Configuration | ROUGE-L | Mean generated tokens | Mean seconds/example |
|---|---:|---:|---:|
| Dense | 29.6917 | 334.91 | 11.59 |
| Query | 29.6261 | 335.08 | 12.89 |
| Prefix | 17.8460 | 246.30 | 8.85 |
| Query + Prefix | 17.8343 | 251.09 | 9.88 |

Sequential Query remains within 0.066 ROUGE-L of dense. Prefix and Query +
Prefix lose 11.846 and 11.857 points respectively, consistent with the
fixed-prefix-capacity diagnosis in `sdar_gov_report_prefix_drop_analysis.md`.

## Comparison with low-confidence dynamic

| Configuration | Five-task sequential F1 | Five-task dynamic F1 | Sequential `gov_report` | Dynamic `gov_report` |
|---|---:|---:|---:|---:|
| Dense | 54.0965 | 51.6306 | 29.6917 | 29.5651 |
| Query | 54.0448 | 51.6828 | 29.6261 | 28.9894 |
| Prefix | 51.7289 | 48.1070 | 17.8460 | 16.9767 |
| Query + Prefix | 51.5483 | 47.9668 | 17.8343 | 16.9700 |

The five-task values above are sample-weighted aggregates. Sequential improves
all four configurations relative to the tested dynamic strategy, but changing
the remasking strategy does not remove the large `gov_report` prefix loss.

Raw reports:

- `/data0/ysy/sparse/sdar_exp/longbench5_sequential_t095_b512_l5_r0p7_20260914`
- `/data0/ysy/sparse/sdar_exp/longbench_sequential_t095_b512_l5_r0p7_20260914`
- `/data0/ysy/sparse/sdar_exp/longbench_dynamic_t095_b512_l5_r0p7_20260913`
- `/data0/ysy/sparse/longbench_gov_report_supplement_20260913/sdar`
