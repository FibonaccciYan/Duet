# LoSA v4: diagnostic progress, acceptance incomplete

n33 physical GPU5, H800 PCIe. Branch codex/integrate-all-20260916,
base fc052c5325181045e436519a06b55a36739b505a plus uncommitted versioned code.
Real gov_report sample115, 8192 prompt tokens, 64 generated-token budget.
One measured repetition after warmup; these are exploratory measurements.
Public MoE applied equally; SDAR public RMSNorm applied equally.
LoSA group_mean, active top5, token budget256, page16 remain unchanged.

| Model | Matched dense seconds | v3 seconds | v4 fast-prefill seconds |
|---|---:|---:|---:|
| LLaDA2.1 Q | 0.781850 | 1.233011 | 1.131582 |
| SDAR low_confidence_dynamic | 1.138059 | 4.294324 | 2.443644 |

SDAR v4 tokens matched v3 on this sample. LLaDA tokens differed;
diagnostic short-output ROUGE .100575 vs .100430 is not quality acceptance.
Sparse forward medians stayed approximately 31ms LLaDA and 42.6ms SDAR.

A subsequent v4 cache-view change avoids DynamicCache.from_legacy_cache
copying prefix KV through concat on every forward. Two CPU ownership/update
unit tests passed. SDAR real8K retest: v3 4.305154s, v4 2.402765s;
sparse forward medians42.792/42.065ms, tokens equal. This is only a small
observed incremental gain; repeated matched runs are needed.

The diagnostic SDAR profiler before the cache-view patch attributes 41.15%
of self CUDA time to mm,21.64% to cat,11.62% to paged prefix attention.
These are operator-time fractions under profiling, not wall-time speedup bounds.

Benchmark now supports --candidate v3|v4; v4 requires explicit numerical
experiment mode. It rejects prompt+generation exceeding configured capacity.
32K protocol remains unresolved; do not silently label32704+64 as32768 input.

Still unmet: >=1.3x vs dense, full-step timing, multi-sample quality, all
8K/16K/32K cells and all methods. Original implementation directories preserved.
Latest GPU holder verified running PID1894097,72992MiB.
