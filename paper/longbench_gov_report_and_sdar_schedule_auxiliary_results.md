# LongBench gov_report Supplement and SDAR Scheduling Study

Date: 2026-09-14. All runs used physical GPU 4. `gov_report` contains 200
examples, uses a maximum generation length of 512, and is scored with the
official LongBench ROUGE-L calculation. The five-task score below is the macro
average of HotpotQA, TriviaQA, NarrativeQA, Qasper, and MultiFieldQA-en F1; the
six-task score is the macro average after adding `gov_report` ROUGE-L.

## Main LongBench configurations

| Model | Method | Five-task macro F1 | gov_report ROUGE-L | Six-task macro |
|---|---:|---:|---:|---:|
| LLaDA | Dense | 51.8849 | 32.6295 | 48.6757 |
| LLaDA | Query | 51.8312 | 32.6238 | 48.6299 |
| LLaDA | Prefix, budget 512 | 50.7777 | 26.9884 | 46.8128 |
| LLaDA | Query + Prefix, budget 512 | 50.5909 | 27.3788 | 46.7222 |
| SDAR | Dense | 51.6287 | 29.5651 | 47.9515 |
| SDAR | Query, layer 5, ratio 0.7 | 51.7542 | 28.9894 | 47.9600 |
| SDAR | Prefix, recursive, budget 512 | 47.9544 | 16.9767 | 42.7914 |
| SDAR | Query + Prefix, recursive, budget 512 | 47.8985 | 16.9700 | 42.7438 |

The SDAR runs use `low_confidence_dynamic`, confidence threshold 0.95,
selection layer 5, and ratio 0.7. The LLaDA sparse runs use layer 1 and ratio
0.7. The raw reports are under
`/data0/ysy/sparse/longbench_gov_report_supplement_20260913`.

## Recursive candidates versus full-KV rescreening

The five-task F1 is the existing 950-example sample-weighted aggregate. The
`gov_report` column is the new 200-example result.

| Prefix policy | Budget | Five-task F1 | Exact match | gov_report ROUGE-L |
|---|---:|---:|---:|---:|
| Recursive | 256 | 46.4075 | 34.0000 | 15.3941 |
| Full rescreen | 256 | 46.6314 | 34.1053 | 15.8657 |
| Recursive | 512 | 48.1070 | 35.7895 | 16.9767 |
| Full rescreen | 512 | 48.3259 | 35.6842 | 18.4709 |
| Recursive | 1024 | 49.6153 | 36.1053 | 19.3745 |
| Full rescreen | 1024 | 49.5859 | 36.1053 | 21.3716 |

Full rescreening changes quality by +0.224, +0.219, and -0.029 F1 points on
the five-task aggregate for budgets 256, 512, and 1024. On `gov_report`, it
improves ROUGE-L by +0.472, +1.494, and +1.997 points respectively.

The table below reports `full-rescreen prefix seconds / recursive prefix
seconds` using the median of four paired runs. Values above 1 mean full
rescreening is slower.

| Budget | Generation | 8K | 16K | 32K |
|---:|---:|---:|---:|---:|
| 256 | 256 | 1.0730 | 1.0859 | 1.0826 |
| 256 | 768 | 1.1416 | 1.1936 | 1.2007 |
| 512 | 256 | 1.0561 | 1.0748 | 1.0796 |
| 512 | 768 | 1.1351 | 1.1851 | 1.1998 |
| 1024 | 256 | 1.0625 | 1.0791 | 1.0750 |
| 1024 | 768 | 1.1254 | 1.1732 | 1.1916 |

Thus full rescreening buys measurable `gov_report` quality at larger budgets,
but costs 5.6%--20.1% latency in these runs. Recursive screening remains the
speed-oriented default; the results do not support describing the two policies
as quality-equivalent on long-form summarization.

## SDAR scheduling and kernel experiments

Nsight Systems on SDAR dense, 32K prompt, generation length 32 identified the
shared `_block_causal_prefill_kernel` as the largest GPU hotspot: 324 captured
calls and 3217.882 ms total. SDAR and LLaDA already call the same Triton kernel;
their surrounding model and cache scheduling differs.

| Candidate | Evidence | Decision |
|---|---|---|
| Match LLaDA's 8192-token prefill chunks instead of SDAR's 4096 | Captured calls 324 -> 180, but kernel time 3217.882 -> 3222.018 ms; one-shot end-to-end 5.7887 -> 5.7572 s; peak memory 20.197 -> 20.605 GiB. Four-run paired speedups were effectively unchanged: gen256 1.0274 -> 1.0268 and gen768 1.0161 -> 1.0153. | Reverted: no independent speed benefit and higher memory. |
| Omit the growing decode attention mask because patched attention ignores it | Existing generic-path test no longer observed the required `(4, 8)` visibility contract. | Reverted before benchmarking. |
| Increase shared prefill kernel `BLOCK_N` from 64 to 128 | Exact test differed in 7895/65536 elements; 32K run slowed from 5.7887 to 9.2481 s. | Reverted. |
| Increase shared prefill kernel from 4 to 8 warps | Exact test differed in 1/65536 elements by 1.22e-4; 32K run slowed from 5.7887 to 7.8401 s. | Reverted. |

No tested LLaDA-style scheduling or shared-kernel change met the acceptance
criterion of stable speedup without a quality risk. The final implementation
therefore retains SDAR's 4096-token long-prefill chunks, the decode mask
contract, `BLOCK_N=64`, and four warps. Relevant profiling artifacts are in
`results/nsys_sdar_schedule_20260914`; the final 30 focused kernel and SDAR
tests pass.
