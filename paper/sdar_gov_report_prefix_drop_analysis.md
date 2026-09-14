# Why SDAR Prefix Sparsity Drops on `gov_report`

Date: 2026-09-14. This note analyzes SDAR with
`low_confidence_dynamic`, confidence threshold 0.95, layer 5, ratio 0.7, and
32-token generation blocks. Unless marked as a diagnostic subset,
`gov_report` results cover all 200 LongBench examples and allow up to 512
generated tokens.

## Conclusion

The main cause is the fixed-size historical-prefix bottleneck, not query
sparsity or EOS early stopping. With budget 512, raw-L1 prefix selection keeps
only 512 historical KV positions per layer: a mean 6.72% of the available
history on `gov_report`. This is poorly matched to long-form summarization,
which needs evidence coverage across the whole document. Recursive candidate
reuse makes the loss larger because discarded positions cannot return, but it
is only a secondary cause: full-KV rescreening still remains far below dense at
the same budget.

The observed repetition and shorter generations are downstream symptoms of
the degraded compact-prefix logits. Forcing generation to continue to 512
tokens does not recover ROUGE-L.

## Main isolation results

| SDAR configuration | Prefix budget | `gov_report` ROUGE-L | Delta from dense |
|---|---:|---:|---:|
| Dense | -- | 29.5651 | -- |
| Query only | -- | 28.9894 | -0.5758 |
| Prefix, recursive candidates | 256 | 15.3941 | -14.1711 |
| Prefix, recursive candidates | 512 | 16.9767 | -12.5885 |
| Prefix, recursive candidates | 1024 | 19.3745 | -10.1907 |
| Prefix, full-KV rescreen | 256 | 15.8657 | -13.6995 |
| Prefix, full-KV rescreen | 512 | 18.4709 | -11.0942 |
| Prefix, full-KV rescreen | 1024 | 21.3716 | -8.1936 |
| Query + prefix, recursive candidates | 512 | 16.9700 | -12.5951 |

Query-only quality is close to dense, while both prefix configurations lose
about 12.6 points at budget 512. Adding query sparsity to prefix sparsity makes
almost no further difference. This isolates historical-prefix pruning as the
dominant intervention.

Full-KV rescreening improves recursive screening by 1.494 points at budget 512
and 1.997 points at budget 1024. Therefore irreversible recursive pruning is a
real contributor. It does not explain most of the gap: even full rescreening
at budget 1024 remains 8.194 points below dense.

## Budget diagnostic

The following controlled run uses the same first 50 `gov_report` examples for
every configuration and full-KV rescreening for the sparse runs.

| Configuration | ROUGE-L | Mean generated tokens |
|---|---:|---:|
| Dense | 29.3343 | 337.60 |
| Full rescreen, budget 256 | 16.3632 | 177.60 |
| Full rescreen, budget 512 | 18.3143 | 184.52 |
| Full rescreen, budget 1024 | 20.6718 | 208.26 |
| Full rescreen, budget 2048 | 23.0976 | 281.56 |
| Full rescreen, budget 4096 | 26.5861 | 302.62 |

Quality recovers monotonically as more historical KV is retained. More
decisively, three of these 50 inputs have at most 4096 tokens, so budget 4096
does not prune them. On those three examples, dense and prefix produce exactly
the same predictions and both score 41.5516 ROUGE-L. On the remaining 47
examples, where pruning still occurs, dense scores 28.5545 and prefix scores
25.6308; only two predictions are identical. Entering the prefix code path is
therefore not sufficient to cause the loss: the loss appears when KV is
actually discarded.

The effect also grows with context length. For recursive budget 512, the
dense-to-prefix loss is 11.56 points below 8K input tokens, 12.67 at 8--16K,
14.59 at 16--24K, and 17.72 at 24K or longer. The Pearson correlation between
input length and loss is 0.17 (0.20 for full-rescreen budget 512).

## EOS and output degeneration

| First-50 diagnostic | EOS early stop | ROUGE-L | Mean generated tokens |
|---|---:|---:|---:|
| Dense | On | 29.3343 | 337.60 |
| Dense | Off | 29.2854 | 512.00 |
| Prefix, recursive budget 512 | On | 16.4084 | 228.50 |
| Prefix, recursive budget 512 | Off | 16.3774 | 512.00 |

Disabling EOS early stopping forces both methods to produce all 512 tokens but
does not improve either score. Early EOS is thus a symptom rather than the
cause.

The full 200-example outputs show the same degeneration:

| Configuration | Mean / median tokens | Repeated 4-gram ratio | Unique-word ratio |
|---|---:|---:|---:|
| Dense | 361.2 / 512.0 | 0.047 | 0.559 |
| Query only | 344.4 / 359.5 | 0.053 | 0.557 |
| Prefix, recursive budget 512 | 243.9 / 151.0 | 0.281 | 0.445 |
| Prefix, full-rescreen budget 512 | 230.1 / 149.5 | 0.216 | 0.483 |
| Query + prefix, recursive budget 512 | 257.0 / 156.5 | 0.293 | 0.437 |

Repetition is not the complete explanation. Even restricting recursive-budget
512 outputs to cases with repeated 4-gram ratio below 0.05 gives only 17.79
ROUGE-L, versus 29.19 for similarly non-repetitive dense outputs. Missing or
distorted document information precedes the visible generation degeneration.

## Comparison with previous NarrativeQA results

The metrics differ by task: NarrativeQA uses answer F1/EM, while `gov_report`
uses ROUGE-L. Deltas from each task's dense baseline are therefore the useful
comparison.

| SDAR configuration | NarrativeQA F1 | Delta | NarrativeQA EM | `gov_report` ROUGE-L | Delta |
|---|---:|---:|---:|---:|---:|
| Dense | 30.7649 | -- | 13.5 | 29.5651 | -- |
| Query only | 30.1672 | -0.5977 | 12.5 | 28.9894 | -0.5758 |
| Prefix, recursive budget 256 | 24.7973 | -5.9676 | 11.5 | 15.3941 | -14.1711 |
| Prefix, recursive budget 512 | 25.5866 | -5.1783 | 11.0 | 16.9767 | -12.5885 |
| Prefix, recursive budget 1024 | 27.0109 | -3.7540 | 11.5 | 19.3745 | -10.1907 |
| Prefix, full-rescreen budget 256 | 25.0414 | -5.7235 | 11.5 | 15.8657 | -13.6995 |
| Prefix, full-rescreen budget 512 | 25.4779 | -5.2870 | 11.0 | 18.4709 | -11.0942 |
| Prefix, full-rescreen budget 1024 | 27.0109 | -3.7540 | 11.5 | 21.3716 | -8.1936 |

At budget 512, recursive prefix loses 5.18 NarrativeQA F1 points but 12.59
`gov_report` ROUGE-L points. The difference is consistent with the generation
requirements. NarrativeQA outputs are very short in these runs (mean 5.73
tokens dense and 4.62 tokens with prefix; no sample reaches its 128-token
limit), and answering can succeed from a small local evidence subset.
`gov_report` dense outputs average 361.2 tokens and span about 11.5 generation
blocks. They require broad source coverage and repeatedly reuse the historical
cache, so both selection omissions and logit drift can accumulate across
blocks.

For reference, LLaDA shows the same direction but a smaller budget-512 loss:

| Model | Task | Dense | Prefix budget 512 | Delta |
|---|---|---:|---:|---:|
| LLaDA | NarrativeQA F1 | 23.8082 | 22.4220 | -1.3862 |
| LLaDA | `gov_report` ROUGE-L | 32.6295 | 26.9884 | -5.6411 |
| SDAR | NarrativeQA F1 | 30.7649 | 25.5866 | -5.1783 |
| SDAR | `gov_report` ROUGE-L | 29.5651 | 16.9767 | -12.5885 |

These results establish that SDAR is less robust to this prefix approximation,
but they do not by themselves identify an SDAR-specific architectural cause.
The interaction between `low_confidence_dynamic` transfer and compact-prefix
logit drift is plausible, especially over many generation blocks, but would
require layer/block-level logit-divergence traces to demonstrate.

## Mechanism in the current implementation

At the first denoising step of each generation block, SDAR performs a dense
refresh over the available full prefix and captures the block queries. It then
uses raw-L1 query--key distance independently at each layer to retain at most
`prefix_token_budget` historical KV positions. The later denoising steps in
that block use this compact cache.

This objective finds keys close to the current queries; it does not optimize
document-section coverage, diversity, or summary salience. A fixed 512-token
subset can therefore be adequate for a localized short answer while omitting
material needed for a report-wide summary. Under recursive screening, the next
block searches only the previously retained positions plus newly generated
positions, so an omitted source token cannot re-enter later.

## Evidence status

- **Established:** prefix KV pruning is the dominant cause; query pruning and
  EOS stopping are not. Increasing budget recovers quality, and no-pruning
  examples exactly match dense.
- **Established contributor:** recursive candidate reuse adds loss, especially
  on `gov_report`, but full rescreening does not close the fixed-budget gap.
- **Observed consequence:** prefix runs become shorter and much more
  repetitive, but low quality remains even after controlling for repetition.
- **Supported mechanism:** raw-L1 nearest-key selection lacks the global
  coverage objective required by long-form summarization, and fixed-budget
  errors recur across many generation blocks.
- **Not yet established:** why SDAR is more sensitive than LLaDA at equal
  budget. Proving that requires aligned dense/sparse logit divergence and
  retained-position coverage measured by block and layer.

Raw full-run reports are in
`/data0/ysy/sparse/longbench_gov_report_supplement_20260913`; the first-50
budget and EOS diagnostics are in
`/data0/ysy/sparse/sdar_exp/gov_report_prefix_diagnostics_20260914`.
