# Auxiliary Results: SDAR Remasking, Prefix Budget, and Per-Layer Selection

## Experimental setup

- Model: SDAR-8B-Chat-b32, FP16, batch size 1.
- Sparse method: Raw-L1 prefix selection, no cost gates, independent prefix
  selection at every layer (`prefix_share_layer_pairs=false`).
- HumanEval: 164 problems, generation length 768, block length 32, 32 steps,
  threshold 1.0. Scores are official pass count / indentation-normalized count.
- Long-context speed: `low_confidence_dynamic`, prefix budget 512, NarrativeQA
  record 4, exact input lengths 8K/16K/32K, generation lengths 256/768, four
  paired repeats per mode. Latency is the median of four repeats.

## HumanEval quality

### Sequential remasking

The matched dense reference is 129/130.

| Prefix budget | Prefix | Query+Prefix |
| ---: | ---: | ---: |
| 256 | 128/128 | 128/129 |
| 512 | 130/131 | 130/132 |
| 1024 | 130/131 | 129/132 |

For comparison, the historical two-layer-sharing results at budget 256 were
124/127 for Prefix and 124/125 for Query+Prefix. Independent per-layer
selection raises these results to 128/128 and 128/129, respectively.

### Low-confidence dynamic remasking

Dense and Query are budget-independent controls and were run once.

| Prefix budget | Dense | Query | Prefix | Query+Prefix |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 121/125 | 115/117 | 118/122 | 115/117 |
| 512 | — | — | 121/125 | 113/115 |
| 1024 | — | — | 121/125 | 114/117 |

Prefix first matches the dynamic dense reference at budget 512. Query and all
Query+Prefix configurations regress relative to dynamic dense. Dynamic dense
also trails the sequential dense reference by 8 official and 5 normalized
passes.

## Long-context latency

Each entry is median seconds / speedup relative to the paired dense mode.

### Generation length 256

| Exact input length | Dense | Query | Prefix | Query+Prefix |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 8.129 / 1.000x | 6.130 / 1.326x | 8.261 / 0.984x | 6.188 / 1.314x |
| 16K | 9.113 / 1.000x | 7.157 / 1.273x | 9.355 / 0.974x | 7.184 / 1.269x |
| 32K | 13.807 / 1.000x | 10.912 / 1.265x | 12.516 / 1.103x | 10.638 / 1.298x |

### Generation length 768

| Exact input length | Dense | Query | Prefix | Query+Prefix |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 23.568 / 1.000x | 17.466 / 1.349x | 23.962 / 0.984x | 17.642 / 1.336x |
| 16K | 24.620 / 1.000x | 18.684 / 1.318x | 25.019 / 0.984x | 18.665 / 1.319x |
| 32K | 32.428 / 1.000x | 23.662 / 1.370x | 28.330 / 1.145x | 21.981 / 1.475x |

Query+Prefix is faster than dense at all six shapes. Prefix alone is slower at
8K and 16K and faster at 32K.

## Peak allocated memory

Values are GiB.

| Generation / input | Dense | Query | Prefix | Query+Prefix |
| --- | ---: | ---: | ---: | ---: |
| 256 / 8K | 20.04 | 22.62 | 20.54 | 20.54 |
| 256 / 16K | 17.95 | 26.70 | 20.06 | 20.09 |
| 256 / 32K | 20.20 | 37.95 | 24.63 | 24.63 |
| 768 / 8K | 20.04 | 22.62 | 20.54 | 20.54 |
| 768 / 16K | 17.95 | 27.05 | 20.17 | 20.23 |
| 768 / 32K | 20.20 | 38.30 | 24.67 | 24.74 |

All four repeats produced the requested output length and stable per-mode output
checksums.

## Artifacts

- HumanEval outputs:
  `/data0/ysy/sparse/sdar_exp/humaneval_per_layer_raw_l1/`
- Speed, generation 256:
  `results/raw_l1_per_layer_dynamic_b512_narrativeqa_sdar_g256_r4.json`
- Speed, generation 768:
  `results/raw_l1_per_layer_dynamic_b512_narrativeqa_sdar_g768_r4.json`
