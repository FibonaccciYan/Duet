# SparseDLM runtime for LLaDA and SDAR

This repository provides one GPU runtime for Query Sparse, Prefix Sparse,
LoSA, and the LLaDA MoE optimization. Checkpoint files are never modified.

## Code layout

```text
src/sparse/
├── __init__.py         public exports
├── api.py              model detection and patch_model(...)
├── sparse_ops.py       shared cache, Adamas, LoSA, RoPE, and online merge
├── triton_kernels.py   canonical CUDA implementations
├── llada_patch.py      LLaDA attention, generation, Query Sparse, and MoE patch
└── sdar_patch.py       SDAR attention, generation, and Query Sparse patch
```

The dependency direction is intentionally one-way:

```text
api -> llada_patch / sdar_patch -> sparse_ops -> triton_kernels
```

The runtime requires CUDA, Triton, and `faster_hadamard_transform`. It does not
silently switch to CPU or PyTorch implementations. The Faster Hadamard extension
used in the current environments comes from:

```text
/data0/ysy/Adamas/kernels/3rdparty/applied-ai/kernels/cuda/inference/hadamard_transform
```

## Inference

```bash
MODEL_TYPE=llada bash scripts/generate.sh
MODEL_TYPE=sdar bash scripts/generate.sh
```

Direct invocation:

```bash
CUDA_VISIBLE_DEVICES=5 /home/ysy/anaconda3/envs/llada/bin/python \
  scripts/generate.py --model llada --pattern patch \
  --query_sparse true --prefix_sparse true --losa false

CUDA_VISIBLE_DEVICES=5 /home/ysy/anaconda3/envs/dream/bin/python \
  scripts/generate.py --model sdar --pattern patch \
  --query_sparse false --prefix_sparse true --losa false
```

The public Python entry point is:

```python
from src.sparse import patch_model

patch_model(
    model,
    model_name="llada",  # or "sdar"
    query_sparse=False,
    prefix_sparse=True,
    prefix_token_budget=1024,
    losa=False,
)
```

## Sparse features

### Query Sparse

Query Sparse selects mask positions at an intermediate decoder layer, then
runs only those positions through the remaining layers. Decoded positions stay
in the selected set so their hidden states and KV entries remain available.

`query_dense_threshold` is an algorithm threshold: Query Sparse is skipped when
the current mask count is at or below it. LLaDA defaults to 4 and SDAR defaults
to 0.

LLaDA also uses `query_min_prefix_length=24576`: below that prefix length the
runtime keeps Prefix Sparse but skips Query Sparse because its fixed selection
cost exceeds the saved short-prefix attention work. Set it to 0 to force Query
Sparse at every context length.

LLaDA uses its native confidence selector. SDAR uses the same configured
`remasking_strategy` for shallow query selection and final token transfer.

### Prefix Sparse

Prefix Sparse runs one dense refresh at the beginning of a block and captures
every layer's real RoPE query. LLaDA selects historical positions independently
for every layer; SDAR shares one Adamas selection within each adjacent layer
pair. Every layer gathers its own K/V values, and the complete current block is
always retained.

Validated deployment budgets are:

- LLaDA: `prefix_token_budget=1024`
- SDAR: `prefix_token_budget=512`

The v1.2-compatible speed profile uses `prefix_token_budget=256`. With Prefix
only, it reaches 128/164 normalized HumanEval for LLaDA and 129/164 for SDAR.

The runtime four-bin boundaries are:

| Model | Hq boundaries | Hk boundaries |
| --- | --- | --- |
| LLaDA | `[-1.35, 0, 1.35]` | `[-2.26, 0, 2.26]` |
| SDAR | `[-1.50, 0, 1.49]` | `[-2.87, 0, 2.86]` |

### LoSA

LoSA splits attention into historical prefix and current-block parts. Prefix
output/LSE state is reused between denoising steps, selected query positions
refresh that state, and the two parts are combined with online softmax.

When `losa_active_topk >= block_length`, LoSA still executes the split
prefix/current calculation and refreshes every position. This is the exact
full-active control and should remain close to dense attention.

The current block uses an explicit short-block PyTorch attention calculation;
historical prefix attention, query-delta scoring, Adamas distance, KV writes,
and LLaDA MoE execution use the canonical Triton kernels.

## Evaluation defaults

`eval_instruct/eval.sh` is the source of truth for evaluation configuration.
The runtime also persists the complete family-specific snapshot as
`model.config.llada_sparse_config` or `model.config.sdar_sparse_config`.
These are independent configurations; a parameter omitted by the caller uses
the default for the detected model family.

| Setting | LLaDA | SDAR |
| --- | ---: | ---: |
| dtype | BF16 | FP16 |
| block length / steps | 32 / 32 | 32 / 32 |
| Query Sparse ratio | 0.7 | 0.5 |
| selection interval | 4 | 1 |
| query dense threshold | 4 | 0 |
| selection layer | 1 | 5 |
| Prefix chunk size | 1024 | 1024 |
| Prefix Sparse default | enabled | disabled |
| LoSA default | disabled | disabled |
| MoE patch | enabled | disabled |

HumanEval matrix:

```bash
CUDA_VISIBLE_DEVICES=3,4,5 bash eval_instruct/run_humaneval_matrix.sh
```

GSM8K matrix:

```bash
CUDA_VISIBLE_DEVICES=3,4,5 bash eval_instruct/run_gsm8k_matrix.sh
```

## Long-context benchmark

```bash
CUDA_VISIBLE_DEVICES=5 /home/ysy/anaconda3/envs/llada/bin/python \
  scripts/bench_long_context.py --model llada --mode prefix \
  --contexts 8192 16384 32768 --gen-length 64 --prefix-token-budget 1024

CUDA_VISIBLE_DEVICES=5 /home/ysy/anaconda3/envs/dream/bin/python \
  scripts/bench_long_context.py --model sdar --mode prefix \
  --contexts 8192 16384 32768 --gen-length 64 --prefix-token-budget 512
```

Matched steady measurements from the calibrated Prefix configurations:

| Model/config | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| LLaDA dense | 6.725s | 12.440s | 26.584s |
| LLaDA Prefix-1024 | 6.244s (1.08x) | 12.290s (1.01x) | 25.427s (1.05x) |
| SDAR dense | 6.456s | 12.489s | 35.621s |
| SDAR Prefix-512 | 6.228s (1.04x) | 12.209s (1.02x) | 34.669s (1.03x) |

Matched GPU 2 measurements for the v1.2-compatible Prefix-256 profile:

| Model/config | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| LLaDA dense | 9.033s | 14.839s | 28.676s |
| LLaDA Prefix-256 | 7.869s (1.15x) | 13.981s (1.06x) | 26.689s (1.07x) |
| SDAR dense | 6.361s | 12.719s | 35.763s |
| SDAR Prefix-256 | 6.408s (0.99x) | 12.559s (1.01x) | 34.735s (1.03x) |

Full HumanEval validation for the calibrated quality configurations:

- LLaDA: 74/164 official, 137/164 indentation-normalized, matching dense.
- SDAR: 129/164 official and 129/164 normalized; dense is 129/164 and 130/164.

Full HumanEval for the Prefix-256 speed profile:

- LLaDA: 71/164 official and 128/164 indentation-normalized.
- SDAR: 127/164 official and 129/164 indentation-normalized.

Latest SDAR prefill/Adamas optimization (GPU 2/3, generation length 64):

| Configuration | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| Dense, Triton prefill | 3.77s | 5.70s | 8.79s |
| Query+Prefix-256, Adamas chunk 1024 | — | — | 7.93s |

The original SDAR dense prefill baseline at 32K was 35.76s; the current dense
Triton path is approximately 4.1x faster.

For fair end-to-end ablation, use `--ablation`. Dense, Query-only, Prefix-only,
and Query+Prefix are run on the same loaded model. Every mode is first warmed
up with the same context and generation length as the measured run; the four
measured orders rotate, context order alternates, output length is fixed with
`eos_early_stop=False`, and the table reports four-run medians. This avoids
charging a new generation shape's compilation cost to dense.

```bash
scripts/bench_long_context.py --model llada --mode query_prefix \
  --contexts 8192 16384 32768 --gen-length 256 \
  --prefix-token-budget 256 --repeats 4 --ablation
```

Times and speedups relative to dense:

| Model / generation / mode | 8K | 16K | 32K |
| --- | ---: | ---: | ---: |
| LLaDA 256 dense | 2.354s | 3.669s | 8.481s |
| LLaDA 256 Query, forced | 3.013s (0.78x) | 3.757s (0.98x) | 7.795s (1.09x) |
| LLaDA 256 Prefix-256 | 2.320s (1.01x) | 3.410s (1.08x) | 6.565s (1.29x) |
| LLaDA 256 adaptive Query+Prefix-256 | 2.334s (1.01x) | 3.416s (1.07x) | 5.938s (1.43x) |
| LLaDA 768 dense | 5.405s | 6.484s | 12.902s |
| LLaDA 768 Query, forced | 5.459s (0.99x) | 7.131s (0.91x) | 11.573s (1.11x) |
| LLaDA 768 Prefix-256 | 5.193s (1.04x) | 5.738s (1.13x) | 8.142s (1.58x) |
| LLaDA 768 adaptive Query+Prefix-256 | 5.228s (1.03x) | 5.761s (1.13x) | 8.362s (1.55x) |
| SDAR-b32 256 dense | 8.288s | 9.213s | 13.783s |
| SDAR-b32 256 Query | 5.939s (1.40x) | 6.926s (1.33x) | 10.593s (1.30x) |
| SDAR-b32 256 Prefix-256 | 8.374s (0.99x) | 9.327s (0.99x) | 12.595s (1.09x) |
| SDAR-b32 256 Query+Prefix-256 | 5.967s (1.39x) | 6.903s (1.33x) | 10.164s (1.36x) |
| SDAR-b32 768 dense | 23.506s | 24.465s | 31.863s |
| SDAR-b32 768 Query | 16.519s (1.42x) | 17.632s (1.39x) | 22.365s (1.42x) |
| SDAR-b32 768 Prefix-256 | 23.738s (0.99x) | 24.721s (0.99x) | 28.179s (1.13x) |
| SDAR-b32 768 Query+Prefix-256 | 16.622s (1.41x) | 17.511s (1.40x) | 20.983s (1.52x) |

LLaDA now defaults to Adamas chunk 1024, and block-causal prefill stops each
query tile at its last visible KV block. Direct paired measurements give
`1.031x / 1.099x / 1.504x` for generation length 256 and
`1.049x / 1.149x / 1.591x` for generation length 768 at 8K/16K/32K. Output
checksums were stable across repeats and matched the former prefill kernel and
chunk-256 runs.

Prefix is the main LLaDA accelerator: it saves 1--8% at 8K/16K and 23--37%
at 32K. Forced LLaDA Query is only beneficial at 32K, so the default adaptive
policy skips it below 24K and makes Query+Prefix faster than dense at every
tested length. Query is the main SDAR accelerator, saving 23--30% by
itself; Prefix is neutral below 32K and adds a further 3--4% of dense latency
savings after Query at 32K. Thus the two selectors have a measurable
interaction and their standalone speedups should not be multiplied.

Adaptive LLaDA Query+Prefix reaches 74/164 official and 132/164
indentation-normalized on HumanEval; forced Query reached 79/164 and 130/164.
SDAR reaches 127/164 official and 129/164 normalized. SDAR-b4 remains slower
than b32 for this configuration.

The runtime stores block-causal structure as implicit metadata, caches the
fixed prompt KV once, and only refreshes the generated suffix for each new
block. Peak LLaDA memory is 33--37 GiB; the former quadratic-mask path used
about 64 GiB at 32K. The routed-MoE kernels use a 32-row tile because a 64-row
tile exceeds the GPU shared-memory limit.

## Supporting tools

- `scripts/bench_sparse_ops.py`: kernel-level sparse operation benchmark.
- `scripts/collect_adamas_hqhk.py`: collect real Hq/Hk distributions.
- `scripts/analyze_llada_layer_overlap.py`: LLaDA query-layer overlap.
- `scripts/analyze_sdar_layer_overlap.py`: SDAR query-layer overlap.
- `scripts/analyze_llada_query_losa_correlation.py`: Query/LoSA correlation.

Historical results are in [`RESULTS.md`](RESULTS.md). Adamas distribution,
lookup-table, and GQA findings are in [`docs/adamas.md`](docs/adamas.md).
