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
| Prefix chunk size | 256 | 1024 |
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

The SDAR Query+Prefix-256 configuration scored 129/164 official and 131/164
indentation-normalized on HumanEval. The original SDAR dense prefill baseline
at 32K was 35.76s; the current dense Triton path is approximately 4.1x faster.

For end-to-end generation, the benchmark uses a fixed output length
(`eos_early_stop=False`) so sparse and dense runs perform the same amount of
work. Representative SDAR 32K runs on GPU 3 are:

| Configuration | gen=256 | gen=768 |
| --- | ---: | ---: |
| Dense | 19.04s | 46.75s |
| Query+Prefix-256 (Adamas chunk 1024) | 15.57s (1.22x) | 36.37s (1.29x) |

These timings include the complete prefill/decode path; checksums and actual
generated-token counts are emitted by `scripts/bench_long_context.py`.

Latest LLaDA compact-prefill results (GPU 4, 32K total context):

| Configuration | gen=256 | gen=768 |
| --- | ---: | ---: |
| Dense | 20.38s | 42.89s |
| Query+Prefix-256 | 11.58s (1.76x) | 18.85s (2.28x) |

The runtime stores block-causal structure as implicit metadata, caches the
fixed prompt KV once, and only refreshes the generated suffix for each new
block. Peak memory is 34--36 GiB; the former quadratic-mask path used about
64 GiB at 32K and could not run the 32K/gen=768 dense case on an 80 GiB GPU.

## Supporting tools

- `scripts/bench_sparse_ops.py`: kernel-level sparse operation benchmark.
- `scripts/collect_adamas_hqhk.py`: collect real Hq/Hk distributions.
- `scripts/analyze_llada_layer_overlap.py`: LLaDA query-layer overlap.
- `scripts/analyze_sdar_layer_overlap.py`: SDAR query-layer overlap.
- `scripts/analyze_llada_query_losa_correlation.py`: Query/LoSA correlation.

Historical results are in [`RESULTS.md`](RESULTS.md). Adamas distribution,
lookup-table, and GQA findings are in [`docs/adamas.md`](docs/adamas.md).
