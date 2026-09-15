# Query-resolved prefix attention-mass visualization

This experiment captures full-attention mass from a controlled LLaDA block-diffusion
trajectory and renders one query-row × prefix-column matrix for every layer and
denoising step. It is intended for paper motivation figures and visual inspection;
it does not run a sparse inference method.

## Directory layout

```text
scripts/attention_mass/
├── collect_attention_mass.py   # Run eager dense LLaDA and reduce attention to NPZ
├── plot_attention_mass.py      # Render individual PNGs and contact sheets
└── README.md
```

Generated data and figures are intentionally kept outside the repository. On the
A800 host, the convention used by the recent runs is:

```text
/root/gs/attention_mass_<dataset>_<context>/
├── data/<task>.jsonl
├── run/manifest.json
├── run/sample_XX_<task>.npz
└── figures_<cmap>/<task>/layer_XX/step_XX.png
```

## Input records

Each `<task>.jsonl` record must contain:

```json
{
  "idx": 0,
  "prompt": "Question: ...",
  "answer": "..."
}
```

A `length` field is used as a cheap prefilter when present. The collector then
tokenizes candidate prompts with the model tokenizer and selects the record whose
actual tokenized length is closest to `--prompt_tokens` while still being at least
that long. Longer prompts are middle-truncated to the exact requested context.

## Collect

Example for the 64-token, token-level setting (`page_size=1`):

```bash
cd /root/gs/SparseDLM_LLaDA_SDAR
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. \
/root/miniforge3/envs/sparse/bin/python scripts/attention_mass/collect_attention_mass.py \
  --data_dir /root/gs/attention_mass_gsm8k_64/data \
  --tasks gsm8k \
  --prompt_tokens 64 \
  --page_size 1 \
  --output_dir /root/gs/attention_mass_gsm8k_64/run
```

Use `--page_size 16` to aggregate every 16 prefix KV tokens into one column.
Use `--page_size 1` when no KV aggregation is desired.

The NPZ matrix shape is:

```text
[layers, steps, block_queries, prefix_tokens / page_size]
```

For the default block size and a 64-token token-level run this is:

```text
[20, 32, 32, 64]
```

The visualization uses `threshold=1.1` by default so one token is revealed per
iteration. This is deliberately controlled for a complete step axis; it is not a
normal-quality decoding configuration.

## Plot

```bash
cd /root/gs/SparseDLM_LLaDA_SDAR
/root/miniforge3/envs/sparse/bin/python scripts/attention_mass/plot_attention_mass.py \
  --base_dir /root/gs/attention_mass_gsm8k_64 \
  --parts run \
  --cmap viridis \
  --scale 4
```

`--parts` accepts one or more run directories under `--base_dir`, such as
`full_a full_b` for a sharded LongBench collection. The renderer writes:

- `layer_XX/step_XX.png`: one matrix per layer and step
- `contact_sheet_layers_by_steps.png`: layers by steps overview

Color normalization is global within each task across all layers and steps.
The default colormap is `viridis`; `magma` remains available through `--cmap magma`.
