# First-block dense attention-mass experiment

The experiment uses ten real 8K-token contexts and up to the first eight complete
generation blocks per requested task. At each block's initial all-mask state,
dense prefix attention averaged
over all block queries ranks the prefix KV tokens independently for every layer.
The top 128, 256, and 512 KV sets remain fixed for the rest of that block.

For each later step, the reported value is the attention mass on that fixed set,
averaged over all block queries and then over layers. Attention is normalized
within the prefix, so every curve is in `[0, 1]`.

## Context selection

The collector reads `<data_dir>/<task>.jsonl`, deterministically chooses ten
distinct real records whose tokenized prompt lengths are closest to but not
shorter than 8192, and middle-truncates each to exactly 8192 tokens.

## Collect

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python scripts/attention_mass/collect_attention_mass.py \
  --data_dir /path/to/longbench \
  --tasks narrativeqa \
  --output_dir /path/to/attention_mass_8k/run \
  --samples_per_task 10 --max_blocks 8 \
  --page_size 1
```

The collector uses the current SparseDLM transfer policy (`threshold=0.5`,
`editing_threshold=0.0`, `num_to_transfer=1`) with EOS early stopping enabled.
Each sample therefore contributes however many complete blocks occur before EOS,
up to eight.
One forward may therefore resolve multiple tokens. The raw NPZ shape is
`[layers, all observed forwards, 32 queries, padded prefix tokens]`.

## Plot

```bash
python scripts/attention_mass/plot_attention_mass.py \
  --base_dir /path/to/attention_mass_8k \
  --parts run \
  --budgets 128 256 512
```

Outputs are `attention_mass_by_budget.png` and a compact NPZ containing both
per-layer curves and their mean. KV indices are selected only in the initial all-mask state; they
are never recalculated at later steps. The x-axis is the number of resolved
tokens on entry to each forward, not the denoising-step index. Each block is
expanded over the tokens resolved by its forward. Blocks are averaged within
each sample first, then the ten sample curves are averaged with equal weight.
For example, a forward that moves the block from 0 to 18 resolved tokens
contributes its measured attention value to token progress 0--17. This gives every
token position one value from every complete block without interpolating
unobserved attention states. Shading is one standard deviation across samples. A final 32-token
point is not synthesized because there is no attention forward after completion.
