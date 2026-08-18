# LLaDA Sparse-KV 0.5 Throughput Benchmark

## Config

- model_path: `/data0/ysy/models/LLaDA2.1-mini`
- gen_length: `4`
- block_length: `4`
- steps: `4`
- sparse mode: `kv`
- sparse ratio: `0.5`
- dense_fallback_mask_count: `0`
- repeats: `1`
- warmup: `0`
- CUDA_VISIBLE_DEVICES: `0`

## Results

| mode | median time s | mean time s | requested tok/s median | requested tok/s mean | actual output tokens |
|---|---:|---:|---:|---:|---|
| vanilla | 1.3897 | 1.3897 | 2.88 | 2.88 | [4] |
| sparse_kv | 1.1197 | 1.1197 | 3.57 | 3.57 | [4] |

Sparse-KV median-time speedup vs vanilla: `1.2411x`

## Sample Outputs

### vanilla

The dust of the

### sparse_kv

The dust of the
