# LoSA v6 implementation and validation

Date: 2026-09-17. Work started 16:39:43 Asia/Shanghai; allotted deadline
18:09:43. Host n33, physical GPU5, H800 PCIe sm90. Repository:
/data0/gs/SparseDLM_LLaDA_SDAR_integration_20260916, branch
codex/integrate-all-20260916. No commit or push.

## Delivery status

Priority 1 is implemented and measured: per-layer, per-block scratch workspace,
original PyTorch top-k/sort with output buffers, reusable query/page/attention
buffers, and version-local full-forward CUDA graph replay. One actual sparse
step primes each block before capture. Capture does not advance algorithm state:
query/prefix output/prefix LSE are snapshotted and restored. Replay does not
reuse states across blocks. All capture costs remain in measured generation time.

Priority 2 is implemented and numerically validated, but NOT accepted as an
end-to-end performance improvement. The selected shared pipeline computes
unscaled QK scores while sharing K loads across queries and preserves the v5
FP32 reduction/FMA order in online softmax and V accumulation. It consumes the
same physical-KV-head union and page order. Current preferred value_share=1
does NOT share V loads across queries. Multi-query V-sharing implementations
are retained as experiments but were slower. Thus the full intended
shared-K-and-V performance objective remains incomplete.

LoSA v5/v4 are preserved. Existing v4 source hashes match the pre-work snapshot.
Runtime registration adds losa_v6 without removing other registrations.
Defaults: graph_replay=True, shared_kv=False, group_mean, fused state enabled.

## Performance

Exact NarrativeQA dataset sample0, generation256, prompt8192/16384/32512.
Same model instance/weights, one full warmup per variant, two measured runs;
table is median of run medians. No quality/selector audit in timing runs.
All listed candidate output tokens and ordinary traces match v5.
Sparse-forward medians are NOT complete denoising loop timings.
Forward-through-transfer arrays are also included in delivery_summary.json.
Graph construction and priming are included in E2E and raw per-call arrays;
medians naturally do not represent the rare capture-heavy call.

|Model|Context label|v5 sparse forward ms|v6 sparse forward ms|Forward speedup|v5 E2E s|v6 E2E s|E2E speedup|
|---|---:|---:|---:|---:|---:|---:|---:|
|LLaDA2.1 Q|8K|29.62|13.12|2.26x|1.428|1.377|1.037x|
|LLaDA2.1 Q|16K|28.86|13.22|2.18x|1.919|1.907|1.007x|
|LLaDA2.1 Q|32K|29.46|13.17|2.24x|3.922|3.706|1.058x|
|SDAR|8K|39.61|17.66|2.24x|2.189|1.922|1.139x|
|SDAR|16K|39.37|18.07|2.18x|3.789|3.460|1.095x|
|SDAR|32K|39.70|18.40|2.16x|7.577|7.270|1.042x|

These are v5-relative, not dense-relative speedups. Two repeats on one sample
are diagnostic evidence, not broad performance acceptance. Small differences
such as 0.7% need more repeats. Prefill and state initialization were deliberately
not changed. Capture remains roughly 0.17-0.24s in the listed runs.

## Correctness and memory checks

- Workspace CPU tests: 3 passed; shape/dtype separation, reuse, tie behavior.
- Stateful workspace tests: FP16/BF16, 257 and8192 prefix tokens, five evolving
  steps; output, LSE, active rows, page union, query/prefix state bitwise equal.
- Shared ordered kernel: 96/96 cases bitwise output/LSE equal.
- Selected shared pipeline: 72/72 cases bitwise output/LSE equal, both model
  head configurations, lengths257/8192/16384/32512, random/scaled/zero queries.
- Real NarrativeQA sample0 at32K/gen256: both models' actual active rows,
  union page tables, tokens and ordinary traces equal for workspace/shared.
- Real sample1 at8K/gen64: same selector/token/trace audits pass both models.
- Graph replay at16K/gen256: same-state logits, persistent query/output/LSE,
  actual selected rows/pages and current KV tails bitwise equal on both models.
- Combined graph/shared path at sample1 8K/gen64: same state checks pass.
- Public load_runtime("losa_v6", family="sdar") smoke: tokens/trace equal.
- Compute Sanitizer: 0 errors for partial-page shared-pipeline cases.

Scope limits: no full dataset quality evaluation; memcheck did not cover every
experimental kernel or every graph configuration. Concurrent use of one model
or layer workspace is unsupported. Shared pipeline specializes D128/page16;
other runtime layouts fall back to the original paged attention.

## Shared-kernel experiments

Tensor Core candidates changed selections and, on LLaDA, tokens: rejected.
SIMD and CUDA shared-memory candidates could be made bitwise equal but were
slower. Shared QK pipeline gained about1.05x FP16 /1.11x BF16 at one60-page
microbenchmark, but the extra launch/intermediate buffer offset this in real
steps. Graph+shared did not outperform graph-only reliably, so shared remains
off by default. Experimental modules are not imported by the default runtime.

## Usage

```python
from src.runtime import load_runtime
runtime = load_runtime("losa_v6", family="sdar")
# graph_replay=True; shared_kv=False by default
# shared_kv=True is a validated experimental ablation, not a speed recommendation.
model, tokenizer = runtime.load()
output = runtime.generate(inputs, gen_length=256, block_length=32, steps=32,
                          threshold=0.95,
                          remasking_strategy="low_confidence_dynamic",
                          eos_early_stop=False)
```

```bash
bash scripts/performance/run_reserved_gpu5.sh \
  /data0/ysy/anaconda3/envs/sparse/bin/python -B \
  scripts/performance/bench_losa_v6_real.py \
  --family sdar --length 16384 --gen 256 \
  --modes v5 v6_graph v6_both --repeats 2 \
  --output results/losa_v6_20260917/reproduce.json
```

Use --audit with eager variants for actual rows/pages, and --verify-graph
with graph variants for same-state replay checks. Audit timings are invalid
for speedup claims. Primary results are JSON files in this directory;
manifest.json records source hashes and delivery_summary.json aggregates
timings and verification status.

GPU5 holder restored after final test: PID4079188, reserved72992MiB,
reserved/running confirmed. No benchmark process intentionally left running.
