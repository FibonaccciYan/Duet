# Step-kernel progress, 2026-09-16

The full Sparse/LoSA/FOCUS objective remains active and incomplete. Tests ran
on n33 physical GPU5 (H800 PCIe, sm90), branch codex/integrate-all-20260916,
with public small-MoE and SDAR direct RMSNorm applied to dense and candidates.
Original implementations remain preserved; no commit/push.

## FOCUS v5 cache ownership

FOCUS v5 avoids constructing a DynamicCache from an immutable prefix when an
existing block/append owner is available. Otherwise it creates per-layer cache
containers aliasing prefix tensors. DynamicLayer append still writes separate
storage; source prefix is unchanged. Three CPU cache isolation tests pass.

Real gov_report sample115, 8K input, 64 generated-token budget, two alternated
runs:

| family | v4 E2E | v5 E2E | v4 full forward+transfer | v5 |
|---|---:|---:|---:|---:|
| LLaDA Q | .935-.944 s | .894 s | 24.84-24.87 ms | 23.32-23.35 ms |
| SDAR low-confidence dynamic | 2.989-2.990 s | 2.646-2.647 s | 36.93-36.94 ms | 31.75 ms |

Token and ordinary traces match. Real16K sample59 audits match all retained
positions: LLaDA39 records and SDAR64 records. Sparse/FOCUS stopping logic was
not modified.

## Exact fused RoPE experiment

`src/kernels/versioned/rope_exact.py` fuses Q/K RoPE and explicitly rounds the
multiplications to the input dtype before the add. Its microbenchmark is about
2.7-4.6x faster than the eager reference over 24 FP16/BF16/FP32 cases, and all
outputs are bitwise equal. `rope_runtime.py` installs per-model clones and
leaves original module globals untouched; installation/restore unit tests pass.

Real SDAR8K with dense and Sparse both using the kernel: output tokens and step
counts match prior runs. Full-step medians were dense20.25-20.37ms and
Sparse21.90-22.03ms, so Sparse remains slower.

Real SDAR16K sample59 two alternated runs:

| mode | normal full-step | exact-RoPE full-step | E2E normal | E2E exact-RoPE |
|---|---:|---:|---:|---:|
| dense v3 | 24.24 ms | 26.23 ms | 2.025-2.030 s | 2.068-2.075 s |
| sparse v3 | 22.75 ms | 22.14 ms | 2.393-2.397 s | 2.380-2.382 s |

Tokens and step counts match for every pair. The kernel is therefore not a
universal public optimization: it helps Sparse slightly but regresses dense by
about 8%. It remains opt-in for the experiment and is not enabled by default.
The likely reason is downstream layout/cache behavior after the fused outputs,
not RoPE arithmetic itself; the next investigation should inspect profiler
kernel launches and tensor strides rather than claim a general gain.

A same-input operator test for native FlashAttention in FOCUS confirms exact
Q/K/cache inputs and output differences only within FP rounding. LLaDA native
FlashAttention changes model trajectories on a real16K case and remains
experimental.

## Current bottleneck evidence

FOCUS SDAR second-denoise profile: GEMM about64% self-CUDA, FlashAttention
about15%, copy about4%. Sparse SDAR full-loop timing (including token scatter)
remains approximately22.9ms versus dense21.3ms at8K; Sparse performs61 versus
28 denoise iterations. Cache views alone do not fix this.

## Unmet acceptance cells

No method has yet demonstrated >=1.3x E2E over matched dense across both models,
all 8K/16K/32K inputs. Full-step instrumentation is not yet unified for all
methods, and the configured 32K window requires an explicit prompt+generation
policy. Quality is not accepted from short diagnostic ROUGE or token equality
alone. Further work must target decoder GEMM/MLP launch overhead, real
multi-sample quality, and valid 32K measurements without changing algorithm
selection or stopping rules.

GPU5 holder restored and verified after the final run: PID2034883,
72992MiB reserved, 10% compute duty cycle.
