# Sparse/Dense v3: public cache views and full SDAR iteration timing

Goal remains active and unmet. Original implementation directories preserved.
New sparse_v3 and dense_v3 use the same public small-MoE, SDAR RMSNorm, and
prefix cache-view optimization. No commit/push.

## Implementation

PrefixViewCache.from_legacy_cache creates per-layer containers referencing
immutable prefix tensors. DynamicLayer.update still concatenates into new
storage, preserving source-prefix ownership. Per-model cloned generation and
cached-forward namespaces replace DynamicCache locally; original helpers and
other model objects remain unchanged. Three CPU ownership/isolation/empty-cache
tests pass. Registered sparse_v3 patch_method and dense_v3 patch/runtime paths.

## Real8K sample115,64generated-token budget, two alternating repetitions

| Family / mode | E2E seconds | Approx forward+transfer ms | Timed forward count |
|---|---:|---:|---:|
| LLaDA dense_v3 | .761 | 26.4 | 17 |
| LLaDA sparse_v3 | .875 | 20.2 | 27 |
| SDAR dense_v3 | 1.141-1.143 | 20.79-20.82 | 28 |
| SDAR sparse_v3 | 1.961-1.964 | 22.40-22.46 | 61 |

LLaDA timed boundary is cached-forward through transfer return, excluding
block-start refresh. Counts above are timed calls, not necessarily total model
calls. LLaDA sparse has ~1.31x speedup within that boundary but slower E2E.
SDAR's old boundary excludes final token scatter; use the new full interval
below when discussing full steps.

V2 and v3 tokens/counts match within all four mode/family comparisons, both
repeats. On real16K sample59, query positions and prefix indices match:
LLaDA35query-selection records /2prefix selections; SDAR34/2;tokens equal.

The cache change helps LLaDA dense more than Sparse: dense_v2 roughly.787s
versus v3.761s;Sparse_v2 roughly.884s versus v3.875s. SDAR gain is tiny.
Do not credit common cache savings as sparse-only acceleration.

## Full SDAR loop iteration instrumentation

A benchmark-only AST clone inserts CUDA events and host clocks immediately
before mask/termination checks and after token scatter. Cache finalization is
recorded separately. No original source or algorithm operations are changed.
The transformer fails if the expected loop/scatter structure changes. Two CPU
tests verify post-scatter observation and unchanged behavior.

Run with paired_dense_sparse_steps.py --family sdar --full-step-timing.
On the8K sample:
- Dense_v3:28denoising iterations,median CUDA21.302ms /host21.303ms.
- Sparse_v3:61iterations,median CUDA22.904ms /host22.905ms.
- Final cache persistence totals~40.95ms for each method across two blocks.
Tokens/counts match the preceding uninstrumented runs for each mode.

This covers the denoising-loop iteration including scatter and host dispatch;
prefill and block setup are separate E2E work. Full-step instrumentation for
LLaDA, LoSA and FOCUS is still incomplete.

Next work: reduce SDAR/Sparse per-step decoder execution cost; address setup
and graph-amortization costs while applying comparable public improvements to
dense. Complete multi-sample quality and the explicit32K-window protocol.
The >=1.3x dense E2E target across three methods/both models/all lengths remains
unachieved; no stopping/selection thresholds were changed to improve timing.

GPU5 reservation verified running PID2000196,72992MiB,10%duty cycle.
