# FOCUS v5 progress: acceptance incomplete

New src/focus_v5 preserves v4 and the original implementation directories.
Registered FocusV5Runtime via load_runtime('focus_v5',...). No commit/push.

Default change: choose an existing block/append cache before constructing any
DynamicCache. The previous code copied the entire prefix via from_legacy_cache
and immediately discarded it when block_cache was supplied. For passes without
an existing owner, v5 uses fresh cache containers referencing immutable prefix
KV; subsequent DynamicLayer concatenation leaves the source prefix unchanged.
Two CPU ownership/update tests pass. Original selection/threshold/stopping rules
remain unchanged, including existing SDAR empty-mask iterations.

## Unprofiled real8K, sample115,64-token generation budget

Public MoE/norm identical within each paired run. Two alternating repetitions.

| Model | v4 E2E seconds | v5 default E2E seconds | v4 forward+transfer ms | v5 forward+transfer ms |
|---|---|---|---|---|
| LLaDA2.1 Q | .9346 / .9441 | .8943 / .8942 | 24.840 / 24.865 | 23.346 / 23.324 |
| SDAR low_confidence_dynamic | 2.9905 / 2.9887 | 2.6468 / 2.6459 | 36.933 / 36.938 | 31.749 / 31.751 |

Token IDs and ordinary trace entries equal in both repetitions;25 LLaDA and64
SDAR steps. These ordinary traces are not full retained-position audits.

Real16K sample59,64generation,actual retained-position audit: default v5 matches
v4 tokens and all39LLaDA/64SDAR retained-position vectors and trace entries.
Audit-enabled timings are diagnostics, not final performance measurements.

## Optional native GQA FlashAttention

FocusV5Runtime(attention_backend='flash') is optional; default remains 'sdpa'.
The private attention path requires the explicit all-visible contract and uses
causal=False with original attention scaling, Q/K/V and physical GQA heads.
SDAR dense already uses this native FlashAttention path for decoding.

SDAR8K unprofiled: v4 2.9693s /36.673ms forward+transfer; v5 flash2.2731s /
26.000ms. Tokens and ordinary traces equal. Still slower than the previous
matched dense1.138s; this is not >=1.3x dense acceptance.
SDAR16K:64actual retained-position vectors and token IDs match paired v4.
LLaDA16K:tokens and retained-position trajectories differ. Short-output ROUGE
is diagnostic only; native flash is not quality-approved.

12 same-input operator cases (FP16/BF16,both model head layouts,1/5/32rows,
8192prefix tokens) preserve selector Q/K inputs and cache tensors exactly.
Attention maxabs<=.000977 FP16,<=.007813 BF16 on these synthetic inputs.
These numerical tests do not establish model-level quality.

## Measurement repair and next bottleneck

Benchmark supports focus-reference4,focus-version5 and backend flags.
Batched prefill now gets its own CUDA-event phase; it was previously omitted
from phase timings although included in E2E. Profiler defaults to the second
denoise forward so first-step cache initialization is not mistaken for steady
work. Profiled runs are excluded from speed acceptance.

SDAR8K v5-flash second-step profile: mm9.478ms /64.43% selfCUDA,
FlashAttention2.154ms /14.65%,copy.538ms /3.66%. These are operator-time
fractions, not wall-time fractions. Next work should reduce decoder launch
and linear/MLP execution costs with identical public optimizations for dense.

The full Sparse/LoSA/FOCUS two-model8K/16K/32K target remains unmet. Full-step
boundary still needs completion; current metric stops after transfer function.
No32K input-window policy or broad task-quality acceptance has been established.

Latest holder verified restored: physicalGPU5,PID1976024,72992MiB,10%duty.
