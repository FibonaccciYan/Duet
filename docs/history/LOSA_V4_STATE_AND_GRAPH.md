# LoSA v4 fused state and graph investigation

Goal remains incomplete: three methods, both models, all requested lengths,
full-step speedups and >=1.3x end-to-end relative to matched dense are not met.
Original implementation directories preserved. No commit/push.

## Current fused-state implementation

`src/kernels/versioned/losa_state.py` merges active-row prefix-state writes,
prefix/block softmax merging, and previous-Q update in one Triton launch.
Original group_mean selection, top5, page union and budgets remain external.
`LoSAV4Runtime(..., losa_fused_state=True)` forwards a per-call flag through
v4 generation/context/attention to this kernel; default false. No global toggle.

An early approximate exp/log implementation changed SDAR's sparse step count
from38 to40 despite identical final tokens. It was replaced with libdevice
exp/log1p and disabled FP fusion. Do not use the early approximate kernel's
measurements as evidence for the final implementation.

18 finite cases covering FP16/BF16/FP32,16/32heads and1/5/32active rows have
bitwise-equal output and LSE, plus exact state writes. Infinity conventions
were also tested. These tests do not establish equivalence for every possible
input or hardware/compiler combination.

## Matched v4 eager comparison, final arithmetic

Real gov_report sample115,8192 input tokens,64 generated-token budget,
SDAR low_confidence_dynamic threshold .95. Two alternating measured repeats:

| Implementation | End-to-end seconds | Sparse forward median ms |
|---|---:|---:|
| v4 unfused | 2.4066 / 2.4082 | 42.3219 / 42.3805 |
| v4 fused | 2.3078 / 2.3064 | 39.7137 / 39.7310 |

Both have38sparse+2initialization+2initial-transfer steps; final tokens equal.
This is ~6.6% forward speedup and ~4.3% end-to-end speedup within v4,
not >=1.3x versus dense. Prior matched dense was1.138s for this sample/budget.

Real sample59,16384input tokens,64generation budget: actual active rows and
union-page traces matched exactly for1188SDAR and480LLaDA attention calls.
Final tokens and step counts matched. These trace-enabled timings are diagnostic.
After replacing the benchmark global switch with the public per-call parameter,
an additional SDAR8K audit matched1404attention calls and final tokens/counts.

## Graph results (separate experiment)

SDAR8K graph-only v4: eager-vs-replay logits and persistent states passed
per-step bitwise checks on the tested sample. With verification disabled,
sparse forward median19.695-19.699ms; forward-through-transfer median20.13ms.
Total2.056-2.119s includes0.498-0.542s graph build for two blocks.
Still slower than dense. Full-step boundary remains incomplete (scatter/outer
control not fully included in forward-through-transfer measurements).

LLaDA graph + early approximate fusion lowered forward time but increased
end-to-end to1.331s, with0.543s capture cost. This is neither final fused-kernel
validation nor a successful E2E result. Graph remains benchmark-only/optional.
Tensor Core attention + flash initialization did not improve graph step time.

## Next work

Reduce capture setup/amortization costs without excluding them from E2E.
Compare against the best equally optimized dense path. Extend final fused
arithmetic to further samples and the explicit32K-window protocol. Complete
full-step instrumentation and general task-quality checks; continue Sparse
and FOCUS work rather than treating LoSA-only gains as goal completion.

GPU5 holder restored and verified PID1939411,72992MiB,10% duty cycle.
