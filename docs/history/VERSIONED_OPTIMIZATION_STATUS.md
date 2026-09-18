# Versioned optimization: incomplete acceptance

This is an engineering status record, not a successful benchmark claim.

## Required scope

- Preserve original `src/sparse`, `src/losa`, `src/losa_v2`, `src/focus`,
  `src/focus_v2`, and `src/dense`.
- Target n33 physical GPU 5; restore its existing reservation after every job.
- LoSA selection uses `group_mean`; retain active-row, page-budget, union,
  and state-update rules.
- LLaDA 2.1 Q profile: threshold 0.7, editing threshold 0.5.
- SDAR experiments so far use `low_confidence_dynamic`, threshold 0.95;
  do not label these as static low-confidence results.
- All three methods must show useful step speedups and at least 1.3x
  end-to-end speedup over matched dense at real 8K/16K/32K contexts on both models.
- Apply public MoE and SDAR RMSNorm improvements to dense too.

## Implemented versioned paths

| Version | Changes | Acceptance limits |
| --- | --- | --- |
| `dense_v2` | Public small-token MoE and SDAR direct RMSNorm launch | Baseline backend must match the compared candidate |
| `sparse_v2` | Public kernels; short-candidate exact Raw L1 reduction | No broad speedup established beyond the original algorithm |
| `losa_v3` | Strided KV attention, valid page tail skipping, grouped FP32 initialization, block KV tail reuse | Full performance matrix incomplete |
| `focus_v3` | GQA views, retained position/mask reuse, cache-only passes, append-only sequential prefill | Much slower than matched dense in tested SDAR 8K |

LoSA v3 also fixes two inherited driver defects: stale logits at the first
step of subsequent blocks and dropped `store_kv` forwarding. These are
correctness fixes, not changes to QUEST or sparsity policy. Old v2 remains
unchanged and is not a valid sole oracle for multi-block generation.

## Evidence boundaries

- Synthetic kernel speedups do not establish model-step or end-to-end speedups.
- The existing real-data runs primarily use gov_report sample 115. They are
  not a multi-sample acceptance suite.
- Prompt lengths are exact tokenized input lengths. Generation is additional.
  A 32768-token prompt plus generation extends beyond the configured 32768
  window and needs an explicit final evaluation policy; do not silently call
  this an in-window 32K result.
- Generation budgets vary across diagnostic files (32, 64, 128). Ratios require
  equal budgets and sample identity.
- LLaDA step timing covers cached forward through transfer return and excludes
  block-start refresh. SDAR timing stops at selection return, before the final
  token scatter. Neither is yet the complete requested final step protocol.
- Earlier Sparse traces were empty. Only runs with `selector_audit=true`
  record actual query positions and prefix indices.
- FOCUS ordinary traces contain counts, not full retained-position vectors.
  Equal counts and tokens do not prove every retained-position choice equal.
- Profile, selector-audit, graph-verification and synchronized diagnostic
  timings are not final benchmark results.
- First-use compilation and graph capture must be reported separately from
  warm execution. Hundreds of graph captures have cost tens of seconds.

## Experimental or rejected paths

- FlashAttention LoSA initialization: real page choices and tokens diverged.
- Maskless FOCUS: LLaDA tokens diverged; do not enable by default.
- Large-chunk FOCUS prefill: KV/tokens diverged. The cause is not proven to be
  an algorithmic visibility change; floating-point batching is a candidate.
- Fused SDAR QKV: not bitwise equal in the tested kernel gate.
- New handwritten RMSNorm: failed a bitwise gate. Direct launching of the
  installed original kernel supersedes it.
- MLP graphs: dynamic routing is supported, but gains are shape-dependent and
  capture overhead is material; not default.
- Full decoder graphs: same-start layer verification passes, but unsynchronized
  generation has changed steps/tokens. Explicit stream dependencies alone did
  not fix it. Synchronization is a diagnostic protection, not root-cause closure.
  Never accept the divergent path's timing as a speedup.

## Reproduction and reservations

Use `/data0/ysy/anaconda3/envs/sparse/bin/python` in the integration worktree.
Run GPU work through:

```bash
bash scripts/performance/run_reserved_gpu5.sh \
  /data0/ysy/anaconda3/envs/sparse/bin/python -B \
  scripts/performance/paired_dense_sparse_steps.py \
  --family sdar --contexts 8192 --gen-length 64 --repeats 3 \
  --output results/paired_sdar.json
```

The wrapper releases only the configured user holder, runs on physical GPU 5,
and restores the saved reservation. Check its exit status, not just log text.
The state directory is `/home/gs/.cache/codex-gpu-reserver-n33`.

Source changes are uncommitted. Newer runs include source hashes in sidecar
manifests; older runs without those hashes cannot prove the exact current
source produced their measurements.

## Remaining work

1. Resolve graph correctness before considering warm graph speedups.
2. Record full retained-position and transfer-decision trajectories.
3. Normalize evaluation boundaries and context-window policy.
4. Run multiple samples, both models, all lengths, full steps and matched dense.
5. Verify quality and all algorithm invariants without relying only on text equality.
6. Demonstrate the requested 1.3x matrix or explicitly report unachieved cells.

The objective is **not complete**.
