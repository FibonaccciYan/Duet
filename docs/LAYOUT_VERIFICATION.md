# Layout migration verification

Date: 2026-09-18. Branch: `codex/integrate-all-20260916`.
Base commit: `fc052c5325181045e436519a06b55a36739b505a`.
The migration starts from the frozen n33 checkout including its uncommitted
source changes, not just from the base commit.

## Location and isolation

- Local worktree: `/Users/lier/codes/SparseDLM_LLaDA_SDAR_integration`.
- Original n33 checkout: `/data0/gs/SparseDLM_LLaDA_SDAR_integration_20260916`.
- Isolated validation checkout: `/data0/gs/integration_layout_final_20260918`.
- Validation interpreter: `/data0/ysy/anaconda3/envs/sparse/bin/python`.
- GPU validation used physical GPU0. No holder was started on GPU0; it was free
  after testing (4 MiB driver overhead, no compute processes).
- The original checkout and its GPU5 quality evaluation were not replaced or
  interrupted. No commit, push, or staging operation was performed.

## Checks performed

| Check | Result |
|---|---|
| Python syntax across src/scripts/tests/eval_instruct | 274 files parsed |
| Existing source AST after normalizing mapped names/imports and docstrings | 96 files matched; no algorithm-statement differences |
| CPU regression suite | 155 passed, 22 skipped |
| GPU regression suite on H800 | 174 passed, 3 skipped |
| Maintained CLI help entry points | 10 passed |
| Shell syntax | passed |
| Git whitespace check | passed |
| Real-input old/new optimized runtime generation | 12 pairs; token outputs identical |
| Real-input old/new reference runtime generation | 6 pairs; token outputs identical |

The 12 optimized pairs cover dense/Sparse/LoSA/FOCUS on LLaDA Q, LLaDA S,
and SDAR low_confidence_dynamic. The six reference pairs cover
Sparse/LoSA/FOCUS on LLaDA Q and SDAR. Each uses NarrativeQA80 sample0's stored
8192 input IDs, generation budget64, block32 and seed42. Tests run the actual
public APIs in separate processes. Some original LLaDA dense/Sparse public APIs
return EOS-trimmed outputs despite disabled early stopping; matching that
existing behavior is intentional, not silently replaced by benchmark semantics.

For all 18 pairs, prompt hashes, generation options and returned token IDs match.
All four optimized methods' available traces match before/after relocation.
Tensor-returning Sparse APIs do not supply a generation trace; `null` is not
evidence of an audited selector trace.

## Residual trace limitation

SDAR reference LoSA returns identical tokens but its page-union trace differs
between separate old/new processes. Follow-up controls also observed this
variation within the unchanged old checkout and within the new checkout:
641 of828 trace records differed in each repeated-process comparison, while
tokens stayed identical. This is evidence that repeat instability exists in
the reference implementation, not proof of the underlying numerical cause.
Do not claim bitwise reference LoSA trace equivalence. The structure migration
does not attempt to change its algorithm or numerical kernels to hide this.

## Scope

This is relocation/import/entry-point verification, not a new full NarrativeQA
benchmark or quality evaluation. 16K/32K and full80-sample performance matrices
were not rerun during reorganization. Prior measurements retain their historical
version labels under `docs/history/` and in external artifacts.

Detailed logs, token outputs, migration mapping and repeat-control comparisons
are retained locally at:

`/Users/lier/codes/d2Cache/artifacts/integration_layout_validation_20260918/`

The isolated remote `verification/` directory retains the per-process logs.
