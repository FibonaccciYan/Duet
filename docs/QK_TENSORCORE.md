# QK Tensor Core prefix selector

`prefix_selector="raw_l1"` remains the Sparse source default. `qk` retains the
FP32 pairwise-tree oracle. `qk_tc` is the technical paper variant: FP16/BF16
Tensor Core dot products with FP32 accumulation, without extra normalization,
scaling, group-mean query, TF32, quantization, or approximate prefiltering.

`src/optimized/sparse/qk_tc_prefix.py` groups work by physical KV head. Each
16-query / 64-key tile computes negative dot scores over D=128 with `tl.dot`,
emits up to `local_budget` candidates per query plus token-score minima, and
feeds the existing lexicographic union, fill, strict trim, and sorted-index
path.

The optimized path requires CUDA sm80+, matching FP16/BF16 Q/K, D=128,
candidate length at most 32768, and local budget at most 8. Other inputs fall
back to the QK reference, never Raw L1. Reference dispatch of `qk_tc` runs the
FP32 QK oracle; stats report the actual execution backend and score definition.

Tensor Core accumulation order can differ from the tree oracle and near-boundary
rounding may change ranked ties. This is an explicit numerical variant, not a
bitwise-equivalence claim.

All three formal Sparse benchmark launchers select `qk_tc` and require
`prefix_strict_budget=true`. This launcher-level enforcement does not change
source-level defaults or claim reproduction of earlier soft-budget tables.
