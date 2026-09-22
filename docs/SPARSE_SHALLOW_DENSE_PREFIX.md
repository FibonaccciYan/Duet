# Optional dense prefix before query selection

`prefix_dense_before_query_selection` is default-off. When enabled together with
query sparsification, layers before the query-selection boundary retain the full
prefix KV instead of applying the token-budget selector:

- LLaDA2.1: layers 0–1 are dense; selection begins at layer 1's output.
- SDAR: layers 0–5 are dense; selection begins at layer 5's output.

The effective boundary follows the model's configured query-selection layer and
is clamped to the model depth in the SDAR driver. Prefix-only runs and disabled
query sparsification keep the historical prefix behavior. If the boundary lands
inside an SDAR shared layer pair, the shallow member retains full prefix KV while
the deep member continues to use that pair's captured representative query; pair
grouping is not silently downgraded.

Full-prefix shallow layers are zero-slice views of the already-built cache. They
report `selection_policy="full_prefix"`, `budget_applied=false`, and
`score_definition="not_computed"` when selection statistics are enabled.

CLI forwarding:

- Unified LongBench: `--set prefix_dense_before_query_selection=true`
- Unified short bench: `--set PREFIX_DENSE_BEFORE_QUERY_SELECTION=true`
- lm-eval model args: `prefix_dense_before_query_selection=true`

The feature only changes which prefix KV is visible before query selection. It
does not alter query positions, confidence, transfer, EOS, RoPE, attention
kernels, current-block KV updates, or prefix selector behavior at deep layers.
