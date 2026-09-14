# SparseDLM research brief

Status: repository-grounded technical inventory at commit `7fda9ee` (2026-09-11).
This is not paper prose. It separates current code, recorded evidence, historical
results, and unsupported claims. “Existing sparse-attention approaches” below
means only baselines and approaches implemented or discussed in this repository;
a literature-level comparison still requires a separate related-work review.

## 1. Research problem

SparseDLM targets the inference cost of block-wise diffusion language models
(DLMs), where every denoising iteration repeatedly processes the unresolved rows
of the current generation block and attends to a potentially long, already fixed
prefix. The current repository asks whether inference can be accelerated without
changing model weights by reducing two different attention dimensions:

- **Query sparsity:** after a shallow layer, run the remaining layers only for a
  subset of current-block rows while always retaining already decoded rows.
- **Prefix sparsity:** replace the full historical prefix KV cache by a small,
  per-layer selected subset shared by the current query block.

The intended comparison is against a **same-runtime dense baseline**, not the
checkpoint's original generation implementation: dense uses the same block-cache
driver and common Triton/MoE optimizations with Query, Prefix, and integrated LoSA
disabled. This avoids attributing shared implementation improvements to sparsity.

Evidence: `src/dense/api.py::patch_model`; `src/sparse/api.py::patch_model`;
`src/sparse/llada_patch.py::_block_cache_generate`;
`src/sparse/sdar_patch.py::_block_diffusion_generate`; README “公平对比协议”.

Supported model families are currently LLaDA2.1-mini (`model_type=llada2_moe`)
and SDAR-8B-Chat-b32 (`model_type=sdar`), both batch size one. Patches are applied
at runtime and do not modify learned weights.

Evidence: `src/sparse/api.py::MODEL_TYPES, resolve_model_family`;
`src/sparse/llada_patch.py::_block_cache_generate` (batch assertion);
`src/sparse/sdar_patch.py::_block_diffusion_generate` (batch check); README opening.

## 2. Key empirical observations

1. **There are two reusable axes of sparsity, but their gains are not additive.**
   Query sparsity reduces suffix-layer rows; Prefix sparsity reduces historical
   KV length. Combining them can change confidence and therefore the denoising
   trajectory, including the number of cached forwards. Under the current exact-
   prompt NarrativeQA protocol, Raw-L1 LLaDA 32K/generation-256 Prefix is 1.655x
   while Query+Prefix is 1.619x; on the current per-layer SDAR dynamic run at
   32K/generation-768, Prefix is 1.145x while Query+Prefix is 1.475x. The
   interaction can therefore help or hurt and
   must be measured end to end. Evidence:
   `results/raw_l1_no_cost_gate_narrativeqa_llada_g{256,768}_r4.json` and
   `results/raw_l1_per_layer_dynamic_b512_narrativeqa_sdar_g{256,768}_r4.json`.

2. **Cost-aware gates remain configurable but are disabled by default.** Prefix
   minimum length, LLaDA Query minimum prefix length, and the small-unresolved-
   Query dense threshold now default to zero for both model families. Evidence:
   `src/sparse/config.py::LLaDASparseConfig, SDARSparseConfig`;
   `src/sparse/llada_patch.py::_block_cache_generate` (local `prefix_sparse`,
   `query_sparse`, and `maskless_attention=query_requested`);
   `src/sparse/sdar_patch.py::_block_diffusion_generate`. The headline ablation
   also sets all three gates to zero, so its gate policy now matches the default.

3. **Raw L1 is the best recorded prefix selector among those covered by the
   current LLaDA long-context quality test.** Across budgets 256/1024
   and Prefix/Query+Prefix, Raw L1 has higher NarrativeQA F1 than floating
   Hadamard L1 and Adamas. At budget 1024, Raw-L1 Prefix reaches F1 0.2575 versus
   dense 0.2525, with 2.033 s versus 2.889 s mean latency. Evidence: README
   “NarrativeQA 真实 32K 端到端对比”;
   `results/e2e_narrativeqa_32k_{dense,raw_l1_budget1024,hadamard_qk_budget1024,adamas_budget1024}/report.json`;
   runner `scripts/profile_longbench_query_steps.py`. This experiment does not
   cover SDAR or the implemented direct-QK selector, and its latency protocol is
   not interchangeable with the current fixed-work speed grid.

4. **Raw-L1 budget 256 is model- and implementation-dependent.** On full HumanEval,
   LLaDA Prefix-only is 68/164 official and 128/164 normalized versus dense
   74/164 and 137/164; LLaDA Query+Prefix is 83/164 and 134/164 and passes the
   repository's stated gate. With current independent per-layer selection, SDAR
   sequential Prefix-only and Query+Prefix are respectively 128/164,128/164 and
   128/164,129/164 versus dense 129/164,130/164; both pass. The older
   124/127 and 124/125 scores came from two-layer index sharing and do not support
   conclusions about the current implementation.
   Evidence: README “HumanEval 精度”; experiment outputs referenced by the README
   under `/data0/ysy/sparse/{llada_exp,sdar_exp}/` (outside this Git repository).

5. **The strongest current speed evidence is an H800 paired four-mode ablation
   on a real NarrativeQA prompt with all cost gates disabled.** Exact input
   lengths are 8K/16K/32K and requested generation is added on top at 256/768.
   Across four-repeat medians, Raw-L1 Query+Prefix is faster than same-runtime
   dense at all six points: LLaDA 1.005–1.619x in the existing run and current
   per-layer SDAR dynamic/budget-512 1.269–1.475x. The SDAR configuration is not
   a quality candidate: Query+Prefix scores 113/115 versus dynamic dense 121/125.
   Evidence: ignored raw reports
   `results/raw_l1_no_cost_gate_narrativeqa_llada_g{256,768}_r4.json` and
   `results/raw_l1_per_layer_dynamic_b512_narrativeqa_sdar_g{256,768}_r4.json`;
   benchmark implementation `scripts/bench_long_context.py::main`.

6. **Long-context latency is not explained by FLOPs alone.** In the current
   generation-256 phase profile, LLaDA dense-prefill GPU time grows from about
   472 ms at 8K to 2,061 ms at 32K, while Raw-L1 Prefix compaction grows only
   from about 53 ms to 67 ms. SDAR Query+Prefix cached-forward time stays near
   4.71--4.72 s while compaction grows from about 56 ms to 83 ms; the remaining
   end-to-end growth occurs outside those two measured phases. Approximation also
   changes LLaDA cached-forward counts and trajectories. Evidence:
   `results/raw_l1_no_cost_gate_narrativeqa_phase_{llada,sdar}_g256_r4.json`.

7. **Mask removal is a material component of LLaDA Query mode.** Query and
   Query+Prefix omit an all-zero cached attention mask, allowing native GQA;
   Dense and Prefix retain a mask. In the current exact-prompt/no-gate paired
   isolation, mask removal lowers Query latency by about 13--20% across the six
   8K/16K/32K x generation-256/768 points. With the full mask retained, Query is
   slower than dense at 8K (0.882x/0.928x), approximately tied at 16K
   (1.008x/0.989x), and faster at 32K (1.179x/1.096x). The separate recorded
   Query-only quality validation remains 85/164 official and 134/164 normalized.
   Evidence:
   `src/sparse/llada_patch.py::_cached_forward` (`maskless_attention`);
   README “Query 全零掩码移除组件验证”;
   `results/raw_l1_no_cost_gate_narrativeqa_llada_fullmask_g{256,768}_r4.json`,
   current Raw-L1 reports, and the README quality table.

8. **Block size 32 remains the stronger 32K Query configuration, but the old
   claim that block 64 is slower than dense is obsolete.** Under the current
   protocol, block-64 Query reaches 1.194x at generation-256 and 1.189x at
   generation-768, versus block-32's 1.470x and 1.357x. Block 64 is also slower
   relative to block 32 at 8K, while its 16K paired speedup is larger. It is
   therefore rejected for the 32K headline configuration, not as universally
   slower than dense. Evidence:
   `results/raw_l1_no_cost_gate_narrativeqa_llada_block64_g{256,768}_r4.json` and
   `results/raw_l1_no_cost_gate_narrativeqa_llada_g{256,768}_r4.json`.

9. **Integrated LoSA is correct in an all-active check but has no reliable
   end-to-end gain and is disabled.** Evidence: README “LoSA”;
   `src/sparse/sparse_ops.py::_merge_attention_states, _losa_active_indices`;
   `tests/test_llada_patch.py` LoSA tests. The separate paper-LoSA runtime is a
   distinct baseline under `src/losa/`, not the integrated SparseDLM option.

## 3. Current SparseDLM algorithm

### 3.1 Shared outer structure

For each generation block, the runtime performs a dense initialization/refresh,
materializes a prefix KV cache, then performs repeated cached denoising forwards.
The first dense pass also supplies per-layer current-block queries used by Prefix
selection. Subsequent blocks may reuse the previous selected prefix indices and
only add the newly available prefix range to the candidate set.

Evidence: `src/sparse/llada_patch.py::_block_cache_generate,
_capture_block_queries`; `src/sparse/sdar_patch.py::_block_diffusion_generate,
_capture_sdar_block_queries`; `src/sparse/sparse_ops.py::_compact_prefix_cache`.

### 3.2 Prefix selection (current default)

For each representative layer, after rotary position encoding, let current-block
queries be `Q` and historical prefix keys be `K`. Raw L1 computes exact
`||q-k||_1` distances in the model's learned Q/K basis. It takes a small nearest
set per query row/head, unions these indices, fills any budget shortfall using the
best minimum distance across queries, sorts the result, and gathers the same
indices from K and V. The final compact cache contains at most `B_p` prefix
tokens per layer. Both models now select independently per layer by default;
SDAR can explicitly restore its historical two-layer index sharing.

Evidence: `src/sparse/sparse_ops.py::_raw_l1_prefix_indices,
_distance_prefix_indices, _compact_prefix_cache` (notably `group_size`).

The rolling optimization uses the previous block's chosen indices plus keys
added since the previous prefix length instead of rescanning every old key.
This is an approximation to reselecting from the complete new prefix: previously
discarded old keys cannot re-enter later. Evidence:
`src/sparse/sparse_ops.py::_compact_prefix_cache` (`previous_indices`,
`previous_length`, `candidates`); state update in both model generation drivers.

Alternative selectors remain experiment-only: exact QK dot product, exact L1
after floating Hadamard transform, and quantized four-bucket Adamas after
Hadamard. The public Python config does not contain a selector field; evaluation
scripts switch selectors by monkey-patching module-level `_prefix_indices`.

Evidence: `src/sparse/sparse_ops.py::_qk_prefix_indices,
_hadamard_qk_prefix_indices, _adamas_prefix_indices, _prefix_indices`;
`scripts/bench_long_context.py::main` lines selecting the function;
`eval_instruct/lm_eval/models/llada.py` `prefix_selector` handling.

### 3.3 LLaDA Query sparsity

After selection layer 1, the selector applies the LM head to unresolved mask
positions, samples confidence, and retains `ceil(ratio × unresolved_masks)`
positions unless more positions exceed the generation confidence threshold, in
which case all above-threshold positions are retained. Already decoded positions
are always retained. On non-refresh iterations, a still-valid cached selection
can be reused. Suffix layers process only retained rows, write their K/V into a
dual cache at the original positions, and compute logits only for selected masks;
only positions with corresponding logits may transfer.

Evidence: `src/sparse/llada_patch.py::_select_positions, _cached_forward,
_transfer_tokens`; `src/sparse/sparse_ops.py::_BlockDualCache,
_dual_cache_from_dense`; guard in `_block_cache_generate` preventing an
uncomputed position from changing.

LLaDA's confidence threshold and editing behavior are generation controls that
also affect sparsity and trajectory. In the current sparse evaluation entrypoint
they default to threshold 0.5, editing threshold 0.0, and at least one transfer;
the model-level generate defaults differ (0.95/0.9). Evidence:
`eval_instruct/eval.sh` defaults and model args;
`src/sparse/llada_patch.py::_block_cache_generate` keyword defaults.

### 3.4 SDAR Query sparsity

The paper-speed configuration uses sequential remasking. For this strategy the
selected positions form a contiguous prefix of the current block: already
sequentially decoded rows plus `ceil(ratio × remaining)` candidate rows, bounded
below by the transfer minimum. This avoids confidence scoring for selection and
allows contiguous slicing. The first step is dense; subsequent suffix layers run
on this selected prefix. The default refresh value `-1` means no explicit late-KV
refresh under the condition implemented in `_block_diffusion_generate`.

Evidence: `src/sparse/sdar_patch.py::_select_positions` sequential branch;
`_sparse_cached_forward`; `_block_diffusion_generate` (`sequential_decoded`,
`refresh_late_kv`); `src/sparse/config.py::SDARSparseConfig`.

Other SDAR remasking strategies are implemented but are not the current speed
candidate: `low_confidence_static`, `low_confidence_dynamic`, and
`entropy_bounded`. Evidence: `src/sparse/sdar_patch.py::select_transfer,
block_diffusion_generate`; choices in `scripts/bench_long_context.py::parse_args`.

### 3.5 Attention and cache execution

Dense prefill uses custom block-causal attention paths and stores KV without
constructing a full square mask. Compact Prefix attention is exact over the
selected KV subset, not an approximation of softmax within that subset. Query
sparsity changes which rows reach suffix layers; Prefix sparsity changes which
historical keys are visible. Current-block attention remains present.

Evidence: `src/sparse/llada_patch.py::_prefill_attention_forward,
_dense_block_prefill, _cached_forward`;
`src/sparse/sdar_patch.py::_prefill_attention_forward,
_sparse_cached_forward`; kernels in `src/sparse/triton_kernels.py`.

For LLaDA, compatible routed MoE blocks are replaced by packed Triton inference:
weights are stacked into non-persistent buffers, original expert weight
references are cleared, and a fused implementation is installed. This is enabled
for both sparse and same-runtime dense LLaDA benchmarks, so it is a common system
optimization rather than the sparse algorithmic contribution.

Evidence: `src/sparse/llada_patch.py::_packed_weights, _pack_block,
patch_moe_experts`; `src/dense/api.py::patch_model`;
`scripts/bench_long_context.py::load`.

## 4. Differences from full attention

| Dimension | Same-runtime dense | SparseDLM |
| --- | --- | --- |
| Current-block query rows | Every row passes every layer | Shallow layers are dense; selected rows only pass suffix layers when Query pruning is active |
| Historical prefix KV | Every prefix position is visible | At most `prefix_token_budget` selected historical positions per representative layer when Prefix is active |
| Selection | None | Query confidence/sequential selector plus Raw-L1 prefix selector |
| Approximation | No sparse row/key removal | Removes suffix query computation and/or historical keys; can change logits, token transfers, EOS, and step count |
| Cache | Full prefix plus current block | Compact prefix; dual cache preserves dense current-block KV and sparse updates |
| LLaDA cached mask | Dense/Prefix keep mask | Query request uses maskless native-GQA path, even when row-pruning length gate is off |
| Shared optimizations | Block cache, custom prefill, optional LLaDA MoE patch | Same shared optimizations plus sparsity |

Evidence: `src/dense/api.py::patch_model`;
`src/sparse/llada_patch.py::_cached_forward, _block_cache_generate`;
`src/sparse/sdar_patch.py::_sparse_cached_forward, _block_diffusion_generate`;
`src/sparse/sparse_ops.py::_BlockDualCache, _compact_prefix_cache`.

The nominal complexity reduction is shape dependent, not a single asymptotic
claim: Prefix replaces a cached-attention key length near `P` by at most `B_p`,
while Query replaces suffix-layer current-block row count `B` by decoded rows
plus approximately `ratio × unresolved rows`. Dense prompt prefill and shallow
layers remain. The repository has not supplied a formal complexity derivation.

## 5. Differences from sparse-attention approaches present in this repository

### FOCUS baseline

FOCUS measures attention importance for unresolved masks at layers 0 and 1,
uses the positive importance delta to retain rows, and scales the target count
from recent average decoded tokens via `alpha`. SparseDLM Query instead uses
shallow LM-head confidence for LLaDA and a sequential contiguous selector for
the current SDAR candidate. FOCUS is query-row sparsity only in this integration;
it does not use Raw-L1 prefix KV selection.

Evidence: `src/focus/algorithm.py::attention_importance,
select_retained_positions`; `src/focus/generation.py::focus_generate`;
`src/sparse/{llada,sdar}_patch.py::_select_positions`.

### Paper-LoSA baseline

The standalone LoSA implementation splits historical-prefix and current-block
attention. It initializes exact prefix attention, detects active query rows by
query drift, uses QUEST min/max page metadata for active rows, reuses cached
prefix output/LSE for stable rows, computes exact block attention, and merges via
online softmax. SparseDLM Prefix instead selects individual token indices using
Raw-L1 Q/K distance and recomputes attention over that compact KV; SparseDLM
Query selects suffix-layer rows by generation confidence/sequential position,
not query drift. Evidence: `src/losa/operators.py` module algorithm and
`losa_attention`; `src/losa/api.py::patch_model`;
`src/sparse/sparse_ops.py::_compact_prefix_cache`.

### Adamas and alternative selectors

Adamas applies a Hadamard transform, quantizes Q/K into four buckets with
model-specific thresholds, and uses integer L1-like distance. Floating
Hadamard-L1 removes quantization but keeps the transform; direct QK selects high
dot-product keys. Current Raw L1 uses neither transform nor quantization and
minimizes exact L1 in learned Q/K coordinates. Evidence:
`src/sparse/sparse_ops.py::_adamas_prefix_indices,
_hadamard_qk_prefix_indices, _qk_prefix_indices, _raw_l1_prefix_indices`;
historical measurements remain in local `results/hqhk_*.json` artifacts.

### Integrated experimental LoSA

The `losa` flag inside `src/sparse/` is not the standalone paper-LoSA baseline.
It caches prefix attention output/LSE per row and refreshes a small active set,
with optional Query/LoSA union. It is disabled in both default sparse configs.
Evidence: `src/sparse/sparse_ops.py::_losa_active_indices,
_queue_losa_active_update`; `src/sparse/llada_patch.py::_losa_attention_forward`;
`src/sparse/config.py`.

### Literature-positioning limitation

The repository contains no citations, bibliography, or systematic comparison
against published sparse-attention/DLM acceleration methods beyond FOCUS, LoSA,
Adamas, and the implemented selectors. Claims of novelty or superiority over
“existing sparse attention” broadly are therefore unsupported by repository
evidence and must wait for literature review and matched experiments.

## 6. Important hyperparameters and controls

### Sparse algorithm defaults

| Parameter | LLaDA | SDAR | Scientific role / source |
| --- | ---: | ---: | --- |
| dtype | BF16 | FP16 | Runtime numerical regime; `scripts/bench_long_context.py::load`, `eval_instruct/eval.sh` |
| block length | 32 | 32 | Current denoising block; evaluation defaults in `eval_instruct/eval.sh` |
| steps | 32 | 32 | Denoising budget per block; same source |
| query ratio | 0.7 | 0.5 | Approximate unresolved rows retained; `src/sparse/config.py` |
| query top-k | 64 | 64 | Sampling confidence top-k argument; `src/sparse/config.py` |
| selection layer (zero-based) | 1 | 5 | Last dense/selector layer; `src/sparse/config.py` |
| selection interval | 4 | 1 | Recompute versus reuse Query selection; `src/sparse/config.py` |
| dense threshold | 0 | 0 | Optional small-query cost gate, disabled by default; `src/sparse/config.py` |
| query minimum prefix | 0 | no field/gate | Optional LLaDA length gate, disabled by default; `src/sparse/config.py`, LLaDA generation driver |
| query sparse default | on | on | `src/sparse/config.py` |
| prefix sparse default | on | off | Conservative runtime default; current sequential per-layer budget-256 quality passes, but matched current speed coverage is incomplete; `src/sparse/config.py` |
| prefix minimum length | 0 | 0 | Optional cost gate, disabled by default; `src/sparse/config.py` |
| prefix token budget | 256 | 256 | Maximum compact historical KV positions; `src/sparse/config.py` |
| prefix chunk size | 1,024 | 1,024 | Retained interface value; Raw L1 currently passes full prefix to `_distance_prefix_indices`, so it does not bound Raw-L1 scan chunks; `src/sparse/config.py`, `src/sparse/sparse_ops.py::_raw_l1_prefix_indices` |
| prefix selector | Raw L1 | Raw L1 | Hard-coded `_prefix_indices`; script-level override; `src/sparse/sparse_ops.py` |
| prefix layer sharing | none | none by default; optional pairs | Engineering cost/quality switch; `src/sparse/config.py`, `src/sparse/sparse_ops.py` |
| SDAR refresh step | n/a | -1 | Late-KV refresh control; `src/sparse/config.py`, SDAR generation driver |
| SDAR deep-only transfer | n/a | false | Transfer restriction; same sources |
| integrated LoSA | off | off | Experimental option; `src/sparse/config.py` |
| LoSA active top-k | 5 | 5 | Rows refreshed; `src/sparse/config.py` |
| LoSA score mode | query | query | Query drift versus key-diagonal weighting; `src/sparse/config.py` |
| LoSA key samples | 32 | 32 | Prefix energy estimator sample count; `src/sparse/config.py` |
| Query-LoSA union | false | absent | LLaDA experimental coupling; `src/sparse/config.py` |
| MoE expert patch | on | absent/off | Common LLaDA system optimization; `src/sparse/config.py`, `eval_instruct/eval.sh` |

### Generation/evaluation controls that can change results

- LLaDA sparse evaluation: temperature 0, threshold 0.5, editing threshold 0,
  `num_to_transfer=1`, mask 156895, EOS 156892. Dense/FOCUS/LoSA use different
  default thresholds (0.95/0.9), so matched method comparisons must explicitly
  decide whether generation policy or only attention approximation is under test.
  Evidence: `eval_instruct/eval.sh`.
- SDAR sparse evaluation: temperature 0, threshold 1.0, sequential remasking,
  entropy budget 0.35 (only operative for entropy-bounded), mask 151669. Dense
  defaults threshold 0.85. Evidence: `eval_instruct/eval.sh`.
- Benchmark axes: exact prompt/context length, requested generation length,
  repeats, seed 42, early-EOS behavior, warmup, mode order, and context order.
  Evidence: `scripts/bench_long_context.py::parse_args, generation_kwargs, main`.
- Quality-task generation limits and few-shot counts differ by task; the current
  wrapper is the authoritative invocation. Evidence: `eval_instruct/eval.sh`
  and task YAML under `eval_instruct/lm_eval/tasks/`.
- Standalone baselines: FOCUS `alpha=1.5`; paper-LoSA page size 16, token budget
  256, active top-k 5, GQA mode `per_query_head`, backend `auto`. Evidence:
  `src/focus/api.py::patch_model`; `src/losa/api.py::patch_model`.

## 7. Experimental evidence currently available

### Current, directly relevant evidence

| Evidence | Scope | Main recorded result | Artifact / runner |
| --- | --- | --- | --- |
| Paired real-prompt long-context ablation | LLaDA plus current per-layer SDAR dynamic/budget-512; H800; one NarrativeQA sample; exact 8K/16K/32K input; requested generation 256/768 added on top; all cost gates zero; four repeats; dense/query/prefix/query+prefix | Raw-L1 Query+Prefix faster at all 12 model/shape points: LLaDA 1.005–1.619x, current SDAR dynamic 1.269–1.475x; SDAR dynamic Query+Prefix fails quality | `results/raw_l1_no_cost_gate_narrativeqa_llada_g{256,768}_r4.json`; `results/raw_l1_per_layer_dynamic_b512_narrativeqa_sdar_g{256,768}_r4.json`; `scripts/bench_long_context.py` |
| Matched selector speed ablation | Same protocol as above; Raw L1, floating Hadamard L1, Adamas, and direct QK | No selector is uniformly fastest. Raw L1 is generally fastest on LLaDA and direct QK is materially slower there; SDAR selector differences are small, with direct QK fastest or tied at generation 768 | `results/{raw_l1,adamas,float_l1,qk}_no_cost_gate_narrativeqa_*_r4.json` |
| Real 32K NarrativeQA selector comparison | LLaDA; 91 examples ≥32640 input tokens; generation cap 128; natural EOS; budgets 256/1024 | Raw L1 best F1 among tested sparse selectors; Prefix-1024 F1 0.2575 vs dense 0.2525 | `results/e2e_narrativeqa_32k_*/report.json`; `scripts/profile_longbench_query_steps.py` |
| Full HumanEval selector validation | LLaDA + SDAR; 164 problems | LLaDA Raw-L1 Query+Prefix passes; current SDAR sequential per-layer Prefix and Query+Prefix pass at budgets 256/512/1024; SDAR dynamic Query and Query+Prefix fail | README HumanEval table; external `/data0/ysy/sparse/{llada_exp,sdar_exp}` outputs; `eval_instruct/eval.sh` |
| Length/phase diagnosis | LLaDA + SDAR; H800; one NarrativeQA sample; exact 8K/16K/32K input; generation 256; no gates; four repeats | Separates measured prefill, cached-forward, and compaction phases; exposes trajectory/call-count differences | `results/raw_l1_no_cost_gate_narrativeqa_phase_{llada,sdar}_g256_r4.json` |
| Maskless Query component | LLaDA; H800; current exact-prompt/no-gate protocol; generation 256/768; plus earlier HumanEval quality result | Mask removal lowers paired Query latency about 13--20%; earlier quality gate passed | `results/raw_l1_no_cost_gate_narrativeqa_llada_fullmask_g{256,768}_r4.json`; current Raw-L1 reports; README quality table |
| Cost-gate search | LLaDA threshold 4 vs 20; long context and LongBench 50-example subset | Threshold 20 improved tested latency/quality but was not adopted to avoid dataset-tuned gate | local `results/query_gate_*.json`, `results/longbench_query_gate_t*/report.json` |
| Block-size ablation | LLaDA Query; current protocol; block 32 vs 64; generation 256/768 | Block 64 is faster than dense but materially weaker at 32K than block 32 | `results/raw_l1_no_cost_gate_narrativeqa_llada_block64_g{256,768}_r4.json`; current Raw-L1 reports |
| Unit/correctness checks | CPU/GPU where available | Cache alignment, selector budgets, dense equivalence at ratio 1, LoSA merge, Triton kernels | `tests/test_llada_patch.py`, `tests/test_sdar_patch.py`, `tests/test_triton_sparse_kernels.py` |

### Historical or non-current evidence (do not mix into the main method table)

- Adamas calibration, selector microbenchmarks, and older throughput/memory
  results remain useful component evidence but use a superseded selector.
  Evidence: local `results/hqhk_*.json` artifacts.
- `results/raw_l1_final_{llada,sdar}_g{256,768}_r4.json` is the superseded
  headline grid: it uses repetitive synthetic prompts, prompt-plus-generation
  context lengths, and production cost gates. It remains a production-gated
  comparison but must not be pooled with the exact-prompt, no-gate grid.
- `results/` is Git-ignored, so the raw reports are present locally but are not
  versioned by the repository. Evidence: `.gitignore` rule for `results/` and
  `git ls-files results` returning no tracked files at brief creation time.

### Existing evaluation infrastructure without a complete reported matrix

The repository can run GSM8K, Minerva Math, HumanEval, MMLU, and a five-task
LongBench subset for sparse/dense/FOCUS/LoSA. However, README does not report a
complete current Raw-L1 four-method matrix. Evidence: `eval_instruct/eval.sh`,
`eval_instruct/run_matrix.sh`, `scripts/run_longbench_quality.py`.

## 8. Implementation details that may affect scientific claims

1. **The fixed-work LLaDA benchmark still truncates the returned tensor at EOS.**
   `scripts/bench_long_context.py::generation_kwargs` sets
   `eos_early_stop=False`, which prevents stopping future blocks, but
   `src/sparse/llada_patch.py::_block_cache_generate` unconditionally calls
   `_trim_to_first_eos` before returning. Raw reports therefore contain, e.g.,
   33 returned tokens for a requested 256. Timing may still cover the scheduled
   generation work, but `generated_tokens_per_second`, output checksum, and any
   statement of equal actual output length do not represent 256 returned tokens.
   The benchmark only checks `<= gen_length`, not equality
   (`scripts/bench_long_context.py::run_once`).

2. **Synthetic “context length” historically means prompt plus requested
   generation window.** Unless `--prompt-lengths` is used, prompt length is
   `context_length - gen_length`; recent commit `7fda9ee` added exact prompt
   lengths. Cross-paper comparisons must say which axis was used. Evidence:
   `scripts/bench_long_context.py::run_once, parse_args`; recent git history.

3. **The current headline grid uses one real prompt, not a representative prompt
   sample.** One NarrativeQA example is middle-truncated to exact 8K/16K/32K
   inputs. This removes repetitive synthetic filler from the headline result but
   does not establish robustness across documents or domains. The older synthetic
   grid remains only a protocol countercheck. Evidence:
   `scripts/bench_long_context.py::exact_prompt, load_narrativeqa_prompt` and
   `results/raw_l1_no_cost_gate_narrativeqa_*_r4.json`.

4. **Sparsity changes outputs and denoising trajectories.** Different modes can
   return different checksums, generated lengths, confidence transfers, and
   cached-call counts. Requested-token latency is not pure kernel throughput.
   Evidence: raw result JSON fields and `scripts/bench_long_context.py` phase metrics.

5. **Raw-L1 “chunk size” is currently ineffective for the distance scan.**
   `_raw_l1_prefix_indices` calls `_distance_prefix_indices` with
   `chunk_size=prefix_length`, ignoring its caller's configured chunk size. This
   makes the selector exact over the candidate prefix but can materialize a large
   distance matrix and affects scalability/memory claims. Evidence:
   `src/sparse/sparse_ops.py::_raw_l1_prefix_indices, _prefix_indices`.

6. **Rolling prefix reuse is not exact full-prefix reselection.** The next block
   selects only among prior winners and newly appended keys; an old loser can
   never become a winner for changed queries. Thus “exact L1” describes distances
   on the current candidate set, not globally exact selection after the first
   rolling reuse. Evidence: `src/sparse/sparse_ops.py::_compact_prefix_cache`.

7. **Historical SDAR results used two-layer index sharing.** Current SDAR
   defaults to independent per-layer selection; the old pair sharing remains an
   explicit engineering switch. New HumanEval results cover the current
   sequential implementation, and new speed results cover dynamic/budget 512;
   historical artifacts still cannot establish current sequential speed.
   Evidence: `src/sparse/sparse_ops.py::_compact_prefix_cache` (`group_size`).

8. **Query mode includes a non-sparsity backend change.** LLaDA Query request
   removes a zero attention mask to enable native GQA. This backend change is
   present alongside row pruning with the new zero-gate default and must remain
   separated in component ablations. Evidence:
   `src/sparse/llada_patch.py::_block_cache_generate, _cached_forward`.

9. **Sparse and dense quality-policy defaults differ in the evaluation wrapper.**
   LLaDA sparse uses threshold/editing 0.5/0.0 versus 0.95/0.9 for other methods;
   SDAR sparse uses 1.0 versus 0.85. A method comparison using defaults jointly
   tests generation policy and attention method. Evidence: `eval_instruct/eval.sh`.

10. **The HumanEval “normalized” metric is a custom post-processing diagnostic.**
    It repairs indentation/code extraction and should not replace official pass@1.
    Both must be reported. Evidence: `eval_instruct/reevaluate_humaneval.py`
    and the README quality gate.

11. **HumanEval evidence locations are partly external and configurations have
    evolved.** Current scores are summarized in README, but raw current outputs
    are outside Git and some older result directories refer to earlier selectors
    and policies. Reproducibility requires copying immutable raw outputs and full
    configs into a versioned artifact. Evidence: README paths, `.gitignore`,
    and recent commits `257406c`–`03297b5`.

12. **Hardware/software evidence is narrow.** Current headline timing reports
    record NVIDIA H800 PCIe, PyTorch 2.5.0+cu124/CUDA 12.4 in artifacts. No
    multi-GPU-family or multi-software-version robustness result is reported.
    Evidence: `results/raw_l1_no_cost_gate_narrativeqa_*.json`.

13. **Peak memory uses allocated memory after one loaded model and shared
    process.** It is useful within the benchmark but is not a clean model-plus-
    runtime residency measurement across independent processes. Evidence:
    `scripts/bench_long_context.py::run_once` (`reset_peak_memory_stats`,
    `max_memory_allocated`) and `main` (one loaded model, modes toggled).

14. **Selector switching is global monkey-patching.** It is adequate for a
    single-process experiment but is not persisted in `SparseDLMConfig`; saved
    config plus selector field in the report must both be used to reproduce a
    run. Evidence: `scripts/bench_long_context.py::main`;
    `src/sparse/config.py`; report construction in benchmark `main`.

15. **The method is inference-only and checkpoint-specific.** There is no
    training/fine-tuning, learned selector, multi-batch support, or generic model
    architecture adapter. Evidence: runtime patch APIs, batch checks, and
    `src/sparse/api.py::MODEL_TYPES`.

## 9. Claims currently unsupported by experiments

The following should not appear as paper claims without new evidence:

- **General state-of-the-art superiority** over sparse attention, DLM inference,
  FOCUS, LoSA, or Adamas. There is no complete matched current four-method table,
  literature review, or external baseline reproduction.
- **Model-general or architecture-general effectiveness.** Only two checkpoints
  are implemented, and the two use materially different Query selectors and
  prefix sharing.
- **Quality-preserving confidence-based Query sparsity for SDAR.** Dynamic Query
  and Query+Prefix explicitly fail the chosen HumanEval gate at the tested
  ratio; their speedups cannot be presented as a quality-preserving result.
- **Uniform quality preservation for LLaDA Prefix-256.** Prefix-only fails full
  HumanEval; Query+Prefix passes only the repository-defined gate and still
  differs from dense on official/normalized metrics.
- **Accuracy improvement over dense.** Some scores are higher, but decoding is
  deterministic on one benchmark set, outputs/trajectories change, and no
  uncertainty or repeated-seed analysis establishes a reliable improvement.
- **Fixed actual-output-token throughput for LLaDA headline timing.** Returned
  outputs are EOS-trimmed despite disabled early block stopping; the reports use
  requested tokens for headline throughput.
- **Kernel-only speedup.** End-to-end latency includes dense prefill, selector,
  changed step count, and changed generation trajectory.
- **Global exact top-L1 selection across every generation block.** Rolling reuse
  excludes previously discarded old keys after the first selection.
- **Linear scaling, asymptotic complexity, or speedup increasing monotonically
  with context.** Existing phase results show non-monotonic behavior and no
  formal/measured scaling law.
- **Robustness across prompt documents and domains.** The headline speed grid now
  covers three exact input lengths but derives all of them from one NarrativeQA
  sample. The complete current LongBench matrix is still absent.
- **Robustness across generation policies, seeds, batch sizes, GPUs, CUDA/PyTorch
  versions, or precision.** These axes have not been systematically evaluated.
- **Training-free equivalence to dense.** The method is training-free, but it is
  approximate and produces different outputs; equivalence exists only in narrow
  correctness tests such as ratio 1.0/all-active paths.
- **General memory reduction.** The current Raw-L1 reports provide a complete
  paired `max_memory_allocated` grid for both models, but it is mixed: 32K
  Query+Prefix lowers LLaDA peak allocation by about 1.5 GiB while increasing
  SDAR peak allocation by about 4.4 GiB. It is not evidence of uniform memory
  reduction or independent-process residency.
- **Ablation completeness.** Raw L1, floating Hadamard L1, Adamas, and direct QK
  now have a matched current speed grid, but ratio, budget, selection
  layer/interval, rolling reuse, and MoE interaction have not all been crossed
  in a controlled current-quality/current-speed study. Maskless GQA has a current
  latency isolation, but not a fully crossed quality study.
- **Reproducibility from Git checkout alone.** Models, datasets, environments,
  Hadamard extension for comparator selectors, and all raw current result files
  are external or ignored.

## Repository/history notes relevant to scientific interpretation

Recent history shows that the present method is the result of several scientific
and systems changes, so results before each change should not be pooled blindly:

- `dcdbfe2`: introduced fair paired sparse ablation.
- `31e7f34`, `00ee7bf`: added LLaDA query/prefix length gating.
- `43432e9`–`86a32d1`: iterated cached attention and native-GQA/mask handling.
- `24ed326`, `4287ca5`: added direct-QK and floating-Hadamard selectors.
- `257406c`, `c03a8f9`: added Raw L1 and made it default.
- `03297b5`: replaced headline results and quality tables with Raw-L1 evidence.
- `325d266`, `7fda9ee`: added real NarrativeQA prompts and exact prompt-length
  benchmarking.

Evidence: `git log -- src/sparse README.md scripts/bench_long_context.py` and the
named commits. The brief itself was prepared from a clean worktree before this
file was added.
