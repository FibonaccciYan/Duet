# Unified scripts verification

2026-09-18; branch `codex/integrate-all-20260916`.
Local source: `/Users/lier/codes/SparseDLM_LLaDA_SDAR_integration`.
Isolated remote check: `/data0/gs/unified_scripts_check_20260918`.

## Delivered

- Twelve shell entry points under `scripts/unified/`.
- Preserved scripts under `scripts/original/`; S-mode-specific scripts removed.
- Configurable shared launcher, task defaults, speed runner and original quality
  evaluator adapters. `eval_instruct` task YAML, prompts and scoring functions
  are unchanged.
- All short benchmark entries select optimized runtime patches.
- SDAR default threshold0.95 and MMLU five-shot for every method/model.
- Repository-local `data/narrativeqa_speed80`:20 files,125207617 bytes. No files
  ignored by Git; largest file26849444 bytes. `.gitattributes` prevents checksum
  damage through newline conversion. No model weights or external data symlinks.

## Defaults / regression

CPU suite: **189 passed,22 skipped**. This includes eight tests executing the
actual `eval.sh` with an intercepted Python executable and verifying optimized
method arguments and `--num_fewshot 5`. It is not merely a JSON-config assertion.

Dataset verifier: **80 paired source IDs,480 exact head/tail crops**, all
manifest checksums passed. Formal speed defaults remain two warmups,three E2E
repeats,256 generation tokens. Independent event passes preserve original
generation/timing drivers; noninitial aggregation excludes only each block's
first step.

## GPU smoke

GPU0 served LLaDA2.1 and GPU2 served SDAR. Both were idle before work and had no
compute processes after work (4MiB driver overhead,0% utilization). No GPU0/2
holder was used. Other GPUs and the production repository were not modified.

| Scope | Result |
|---|---|
| LongBench default five QA datasets | Both models x four methods passed; one row per task,context budget1024 |
| Speed 8K/16K/32K | Both models x four methods x three lengths passed; one sample,one E2E,two warmups,gen32 for smoke |
| Short GSM8K | Both models x four optimized methods passed; one sample,gen32 |
| Short HumanEval | Both models x four optimized methods passed; one sample,gen32; official and normalized diagnostic outputs retained |
| Short Math | Both models x four optimized methods passed; each of seven actual subtasks one sample,gen32 |
| Short MMLU | Both models FOCUS passed all57 subjects with one sample per subject; other methods not GPU-rerun after the user requested no rerun |

Dense/Sparse/LoSA initial MMLU attempts failed on an existing datasets3.6 cache
schema mismatch (`List` feature serialized by a newer datasets version).
The compatibility layer handles only equivalent primitive List[Value] metadata,
not struct/list rewrites; it does not change sample records. All57 subject
caches were subsequently read and validated on CPU. Five-shot command dispatch
is covered for all eight model/method combinations. **Do not describe the six
unrerun MMLU combinations as successful GPU runs.**

Initial online GSM8K/metric fetch attempts were stopped after n33 network
unavailability was established. Replacements used the same existing cached data
and metric modules with hub/datasets/evaluate offline flags. Two supervised
agents completed only previously missing Math smoke jobs on their respective
cards; they did not run full evaluation, rerun MMLU or start holders.

Smoke generation budgets are explicit overrides and do not change formal
defaults. Smoke accuracy (often zero with gen32) is not a quality result.
Speed smoke checks tokens,output lengths and block counts against independent
event timing. Reference LongBench remains the default protocol; optimized
LongBench selection is explicit, while short/speed are optimized by default.

## Evidence

Remote:

- `cpu_tests_final.log`
- `smoke_inventory.json`
- `mmlu_cache_preflight.json`
- `results/verified_long_*`
- `results/verified_speed_*`
- Successful short task outputs across `results/verified_short_*`,
  `results/smoke_short_offline_*`, `results/smoke_short_final_*`,
  and `results/supplement_math_*`

Local audit artifacts:
`/Users/lier/codes/d2Cache/artifacts/scripts_unification_20260918/`.

No full evaluation, Git commit, staging or push was performed in this task.
