# Short benchmark scoring protocols

## GSM8K (task version 3.1)

`flexible-extract` still chooses the last numeric match. Each match must contain
at least one digit; trailing `$$` and `..` cannot become answers. Signed decimal
numbers and comma-separated numbers are supported. This is not a boxed-only or
semantic grading protocol. `strict-match`, prompts, few-shot, generation settings
and metric normalization are unchanged. Historical v3.0 scores must be labeled
separately; rescore saved responses instead of regenerating.

## MATH-500 (task version 2.0)

- `math_verify`: parse the raw LaTeX gold inside `$...$`, then verify against the
  full response. This fixes empty/partial gold extraction while preserving the
  full-response prediction parsing policy.
- `exact_match`: literal equality of the extracted last box and gold, stripping
  surrounding whitespace only. This replaces the incompatible Minerva fixed
  sentence extractor. Symbolic equivalence belongs to the math_verify metrics.

The last-box extractor accepts braced boxed/fbox, handles nested and escaped
braces, and rejects unclosed final boxes without falling back to an earlier box.
These are explicit repository protocols, not a claim of official OpenCompass
parity. The existing metric names are replaced in place; no legacy metrics are
emitted. Task versions distinguish pre-fix and post-fix results.
No generation, model, dataset, or decoding defaults are changed by this fix.
Minerva MATH tasks are unchanged. Use `--tasks math500` for MATH-500; `math`
selects the separate Minerva task group.

CPU validation environment: math-verify 0.9.0, sympy 1.13.1,
latex2sympy2_extended 1.11.0. Record dependency versions with reported scores.
Run `python -m unittest discover -s tests -p test_short_bench_scoring.py`.
The audit also checks all 500 actual golds against identical boxed predictions
and rescores historical saved responses; no GPU is required.
