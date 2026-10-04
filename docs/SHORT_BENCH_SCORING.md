# Short benchmark scoring protocols

## GSM8K (task version 3.1)

`flexible-extract` chooses the last numeric match. Each match contains at least
one digit; signed decimals and comma-separated numbers are supported. Prompts,
few-shot settings, generation settings, metric normalization, and
`strict-match` remain unchanged. Historical v3.0 scores must be labeled
separately.

## MATH-500 (task version 2.0)

- `math_verify`: parse raw LaTeX gold inside `$...$`, then verify against the
  full response; prediction parsing also uses the full response.
- `exact_match`: literal equality of the extracted last box and gold after
  stripping surrounding whitespace only.

The last-box extractor accepts braced boxed/fbox forms, handles nested and
escaped braces, and rejects an unclosed final box instead of selecting an
earlier box. These are explicit repository protocols, not a claim of official
OpenCompass parity. `math500` selects MATH-500; the unrelated Minerva task
group is not distributed in this paper release. Report `math_verify` as the
primary score and label literal `exact_match` separately.

Record `math-verify`, `sympy`, and `latex2sympy2_extended` versions with any
reported score.

## HumanEval

The formal entry runs `humaneval_instruct` and then
`eval_instruct/reevaluate_humaneval.py` on its saved generations. The reported
score is the indentation-normalized pass@1, not the preliminary harness score.
The existing code extraction, normalization, sanitization, and execution
timeout policy are preserved. HumanEval executes generated code and should be
run in an isolated environment.

## MMLU

Use `mmlu_generative` with five demonstrations. The existing first-standalone
A-D extraction policy is unchanged.
