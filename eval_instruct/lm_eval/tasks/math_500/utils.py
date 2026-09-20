"""MATH-500 scoring v2. See docs/SHORT_BENCH_SCORING.md for protocols."""
import re

from math_verify import parse, verify


def last_boxed_answer(text):
    """Extract the last braced boxed/fbox answer, including nested braces.

    A malformed final box is invalid; do not silently reuse an earlier box.
    """
    matches = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        # Escaped literal braces do not delimit a TeX argument.
        slashes = 0
        j = i - 1
        while j >= 0 and text[j] == "\\":
            slashes += 1
            j -= 1
        if slashes % 2:
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def process_results(doc, results):
    prediction = results[0]
    answer = doc["answer"].strip()
    gold = parse("$" + answer + "$")
    predicted = parse(prediction)
    boxed = last_boxed_answer(prediction)
    # Literal boxed equality, not symbolic equivalence. Keep internal whitespace
    # significant (e.g. within \\text{}).
    exact = boxed is not None and boxed == answer
    return {
        "exact_match": int(exact),
        "math_verify": int(bool(verify(gold, predicted))),
    }
