"""MATH-500 scoring: reuse the minerva math_verify pipeline.

The official SDAR eval uses OpenCompass MATHVerifyEvaluator (extract the last
\\boxed{} content, parse LaTeX, verify symbolically).  The minerva_math
tasks in this fork implement the same math_verify metric; we reuse it here.
"""
import sys
from pathlib import Path

_TASKS = Path(__file__).resolve().parents[1]
_MINERVA = _TASKS / "minerva_math"
if str(_MINERVA) not in sys.path:
    sys.path.insert(0, str(_MINERVA))

from utils import (  # noqa: E402,F401
    get_unnormalized_answer,
    is_equiv,
    normalize_final_answer,
    process_results,
)
