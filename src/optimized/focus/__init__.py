"""Correctness-first optimized FOCUS runtime.

This package preserves the FOCUS selection semantics while running the model
through a single-projection two-stage forward path.  It is intentionally kept
separate from ``src.reference.focus`` so the portable implementation can continue to act
as the reference oracle.
"""

from .api import FocusOptimizedRuntime, patch_model
from .generation import focus_optimized_generate
from .model import focus_optimized_forward

__all__ = [
    "FocusOptimizedRuntime",
    "focus_optimized_forward",
    "focus_optimized_generate",
    "patch_model",
]
