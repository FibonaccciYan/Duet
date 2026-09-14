"""Correctness-first FOCUS v2 runtime.

This package preserves the FOCUS selection semantics while running the model
through a single-projection two-stage forward path.  It is intentionally kept
separate from ``src.focus`` so the portable implementation can continue to act
as the reference oracle.
"""

from .api import FocusV2Runtime, patch_model
from .generation import focus_v2_generate
from .model import focus_v2_forward

__all__ = [
    "FocusV2Runtime",
    "focus_v2_forward",
    "focus_v2_generate",
    "patch_model",
]
