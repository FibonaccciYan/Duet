"""FOCUS v2 algorithm re-exports.

The numerical semantics are intentionally shared with the portable FOCUS
reference in :mod:`src.focus.algorithm`.  v2 changes the model execution path,
not the selection rules.
"""

from src.focus.algorithm import (
    FocusDecodeState,
    attention_importance,
    select_retained_positions,
)

__all__ = [
    "FocusDecodeState",
    "attention_importance",
    "select_retained_positions",
]
