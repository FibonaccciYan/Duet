import pytest
from src.optimized.focus.backend import resolve_attention_backend
from src.optimized.focus.api import FocusOptimizedRuntime
from src.runtime import load_runtime

def test_defaults():
    assert FocusOptimizedRuntime(family="llada").attention_backend == "flash"
    assert FocusOptimizedRuntime(family="sdar").attention_backend == "sdpa"
    assert load_runtime("focus_optimized",family="llada").attention_backend == "flash"

def test_override_preserved():
    assert FocusOptimizedRuntime(family="llada",attention_backend="sdpa").attention_backend == "sdpa"
    assert resolve_attention_backend("sdar","flash") == "flash"

@pytest.mark.parametrize("family,backend",[("bad","auto"),("llada","bad")])
def test_invalid(family,backend):
    with pytest.raises(ValueError):resolve_attention_backend(family,backend)
