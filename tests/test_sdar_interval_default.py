import inspect

from src.reference.sparse.config import SDARSparseConfig, LLaDASparseConfig, apply_overrides
from src.reference.sparse.sdar_patch import patch_sdar_model


def test_sdar_selection_interval_default():
    assert SDARSparseConfig().selection_interval == 4
    assert inspect.signature(patch_sdar_model).parameters["selection_interval"].default == 4


def test_explicit_interval_one_remains_supported():
    config = apply_overrides(SDARSparseConfig(), {"selection_interval": 1})
    assert config.selection_interval == 1
    assert SDARSparseConfig().selection_interval == 4


def test_llada_defaults_unchanged():
    assert LLaDASparseConfig().selection_interval == 4
