from pathlib import Path
from src.dense.api import DenseRuntime, patch_model
from src.optimized.dense.api import DenseOptimizedRuntime, patch_model as final_patch
from src.runtime import load_runtime

def test_dense_alias_is_final():
    assert DenseRuntime is DenseOptimizedRuntime
    assert patch_model is final_patch
    assert type(load_runtime("dense", family="sdar")) is DenseOptimizedRuntime

def test_only_retained_packages():
    root = Path(__file__).resolve().parents[1] / "src"
    assert not (root / "sparse_v2").exists()
    assert not (root / "dense_v2").exists()
    assert (root / "reference" / "sparse").is_dir()
    assert (root / "optimized" / "sparse").is_dir()
