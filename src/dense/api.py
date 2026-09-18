"""Compatibility exports only: all dense names use final efficient optimized Dense."""
from src.optimized.dense.api import (DEFAULT_MODEL_PATHS, DenseOptimizedRuntime, load_model_and_tokenizer, patch_model, set_seed)
DenseRuntime = DenseOptimizedRuntime
