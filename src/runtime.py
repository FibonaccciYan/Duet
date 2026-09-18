from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .dense.api import DenseRuntime
from .optimized.dense.api import DenseOptimizedRuntime, patch_model as patch_dense_optimized_model
from .optimized.sparse.api import patch_model as patch_sparse_optimized_model
from .dense.api import patch_model as patch_dense_model
from .reference.focus.api import FocusRuntime
from .reference.focus.api import patch_model as patch_focus_model
from .optimized.focus.api import FocusOptimizedRuntime, patch_model as patch_focus_optimized_model
from .reference.losa.api import LoSARuntime
from .reference.losa.api import patch_model as patch_losa_model
from .optimized.losa.api import LoSAOptimizedRuntime, patch_model as patch_losa_optimized_model
from .reference.sparse.api import patch_model as patch_sparse_model


RuntimeKind = Literal["dense", "dense_optimized", "losa", "focus", "focus_optimized", "losa_optimized"]
MethodKind = Literal["sparse", "sparse_optimized", "dense_optimized", "dense", "focus", "focus_optimized", "losa", "losa_optimized"]


def method_source_directory(method: str) -> str:
    """Source directory for benchmark manifests, independent of public labels."""
    if method in {"dense", "dense_optimized"}:
        return "src/optimized/dense"
    if method.endswith("_optimized") and method.removesuffix("_optimized") in {
        "sparse", "losa", "focus"
    }:
        return f"src/optimized/{method.removesuffix('_optimized')}"
    if method in {"sparse", "losa", "focus"}:
        return f"src/reference/{method}"
    raise ValueError(f"unknown method: {method}")


def patch_method(model, method: MethodKind, model_name: str = "auto", **kwargs):
    if method == "losa_optimized":
        return patch_losa_optimized_model(model, model_name=model_name, **kwargs)
    if method == "sparse_optimized":
        return patch_sparse_optimized_model(model, model_name=model_name, **kwargs)
    if method == "dense_optimized":
        return patch_dense_optimized_model(model, model_name=model_name, **kwargs)
    if method == "sparse":
        return patch_sparse_model(model, model_name=model_name, **kwargs)
    if method == "dense":
        return patch_dense_model(model, model_name=model_name, **kwargs)
    if method == "focus_optimized":
        return patch_focus_optimized_model(model, model_name=model_name, **kwargs)
    if method == "focus":
        return patch_focus_model(model, model_name=model_name, **kwargs)
    if method == "losa":
        return patch_losa_model(model, model_name=model_name, **kwargs)
    raise ValueError(f"unknown method: {method}")


def load_runtime(kind: RuntimeKind, **kwargs):
    if kind == "dense_optimized":
        return DenseOptimizedRuntime(**kwargs)
    if kind == "focus_optimized":
        return FocusOptimizedRuntime(**kwargs)
    if kind == "dense":
        return DenseRuntime(**kwargs)
    if kind == "losa":
        return LoSARuntime(**kwargs)
    if kind == "losa_optimized":
        return LoSAOptimizedRuntime(**kwargs)
    if kind == "focus":
        return FocusRuntime(**kwargs)
    raise ValueError(f"unknown runtime kind: {kind}")


@dataclass(frozen=True)
class RuntimeSpec:
    kind: RuntimeKind
    family: str
    kwargs: dict

    def instantiate(self):
        return load_runtime(self.kind, family=self.family, **self.kwargs)
