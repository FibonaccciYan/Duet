from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .dense.api import DenseRuntime
from .dense.api import patch_model as patch_dense_model
from .focus.api import FocusRuntime
from .focus.api import patch_model as patch_focus_model
from .losa.api import LoSARuntime
from .losa.api import patch_model as patch_losa_model
from .sparse.api import patch_model as patch_sparse_model


RuntimeKind = Literal["dense", "losa", "focus"]
MethodKind = Literal["sparse", "dense", "focus", "losa"]


def patch_method(model, method: MethodKind, model_name: str = "auto", **kwargs):
    if method == "sparse":
        return patch_sparse_model(model, model_name=model_name, **kwargs)
    if method == "dense":
        return patch_dense_model(model, model_name=model_name)
    if method == "focus":
        return patch_focus_model(model, model_name=model_name, **kwargs)
    if method == "losa":
        return patch_losa_model(model, model_name=model_name, **kwargs)
    raise ValueError(f"unknown method: {method}")


def load_runtime(kind: RuntimeKind, **kwargs):
    if kind == "dense":
        return DenseRuntime(**kwargs)
    if kind == "losa":
        return LoSARuntime(**kwargs)
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
