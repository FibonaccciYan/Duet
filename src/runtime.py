from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .dense.api import DenseRuntime
from .focus.api import FocusRuntime
from .losa.api import LoSARuntime


RuntimeKind = Literal["dense", "losa", "focus"]


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
