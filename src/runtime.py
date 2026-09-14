from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .dense.api import DenseRuntime
from .focus.api import FocusRuntime
from .focus_v2.api import FocusV2Runtime
from .losa.api import LoSARuntime
from .losa_v2.api import LoSAV2Runtime


RuntimeKind = Literal["dense", "losa", "losa_v2", "focus", "focus_v2"]


def load_runtime(kind: RuntimeKind, **kwargs):
    if kind == "dense":
        return DenseRuntime(**kwargs)
    if kind == "losa":
        return LoSARuntime(**kwargs)
    if kind == "losa_v2":
        return LoSAV2Runtime(**kwargs)
    if kind == "focus":
        return FocusRuntime(**kwargs)
    if kind == "focus_v2":
        return FocusV2Runtime(**kwargs)
    raise ValueError(f"unknown runtime kind: {kind}")


@dataclass(frozen=True)
class RuntimeSpec:
    kind: RuntimeKind
    family: str
    kwargs: dict

    def instantiate(self):
        return load_runtime(self.kind, family=self.family, **self.kwargs)
