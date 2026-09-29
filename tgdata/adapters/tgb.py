from __future__ import annotations

from typing import Any

from ..schema import TemporalGraph


def to_tgb(g: TemporalGraph, **kwargs: Any) -> Any:
    raise NotImplementedError("continuous-time datasets are out of scope for schema v1")
