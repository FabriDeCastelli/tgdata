from __future__ import annotations

from typing import Any

from ..schema import TemporalGraph
from .base import TASKS, ConcatTasks, Task, passthrough, register_task
from .classification import GraphClassification
from .forecasting import NodeForecasting
from .regression import NodeRegression


def available(g: TemporalGraph) -> list[str]:
    return [name for name, cls in TASKS.items() if cls.applies(g)]


def make_task(g: TemporalGraph, name: str = "default", **params: Any) -> Task:
    if name == "default":
        spec = g.meta["default_task"]
        name, params = spec["name"], {**spec["params"], **params}
    return TASKS[name](g, **params)


__all__ = ["TASKS", "ConcatTasks", "GraphClassification", "NodeForecasting", "NodeRegression",
           "Task", "available", "make_task", "passthrough", "register_task"]
