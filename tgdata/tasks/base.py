from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
from torch.utils.data import Dataset

from ..schema import TemporalGraph

Sample = dict[str, Any]
TASKS: dict[str, type[Task]] = {}


def register_task(cls: type[Task]) -> type[Task]:
    TASKS[cls.name] = cls
    return cls


class Task(Dataset):
    """A dataset view whose samples and targets are all fixed when it is built.

    Subclasses set `self.anchors` (the step each sample predicts) in `__init__`; nothing
    random or data-dependent happens in `__getitem__` beyond reading those anchors.
    """

    name: ClassVar[str]
    collate: ClassVar[Callable[[list[Sample]], Sample]]
    anchors: np.ndarray

    @classmethod
    def applies(cls, g: TemporalGraph) -> bool:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.anchors)


def select_anchors(
    g: TemporalGraph,
    anchors: np.ndarray,
    split: str,
    split_name: str,
    first: np.ndarray,
    last: np.ndarray,
    strict: bool,
) -> np.ndarray:
    """Anchors of `split` under `g.splits[split_name]`.

    `first`/`last` are, per anchor, the first input step and the last target step. A sample
    belongs to a split when its targets lie inside it; with `strict` its inputs must too.
    """
    spec = g.splits[split_name]
    if spec.nodes is not None:
        raise ValueError(f"split {split_name!r} splits nodes, not time")
    if spec.fractions is not None and spec.over == "samples":
        start, end = spec.resolve(len(anchors))[split]
        return anchors[start:end]
    if spec.fractions is not None:
        start, end = spec.resolve(g.num_steps)[split]
        target_first, target_last = anchors, last
    else:
        assert spec.boundaries is not None and g.timestamps is not None
        start, end = spec.boundaries[split]
        ts = np.asarray(g.timestamps)
        target_first, target_last = ts[anchors], ts[last]
        first = ts[first]
    keep = (target_first >= start) & (target_last < end)
    if strict:
        keep &= first >= start
    return anchors[keep]
