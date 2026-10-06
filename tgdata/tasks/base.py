from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from ..schema import TemporalGraph

Sample = dict[str, Any]
TASKS: dict[str, type[Task]] = {}


def passthrough(batch: Sample) -> Sample:
    """Collate for tasks whose `__getitems__` already returns a whole batch."""
    return batch


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

    def loader(self, batch_size: int, shuffle: bool = False, drop_last: bool = False,
               **kwargs: Any) -> DataLoader:
        """A standard DataLoader set up for this task (usable as is by Lightning).

        Batched tasks build batches on their device, so there are no worker processes.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                          collate_fn=self.collate, num_workers=0, **kwargs)


class ConcatTasks(ConcatDataset):
    """Tasks of several datasets behind one index space, for `MultiDatasetSampler` batches.

    Each batch must come from a single task; it is built by that task's `__getitems__`.
    """

    collate = staticmethod(passthrough)

    def __getitems__(self, indices: Sequence[int]) -> Sample:
        d = bisect.bisect_right(self.cumulative_sizes, indices[0])
        start = self.cumulative_sizes[d - 1] if d else 0
        local = [i - start for i in indices]
        if not 0 <= min(local) <= max(local) < len(self.datasets[d]):
            raise ValueError("a batch spans several tasks; sample it with MultiDatasetSampler")
        return self.datasets[d].__getitems__(local)

    def loader(self, batch_size: int, num_batches: int, temperature: float = 1.0,
               seed: int = 0) -> DataLoader:
        from ..sampling import MultiDatasetSampler

        sampler = MultiDatasetSampler([len(t) for t in self.datasets], batch_size, num_batches,
                                      temperature=temperature, seed=seed)
        return DataLoader(self, batch_sampler=sampler, collate_fn=self.collate, num_workers=0)


def select_anchors(
    g: TemporalGraph,
    anchors: np.ndarray,
    split: str | None,
    split_name: str,
    first: np.ndarray,
    last: np.ndarray,
    strict: bool | None,
) -> np.ndarray:
    """Anchors of `split` under `g.splits[split_name]`; all anchors, in order, when `split` is None.

    `first`/`last` are, per anchor, the first input step and the last target step. A sample
    belongs to a split when its targets lie inside it; with `strict` (default: the split's own
    setting) its inputs must too. A holdout part holds the windows entirely inside it; the
    other parts split the windows that end before it.
    """
    if split is None:
        return anchors
    spec = g.splits[split_name]
    strict = spec.strict if strict is None else strict
    if spec.nodes is not None:
        raise ValueError(f"split {split_name!r} splits nodes, not time")
    if spec.holdout is not None:
        cut = spec.holdout_start(g.num_steps)
        if split in spec.holdout:
            return anchors[first >= cut]
        anchors = anchors[last < cut]
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
