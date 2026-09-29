from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch

from ..sampling import collate_pad
from ..schema import TemporalGraph, _mask_as_x, adjacency
from ..time import to_steps
from .base import Sample, Task, register_task, select_anchors


@register_task
class NodeForecasting(Task):
    """Predict steps [t, t + horizon) of `x` (or of node target `target`) from [t - window, t)."""

    name = "node_forecasting"
    collate = staticmethod(collate_pad)

    def __init__(
        self,
        g: TemporalGraph,
        window: int | str,
        horizon: int | str,
        split: str = "train",
        splits: str = "default",
        stride: int = 1,
        strict: bool = False,
        target: str | None = None,
        normalize: Literal["channel", "node"] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
        adjacency_kind: str | None = None,
    ) -> None:
        if not self.applies(g):
            raise ValueError(f"{g.name} has no node time series to forecast")
        self.g, self.transform = g, transform
        self.window = to_steps(window, g.meta["freq"])
        self.horizon = to_steps(horizon, g.meta["freq"])
        candidates = np.arange(self.window, g.num_steps - self.horizon + 1, stride)
        self.anchors = select_anchors(g, candidates, split, splits,
                                      first=candidates - self.window,
                                      last=candidates + self.horizon - 1, strict=strict)
        self.target = None if target is None else g.y[target]
        self.target_channels = g.meta.get("target_channels")
        self.scale = _scale(g, normalize) if normalize else None
        if g.edge_index is not None:
            self.edge_index, self.edge_weight = adjacency(g, adjacency_kind)
            if g.is_static:
                self.static_edges = _edges(self.edge_index, self.edge_weight)

    @classmethod
    def applies(cls, g: TemporalGraph) -> bool:
        return g.time_mode == "discrete" and g.x is not None

    def __getitem__(self, i: int) -> Sample:
        g, t = self.g, int(self.anchors[i])
        lo, hi = t - self.window, t + self.horizon
        x = np.asarray(g.x[lo:hi], dtype=np.float32)
        if self.scale is not None:
            x = (x - self.scale[0]) / self.scale[1]
        if self.target is not None:
            y = np.asarray(self.target.values[t:hi])
        elif self.target_channels is not None:
            y = x[self.window:, :, self.target_channels]
        else:
            y = x[self.window:]
        sample: Sample = {"x": torch.from_numpy(x[: self.window].copy()),
                          "y": torch.from_numpy(np.ascontiguousarray(y)), "t": t}
        if g.mask is not None:
            mask = torch.from_numpy(_mask_as_x(np.asarray(g.mask[lo:hi]), x).copy())
            sample["mask_x"] = mask[: self.window]
            mask_y = mask[self.window:]
            if self.target is None and self.target_channels is not None and mask.shape[-1] > 1:
                mask_y = mask_y[..., self.target_channels]
            sample["mask_y"] = mask_y
        if g.covariates is not None:
            sample["covariates"] = torch.from_numpy(np.array(g.covariates[lo:hi], np.float32))
        if g.timestamps is not None:
            sample["timestamps"] = torch.from_numpy(np.array(g.timestamps[lo:hi]))
        if g.edge_index is not None:
            sample.update(self.static_edges if g.is_static else self._dynamic_edges(lo, t))
        return self.transform(sample) if self.transform else sample

    def _dynamic_edges(self, lo: int, hi: int) -> Sample:
        ptr = np.array(self.g.edge_ptr[lo : hi + 1], dtype=np.int64)
        a, b = int(ptr[0]), int(ptr[-1])
        return {**_edges(self.edge_index[:, a:b], self.edge_weight[a:b]),
                "edge_ptr": torch.from_numpy(ptr - a)}


def _edges(edge_index: np.ndarray, edge_weight: np.ndarray) -> Sample:
    return {"edge_index": torch.from_numpy(np.array(edge_index, dtype=np.int64)),
            "edge_weight": torch.from_numpy(np.array(edge_weight, dtype=np.float32))}


def _scale(g: TemporalGraph, normalize: str) -> tuple[np.ndarray, np.ndarray]:
    key = "stats" if normalize == "channel" else "stats_node"
    if key not in g.meta:
        raise KeyError(f"meta has no {key!r}; compute it with tgdata.compute_stats")
    mean = np.asarray(g.meta[key]["mean"], dtype=np.float32)
    std = np.asarray(g.meta[key]["std"], dtype=np.float32)
    return mean, np.where(std > 0, std, 1.0).astype(np.float32)
