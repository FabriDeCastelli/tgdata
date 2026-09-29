from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import TemporalGraph, _mask_as_x, adjacency

Sample = dict[str, Any]


class WindowDataset(Dataset):
    """Target steps lie inside `split`; inputs may reach into earlier steps unless `strict`."""

    def __init__(
        self,
        g: TemporalGraph,
        in_len: int,
        out_len: int,
        split: str = "train",
        stride: int = 1,
        strict: bool = False,
        normalize: Literal["channel", "node"] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
        adjacency_kind: str | None = None,
    ) -> None:
        if g.time_mode != "discrete":
            raise ValueError("windowing needs a discrete-time graph")
        self.g, self.in_len, self.out_len = g, in_len, out_len
        self.transform = transform
        start, end = g.splits[split]
        first_target = start + in_len if strict else max(start, in_len)
        self.targets = np.arange(first_target, end - out_len + 1, stride)
        self.scale = _scale(g, normalize) if normalize else None
        self.target_channels = g.meta.get("target_channels")
        if g.edge_index is not None:
            self.edge_index, self.edge_weight = adjacency(g, adjacency_kind)
            if g.is_static:
                self.static_edges = _edges(self.edge_index, self.edge_weight)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> Sample:
        g, t = self.g, int(self.targets[i])
        lo, hi = t - self.in_len, t + self.out_len
        x = np.asarray(g.x[lo:hi], dtype=np.float32)
        if self.scale is not None:
            x = (x - self.scale[0]) / self.scale[1]
        if g.y is not None:
            y = np.asarray(g.y[t:hi])
        elif self.target_channels is not None:
            y = x[self.in_len:, :, self.target_channels]
        else:
            y = x[self.in_len:]
        sample: Sample = {"x": torch.from_numpy(x[: self.in_len].copy()),
                          "y": torch.from_numpy(np.ascontiguousarray(y)), "t": t}
        if g.mask is not None:
            mask = torch.from_numpy(_mask_as_x(np.asarray(g.mask[lo:hi]), x).copy())
            sample["mask_x"] = mask[: self.in_len]
            mask_y = mask[self.in_len:]
            if g.y is None and self.target_channels is not None and mask.shape[-1] > 1:
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
        weight = None if self.edge_weight is None else self.edge_weight[a:b]
        return {**_edges(self.edge_index[:, a:b], weight),
                "edge_ptr": torch.from_numpy(ptr - a)}


def _edges(edge_index: np.ndarray, edge_weight: np.ndarray | None) -> Sample:
    out: Sample = {"edge_index": torch.from_numpy(np.array(edge_index, dtype=np.int64))}
    if edge_weight is not None:
        out["edge_weight"] = torch.from_numpy(np.array(edge_weight, dtype=np.float32))
    return out


def _scale(g: TemporalGraph, normalize: str) -> tuple[np.ndarray, np.ndarray]:
    key = "stats" if normalize == "channel" else "stats_node"
    if key not in g.meta:
        raise KeyError(f"meta has no {key!r}; compute it with tgdata.compute_stats")
    mean = np.asarray(g.meta[key]["mean"], dtype=np.float32)
    std = np.asarray(g.meta[key]["std"], dtype=np.float32)
    return mean, np.where(std > 0, std, 1.0).astype(np.float32)
