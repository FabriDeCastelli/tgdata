from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
import torch

from ..device import decode, on_device, resolve_device
from ..schema import TemporalGraph
from ..time import to_steps
from .base import Sample, Task, passthrough, register_task, select_anchors

# Keys whose first dimension is the batch; the rest (static edges) are shared by the batch.
BATCHED = ("x", "y", "mask_x", "mask_y", "covariates", "timestamps", "t")


@register_task
class NodeForecasting(Task):
    """Predict steps [t, t + horizon) of `x` (or of node target `target`) from [t - window, t).

    Batches are cut on `device` (CUDA when available) from arrays uploaded once, by
    `__getitems__`, which torch's DataLoader calls with a whole batch of indices.
    """

    name = "node_forecasting"
    collate = staticmethod(passthrough)

    def __init__(
        self,
        g: TemporalGraph,
        window: int | str,
        horizon: int | str,
        split: str = "train",
        splits: str = "default",
        stride: int = 1,
        strict: bool | None = None,
        target: str | None = None,
        normalize: Literal["channel", "node"] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
        adjacency_kind: str | None = None,
        device: str | torch.device | None = None,
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
        self.device = resolve_device(device)
        self.data = on_device(g, self.device, adjacency_kind)
        self.anchor_steps = torch.as_tensor(self.anchors, device=self.device)
        self.offsets = torch.arange(-self.window, self.horizon, device=self.device)
        self.target = None if target is None else torch.as_tensor(
            np.asarray(g.y[target].values), device=self.device)
        channels = g.meta.get("target_channels")
        self.target_channels = None if channels is None else torch.as_tensor(
            channels, device=self.device)
        self.scale = None
        if normalize:
            mean, std = _scale(g, normalize)
            self.scale = (torch.from_numpy(mean).to(self.device),
                          torch.from_numpy(std).to(self.device))

    @classmethod
    def applies(cls, g: TemporalGraph) -> bool:
        return g.time_mode == "discrete" and g.x is not None

    def __getitems__(self, indices: Sequence[int]) -> Sample:
        data, w = self.data, self.window
        t = self.anchor_steps[torch.as_tensor(indices, device=self.device)]
        rows = t[:, None] + self.offsets  # [B, window + horizon]
        raw = data["x"][rows]  # [B, window + horizon, N, F], stored dtype
        x = decode(raw, data.get("x_divisor"))
        if self.scale is not None:
            x = (x - self.scale[0]) / self.scale[1]
        if self.target is not None:
            y = self.target[rows[:, w:]]
        elif self.target_channels is not None:
            y = x[:, w:, :, self.target_channels]
        else:
            y = x[:, w:]
        batch: Sample = {"x": x[:, :w], "y": y, "t": t}
        if data["mask_derived"] or "mask" in data:
            mask = raw[..., 0] != 0 if data["mask_derived"] else data["mask"][rows]
            mask = mask.unsqueeze(-1) if mask.ndim == 3 else mask
            mask_y = mask[:, w:]
            if self.target is None and self.target_channels is not None and mask.shape[-1] > 1:
                mask_y = mask_y[..., self.target_channels]
            batch["mask_x"], batch["mask_y"] = mask[:, :w], mask_y
        if "covariates" in data:
            batch["covariates"] = decode(data["covariates"][rows], data.get("covariates_divisor"))
        if "timestamps" in data:
            batch["timestamps"] = data["timestamps"][rows]
        if "edge_index" in data:
            batch.update(self._edges(rows))
        return self.transform(batch) if self.transform else batch

    def __getitem__(self, i: int) -> Sample:
        batch = self.__getitems__([i])
        sample = {k: v[0] if k in BATCHED else v for k, v in batch.items()}
        if "edges" in sample:
            sample.update(sample.pop("edges")[0])
        sample["t"] = int(sample["t"])
        return sample

    def _edges(self, rows: torch.Tensor) -> Sample:
        data = self.data
        if "edge_ptr" not in data:
            return {"edge_index": data["edge_index"], "edge_weight": data["edge_weight"]}
        ptr, w = data["edge_ptr"], self.window
        edges = []
        for lo, hi in zip(rows[:, 0].tolist(), rows[:, w].tolist(), strict=True):
            a, b = int(ptr[lo]), int(ptr[hi])
            edges.append({"edge_index": data["edge_index"][:, a:b],
                          "edge_weight": data["edge_weight"][a:b],
                          "edge_ptr": ptr[lo:hi + 1] - a})
        return {"edges": edges}


def _scale(g: TemporalGraph, normalize: str) -> tuple[np.ndarray, np.ndarray]:
    key = "stats" if normalize == "channel" else "stats_node"
    if key not in g.meta:
        raise KeyError(f"meta has no {key!r}; compute it with tgdata.compute_stats")
    mean = np.asarray(g.meta[key]["mean"], dtype=np.float32)
    std = np.asarray(g.meta[key]["std"], dtype=np.float32)
    return mean, np.where(std > 0, std, 1.0).astype(np.float32)
