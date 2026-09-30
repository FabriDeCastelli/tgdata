from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import torch

from ..device import on_device, resolve_device, window_edges
from ..schema import TemporalGraph
from ..transforms import FEATURES, TARGETS
from .base import Sample, Task, passthrough, register_task, select_anchors

BATCHED = ("x", "y", "t")


@register_task
class NodeRegression(Task):
    """Predict a stored node target from the window of snapshots [s - window + 1, s].

    The target is the one at step min(s + offset, T - 1): PyTorch Geometric Temporal pairs
    snapshot s with the label of s + offset and reuses the last label at the end, and this
    reproduces that pairing sample for sample. `features` and `target_transform` name
    preprocessing from `tgdata.transforms`, applied per batch.
    """

    name = "node_regression"
    collate = staticmethod(passthrough)

    def __init__(
        self,
        g: TemporalGraph,
        target: str,
        window: int = 1,
        offset: int = 0,
        split: str | None = "train",
        splits: str = "default",
        strict: bool | None = None,
        features: str | None = None,
        target_transform: str | None = None,
        transform: Callable[[Sample], Sample] | None = None,
        adjacency_kind: str | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        tgt = g.y[target]
        if not self.applies(g) or tgt.level != "node" or tgt.static or tgt.steps is not None:
            raise ValueError(f"{target!r} is not a per-step node target with node signals")
        self.g, self.window, self.transform = g, window, transform
        last_step = g.num_steps - 1
        candidates = np.arange(window - 1, g.num_steps)
        target_steps = np.minimum(candidates + offset, last_step)
        self.anchors = select_anchors(g, candidates, split, splits,
                                      first=candidates - window + 1,
                                      last=target_steps, strict=strict)
        self.device = resolve_device(device)
        self.data = on_device(g, self.device, adjacency_kind)
        self.anchor_steps = torch.as_tensor(self.anchors, device=self.device)
        self.target_steps = torch.as_tensor(np.minimum(self.anchors + offset, last_step),
                                            device=self.device)
        self.target = torch.as_tensor(np.array(tgt.values), device=self.device)
        self.features = None if features is None else FEATURES[features]
        self.target_transform = None if target_transform is None else TARGETS[target_transform]
        self.offsets = torch.arange(-window + 1, 1, device=self.device)

    @classmethod
    def applies(cls, g: TemporalGraph) -> bool:
        return (g.time_mode == "discrete" and g.x is not None
                and any(t.level == "node" and not t.static for t in g.y.values()))

    @property
    def window_starts(self) -> np.ndarray:
        return self.anchors - self.window + 1

    def __getitems__(self, indices: Sequence[int]) -> Sample:
        data = self.data
        idx = torch.as_tensor(indices, device=self.device)
        s = self.anchor_steps[idx]
        raw = data["x"][s[:, None] + self.offsets]  # [B, window, N, F]
        values = raw.to(torch.float64)
        if data.get("x_divisor") is not None:
            values = values / data["x_divisor"].to(torch.float64)
        x = values.to(torch.float32) if self.features is None else self.features(values)
        y = self.target[self.target_steps[idx]]
        y = y.to(torch.float32) if self.target_transform is None else self.target_transform(y)
        batch: Sample = {"x": x, "y": y, "t": s}
        if "edge_ptr" in data:
            batch.update(window_edges(data, s - self.window + 1, self.window, self.g.num_nodes))
        elif "edge_index" in data:
            batch.update(edge_index=data["edge_index"], edge_weight=data["edge_weight"])
        return self.transform(batch) if self.transform else batch

    def __getitem__(self, i: int) -> Sample:
        batch = self.__getitems__([i])
        sample = {k: v[0] if k in BATCHED else v for k, v in batch.items()}
        sample["t"] = int(sample["t"])
        return sample
