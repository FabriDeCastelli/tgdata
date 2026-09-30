from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from ..sampling import collate_concat
from ..schema import TemporalGraph, adjacency
from .base import Sample, Task, register_task, select_anchors


@register_task
class GraphClassification(Task):
    """Predict the graph target at step t from snapshots [t - window + 1, t].

    Nodes are those with an edge anywhere in the window, relabelled to 0..n-1 in id order;
    `node_ids` maps them back and `num_nodes_total` is the size of the full node set.
    """

    name = "graph_classification"
    collate = staticmethod(collate_concat)

    def __init__(
        self,
        g: TemporalGraph,
        target: str,
        window: int = 1,
        split: str | None = "train",
        splits: str = "default",
        strict: bool | None = None,
        transform: Callable[[Sample], Sample] | None = None,
        adjacency_kind: str | None = None,
    ) -> None:
        tgt = g.y[target]
        if not self.applies(g) or tgt.level != "graph" or tgt.static:
            raise ValueError(f"{target!r} is not a per-step graph target of a snapshot graph")
        self.g, self.window, self.transform = g, window, transform
        steps = np.arange(g.num_steps) if tgt.steps is None else np.asarray(tgt.steps)
        rows = np.flatnonzero(steps >= window - 1)
        anchors = select_anchors(g, steps[rows], split, splits, first=steps[rows] - window + 1,
                                 last=steps[rows], strict=strict)
        row_of = dict(zip(steps[rows].tolist(), rows.tolist(), strict=True))
        self.anchors = anchors
        self.values = np.asarray(tgt.values)[[row_of[int(t)] for t in anchors]]
        self.edge_index, self.edge_weight = adjacency(g, adjacency_kind)
        self.edge_ptr = np.asarray(g.edge_ptr)
        self._local = np.empty(g.num_nodes, dtype=np.int64)

    @classmethod
    def applies(cls, g: TemporalGraph) -> bool:
        return (g.time_mode == "discrete" and g.edge_ptr is not None
                and any(t.level == "graph" and not t.static for t in g.y.values()))

    def __getitem__(self, i: int) -> Sample:
        t = int(self.anchors[i])
        ptr = self.edge_ptr[t - self.window + 1 : t + 2].astype(np.int64)
        a, b = int(ptr[0]), int(ptr[-1])
        edges = np.asarray(self.edge_index[:, a:b], dtype=np.int64)
        node_ids = self._active_nodes(edges)
        self._local[node_ids] = np.arange(len(node_ids))
        sample: Sample = {
            "edge_index": torch.from_numpy(self._local[edges]),
            "edge_weight": torch.from_numpy(np.array(self.edge_weight[a:b], dtype=np.float32)),
            "edge_ptr": torch.from_numpy(ptr - a),
            "node_ids": torch.from_numpy(node_ids),
            "num_nodes_total": self.g.num_nodes,
            "y": torch.as_tensor(self.values[i]),
            "t": t,
        }
        if self.g.node_features is not None:
            feats = np.asarray(self.g.node_features[node_ids], dtype=np.float32)
            sample["node_features"] = torch.from_numpy(feats)
        if self.g.x is not None:
            x = np.asarray(self.g.x[t - self.window + 1 : t + 1][:, node_ids], dtype=np.float32)
            sample["x"] = torch.from_numpy(x)
        return self.transform(sample) if self.transform else sample

    def _active_nodes(self, edges: np.ndarray) -> np.ndarray:
        # O(E) dedup: whichever repeated write survives, exactly one position per node matches.
        touched = edges.ravel()
        position = np.arange(len(touched))
        self._local[touched] = position
        return np.sort(touched[self._local[touched] == position])
