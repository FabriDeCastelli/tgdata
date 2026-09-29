from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..schema import TemporalGraph, adjacency


def to_pyg(g: TemporalGraph, adjacency_kind: str | None = None) -> Any:
    """Static graphs become one Data with x [T, N, F]; dynamic graphs a list of per-step Data."""
    from torch_geometric.data import Data

    if g.time_mode != "discrete":
        raise NotImplementedError("continuous graphs: use to_tgb")
    x = torch.from_numpy(np.array(g.x, dtype=np.float32))
    mask = None if g.mask is None else torch.from_numpy(np.array(g.mask))
    ei, w = adjacency(g, adjacency_kind) if g.edge_index is not None else (None, None)
    if g.is_static:
        return Data(x=x, mask=mask, num_nodes=g.num_nodes, **_edges(ei, w))
    ptr = g.edge_ptr
    return [
        Data(x=x[step], mask=None if mask is None else mask[step], num_nodes=g.num_nodes,
             **_edges(ei[:, ptr[step] : ptr[step + 1]],
                      None if w is None else w[ptr[step] : ptr[step + 1]]))
        for step in range(g.num_steps)
    ]


def _edges(ei: np.ndarray | None, w: np.ndarray | None) -> dict[str, Any]:
    return {
        "edge_index": None if ei is None else torch.from_numpy(np.array(ei, dtype=np.int64)),
        "edge_weight": None if w is None else torch.from_numpy(np.array(w, dtype=np.float32)),
    }
