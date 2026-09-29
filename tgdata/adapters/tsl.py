from __future__ import annotations

from typing import Any

import numpy as np

from ..schema import TemporalGraph, _mask_as_x, adjacency


def to_tsl(g: TemporalGraph, window: int = 12, horizon: int = 12,
           adjacency_kind: str | None = None, **kwargs: Any) -> Any:
    import pandas as pd
    from tsl.data import SpatioTemporalDataset

    if g.time_mode != "discrete" or not g.is_static:
        raise NotImplementedError("tsl needs a discrete graph with static edges")
    x = np.asarray(g.x, dtype=np.float32)
    mask = None if g.mask is None else np.broadcast_to(_mask_as_x(np.asarray(g.mask), x), x.shape)
    index = None if g.timestamps is None else pd.to_datetime(np.asarray(g.timestamps), unit="s")
    connectivity = None
    if g.edge_index is not None:
        ei, w = adjacency(g, adjacency_kind)
        connectivity = (np.asarray(ei), None if w is None else np.asarray(w))
    covariates = None if g.covariates is None else {"u": np.asarray(g.covariates, np.float32)}
    return SpatioTemporalDataset(target=x, index=index, mask=mask, connectivity=connectivity,
                                 covariates=covariates, window=window, horizon=horizon,
                                 name=g.name, **kwargs)
