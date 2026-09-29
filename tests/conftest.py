import os

import numpy as np
import pytest

# Tests compare batches with numpy arrays; GPU tests pass `device` explicitly.
os.environ.setdefault("TGDATA_DEVICE", "cpu")

from tgdata import Split, Target, TemporalGraph, compute_stats

T, N, F = 50, 6, 2
SPLIT_60_20_20 = Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2})


@pytest.fixture
def static_graph() -> TemporalGraph:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(T, N, F)).astype(np.float32)
    mask = rng.random((T, N)) > 0.1
    g = TemporalGraph(
        name="toy_static", domain="synthetic", tasks=["node_forecasting"], time_mode="discrete",
        x=x, mask=mask,
        timestamps=np.arange(T, dtype=np.int64) * 300,
        covariates=rng.normal(size=(T, 3)).astype(np.float32),
        edge_index=np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64),
        edge_weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        splits={"default": SPLIT_60_20_20,
                "70/10/20": Split(fractions={"train": 0.7, "val": 0.1, "test": 0.2})},
        meta={"freq": "5min", "edge_weight": {"kind": "distance", "observed": True},
              "default_task": {"name": "node_forecasting",
                               "params": {"window": 4, "horizon": 3}}},
    )
    g.meta["stats"] = compute_stats(g)
    g.meta["stats_node"] = compute_stats(g, per_node=True)
    return g


@pytest.fixture
def dynamic_graph() -> TemporalGraph:
    rng = np.random.default_rng(1)
    counts = rng.integers(0, 5, size=T)
    E = int(counts.sum())
    return TemporalGraph(
        name="toy_dynamic", domain="synthetic", tasks=["node_forecasting"], time_mode="discrete",
        x=rng.normal(size=(T, N, F)).astype(np.float32),
        edge_index=rng.integers(0, N, size=(2, E)).astype(np.int64),
        edge_weight=rng.random(E).astype(np.float32),
        edge_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        splits={"default": SPLIT_60_20_20},
        meta={"freq": "1h", "edge_weight": {"kind": "interaction_count", "observed": True}},
    )


@pytest.fixture
def snapshot_graph() -> TemporalGraph:
    """Featureless snapshots over 100 potential nodes, most of them inactive at any step."""
    rng = np.random.default_rng(3)
    steps, n = 20, 100
    counts = rng.integers(1, 6, size=steps)
    E = int(counts.sum())
    return TemporalGraph(
        name="toy_snapshots", domain="synthetic", tasks=["graph_classification"],
        time_mode="discrete",
        edge_index=rng.integers(0, n, size=(2, E)).astype(np.int64),
        edge_weight=np.ones(E, dtype=np.float32),
        edge_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        timestamps=np.arange(steps, dtype=np.int64) * 86400,
        y={"growth": Target(level="graph", kind="class", num_classes=2,
                            values=rng.integers(0, 2, size=steps).astype(np.int64))},
        splits={"default": Split(fractions={"train": 0.7, "val": 0.15, "test": 0.15},
                                 over="samples")},
        meta={"freq": "1D", "num_nodes": n,
              "edge_weight": {"kind": "transfer_count", "observed": True},
              "default_task": {"name": "graph_classification",
                               "params": {"target": "growth", "window": 1}}},
    )


@pytest.fixture
def continuous_graph() -> TemporalGraph:
    rng = np.random.default_rng(2)
    n = 100
    return TemporalGraph(
        name="toy_events", domain="synthetic", tasks=["link_prediction"], time_mode="continuous",
        src=rng.integers(0, N, n).astype(np.int64),
        dst=rng.integers(0, N, n).astype(np.int64),
        t=np.sort(rng.integers(0, 10_000, n)).astype(np.int64),
        msg=rng.normal(size=(n, 4)).astype(np.float32),
        node_features=rng.normal(size=(N, 3)).astype(np.float32),
        splits={"default": Split(fractions={"train": 0.7, "val": 0.15, "test": 0.15})},
    )
