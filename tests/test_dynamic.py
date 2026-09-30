import numpy as np
import pytest
import torch

from tgdata import Split, Target, TemporalGraph
from tgdata.converters import twittertennis
from tgdata.device import window_edges
from tgdata.transforms import log1p, pygt_encoded


def per_sample_edges(data, first, window, num_nodes):
    """The per-sample list construction, merged into the same step-major layout."""
    ptr, parts, weights, counts = data["edge_ptr"].tolist(), [], [], []
    for k in range(window):
        n = 0
        for b, s in enumerate(first.tolist()):
            lo, hi = ptr[s + k], ptr[s + k + 1]
            parts.append(data["edge_index"][:, lo:hi] + b * num_nodes)
            weights.append(data["edge_weight"][lo:hi])
            n += hi - lo
        counts.append(n)
    return {"edge_index": torch.cat(parts, 1), "edge_weight": torch.cat(weights),
            "edge_ptr": torch.tensor([0, *np.cumsum(counts)])}


@pytest.mark.parametrize("window", [1, 3, 7])
def test_window_edges_match_the_per_sample_construction(window):
    rng = np.random.default_rng(window)
    counts = rng.integers(0, 6, size=20)  # includes empty snapshots
    ptr = torch.as_tensor(np.concatenate([[0], np.cumsum(counts)]))
    data = {"edge_ptr": ptr, "edge_index": torch.randint(9, (2, int(ptr[-1]))),
            "edge_weight": torch.rand(int(ptr[-1]))}
    first = torch.as_tensor(rng.integers(0, 20 - window + 1, size=5))
    fast, slow = window_edges(data, first, window, 9), per_sample_edges(data, first, window, 9)
    for key in fast:
        assert torch.equal(fast[key], slow[key]), key


def tennis_json(T=6, N=5):
    rng = np.random.default_rng(0)
    data = {"time_periods": T, "node_ids": {f"acct{i}": i for i in range(N)}}
    for t in range(T):
        e = rng.integers(0, N, size=(rng.integers(1, 5), 2))
        data[str(t)] = {"index": t, "edges": e.tolist(),
                        "weights": rng.integers(1, 4, size=len(e)).tolist(),
                        "X": np.stack([rng.integers(0, 60, N),
                                       rng.integers(0, 11, N) / 10], 1).tolist(),
                        "y": rng.integers(0, 100, N).tolist()}
    return data


def test_twittertennis_default_task_pairs_snapshots_like_pyg():
    g = twittertennis.build("rg17", tennis_json(), {"url": "x"})
    task = g.task(split=None)
    assert len(task) == 6
    raw_y = np.asarray(g.y["mentions"].values)
    for t in range(6):
        s = task[t]
        assert s["x"].shape == (1, 5, 16)
        target_step = min(t + 1, 5)                          # PyG: min(time + offset, T - 1)
        np.testing.assert_array_equal(s["y"], np.log(1.0 + raw_y[target_step]).astype(np.float32))
        lo, hi = g.edge_ptr[t], g.edge_ptr[t + 1]
        np.testing.assert_array_equal(s["edge_index"], g.edge_index[:, lo:hi])
    assert [len(g.task(split=p)) for p in ("train", "val", "test")] == [4, 1, 1]


def test_pygt_encoding_bins_in_float64():
    x = torch.tensor([[0.0, 0.7], [1.0, 1.0], [300.0, 0.3]], dtype=torch.float64)
    out = pygt_encoded(x)
    assert out.dtype == torch.float32 and out.shape == (3, 16)
    assert out[0, :5].argmax() == 0 and out[0, 5:].argmax() == 7      # float32 0.7 would give 6
    assert out[1, :5].argmax() == 1 and out[1, 5:].argmax() == 10
    assert out[2, :5].argmax() == 4 and out[2, 5:].argmax() == 3      # capped at the cutoff
    assert torch.equal(log1p(torch.tensor([0, 9])), torch.tensor([0.0, np.log(10.0)]).float())


def test_node_regression_windows_and_positions():
    T, N = 10, 3
    g = TemporalGraph(
        name="toy", domain="synthetic", tasks=["node_regression"], time_mode="discrete",
        x=np.arange(T * N, dtype=np.float64).reshape(T, N, 1),
        edge_index=np.zeros((2, 0), np.int64), edge_weight=np.zeros(0, np.float32),
        edge_ptr=np.zeros(T + 1, np.int64),
        y={"v": Target(level="node", kind="regression", values=np.arange(T * N).reshape(T, N))},
        splits={"default": Split(fractions={"train": 0.8, "val": 0.1, "test": 0.1},
                                 over="samples")},
        meta={"freq": "1h", "edge_weight": {"kind": "none", "observed": True}})
    task = g.task("node_regression", target="v", window=3, offset=2, split=None)
    assert len(task) == T - 2
    s = task[len(task) - 1]                                   # window ends at the last step
    np.testing.assert_array_equal(s["x"][:, :, 0], np.asarray(g.x)[7:10, :, 0])
    np.testing.assert_array_equal(s["y"], g.y["v"].values[9])  # clamped to the last step
    np.testing.assert_array_equal(task.window_starts, np.arange(T - 2))
