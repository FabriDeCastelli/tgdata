import numpy as np
import pytest
import torch

from tgdata import make_splits


def test_make_splits_fractions():
    assert make_splits(100, 0.1, 0.2) == {"train": (0, 70), "val": (70, 80), "test": (80, 100)}
    assert make_splits(100, 0.0, 0.2) == {"train": (0, 80), "test": (80, 100)}
    with pytest.raises(ValueError):
        make_splits(10, 0.5, 0.5)


def test_with_splits(static_graph):
    assert static_graph.with_splits("70/10/20").splits["train"] == (0, 35)
    assert static_graph.with_splits(val_fraction=0.1, test_fraction=0.1).splits["test"] == (45, 50)


def test_window_counts_and_boundaries(static_graph):
    start, end = static_graph.splits["test"]
    ds = static_graph.window(4, 3, "test")
    assert len(ds) == end - start - 3 + 1
    assert ds[0]["t"] == start
    strict = static_graph.window(4, 3, "test", strict=True)
    assert len(strict) == len(ds) - 4
    assert strict[0]["t"] == start + 4
    assert static_graph.window(4, 3, "train")[0]["t"] == 4


def test_window_contents(static_graph):
    s = static_graph.window(4, 3, "val", stride=2)[1]
    t = s["t"]
    assert s["x"].shape == (4, 6, 2) and s["y"].shape == (3, 6, 2)
    np.testing.assert_array_equal(s["x"], static_graph.x[t - 4 : t])
    np.testing.assert_array_equal(s["y"], static_graph.x[t : t + 3])
    np.testing.assert_array_equal(s["mask_y"][..., 0], static_graph.mask[t : t + 3])
    assert s["covariates"].shape == (7, 3)
    assert s["edge_index"].shape == (2, 5)


def test_target_channels_and_stored_y(static_graph):
    static_graph.meta["target_channels"] = [1]
    s = static_graph.window(4, 3)[0]
    np.testing.assert_array_equal(s["y"][..., 0], static_graph.x[s["t"] : s["t"] + 3, :, 1])
    static_graph.y = np.arange(50 * 6).reshape(50, 6, 1)
    s = static_graph.window(4, 3)[0]
    np.testing.assert_array_equal(s["y"], static_graph.y[s["t"] : s["t"] + 3])


@pytest.mark.parametrize("mode", ["channel", "node"])
def test_normalize(static_graph, mode):
    key = "stats" if mode == "channel" else "stats_node"
    mean = np.asarray(static_graph.meta[key]["mean"])
    std = np.asarray(static_graph.meta[key]["std"])
    s = static_graph.window(4, 3, normalize=mode)[0]
    t = s["t"]
    expected = (static_graph.x[t - 4 : t] - mean) / std
    np.testing.assert_allclose(s["x"], expected, rtol=1e-5, atol=1e-6)


def test_transform_hook(static_graph):
    ds = static_graph.window(4, 3, transform=lambda s: {**s, "x": s["x"] * 0})
    assert torch.count_nonzero(ds[0]["x"]) == 0


def test_dynamic_edges_are_sliced_per_input_step(dynamic_graph):
    s = dynamic_graph.window(4, 2, "val")[0]
    t = s["t"]
    ptr = dynamic_graph.edge_ptr
    assert s["edge_ptr"].tolist() == (ptr[t - 4 : t + 1] - ptr[t - 4]).tolist()
    np.testing.assert_array_equal(s["edge_index"], dynamic_graph.edge_index[:, ptr[t - 4] : ptr[t]])
    last_step, _ = dynamic_graph.edges_at(t - 1)
    np.testing.assert_array_equal(s["edge_index"][:, s["edge_ptr"][-2] :], last_step)


def test_continuous_graph_cannot_be_windowed(continuous_graph):
    with pytest.raises(ValueError):
        continuous_graph.window(4, 2)
