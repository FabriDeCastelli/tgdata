import numpy as np
import pytest
import torch

import tgdata
from tgdata import Split, Target
from tgdata.time import to_steps


def test_split_resolve_floors_cumulatively():
    split = Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2})
    assert split.resolve(17833) == {"train": (0, 10699), "val": (10699, 14266),
                                    "test": (14266, 17833)}
    split = Split(fractions={"train": 0.7, "val": 0.1, "test": 0.2})
    assert split.resolve(100) == {"train": (0, 70), "val": (70, 80), "test": (80, 100)}


def test_with_split(static_graph):
    g = static_graph.with_split("70/10/20")
    assert g.splits["default"].resolve(50)["train"] == (0, 35)
    g = static_graph.with_split(val_fraction=0.1, test_fraction=0.1)
    assert g.splits["default"].resolve(50)["test"] == (45, 50)


def test_durations_become_whole_steps():
    assert to_steps("1h", "5min") == 12 and to_steps(12, "5min") == 12
    assert to_steps("2Y", "1Y") == 2 and to_steps("1W", "1D") == 7
    for bad in ("7min", "1M"):
        with pytest.raises(ValueError):
            to_steps(bad, "5min")


def test_window_counts_and_boundaries(static_graph):
    start, end = static_graph.splits["default"].resolve(50)["test"]
    ds = static_graph.window(4, 3, "test")
    assert len(ds) == end - start - 3 + 1
    assert ds[0]["t"] == start
    strict = static_graph.window(4, 3, "test", strict=True)
    assert len(strict) == len(ds) - 4
    assert strict[0]["t"] == start + 4
    assert static_graph.window(4, 3, "train")[0]["t"] == 4
    assert len(static_graph.window("20min", "15min", "test")) == len(ds)


def test_split_over_samples_cuts_the_sample_list(static_graph):
    static_graph.splits["default"] = Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2},
                                           over="samples")
    parts = [static_graph.window(4, 3, s).anchors for s in ("train", "val", "test")]
    all_anchors = np.arange(4, 50 - 3 + 1)
    np.testing.assert_array_equal(np.concatenate(parts), all_anchors)
    assert [len(p) for p in parts] == [26, 9, 9]


def test_split_by_dates(static_graph):
    static_graph.splits["default"] = Split(boundaries={"train": (0, 9000), "test": (9000, 15000)})
    anchors = static_graph.window(4, 3, "test").anchors
    assert anchors[0] == 30 and anchors[-1] == 47


def test_default_task(static_graph):
    task = static_graph.task()
    assert (task.window, task.horizon) == (4, 3)
    assert tgdata.tasks.available(static_graph) == ["node_forecasting"]


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
    values = np.arange(50 * 6, dtype=np.float32).reshape(50, 6, 1)
    static_graph.y["demand"] = Target(level="node", kind="regression", values=values)
    s = static_graph.window(4, 3, target="demand")[0]
    np.testing.assert_array_equal(s["y"], values[s["t"] : s["t"] + 3])


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


def test_anchors_match_v1_windows(static_graph):
    bounds = static_graph.splits["default"].resolve(50)
    for split, (start, end) in bounds.items():
        for strict in (False, True):
            first = start + 4 if strict else max(start, 4)
            v1 = np.arange(first, end - 3 + 1)
            task = static_graph.window(4, 3, split, strict=strict)
            np.testing.assert_array_equal(task.anchors, v1)


def test_samples_from_disk_are_writable(static_graph, tmp_path):
    tgdata.save(static_graph, tmp_path)
    s = tgdata.load_dir(tmp_path).window(4, 3)[0]
    for key in ("x", "y", "mask_x", "mask_y", "covariates", "edge_index", "edge_weight"):
        s[key].copy_(s[key].clone())
