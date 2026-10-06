import numpy as np
import pyarrow as pa
import pytest
import torch

import tgdata
from tgdata import Split
from tgdata.converters import mobins

DAYS, N = 80, 3


def tables(days=DAYS, n=N, seed=0):
    """A synthetic Epidemic-NYC-shaped release: daily steps, one channel."""
    rng = np.random.default_rng(seed)
    stamps = [str(np.datetime64("2020-03-01") + d) for d in range(days)]
    nodes = {"datetime": stamps} | {f"N{i}_INFECTION": rng.integers(0, 50, days).astype(float)
                                    for i in range(n)}
    od = {"datetime": stamps} | {f"N{i}_N{j}": rng.integers(0, 300, days)
                                 for i in range(n) for j in range(n)}
    adj = np.eye(n, dtype=int)
    adj[0, 1] = adj[1, 0] = 1
    network = {"INDEX": [f"N{i}" for i in range(n)]} | {f"N{j}": adj[:, j] for j in range(n)}
    return pa.table(nodes), pa.table(od), pa.table(network)


def mobins_windows(values, seq, pred, test_ratio=0.25, train_ratio=0.8):
    """MOBINS data_loader.py _make_windowing_and_loader, unscaled, on [days, dim] values."""
    test_start = len(values) - int(len(values) * test_ratio)

    def windows(part):
        return [(part[i:i + seq], part[i + seq:i + seq + pred])
                for i in range(len(part) - seq - pred + 1)]

    before, test = windows(values[:test_start]), windows(values[test_start:])
    cut = int(train_ratio * len(before))
    return {"train": before[:cut], "val": before[cut:], "test": test}


@pytest.fixture
def graph():
    return mobins.build("mobins-epi-nyc", *tables(), {"MOBINS.zip": "x"})


def test_layout(graph):
    assert graph.x.shape == (DAYS, N, 1) and graph.domain == "epidemic"
    assert graph.y["od"].level == "pair" and graph.y["od"].values.shape == (DAYS, N, N)
    assert graph.y["od"].values.dtype == np.int16
    assert sorted(map(tuple, graph.edge_index.T.tolist())) == [(0, 0), (0, 1), (1, 0), (1, 1),
                                                              (2, 2)]
    nodes, od, _ = tables()
    assert graph.y["od"].values[5, 1, 2] == od.column("N1_N2")[5].as_py()
    assert graph.x[7, 2, 0] == nodes.column("N2_INFECTION")[7].as_py()


def test_default_task_matches_mobins_windows(graph):
    nodes, od, _ = tables()
    values = np.concatenate([np.stack([t.column(i).to_numpy() for i in range(1, t.num_columns)],
                                      1) for t in (nodes, od)], 1).astype(np.float32)
    for pred_day in (7, 14):
        expected = mobins_windows(values, 4, pred_day)
        for split, pairs in expected.items():
            task = graph.task(split=split, horizon=f"{pred_day}D")
            batch = task.__getitems__(list(range(len(task))))
            assert batch["x_od"].shape == (len(pairs), 4, N, N)
            assert batch["y_od"].dtype == torch.float32
            x = torch.cat([batch["x"].flatten(2), batch["x_od"].flatten(2)], -1).numpy()
            y = torch.cat([batch["y"].flatten(2), batch["y_od"].flatten(2)], -1).numpy()
            np.testing.assert_array_equal(x, np.stack([p[0] for p in pairs]))
            np.testing.assert_array_equal(y, np.stack([p[1] for p in pairs]))


def test_hourly_windows_start_once_a_day():
    split = Split(fractions={"train": 0.8, "val": 0.2}, over="samples", holdout={"test": 0.25},
                  unit=24)
    assert split.holdout_start(730 * 24) == 548 * 24  # MOBINS Seoul: test from day 548


def test_holdout_and_pairs_survive_roundtrip_and_selection(graph, tmp_path):
    tgdata.save(graph, tmp_path)
    g = tgdata.load_dir(tmp_path)
    assert g.splits["default"].holdout == {"test": 0.25}
    np.testing.assert_array_equal(g.y["od"].values, graph.y["od"].values)
    sub = g.select_nodes(np.array([0, 2]))
    np.testing.assert_array_equal(sub.y["od"].values, graph.y["od"].values[:, [0, 2]][:, :, [0, 2]])
    tgdata.validate(sub)


def test_holdout_must_split_windows(graph):
    graph.splits["bad"] = Split(fractions={"train": 0.8, "val": 0.2}, holdout={"test": 0.25})
    with pytest.raises(ValueError, match="holdout"):
        tgdata.validate(graph)


def test_rejects_unexpected_columns():
    nodes, od, network = tables()
    with pytest.raises(ValueError, match="column order"):
        mobins.build("mobins-epi-nyc", nodes, od.select([0, 2, 1, *range(3, od.num_columns)]),
                     network, {})
