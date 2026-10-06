import math

import numpy as np
import pyarrow as pa
import pytest

import tgdata
from tgdata.converters import mint

T, DAYS = 40, 40 + 6 + 10  # snapshots, days with transfers (labels look 16 days past a start)
FIRST = np.datetime64("2021-03-01")


def release(token="IOTX", T=T, seed=0):
    """A synthetic MiNT network built the way ScalingTGNs/script/utils/TGS.py builds it."""
    rng = np.random.default_rng(seed)
    per_day = rng.integers(0, 6, DAYS + T)
    addr = [f"0x{i:040x}" for i in range(12)]
    rows = [(addr[rng.integers(12)], addr[rng.integers(12)], float(10 ** rng.integers(3, 28)),
             d) for d in range(len(per_day)) for _ in range(per_day[d])]
    day = np.array([r[3] for r in rows])
    cols = {k: [] for k in mint.COLUMNS}
    for t in range(1, T + 1):
        for i in np.flatnonzero((day >= t - 1) & (day < t - 1 + 7)):
            for k, v in zip(mint.COLUMNS, (*rows[i][:3], str(FIRST + int(day[i])), t), strict=True):
                cols[k].append(v)
    counts = np.bincount(day, minlength=len(per_day))
    labels = np.array([int(counts[t + 9:t + 16].sum() > counts[t - 1:t + 6].sum())
                       for t in range(1, T + 1)])
    edges = pa.table(cols, schema=pa.schema([("source", pa.string()), ("destination", pa.string()),
                                             ("weight", pa.float64()), ("date", pa.string()),
                                             ("snapshot", pa.int64())]))
    return edges, labels


@pytest.fixture(scope="module")
def made():
    edges, labels = release()
    return edges, labels, mint.build("IOTX", edges, labels, {"x": "y"})


def test_every_row_of_the_release_is_stored(made):
    edges, labels, g = made
    assert g.name == "mint-iotx" and g.meta["role"] == "train" and g.meta["train_rank"] == 1
    src, dst = g.edge_index
    np.testing.assert_array_equal(g.node_table["address"][src], edges.column("source").to_numpy())
    np.testing.assert_array_equal(g.node_table["address"][dst],
                                  edges.column("destination").to_numpy())
    np.testing.assert_array_equal(g.edge_weight, edges.column("weight").to_numpy()
                                  .astype(np.float32))
    snapshot = edges.column("snapshot").to_numpy()
    np.testing.assert_array_equal(np.repeat(np.arange(1, T + 1), np.diff(g.edge_ptr)), snapshot)
    np.testing.assert_array_equal(g.y["edge_gs"].values, labels)
    assert g.y["edge_gs"].values.dtype == np.uint8 and g.num_steps == T


def test_dates_are_recoverable_from_day_counts(made):
    edges, _, g = made
    days = np.repeat(np.arange(len(g.meta["day_counts"])), g.meta["day_counts"])
    dates = [str(np.datetime64(g.meta["first_date"]) + int(d)) for d in days]
    released = edges.column("date").to_pylist()
    for t in (1, 2, T):
        start = int(np.sum(g.meta["day_counts"][:t - 1]))
        size = int(g.edge_ptr[t] - g.edge_ptr[t - 1])
        assert released[g.edge_ptr[t - 1]:g.edge_ptr[t]] == dates[start:start + size]


def test_timestamps_are_window_starts(made):
    _, _, g = made
    assert g.timestamps[0] == int(FIRST.astype("datetime64[s]").astype(np.int64))
    assert (np.diff(g.timestamps) == 86400).all()


def mint_ranges(n, test_ratio=0.15, val_ratio=0.15):
    """ScalingTGNs train_foundation_tgc_64.py and test_foundation_tgc_64.py."""
    test, val = math.floor(n * test_ratio), math.floor(n * val_ratio)
    return {"train": range(0, n - test - val), "val": range(n - test - val, n - test),
            "test": range(n - test, n), "zero-shot": range(n - test - val, n)}


@pytest.mark.parametrize("n", [20, 100, 107, 260, 580])
def test_split_is_the_codes_not_70_15_15(n):
    ts = np.arange(n) * 86400
    expected = mint_ranges(n)
    for role in ("train", "test"):
        splits = mint._splits(ts, role)
        for name, key in (("default", ("train", "val", "test")), ("zero-shot", ("test",))):
            if name not in splits:
                assert role == "train"
                continue
            for part in key:
                lo, hi = splits[name].boundaries[part]
                got = [t for t in range(n) if lo <= ts[t] < hi]
                want = expected["zero-shot" if name == "zero-shot" else part]
                assert got == list(want)


def test_n107_differs_from_fractions():
    assert mint_ranges(107)["train"].stop == 75 != math.floor(0.7 * 107)


def test_graph_classification_yields_each_snapshot(made):
    edges, labels, g = made
    task = g.task()
    assert task.__class__.__name__ == "GraphClassification"
    for split in ("train", "val", "test"):
        assert len(task) and len(g.task(split=split))
    task = g.task(split="train")
    n_train = len(task)
    assert n_train == mint_ranges(T)["train"].stop
    sample = task[3]
    sl = slice(g.edge_ptr[3], g.edge_ptr[4])
    nodes = sample["node_ids"]
    np.testing.assert_array_equal(nodes[sample["edge_index"][0]], g.edge_index[0, sl])
    assert int(sample["y"]) == labels[3]


def test_zero_shot_only_on_test_networks():
    edges, labels = release()
    g = mint.build("MIR", edges, labels, {})
    assert g.meta["role"] == "test" and g.meta["train_rank"] is None
    assert set(g.splits) == {"default", "zero-shot"}
    assert "zero-shot" not in mint.build("IOTX", edges, labels, {}).splits


def test_roles_are_the_papers():
    assert len(mint.TRAIN) == 64 and len(mint.TEST) == 20
    assert not set(mint.TRAIN) & set(mint.TEST) and len(mint.ADDRESS) == 84
    assert mint.TRAIN[:2] == ("IOTX", "NOIA") and mint.TEST[1] == "DOGE2.0"


def test_roundtrip_and_card(made, tmp_path):
    _, _, g = made
    tgdata.save(g, tmp_path)
    back = tgdata.load_dir(tmp_path)
    np.testing.assert_array_equal(back.edge_weight, g.edge_weight)
    np.testing.assert_array_equal(back.y["edge_gs"].values, g.y["edge_gs"].values)
    assert back.splits["default"].boundaries == g.splits["default"].boundaries
    assert "- role:train" in (tmp_path / "README.md").read_text()


def test_rejects_a_release_that_is_not_the_sliding_window():
    edges, labels = release()
    with pytest.raises(ValueError, match="labels"):
        mint.build("IOTX", edges, labels[:-1], {})
    snap = edges.column("snapshot").to_numpy().copy()
    snap[snap == 5] = 6
    with pytest.raises(ValueError, match="contiguous|outside|disagree"):
        mint.build("IOTX", edges.set_column(4, "snapshot", pa.array(snap)), labels, {})


def test_registry_lists_the_papers_networks_by_role():
    train = tgdata.list(domain="transaction", role="train", offline=True)
    test = tgdata.list(domain="transaction", role="test", offline=True)
    assert train == sorted(mint.name_of(t) for t in mint.TRAIN)
    assert test == sorted(mint.name_of(t) for t in mint.TEST)
    assert tgdata.list(role="train", offline=True) == train
