import dataclasses

import numpy as np
import pytest

import tgdata
from tgdata import Split, Target

from .conftest import SPLIT_60_20_20


def assert_same(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    elif dataclasses.is_dataclass(a):
        for f in dataclasses.fields(a):
            assert_same(getattr(a, f.name), getattr(b, f.name))
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    else:
        assert a == b


@pytest.mark.parametrize(
    "fixture", ["static_graph", "dynamic_graph", "snapshot_graph", "continuous_graph"]
)
def test_save_load_roundtrip(fixture, request, tmp_path):
    g = request.getfixturevalue(fixture)
    tgdata.save(g, tmp_path)
    h = tgdata.load_dir(tmp_path)
    tgdata.validate(h)
    for f in dataclasses.fields(g):
        if f.name != "meta":
            assert_same(getattr(g, f.name), getattr(h, f.name))
    assert h.meta.keys() == g.meta.keys()


def test_node_split_and_static_target_roundtrip(snapshot_graph, tmp_path):
    g = snapshot_graph
    g.splits["by_node"] = Split(nodes={"train": np.arange(0, 60), "test": np.arange(60, 100)})
    g.y["field"] = Target(level="node", kind="class", static=True, num_classes=3,
                          values=np.arange(10) % 3, index=np.arange(0, 100, 10))
    tgdata.save(g, tmp_path)
    h = tgdata.load_dir(tmp_path)
    assert_same(g.splits, h.splits)
    assert_same(g.y, h.y)


def test_discrete_arrays_are_memory_mapped(static_graph, tmp_path):
    tgdata.save(static_graph, tmp_path)
    assert isinstance(tgdata.load_dir(tmp_path).x, np.memmap)


def test_card_has_hub_tags(static_graph, tmp_path):
    tgdata.save(static_graph, tmp_path)
    card = (tmp_path / "README.md").read_text()
    for tag in ("domain:synthetic", "time:discrete", "task:node_forecasting"):
        assert f"- {tag}\n" in card


@pytest.mark.parametrize(
    "change, message",
    [
        ({"splits": {"train": SPLIT_60_20_20}}, "'default'"),
        ({"splits": {"default": Split(fractions={"train": 0.7, "test": 0.2})}}, "sum to 1"),
        ({"splits": {"default": Split(boundaries={"train": (0, 30), "val": (20, 40)})}},
         "non-overlapping"),
        ({"splits": {"default": Split()}}, "exactly one"),
        ({"y": {"bad": Target(level="graph", kind="class", values=np.zeros(3, np.int64),
                              num_classes=2)}}, "steps"),
        ({"y": {"bad": Target(level="node", kind="class", values=np.zeros((50, 6)),
                              num_classes=2)}}, "integer values"),
        ({"meta": {}}, "meta.freq"),
        ({"mask": np.ones((50, 6), dtype=np.float32)}, "mask must be bool"),
        ({"edge_ptr": np.zeros(51, dtype=np.int64)}, "edge_ptr"),
        ({"edge_index": np.array([[0], [9]])}, "outside"),
        ({"timestamps": np.zeros(50, dtype=np.int64)}, "strictly increasing"),
        ({"x": np.zeros((50, 6), dtype=np.float32)}, "[T, N, F]"),
    ],
)
def test_validate_rejects(static_graph, change, message):
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        tgdata.validate(dataclasses.replace(static_graph, **change))


def test_validate_rejects_unsorted_events(continuous_graph):
    g = dataclasses.replace(continuous_graph, t=continuous_graph.t[::-1].copy())
    with pytest.raises(ValueError, match="not sorted"):
        tgdata.validate(g)


def test_offline_registry():
    traffic = tgdata.list(domain="traffic_flow", offline=True)
    assert traffic == ["pems03", "pems04", "pems07", "pems08"]
    assert tgdata.info("pems08", offline=True)["time_mode"] == "discrete"


def test_featureless_graph_needs_num_nodes(snapshot_graph):
    g = dataclasses.replace(snapshot_graph, meta={k: v for k, v in snapshot_graph.meta.items()
                                                  if k != "num_nodes"})
    with pytest.raises(ValueError, match="num_nodes"):
        tgdata.validate(g)
