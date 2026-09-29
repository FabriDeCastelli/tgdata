import numpy as np
import pytest

import tgdata
from tgdata import Split, Target, compute_stats, registry


def test_contiguous_selection_is_a_view_with_induced_edges(static_graph, tmp_path):
    tgdata.save(static_graph, tmp_path)
    g = tgdata.load_dir(tmp_path)
    sub = g.select_nodes(np.array([1, 2, 3]))
    assert np.shares_memory(sub.x, g.x) and sub.x.shape == (50, 3, 2)
    assert sub.edge_index.tolist() == [[0, 1], [1, 2]]
    np.testing.assert_array_equal(sub.edge_weight, [2.0, 3.0])
    tgdata.validate(sub)


def test_named_set_copies_only_its_columns(static_graph):
    static_graph.node_sets = {"ends": np.array([0, 5])}
    sub = static_graph.select_nodes("ends")
    assert sub.name == "toy_static-ends" and sub.num_nodes == 2
    np.testing.assert_array_equal(sub.x, static_graph.x[:, [0, 5]])
    assert sub.edge_index.shape == (2, 0) and sub.node_sets == {}


def test_pooled_stats_equal_recomputed_stats(static_graph):
    sub = static_graph.select_nodes(np.array([0, 2, 3]))
    direct = compute_stats(sub)
    np.testing.assert_allclose(sub.meta["stats"]["mean"], direct["mean"], rtol=1e-10)
    np.testing.assert_allclose(sub.meta["stats"]["std"], direct["std"], rtol=1e-10)
    assert sub.meta["stats"]["count"] == direct["count"]
    np.testing.assert_allclose(sub.meta["stats_node"]["mean"],
                               compute_stats(sub, per_node=True)["mean"])


def test_dynamic_edges_and_node_targets_follow_the_selection(dynamic_graph):
    ids = np.array([0, 1, 2, 3])
    dynamic_graph.y["load"] = Target(level="node", kind="regression",
                                     values=np.arange(50 * 6, dtype=np.float32).reshape(50, 6))
    sub = dynamic_graph.select_nodes(ids)
    tgdata.validate(sub)
    for t in (0, 17, 49):
        full, _ = dynamic_graph.edges_at(t)
        inside = np.isin(full, ids).all(axis=0)
        np.testing.assert_array_equal(sub.edges_at(t)[0], full[:, inside])
    np.testing.assert_array_equal(sub.y["load"].values, dynamic_graph.y["load"].values[:, :4])


def test_rounded_split_sizes():
    split = Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2}, rounding="round")
    # LargeST: num_train = round(n * 0.6), num_val = round(n * 0.2), test takes the rest.
    assert split.resolve(105097) == {"train": (0, 63058), "val": (63058, 84077),
                                     "test": (84077, 105097)}
    assert split.resolve(5) == {"train": (0, 3), "val": (3, 4), "test": (4, 5)}


def test_node_table_and_sets_roundtrip(static_graph, tmp_path):
    static_graph.node_table = {"district": np.array([3, 3, 4, 4, 4, 5]),
                               "fwy": np.array(["I5-N", "I5-S", "I80", "I80", "SR1", "US101"])}
    static_graph.node_sets = {"d4": np.array([2, 3, 4])}
    tgdata.save(static_graph, tmp_path)
    g = tgdata.load_dir(tmp_path)
    assert g.node_table["fwy"].tolist() == static_graph.node_table["fwy"].tolist()
    sub = g.select_nodes("d4")
    assert sub.node_table["district"].tolist() == [4, 4, 4]


def test_invalid_node_set(static_graph):
    static_graph.node_sets = {"bad": np.array([3, 1])}
    with pytest.raises(ValueError, match="node_sets"):
        tgdata.validate(static_graph)


@pytest.fixture
def fake_registry(monkeypatch):
    entries = {
        "whole": {"domain": "traffic_flow", "tasks": ["node_forecasting"],
                  "time_mode": "discrete", "pool": False},
        "whole-a": {"alias_of": "whole", "nodes": "a", "domain": "traffic_flow",
                    "tasks": ["node_forecasting"], "time_mode": "discrete"},
        "other": {"domain": "epidemics", "tasks": ["node_forecasting"], "time_mode": "discrete"},
        "orphan-a": {"alias_of": "missing", "nodes": "a", "domain": "epidemics",
                     "tasks": [], "time_mode": "discrete"},
    }
    monkeypatch.setattr(registry, "packaged_registry", lambda: entries)


def test_aliases_domains_and_pool(fake_registry):
    assert tgdata.list(offline=True) == ["other", "whole", "whole-a"]
    assert tgdata.list(domain="traffic_flow", pool=True, offline=True) == ["whole-a"]
    assert tgdata.domains(offline=True) == {"epidemics": ["other"],
                                            "traffic_flow": ["whole", "whole-a"]}
    assert tgdata.resolve("whole-a", None) == ("whole", "a")
    assert tgdata.resolve("whole", "a") == ("whole", "a")
    with pytest.raises(ValueError):
        tgdata.resolve("whole-a", "b")
