import dataclasses

import numpy as np
import pytest

import tgdata
from tgdata.converters import drop_duplicate_edges


def test_static_duplicates_are_rejected(static_graph):
    ei = np.concatenate([static_graph.edge_index, static_graph.edge_index[:, :1]], axis=1)
    w = np.concatenate([static_graph.edge_weight, static_graph.edge_weight[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        tgdata.validate(dataclasses.replace(static_graph, edge_index=ei, edge_weight=w))


def test_dynamic_duplicates_are_kept(dynamic_graph):
    ei = dynamic_graph.edge_index.copy()
    ei[:, 1] = ei[:, 0]
    tgdata.validate(dataclasses.replace(dynamic_graph, edge_index=ei))


def test_drop_duplicate_edges():
    edges = [(0, 1, 2.0), (1, 2, 3.0), (0, 1, 2.0)]
    assert drop_duplicate_edges(edges) == edges[:2]
    with pytest.raises(ValueError, match="repeats"):
        drop_duplicate_edges([(0, 1, 2.0), (0, 1, 5.0)])


def test_adjacency_follows_meta(static_graph):
    ei, w = tgdata.adjacency(static_graph)
    np.testing.assert_array_equal(w, static_graph.edge_weight)
    static_graph.meta["adjacency"] = {"kind": "binary", "formula": "w_ij = 1"}
    assert tgdata.adjacency(static_graph)[1].tolist() == [1.0] * 5
    assert static_graph.window(4, 3)[0]["edge_weight"].tolist() == [1.0] * 5
    assert tgdata.adjacency(static_graph, "raw")[1] is static_graph.edge_weight


def test_gaussian_adjacency(static_graph):
    static_graph.meta["adjacency"] = {"kind": "gaussian", "sigma": 2.0, "threshold": 0.1}
    ei, w = tgdata.adjacency(static_graph)
    np.testing.assert_allclose(w, np.exp(-((np.array([1.0, 2.0, 3.0]) / 2.0) ** 2)), rtol=1e-6)
    assert ei.shape == (2, 3)


def test_binary_adjacency_on_dynamic_windows(dynamic_graph):
    dynamic_graph.meta["adjacency"] = {"kind": "binary", "formula": "w_ij = 1"}
    s = dynamic_graph.window(4, 2)[0]
    assert (s["edge_weight"] == 1).all() and s["edge_ptr"][-1] == s["edge_index"].shape[1]


@pytest.mark.parametrize(
    "change, meta, message",
    [
        ({"edge_weight": None}, {}, "edge_weight must be given"),
        ({}, {"edge_weight": {"kind": "distance"}}, "'kind' and 'observed'"),
        ({}, {"edge_weight": {"kind": "gaussian", "observed": False}}, "no 'formula'"),
        ({}, {"adjacency": {"kind": "binary"}}, "needs a 'formula'"),
    ],
)
def test_edge_metadata_is_required(static_graph, change, meta, message):
    g = dataclasses.replace(static_graph, meta={**static_graph.meta, **meta}, **change)
    with pytest.raises(ValueError, match=message):
        tgdata.validate(g)
