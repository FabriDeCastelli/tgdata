import numpy as np
import torch
from torch.utils.data import DataLoader

import tgdata
from tgdata.sampling import collate_concat


def window_edges(g, t, window):
    lo, hi = g.edge_ptr[t - window + 1], g.edge_ptr[t + 1]
    return g.edge_index[:, lo:hi]


def test_default_task_is_the_source_task(snapshot_graph):
    task = snapshot_graph.task()
    assert task.window == 1
    assert tgdata.tasks.available(snapshot_graph) == ["graph_classification"]


def test_samples_split_over_anchors(snapshot_graph):
    parts = [snapshot_graph.task("graph_classification", target="growth", window=3, split=s)
             for s in ("train", "val", "test")]
    anchors = np.concatenate([p.anchors for p in parts])
    np.testing.assert_array_equal(anchors, np.arange(2, 20))
    assert [len(p) for p in parts] == [12, 3, 3]


def test_sample_holds_window_of_active_nodes(snapshot_graph):
    g = snapshot_graph
    task = g.task("graph_classification", target="growth", window=3, split="test")
    s = task[1]
    t = s["t"]
    raw = window_edges(g, t, 3)
    np.testing.assert_array_equal(s["node_ids"], np.unique(raw))
    np.testing.assert_array_equal(s["node_ids"][s["edge_index"]], raw)
    assert s["edge_ptr"].tolist() == (g.edge_ptr[t - 2 : t + 2] - g.edge_ptr[t - 2]).tolist()
    assert s["y"] == g.y["growth"].values[t]
    assert s["num_nodes_total"] == 100


def test_labels_are_fixed_at_construction(snapshot_graph):
    task = snapshot_graph.task("graph_classification", target="growth", window=2)
    snapshot_graph.y["growth"].values[:] = 7
    assert task[0]["y"] != 7


def test_collate_concat_merges_steps(snapshot_graph):
    task = snapshot_graph.task("graph_classification", target="growth", window=3)
    samples = [task[i] for i in range(4)]
    batch = collate_concat(samples)
    sizes = [len(s["node_ids"]) for s in samples]
    assert batch["batch"].bincount().tolist() == sizes
    assert batch["y"].shape == (4,) and len(batch["edge_ptr"]) == 4
    offsets = np.cumsum([0, *sizes[:-1]])
    for k in range(3):
        lo, hi = batch["edge_ptr"][k], batch["edge_ptr"][k + 1]
        expected = torch.cat([s["edge_index"][:, s["edge_ptr"][k] : s["edge_ptr"][k + 1]] + o
                              for s, o in zip(samples, offsets, strict=True)], 1)
        assert torch.equal(batch["edge_index"][:, lo:hi], expected)


def test_loader_uses_task_collate(snapshot_graph):
    task = snapshot_graph.task("graph_classification", target="growth", window=2)
    batch = next(iter(DataLoader(task, batch_size=5, collate_fn=task.collate)))
    assert batch["y"].shape == (5,)
