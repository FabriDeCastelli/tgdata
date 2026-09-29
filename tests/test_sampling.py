import dataclasses
from collections import Counter

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from tgdata.sampling import MultiDatasetSampler, collate_pad, subsample_nodes


def test_sampler_temperature():
    sizes = [900, 100]
    def share(temperature):
        sampler = MultiDatasetSampler(sizes, 4, 2000, temperature, seed=0)
        return Counter(b[0] < 900 for b in sampler)[True] / 2000
    assert abs(share(1.0) - 0.9) < 0.03
    assert abs(share(1e6) - 0.5) < 0.04


def test_sampler_batches_stay_in_one_dataset():
    for batch in MultiDatasetSampler([10, 20, 30], 8, 50, 2.0):
        owners = {int(np.searchsorted([10, 30, 60], i, side="right")) for i in batch}
        assert len(owners) == 1


def test_collate_pads_variable_nodes(static_graph):
    g = static_graph
    small = dataclasses.replace(g, x=g.x[:, :4], mask=g.mask[:, :4],
                                edge_index=g.edge_index[:, :3], edge_weight=g.edge_weight[:3])
    datasets = [static_graph.window(4, 3), small.window(4, 3)]
    loader = DataLoader(ConcatDataset(datasets), collate_fn=collate_pad,
                        batch_sampler=MultiDatasetSampler([len(d) for d in datasets], 2, 10))
    for batch in loader:
        assert batch["x"].shape[1:] == (4, batch["node_mask"].shape[1], 2)
    mixed = collate_pad([datasets[0][0], datasets[1][0]])
    assert mixed["x"].shape == (2, 4, 6, 2)
    assert mixed["node_mask"].sum(1).tolist() == [6, 4]
    assert mixed["x"][1, :, 4:].abs().sum() == 0
    assert not mixed["mask_x"][1, :, 4:].any()
    assert mixed["node_ids"][1].tolist() == [0, 1, 2, 3, -1, -1]
    assert len(mixed["edges"]) == 2


def test_subgraph_sampling_static(static_graph):
    s = static_graph.window(4, 3)[0]
    sub = subsample_nodes(s, 3, torch.Generator().manual_seed(0))
    keep = sub["node_ids"]
    assert sub["x"].shape == (4, 3, 2)
    torch.testing.assert_close(sub["x"], s["x"][:, keep])
    original = {tuple(e) for e in s["edge_index"].T.tolist()}
    for a, b in sub["edge_index"].T.tolist():
        assert (keep[a].item(), keep[b].item()) in original
    assert len(sub["edge_weight"]) == sub["edge_index"].shape[1]


def test_subgraph_sampling_dynamic(dynamic_graph):
    s = dynamic_graph.window(4, 2)[0]
    sub = subsample_nodes(s, 4, torch.Generator().manual_seed(0))
    ptr = sub["edge_ptr"]
    assert ptr[0] == 0 and ptr[-1] == sub["edge_index"].shape[1] and len(ptr) == 5
    batch = collate_pad([s, s], max_nodes=4)
    assert batch["x"].shape == (2, 4, 4, 2)
