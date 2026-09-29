import numpy as np
import pytest
import torch

import tgdata
from tgdata import ConcatTasks
from tgdata.device import resolve_device
from tgdata.encoding import compact

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def integer_graph(g):
    g.x = np.round(np.abs(g.x) * 100).astype(np.float32)
    g.mask = g.x[..., 0] != 0
    return g


def test_batch_equals_stacked_samples(static_graph):
    task = static_graph.task(split="train", normalize="channel")
    idx = [0, 7, 3]
    batch = task.__getitems__(idx)
    for key in ("x", "y", "mask_x", "mask_y", "covariates", "timestamps"):
        stacked = torch.stack([task[i][key] for i in idx])
        assert torch.equal(batch[key], stacked), key
    assert batch["t"].tolist() == [task[i]["t"] for i in idx]
    assert batch["edge_index"].shape == (2, 5)


def test_dataloader_fetches_whole_batches(static_graph, monkeypatch):
    task = static_graph.task(split="train")
    calls = []
    original = type(task).__getitems__
    monkeypatch.setattr(type(task), "__getitems__",
                        lambda self, idx: calls.append(len(idx)) or original(self, idx))
    batches = list(task.loader(batch_size=8, shuffle=True))
    assert calls == [8] * (len(task) // 8) + ([len(task) % 8] if len(task) % 8 else [])
    assert batches[0]["x"].shape == (8, 4, 6, 2)


def test_concat_tasks_route_each_batch(static_graph, dynamic_graph):
    tasks = ConcatTasks([static_graph.task(split="train"), dynamic_graph.task("node_forecasting",
                                                                           window=4, horizon=3)])
    batch = next(iter(tasks.loader(batch_size=4, num_batches=1, seed=0)))
    assert batch["x"].shape[0] == 4
    with pytest.raises(ValueError, match="spans"):
        tasks.__getitems__([0, len(tasks.datasets[0])])


def test_default_device(monkeypatch):
    monkeypatch.delenv("TGDATA_DEVICE", raising=False)
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device(None).type == expected
    assert resolve_device("cpu").type == "cpu"


@cuda
def test_gpu_batches_equal_cpu_batches(static_graph, tmp_path):
    tgdata.save(compact(integer_graph(static_graph)), tmp_path)
    g = tgdata.load_dir(tmp_path)
    cpu = g.task(split="test", normalize="node", device="cpu")
    gpu = g.task(split="test", normalize="node", device="cuda")
    assert gpu.data["x"].dtype == torch.int16 and gpu.data["x"].is_cuda
    idx = list(range(len(cpu)))
    a, b = cpu.__getitems__(idx), gpu.__getitems__(idx)
    for key, value in a.items():
        assert b[key].is_cuda and torch.equal(value, b[key].cpu()), key


@cuda
def test_splits_of_one_graph_share_the_device_copy(static_graph):
    train = static_graph.task(split="train", device="cuda")
    test = static_graph.task(split="test", device="cuda")
    assert train.data["x"] is test.data["x"]
