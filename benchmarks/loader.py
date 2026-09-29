"""How fast tgdata feeds a GPU, against tsl and precomputed windows.

Every contender ends with the batch on the GPU. Two measurements per dataset:
  loader: samples per second, iterating the DataLoader and moving batches to the GPU;
  train:  milliseconds per Lightning training step of a small GCN-GRU, against a
          reference that reuses one batch already on the GPU (the model's compute alone).

Needs pytorch_lightning and tsl (torch-spatiotemporal), whose torch_sparse dependency pins
torch; the published numbers used torch 2.2.2 + cu121 on one A100 40GB:
  python benchmarks/loader.py --data DIR [--datasets pems07 largest] [--batch-size 64]
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import tgdata

WINDOW = HORIZON = 12
Unpack = Callable[[Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def tgdata_loader(g: tgdata.TemporalGraph, batch_size: int) -> tuple[DataLoader, Unpack]:
    task = g.task(split="train", normalize="channel", device="cuda")
    return (task.loader(batch_size, shuffle=True, drop_last=True),
            lambda b: (b["x"], b["y"], b["mask_y"]))


def tsl_loader(g: tgdata.TemporalGraph, batch_size: int, workers: int
               ) -> tuple[DataLoader, Unpack]:
    from tsl.data import SpatioTemporalDataset
    from tsl.data.loader import StaticGraphLoader
    from tsl.data.preprocessing import StandardScaler

    x = np.asarray(g.x)
    ds = SpatioTemporalDataset(target=x, mask=np.asarray(g.mask)[..., None], window=WINDOW,
                               horizon=HORIZON)
    ds.add_scaler("target", StandardScaler(axis=(0, 1)).fit(x))
    train = torch.utils.data.Subset(ds, range(int(len(ds) * 0.6)))
    loader = StaticGraphLoader(train, batch_size=batch_size, shuffle=True, drop_last=True,
                               num_workers=workers, pin_memory=True,
                               persistent_workers=workers > 0)
    return loader, lambda b: (b.x, b.y, b.mask)


def windows_loader(g: tgdata.TemporalGraph, batch_size: int) -> tuple[DataLoader, Unpack]:
    """ASTGCN/STSGCN practice: every window materialised in memory, served by a TensorDataset."""
    x = np.asarray(g.x)
    mean, std = x[np.asarray(g.mask)].mean(), x[np.asarray(g.mask)].std()
    anchors = g.task(split="train", device="cpu").anchors
    rows = anchors[:, None] + np.arange(-WINDOW, HORIZON)
    windows = torch.from_numpy(x[rows])
    inputs = (windows[:, :WINDOW] - mean) / std
    masks = torch.from_numpy(np.asarray(g.mask)[rows[:, WINDOW:]])[..., None]
    data = TensorDataset(inputs, windows[:, WINDOW:], masks)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True,
                        pin_memory=True)
    return loader, lambda b: (b[0], b[1], b[2])


def loader_throughput(loader: DataLoader, unpack: Unpack, n: int) -> float:
    it = iter(loader)
    for _ in zip(range(3), it, strict=False):  # warm up workers and the device cache
        pass
    torch.cuda.synchronize()
    start, samples = time.perf_counter(), 0
    for _, batch in zip(range(n), it, strict=False):
        x, y, mask = (t.cuda(non_blocking=True) for t in unpack(batch))
        samples += x.shape[0]
    torch.cuda.synchronize()
    return samples / (time.perf_counter() - start)


class GCNGRU(nn.Module):
    """A GRU over each node's window, then one graph convolution over the sensor graph."""

    def __init__(self, adjacency: torch.Tensor, hidden: int = 64) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.gru = nn.GRU(1, hidden, batch_first=True)
        self.conv = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, n, f = x.shape
        _, h = self.gru(x.permute(0, 2, 1, 3).reshape(b * n, w, f))
        h = h[-1].view(b, n, -1)
        h = torch.relu(self.conv(torch.einsum("ij,bjh->bih", self.adjacency, h))) + h
        return self.head(h).permute(0, 2, 1).unsqueeze(-1)  # [B, horizon, N, 1]


def train_step_ms(model: nn.Module, batches: Iterable[Any], unpack: Unpack, steps: int) -> float:
    import pytorch_lightning as pl

    class Module(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def training_step(self, batch: Any, _: int) -> torch.Tensor:
            x, y, mask = unpack(batch)
            error = (self.model(x) - y).abs() * mask
            return error.sum() / mask.sum().clamp(min=1)

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    class Timer(pl.Callback):
        def __init__(self) -> None:
            self.times: list[float] = []

        def on_train_batch_end(self, *args: Any) -> None:
            torch.cuda.synchronize()
            self.times.append(time.perf_counter())

    timer = Timer()
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_steps=steps, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False,
                         enable_model_summary=False, callbacks=[timer])
    trainer.fit(Module(), train_dataloaders=batches)
    gaps = np.diff(timer.times[5:])  # drop warm-up steps
    return float(np.median(gaps) * 1e3)


def repeat(batch: Any, n: int) -> DataLoader:
    """One batch, already on the GPU, served n times: the model's compute with no loading."""

    class Same(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return n

        def __getitem__(self, _: int) -> Any:
            return batch

    return DataLoader(Same(), batch_size=None)


def normalized_adjacency(g: tgdata.TemporalGraph) -> torch.Tensor:
    ei, w = tgdata.adjacency(g)
    a = torch.zeros(g.num_nodes, g.num_nodes)
    a[torch.as_tensor(ei[0]), torch.as_tensor(ei[1])] = torch.as_tensor(w)
    a += torch.eye(g.num_nodes)
    d = a.sum(1).rsqrt()
    return (d[:, None] * a * d[None, :]).cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--datasets", nargs="+", default=["pems07", "largest"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__} | "
          f"batch {args.batch_size}, window {WINDOW} -> {HORIZON}")
    for name in args.datasets:
        g = tgdata.load_dir(f"{args.data}/{name}")
        adjacency = normalized_adjacency(g)
        contenders = {"tgdata (GPU)": partial(tgdata_loader, g, args.batch_size),
                      "tsl, 0 workers": partial(tsl_loader, g, args.batch_size, 0),
                      "tsl, 8 workers": partial(tsl_loader, g, args.batch_size, 8)}
        if g.num_nodes < 2000:  # every window in memory: about 52 GB for LargeST CA
            contenders["precomputed windows"] = partial(windows_loader, g, args.batch_size)
        print(f"\n{name}: {g.num_nodes} nodes, {g.num_steps} steps")
        reference = None
        for label, build in contenders.items():
            loader, unpack = build()
            throughput = loader_throughput(loader, unpack, n=args.steps)
            torch.manual_seed(0)
            step = train_step_ms(GCNGRU(adjacency), loader, unpack, args.steps)
            if reference is None:
                batch = next(iter(loader))
                fixed = tuple(t.cuda() for t in unpack(batch))
                torch.manual_seed(0)
                reference = train_step_ms(GCNGRU(adjacency), repeat(fixed, args.steps + 1),
                                          lambda b: b, args.steps)
                print(f"  {'reference (batch already on GPU)':34s} {'':>14s}   "
                      f"{reference:7.2f} ms/step")
            print(f"  {label:34s} {throughput:10.0f} /s   {step:7.2f} ms/step "
                  f"({step / reference - 1:+6.1%} vs reference)")


if __name__ == "__main__":
    main()
