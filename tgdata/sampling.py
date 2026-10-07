from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler

EDGE_KEYS = ("edge_index", "edge_weight", "edge_ptr")


class MultiDatasetSampler(Sampler[list[int]]):
    """Batches of global indices into a ConcatDataset, each drawn from one dataset.

    Dataset i is picked with probability proportional to size_i ** (1 / temperature):
    1 is size-proportional, larger values flatten towards uniform.

    Every pass draws new batches: the draws are seeded with `seed + epoch` and the epoch advances by
    one per pass, because a trainer does not always call `set_epoch` (Lightning reaches the sampler
    and the batch sampler's `sampler`, but not a batch sampler such as this one). `set_epoch` sets
    the epoch, to replay a pass or to resume at one.
    """

    def __init__(
        self,
        sizes: Sequence[int],
        batch_size: int,
        num_batches: int,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.sizes = torch.as_tensor(sizes, dtype=torch.float64)
        self.offsets = torch.cumsum(torch.as_tensor([0, *sizes[:-1]]), 0).tolist()
        self.batch_size, self.num_batches = batch_size, num_batches
        self.probs = self.sizes ** (1.0 / temperature)
        self.probs /= self.probs.sum()
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        gen = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        picks = torch.multinomial(self.probs, self.num_batches, replacement=True, generator=gen)
        for d in picks.tolist():
            local = torch.randint(int(self.sizes[d]), (self.batch_size,), generator=gen)
            yield (local + self.offsets[d]).tolist()


def collate_concat(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch snapshot-window samples the PyG way: nodes concatenated, a `batch` vector, and
    edges merged step by step so one `edge_ptr [w + 1]` indexes window step k for all samples.
    """
    sizes = torch.as_tensor([len(s["node_ids"]) for s in samples])
    offsets = torch.cumsum(sizes, 0) - sizes
    window = len(samples[0]["edge_ptr"]) - 1
    steps = torch.cat([torch.repeat_interleave(torch.arange(window), s["edge_ptr"].diff())
                       for s in samples])
    owner = torch.cat([torch.full((s["edge_index"].shape[1],), b)
                       for b, s in enumerate(samples)])
    order = torch.argsort(steps * len(samples) + owner, stable=True)
    edge_index = torch.cat([s["edge_index"] + off for s, off in zip(samples, offsets,
                                                                     strict=True)], 1)
    counts = torch.bincount(steps, minlength=window)
    out: dict[str, Any] = {
        "edge_index": edge_index[:, order],
        "edge_weight": torch.cat([s["edge_weight"] for s in samples])[order],
        "edge_ptr": torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)]),
        "batch": torch.repeat_interleave(torch.arange(len(samples)), sizes),
        "node_ids": torch.cat([s["node_ids"] for s in samples]),
        "num_nodes_total": torch.as_tensor([s["num_nodes_total"] for s in samples]),
        "y": torch.stack([s["y"] for s in samples]),
        "t": torch.as_tensor([s["t"] for s in samples]),
    }
    if "node_features" in samples[0]:
        out["node_features"] = torch.cat([s["node_features"] for s in samples])
    if "x" in samples[0]:
        out["x"] = torch.cat([s["x"] for s in samples], dim=1)
    return out
