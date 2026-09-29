from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler

NODE_KEYS = ("x", "y", "mask_x", "mask_y")
EDGE_KEYS = ("edge_index", "edge_weight", "edge_ptr")


class MultiDatasetSampler(Sampler[list[int]]):
    """Batches of global indices into a ConcatDataset, each drawn from one dataset.

    Dataset i is picked with probability proportional to size_i ** (1 / temperature):
    1 is size-proportional, larger values flatten towards uniform.
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
        picks = torch.multinomial(self.probs, self.num_batches, replacement=True, generator=gen)
        for d in picks.tolist():
            local = torch.randint(int(self.sizes[d]), (self.batch_size,), generator=gen)
            yield (local + self.offsets[d]).tolist()


def collate_pad(
    samples: list[dict[str, Any]],
    max_nodes: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Pads the node dimension (dim 1) to the largest N in the batch.

    With `max_nodes`, graphs above that size are replaced by a random induced subgraph.
    Edges stay per-sample in `edges` so static, dynamic and subsampled graphs share one format.
    """
    if max_nodes is not None:
        samples = [subsample_nodes(s, max_nodes, generator) for s in samples]
    sizes = [s["x"].shape[1] for s in samples]
    n_max = max(sizes)
    out: dict[str, Any] = {
        "node_mask": torch.stack([torch.arange(n_max) < n for n in sizes]),
        "node_ids": torch.stack([_pad(s.get("node_ids", torch.arange(n)), n_max, 0, -1)
                                 for s, n in zip(samples, sizes, strict=True)]),
        "t": torch.as_tensor([s["t"] for s in samples]),
    }
    for key in (*NODE_KEYS, "covariates"):
        if key in samples[0]:
            node_dim = 1 if samples[0][key].dim() >= 3 else None
            out[key] = torch.stack([_pad(s[key], n_max, node_dim) for s in samples])
    if "timestamps" in samples[0]:
        out["timestamps"] = torch.stack([s["timestamps"] for s in samples])
    if "edge_index" in samples[0]:
        out["edges"] = [{k: s[k] for k in EDGE_KEYS if k in s} for s in samples]
    return out


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


def subsample_nodes(
    sample: dict[str, Any], max_nodes: int, generator: torch.Generator | None = None
) -> dict[str, Any]:
    n = sample["x"].shape[1]
    if n <= max_nodes:
        return sample
    keep = torch.randperm(n, generator=generator)[:max_nodes].sort().values
    out = dict(sample, node_ids=keep)
    for key in (*NODE_KEYS, "covariates"):
        if key in sample and sample[key].dim() >= 3:
            out[key] = sample[key][:, keep]
    if "edge_index" in sample:
        out.update(_induced_edges(sample, keep, n))
    return out


def _induced_edges(sample: dict[str, Any], keep: torch.Tensor, n: int) -> dict[str, Any]:
    relabel = torch.full((n,), -1, dtype=torch.long)
    relabel[keep] = torch.arange(len(keep))
    ei = relabel[sample["edge_index"]]
    kept = (ei >= 0).all(0)
    out: dict[str, Any] = {"edge_index": ei[:, kept]}
    if "edge_weight" in sample:
        out["edge_weight"] = sample["edge_weight"][kept]
    if "edge_ptr" in sample:
        ptr = sample["edge_ptr"]
        step = torch.repeat_interleave(torch.arange(len(ptr) - 1), ptr.diff())
        counts = torch.bincount(step[kept], minlength=len(ptr) - 1)
        out["edge_ptr"] = torch.cat([torch.zeros(1, dtype=ptr.dtype), counts.cumsum(0)])
    return out


def _pad(t: torch.Tensor, n_max: int, dim: int | None, value: float = 0) -> torch.Tensor:
    if dim is None or t.shape[dim] == n_max:
        return t
    shape = list(t.shape)
    shape[dim] = n_max - t.shape[dim]
    return torch.cat([t, torch.full(shape, value, dtype=t.dtype)], dim)
