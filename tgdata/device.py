"""Datasets resident on a device, uploaded once and shared by every task on the same graph."""
from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import torch

from .encoding import DerivedMask, Encoded
from .schema import TemporalGraph, adjacency

_warned = False


def resolve_device(device: str | torch.device | None) -> torch.device:
    """`device`, else $TGDATA_DEVICE, else CUDA when available and the CPU otherwise."""
    global _warned
    if device is not None:
        return torch.device(device)
    if "TGDATA_DEVICE" in os.environ:
        return torch.device(os.environ["TGDATA_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if not _warned:
        warnings.warn("CUDA is not available; tgdata tasks build batches on the CPU", stacklevel=3)
        _warned = True
    return torch.device("cpu")


def on_device(g: TemporalGraph, device: torch.device, adjacency_kind: str | None) -> dict[str, Any]:
    """The graph's arrays as tensors on `device`, in their stored (compact) dtypes.

    Cached on the graph, so the train, val and test tasks of one graph share one copy.
    Encoded arrays come with a `<name>_divisor` to decode them; a derived mask is absent.
    """
    cache = g.__dict__.setdefault("_device_cache", {})
    key = (str(device), adjacency_kind)
    if key in cache:
        return cache[key]
    data: dict[str, Any] = {}
    for name in ("x", "covariates", "mask", "timestamps"):
        values = getattr(g, name)
        if values is None or isinstance(values, DerivedMask):
            continue
        if isinstance(values, Encoded):
            data[f"{name}_divisor"] = (None if values.decimals == [0] * len(values.decimals)
                                       else torch.from_numpy(values.divisor).to(device))
            values = values.raw
        data[name] = _upload(values, device)
    data["mask_derived"] = isinstance(g.mask, DerivedMask)
    if g.edge_index is not None:
        edge_index, edge_weight = adjacency(g, adjacency_kind)
        data["edge_index"] = _upload(edge_index, device, torch.long)
        data["edge_weight"] = _upload(edge_weight, device, torch.float32)
        if g.edge_ptr is not None:
            data["edge_ptr"] = _upload(g.edge_ptr, device, torch.long)
    cache[key] = data
    return data


def decode(raw: torch.Tensor, divisor: torch.Tensor | None) -> torch.Tensor:
    values = raw.to(torch.float32)
    return values if divisor is None else values / divisor


def _upload(values: np.ndarray, device: torch.device, dtype: torch.dtype | None = None
            ) -> torch.Tensor:
    # np.array copies: a memmap is read-only, and .to() is a no-op on the CPU.
    tensor = torch.from_numpy(np.array(values))
    return tensor.to(device=device, dtype=dtype or tensor.dtype)
