"""Preprocessing that sources apply to their raw data, reproduced exactly and applied per batch.

Feature transforms take the raw node signals in float64 (the precision the sources compute in,
so bin edges fall where theirs do) and return float32; target transforms likewise.
"""
from __future__ import annotations

import torch


def pygt_encoded(x: torch.Tensor, log_degree_cutoff: int = 4) -> torch.Tensor:
    """PyTorch Geometric Temporal's TwitterTennis `feature_mode="encoded"`: one-hot of
    min(ceil(log(degree + 1)), 4) and of floor(10 * transitivity), 5 + 11 = 16 columns
    (torch_geometric_temporal/dataset/twitter_tennis.py, encode_features)."""
    degree = torch.clamp(torch.ceil(torch.log(x[..., 0] + 1.0)), max=log_degree_cutoff)
    transitivity = torch.floor(x[..., 1] * 10)
    return torch.cat([_one_hot(degree, log_degree_cutoff + 1), _one_hot(transitivity, 11)], -1)


def log1p(y: torch.Tensor) -> torch.Tensor:
    """log(1 + y) computed as np.log(1.0 + y), the form PyG Temporal uses for its targets."""
    return torch.log(1.0 + y.to(torch.float64)).to(torch.float32)


def _one_hot(values: torch.Tensor, classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(values.to(torch.long), classes).to(torch.float32)


FEATURES = {"pygt_encoded": pygt_encoded}
TARGETS = {"log1p": log1p}
