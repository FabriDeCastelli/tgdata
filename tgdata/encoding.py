"""Lossless compact storage for node signals.

An `Encoded` array keeps integers on disk and on the device and decodes to float32 when
indexed, so every consumer sees the float32 values the source holds. A `DerivedMask` is not
stored at all: it is recomputed from `x` by the rule the dataset's source uses.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# Types torch supports fully on every device, so files load onto the GPU without conversion.
INTEGER_TYPES = (np.uint8, np.int8, np.int16, np.int32)
MAX_DECIMALS = 6


class Encoded:
    """`raw` integers with per-channel (last axis) decimals: value = raw / 10**decimals."""

    def __init__(self, raw: np.ndarray, decimals: list[int]) -> None:
        self.raw, self.decimals = raw, list(decimals)
        self.divisor = np.array([10**d for d in decimals], dtype=np.float32)

    shape = property(lambda self: self.raw.shape)
    ndim = property(lambda self: self.raw.ndim)
    dtype = np.dtype(np.float32)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: Any) -> np.ndarray:
        index = index if isinstance(index, tuple) else (index,)
        if Ellipsis in index or len(index) > self.ndim:
            raise IndexError("index the leading axes explicitly")
        full = len(index) == self.ndim
        lead, channels = (index[:-1], index[-1]) if full else (index, slice(None))
        return decode(np.asarray(self.raw[lead]), self.divisor)[..., channels]

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        values = decode(np.asarray(self.raw), self.divisor)
        return values if dtype is None else values.astype(dtype)

    def take_nodes(self, at: slice | np.ndarray) -> Encoded:
        return Encoded(self.raw[:, at], self.decimals)

    def spec(self) -> dict[str, Any]:
        return {"dtype": str(self.raw.dtype), "decimals": self.decimals}


class DerivedMask:
    """mask = x[..., 0] != 0, computed on access from the (encoded) node signal."""

    rule = "x[..., 0] != 0"

    def __init__(self, x: Encoded | np.ndarray) -> None:
        self.x = x

    shape = property(lambda self: self.x.shape[:2])
    ndim = 2
    dtype = np.dtype(bool)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: Any) -> np.ndarray:
        raw = self.x.raw if isinstance(self.x, Encoded) else self.x
        index = index if isinstance(index, tuple) else (index,)
        index = index + (slice(None),) * (2 - len(index))
        return np.asarray(raw[(*index, 0)]) != 0

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self[:] if dtype is None else self[:].astype(dtype)

    def take_nodes(self, at: slice | np.ndarray) -> DerivedMask:
        return DerivedMask(self.x.take_nodes(at) if isinstance(self.x, Encoded) else self.x[:, at])


def decode(raw: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    # Division, not multiplication by 10**-d: IEEE division rounds correctly, so decoding gives
    # exactly the float32 of the source's decimal value.
    values = raw.astype(np.float32)
    return values if (divisor == 1).all() else values / divisor


def encode(values: np.ndarray, chunk: int = 4096) -> Encoded | None:
    """The smallest lossless integer encoding of float `values` [..., C], or None.

    Lossless means decoding reproduces float32(values) bit for bit.
    """
    values = np.asarray(values)
    target = values.astype(np.float32)
    decimals = []
    for c in range(values.shape[-1]):
        column, expected = values[..., c], target[..., c]
        found = next((d for d in range(MAX_DECIMALS + 1)
                      if np.array_equal(decode(np.round(column * 10**d), np.float32(10**d)),
                                        expected)), None)
        if found is None:
            return None
        decimals.append(found)
    scaled = np.round(values * np.array([10**d for d in decimals]))
    kind = next((t for t in INTEGER_TYPES if np.iinfo(t).min <= scaled.min()
                 and scaled.max() <= np.iinfo(t).max), None)
    if kind is None:
        return None
    encoded = Encoded(scaled.astype(kind), decimals)
    for lo in range(0, len(values), chunk):
        if not np.array_equal(encoded[lo:lo + chunk], target[lo:lo + chunk]):
            return None
    return encoded


def mask_is_derivable(mask: np.ndarray, x: np.ndarray | Encoded) -> bool:
    return bool(np.array_equal(np.asarray(mask), np.asarray(DerivedMask(x))))


def compact(g: Any) -> Any:
    """`g` with `x` and `covariates` encoded where lossless and a derivable mask dropped."""
    changes: dict[str, Any] = {}
    for name in ("x", "covariates"):
        values = getattr(g, name)
        if values is not None and not isinstance(values, Encoded):
            encoded = encode(np.asarray(values))
            if encoded is not None:
                changes[name] = encoded
    x = changes.get("x", g.x)
    mask = g.mask
    if (mask is not None and x is not None and not isinstance(mask, DerivedMask)
            and mask_is_derivable(mask, x)):
        changes["mask"] = DerivedMask(x)
    return dataclasses.replace(g, **changes)
