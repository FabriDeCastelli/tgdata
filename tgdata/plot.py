"""Quick looks at node signals. Needs matplotlib (`pip install tgdata[plot]`)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .schema import TemporalGraph, _mask_as_x

Step = int | str | None


def plot_signals(
    g: TemporalGraph,
    nodes: Sequence[int],
    start: Step = None,
    end: Step = None,
    channel: int = 0,
    field: str = "x",
    ax: Any = None,
) -> Any:
    """Line per node over steps [start, end); dates or step indices, missing readings as gaps.

    `start`/`end` are step indices or ISO dates (with timestamps). `field` is "x" or
    "covariates".
    """
    import matplotlib.pyplot as plt

    lo, hi = _step(g, start, 0), _step(g, end, g.num_steps)
    values = np.array(getattr(g, field)[lo:hi][:, list(nodes), channel], dtype=np.float64)
    if field == "x" and g.mask is not None:
        valid = _mask_as_x(np.asarray(g.mask[lo:hi]), np.asarray(g.x[lo:hi]))
        values[~valid[:, list(nodes), min(channel, valid.shape[-1] - 1)]] = np.nan
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    time = _time(g, lo, hi)
    for i, node in enumerate(nodes):
        ax.plot(time, values[:, i], lw=1, label=f"node {node}")
    ax.set_xlim(time[0], time[-1])
    name = _channel_name(g, field, channel)
    unit = g.meta.get("units", {}).get(name)
    ax.set_ylabel(f"{name} [{unit}]" if unit else name)
    ax.set_xlabel("time" if g.timestamps is not None else "step")
    ax.set_title(g.name)
    if len(nodes) <= 10:
        ax.legend(loc="upper right", fontsize="small")
    return ax


def plot_heatmap(
    g: TemporalGraph,
    start: Step = None,
    end: Step = None,
    channel: int = 0,
    ax: Any = None,
) -> Any:
    """All nodes over steps [start, end) as an image; missing readings are left blank."""
    import matplotlib.pyplot as plt

    assert g.x is not None
    lo, hi = _step(g, start, 0), _step(g, end, g.num_steps)
    values = np.array(g.x[lo:hi, :, channel], dtype=np.float64)
    if g.mask is not None:
        valid = np.asarray(_mask_as_x(g.mask[lo:hi], g.x[lo:hi]))
        values[~valid[..., min(channel, valid.shape[-1] - 1)]] = np.nan
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(values.T, aspect="auto", interpolation="nearest", cmap="viridis",
                      extent=(lo, hi, g.num_nodes, 0))
    name = _channel_name(g, "x", channel)
    ax.figure.colorbar(image, ax=ax, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("node")
    ax.set_title(g.name)
    return ax


def plot_forecast(
    sample: dict[str, Any],
    nodes: Sequence[int],
    prediction: Any = None,
    channel: int = 0,
    axes: Any = None,
) -> Any:
    """Input window, target and optional prediction of a node-forecasting sample, per node.

    `prediction` has the shape of `sample["y"]`; missing targets (`mask_y`) are not drawn.
    """
    import matplotlib.pyplot as plt

    x = _numpy(sample["x"])[..., channel].astype(np.float64)
    y = _numpy(sample["y"])[..., channel].astype(np.float64)
    if "mask_y" in sample:
        y[~_numpy(sample["mask_y"])[..., 0].astype(bool)] = np.nan
    window, horizon = len(x), len(y)
    past, future = np.arange(-window, 0), np.arange(horizon)
    if axes is None:
        _, axes = plt.subplots(len(nodes), 1, figsize=(8, 2.2 * len(nodes)), sharex=True,
                               squeeze=False)
        axes = axes[:, 0]
    for ax, node in zip(axes, nodes, strict=True):
        ax.plot(past, x[:, node], color="0.4", lw=1.2, label="input")
        ax.plot(future, y[:, node], color="C0", lw=1.5, marker=".", label="target")
        if prediction is not None:
            pred = _numpy(prediction)[..., channel].astype(np.float64)
            ax.plot(future, pred[:, node], color="C3", lw=1.5, ls="--", label="prediction")
        ax.axvline(-0.5, color="0.7", lw=0.8)
        ax.set_ylabel(f"node {node}")
    axes[0].legend(loc="upper left", fontsize="small")
    axes[0].set_title(f"t = {sample['t']}")
    axes[-1].set_xlabel("steps from t")
    return axes


def _numpy(values: Any) -> np.ndarray:
    """Tensors (possibly on GPU, possibly requiring grad) or arrays, as numpy."""
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _step(g: TemporalGraph, value: Step, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if g.timestamps is None:
        raise ValueError("dates need timestamps; pass step indices")
    seconds = np.datetime64(value, "s").astype(np.int64)
    return int(np.searchsorted(np.asarray(g.timestamps), seconds))


def _time(g: TemporalGraph, lo: int, hi: int) -> np.ndarray:
    if g.timestamps is None:
        return np.arange(lo, hi)
    return np.asarray(g.timestamps[lo:hi]).astype("datetime64[s]")


def _channel_name(g: TemporalGraph, field: str, channel: int) -> str:
    names = g.meta.get("channels" if field == "x" else "covariate_channels", [])
    return names[channel] if channel < len(names) else f"{field}[{channel}]"
