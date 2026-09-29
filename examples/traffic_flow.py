# %% [markdown]
# # Traffic flow forecasting with tgdata
#
# Load the traffic-flow datasets, look at their node signals and graph, and turn them into
# batches of forecasting samples.
#
# Setup (once): install tgdata with plotting and log in to Hugging Face, since the
# datasets are private to the `tgdata-hub` organisation.
#
# ```bash
# pip install "tgdata[plot] @ git+ssh://git@github.com/FabriDeCastelli/tgdata.git"
# hf auth login
# ```
#
# To open this file as a notebook: `jupytext --to notebook traffic_flow.py`, or run its
# cells directly in VS Code / PyCharm.

# %%
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import tgdata
from tgdata.encoding import DerivedMask
from tgdata.plot import plot_forecast, plot_heatmap, plot_signals

# Downloads land here once, then load offline. On a shared server, point it at a large disk.
ROOT = Path(os.environ.get("TGDATA_ROOT", Path.home() / ".cache" / "tgdata"))
torch.manual_seed(0)
print("tgdata", tgdata.__version__)

# %% [markdown]
# ## What is available
#
# Datasets are grouped by domain. Names such as `largest-d5` are node subsets of a stored
# dataset (here one PeMS district of LargeST); they load like any other dataset.

# %%
for domain, names in tgdata.domains().items():
    print(f"{domain}: {names}")
print("pretraining pool:", tgdata.list(domain="traffic_flow", pool=True))

# %% [markdown]
# ## Load one dataset
#
# `load` downloads PEMS08 (38 MB) the first time and memory-maps it afterwards, so arrays
# are read from disk only when sliced.

# %%
g = tgdata.load("pems08", root=ROOT)
print(g.name, "|", g.domain, "|", g.meta["source"])
print(f"x          {g.x.shape}  {g.x.dtype}   [steps, nodes, channels] = {g.meta['channels']}, "
      f"stored as {getattr(g.x, 'raw', g.x).dtype}")
print(f"covariates {g.covariates.shape}   {g.meta['covariate_channels']}")
stored = "recomputed from x" if isinstance(g.mask, DerivedMask) else "stored"
print(f"mask       {g.mask.shape}      observed: {np.asarray(g.mask).mean():.2%} "
      f"({g.meta['mask_rule']}, {stored})")
print(f"timestamps {g.timestamps.shape}  every {g.meta['freq']}, "
      f"{np.datetime64(int(g.timestamps[0]), 's')} to {np.datetime64(int(g.timestamps[-1]), 's')}")
print(f"edges      {g.edge_index.shape}  weights: {g.meta['edge_weight']['kind']}, "
      f"used as {g.meta['adjacency']['kind']} ({g.meta['adjacency']['formula']})")
print("default task:", g.meta["default_task"])

# %% [markdown]
# ## Node signals
#
# One week of flow for three sensors. Missing readings are drawn as gaps.

# %%
plot_signals(g, nodes=[0, 50, 120], start="2016-08-01", end="2016-08-08")
plt.show()

# %% [markdown]
# All sensors over the same week: daily cycles show up as vertical bands, missing readings
# as white.

# %%
start = int(np.searchsorted(g.timestamps, np.datetime64("2016-08-01", "s").astype(np.int64)))
plot_heatmap(g, start=start, end=start + 7 * 288)
plt.show()

# %% [markdown]
# Mean flow by time of day over the training steps, and the sensor graph as an adjacency
# matrix (edges as the ASTGCN paper uses them).

# %%
steps_per_day = 288
train_end = g.splits["default"].resolve(g.num_steps)["train"][1]
x = np.where(g.mask[:train_end], g.x[:train_end, :, 0], np.nan)
days = train_end // steps_per_day
daily = np.nanmean(x[: days * steps_per_day].reshape(days, steps_per_day, -1), axis=(0, 2))

edge_index, edge_weight = tgdata.adjacency(g)
A = np.zeros((g.num_nodes, g.num_nodes))
A[edge_index[0], edge_index[1]] = edge_weight

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(np.arange(steps_per_day) / 12, daily)
ax1.set(xlabel="hour of day", ylabel="mean flow [vehicles/5min]", title="Daily profile")
ax2.spy(A, markersize=1.5)
ax2.set(title=f"Adjacency: {edge_index.shape[1]} directed edges")
plt.show()

# %% [markdown]
# ## Forecasting samples
#
# `g.task()` builds the task the dataset's source defines: 12 steps (1 hour) in, 12 out,
# with the ASTGCN split. Each sample is a dictionary of tensors.

# %%
train, test = g.task(split="train"), g.task(split="test")
print(f"train {len(train)} samples, test {len(test)} samples")

sample = test[0]
for key, value in sample.items():
    print(f"{key:12s} {tuple(value.shape) if hasattr(value, 'shape') else value}")

# %% [markdown]
# A persistence baseline (repeat the last input value) against the target, for two sensors.

# %%
persistence = sample["x"][-1:].expand_as(sample["y"])
plot_forecast(sample, nodes=[0, 50], prediction=persistence)
plt.show()

# %% [markdown]
# The input window of one sample as an image, nodes by time.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
image = ax.imshow(sample["x"][..., 0].T.cpu(), aspect="auto", cmap="viridis")
fig.colorbar(image, ax=ax, label="flow")
ax.set(xlabel="input step", ylabel="node", title=f"x of sample t = {sample['t']}")
plt.show()

# %% [markdown]
# Batches are built on the GPU when CUDA is available (the CPU otherwise): the dataset is
# uploaded once and each batch is cut there. `task.loader` is a standard torch DataLoader, so
# it also works as is in a Lightning Trainer. `normalize="channel"` scales by the
# training-split statistics stored with the dataset.

# %%
task = g.task(split="train", normalize="channel")
print("batches on", task.device)
loader = task.loader(batch_size=32, shuffle=True)
batch = next(iter(loader))
for key, value in batch.items():
    shape = tuple(value.shape) if hasattr(value, "shape") else f"list of {len(value)}"
    print(f"{key:12s} {shape}")
print("normalised x: mean", batch["x"].mean().item(), "std", batch["x"].std().item())

# %% [markdown]
# Masked MAE of persistence on the test split, in vehicles per 5 minutes.

# %%
errors, counts = 0.0, 0
for batch in test.loader(batch_size=256):
    prediction = batch["x"][:, -1:].expand_as(batch["y"])
    valid = batch["mask_y"].expand_as(batch["y"])
    errors += (prediction - batch["y"]).abs()[valid].sum().item()
    counts += valid.sum().item()
print(f"persistence masked MAE on PEMS08 test: {errors / counts:.2f}")

# %% [markdown]
# ## Several datasets at once
#
# Pretraining draws each batch from one dataset, chosen with probability proportional to
# size^(1/temperature), so batches need no padding. Only the small
# PeMS datasets are used here; set `INCLUDE_LARGEST = True` to add the nine LargeST districts
# (a one-time 4.3 GB download).

# %%
INCLUDE_LARGEST = False

names = tgdata.list(domain="traffic_flow", pool=True)
if not INCLUDE_LARGEST:
    names = [n for n in names if not n.startswith("largest")]
pool = tgdata.ConcatTasks([tgdata.load(n, root=ROOT).task(split="train", normalize="channel")
                           for n in names])
loader = pool.loader(batch_size=16, num_batches=200, temperature=2.0, seed=0)

for name, t, p in zip(names, pool.datasets, loader.batch_sampler.probs, strict=True):
    print(f"{name:14s} {len(t):6d} samples, {t.g.num_nodes:4d} nodes, picked with p = {p:.2f}")
start = time.perf_counter()
samples = sum(batch["x"].shape[0] for batch in loader)
if torch.cuda.is_available():
    torch.cuda.synchronize()
print(f"{samples} samples in {time.perf_counter() - start:.2f} s, on {pool.datasets[0].device}")
