<p align="center"><img src="docs/logo.png" alt="tgdata" width="140"></p>

<h1 align="center">tgdata</h1>

<p align="center">Temporal-graph datasets in one format, downloaded on demand and batched on the GPU.</p>

## Install

In a uv project (nothing is cloned into it; `uv.lock` pins the version):

```bash
uv add "tgdata[plot] @ git+ssh://git@github.com/FabriDeCastelli/tgdata.git" --tag v0.2.2
hf auth login   # the datasets are private to the tgdata-hub organisation
```

With pip: `pip install "tgdata[plot] @ git+ssh://git@github.com/FabriDeCastelli/tgdata.git@v0.2.2"`.
Add `cu128` to the extras for CUDA 12.8 drivers, or `cpu` for a machine without a GPU; see
[installing](docs/contributing.md#installing).

## What you can do

**Find datasets**

```python
import tgdata

tgdata.domains()                    # {'traffic_flow': ['largest', 'largest-d10', ..., 'pems08']}
tgdata.list(domain="traffic_flow")  # names only
```

**Load one** (downloaded the first time, then read from disk)

```python
g = tgdata.load("pems08")
g.x.shape                           # (17856, 170, 1): steps, nodes, channels
g.edge_index.shape                  # (2, 277)
```

**Train on it** (batches are built on the GPU)

```python
train = g.task(split="train", normalize="channel")   # the task the dataset's paper defines
for batch in train.loader(batch_size=64, shuffle=True):
    prediction = model(batch["x"])                   # batch["x"]: [64, 12, 170, 1]
    error = (prediction - batch["y"]).abs() * batch["mask_y"]
    loss = error.sum() / batch["mask_y"].sum()       # MAE over observed readings
```

The loader is a standard PyTorch `DataLoader`, so `trainer.fit(model, train.loader(64))` works in
Lightning as is.

**Pretrain on a whole domain** (each batch comes from one dataset)

```python
pool = tgdata.ConcatTasks([d.task(normalize="channel")
                           for d in tgdata.load_domain("traffic_flow").values()])
for batch in pool.loader(batch_size=64, num_batches=10_000):
    ...
```

**Plot it**

```python
from tgdata.plot import plot_forecast, plot_signals

plot_signals(g, nodes=[0, 50], start="2016-08-01", end="2016-08-08")
plot_forecast(g.task(split="test")[0], nodes=[0, 50])
```

A complete walk-through is in [`examples/traffic_flow.py`](examples/traffic_flow.py), which
opens as a notebook (`jupytext --to notebook examples/traffic_flow.py`).

## Learn more

- [Datasets](docs/datasets.md): what is available, and where each dataset comes from.
- [Tasks and batching](docs/tasks.md): task options, batch contents, GPU performance.
- [Data format](docs/format.md): how datasets are stored, split and described.
- [Contributing](docs/contributing.md): development setup, adding a dataset, hosting.
