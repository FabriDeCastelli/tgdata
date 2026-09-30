# Tasks and batching

## Tasks

A task turns a dataset into training samples. Everything it needs (which samples exist, which
split each belongs to, the targets) is computed once when it is built, so iterating it is only
indexing.

```python
task = g.task()                                              # the dataset's own task
task = g.task("node_forecasting", window=12, horizon=12,    # or choose the parameters
              split="test", normalize="channel")
tgdata.tasks.available(g)                                    # task types this dataset supports
```

| task | a sample is | options |
|---|---|---|
| `node_forecasting` | the last `window` steps of every node, and the next `horizon` steps to predict | `window`, `horizon` (steps, or a duration such as `"1h"`), `split`, `normalize`, `stride`, `strict`, `target`, `device` |
| `node_regression` | the last `window` snapshots, and a stored node target at `offset` steps after the last one (clamped to the final step, as PyG Temporal does) | `target`, `window`, `offset`, `features`, `target_transform`, `split` |
| `graph_classification` | the last `window` snapshots, restricted to the nodes active in them, and the graph's label | `target`, `window`, `split`, `strict` |

- `split` is `"train"`, `"val"` or `"test"` of the dataset's default split, or `None` for the
  whole series; `splits=` picks another stored split, and `g.with_split("70/10/20")` makes one
  the default.
- A sample belongs to the split its targets fall in; its inputs may reach back into the previous
  split unless `strict=True` (some sources, such as STSGCN, make that the default).
- `normalize="channel"` or `"node"` standardises with the training statistics stored in the
  dataset; `transform=` takes any function of the batch instead.

## Batches

Tasks build batches on the GPU when one is available, and on the CPU otherwise (`device=` or the
`TGDATA_DEVICE` environment variable overrides this). Each dataset is uploaded once and shared by
all its tasks.

```python
loader = task.loader(batch_size=64, shuffle=True)    # a standard torch DataLoader
batch = next(iter(loader))
```

| key | shape | content |
|---|---|---|
| `x` | `[B, window, N, F]` | inputs, float32 |
| `y` | `[B, horizon, N, F]` | targets |
| `mask_x`, `mask_y` | `[B, window, N, 1]`, `[B, horizon, N, 1]` | `True` where a reading was observed |
| `covariates` | `[B, window + horizon, N, C]` | other measured channels, if any |
| `timestamps` | `[B, window + horizon]` | epoch seconds (for PeMS, local wall time; see `g.meta["timestamps_tz"]`) |
| `t` | `[B]` | the first target step of each sample |
| `edge_index`, `edge_weight` | `[2, E]`, `[E]` | the graph, once per batch |
| `edge_ptr` | `[window + 1]` | graphs that change over time only: see below |

For graphs whose edges change at every snapshot, a batch carries the edges of all its windows in
one set of tensors, built on the GPU without a Python loop. The edges of window step k of every
sample lie between `edge_ptr[k]` and `edge_ptr[k + 1]`, and sample b's node ids are shifted by
`b * N`, so one graph convolution per step covers the whole batch:

```python
x = batch["x"]                                            # [B, window, N, F]
B, w, N, F = x.shape
for k in range(w):
    lo, hi = batch["edge_ptr"][k], batch["edge_ptr"][k + 1]
    h = conv(x[:, k].reshape(B * N, F), batch["edge_index"][:, lo:hi],
             batch["edge_weight"][lo:hi])                 # [B * N, hidden]
```

`benchmarks/dynamic_edges.py` compares this with slicing each sample's snapshots in a Python
loop: the result is bit-identical, and 2.3x to 25x faster on an A100 for batches of 64 windows
of 1 to 12 snapshots.

`task[i]` returns one sample with the same keys and no batch dimension. Batches stay on the
device they were built on: call `.cpu()` before using numpy or matplotlib (tgdata's plotting
functions do this for you).

## Every window of a series, in order

`split=None` gives every window of the whole series, and `horizon=0` makes windows without a
target. With `shuffle=False` (the default) the windows come in series order: window k holds
steps k, ..., k + window - 1. Every batch also carries `t`, the step right after each window,
so `t - window` is its position even when batches are shuffled.

This is how to embed every window of a series and save the result to disk:

```python
task = g.task("node_forecasting", window=12, horizon=0, split=None)   # T - 11 windows
out = np.lib.format.open_memmap("emb.npy", mode="w+", dtype=np.float32,
                                shape=(len(task), g.num_nodes, dim))
with torch.no_grad():
    for batch in task.loader(batch_size=256):                  # add shuffle=True if needed
        out[(batch["t"] - task.window).cpu().numpy()] = encoder(batch["x"]).cpu().numpy()
out.flush()                                                     # row k: window starting at step k
```

The original train/val/test samples map onto these rows without any shift: `window_starts` is
each sample's row, so every split keeps exactly the samples, and the sizes, of the source's split.

```python
emb = np.load("emb.npy", mmap_mode="r")
for split in ("train", "val", "test"):
    task = g.task(split=split)                  # the source's task, e.g. 12 steps in, 12 out
    inputs = emb[task.window_starts]            # one embedding per sample of that split
    # the same rows, batch by batch: emb[(batch["t"] - task.window).cpu().numpy()]
```

## Several datasets

`ConcatTasks` joins tasks of different datasets. Every batch comes from a single dataset, chosen
at random with probability proportional to its size^(1/temperature): `temperature=1` follows
dataset sizes, larger values move towards picking datasets equally often.

```python
pool = tgdata.ConcatTasks([tgdata.load(n).task() for n in ["pems04", "pems08"]])
loader = pool.loader(batch_size=64, num_batches=1_000, temperature=2.0)
```

## Speed

`benchmarks/loader.py` compares tgdata with tsl's `SpatioTemporalDataset` and with windows
precomputed in memory (what the ASTGCN and STSGCN code does), on one A100, training the same
GCN-GRU with Lightning. The last three columns are the slowdown of a training step against
feeding the model a batch already on the GPU.

| dataset (batch size) | batches from tgdata | from tsl, 0 / 8 workers | step: tgdata | step: tsl, 0 / 8 workers | step: precomputed windows |
|---|---|---|---|---|---|
| PEMS07 (64) | 265,873 samples/s | 26,224 / 10,072 samples/s | +1.0% | +9.7% / +4.9% | +25.0% |
| LargeST CA (16) | 69,304 samples/s | 5,740 / 1,402 samples/s | +0.5% | +5.2% / +3.4% | – |
