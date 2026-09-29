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
| `graph_classification` | the last `window` snapshots, restricted to the nodes active in them, and the graph's label | `target`, `window`, `split`, `strict` |

- `split` is `"train"`, `"val"` or `"test"` of the dataset's default split; `splits=` picks
  another stored split, and `g.with_split("70/10/20")` makes one the default.
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

`task[i]` returns one sample with the same keys and no batch dimension. Batches stay on the
device they were built on: call `.cpu()` before using numpy or matplotlib (tgdata's plotting
functions do this for you).

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
