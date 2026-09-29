# tgdata

Temporal-graph datasets in one canonical format, stored as one Hugging Face dataset repo each.

```python
import tgdata

tgdata.list(domain="traffic_flow")                 # from Hub tags, falls back to registry.json
tgdata.domains()                                   # {domain: [names]}
tgdata.load_domain("traffic_flow")                 # the domain's pretraining pool
tgdata.info("pems08")                              # meta.json of the repo
g = tgdata.load("pems08", root="/data/tgdata")     # local copy if present, else the Hub; mmap
g.task()                                           # the task the dataset's source defines
g.task("node_forecasting", window="1h", horizon=12, split="test")  # 12 steps of 5 min
tgdata.tasks.available(g)                          # task types this dataset supports
g.with_split("70/10/20")                           # make a named split the default
tgdata.load("largest-d5")                          # a named node subset: a view, nothing copied
g.to_pyg(); g.to_tsl()
```

Install with `pip install "tgdata @ git+ssh://git@github.com/FabriDeCastelli/tgdata.git@v0.1.0"`,
adding `[plot]` for plotting and `[pyg]`, `[tsl]` or `[tgb]` for the adapters.

## Supported datasets

| name | domain | default task | nodes | steps | dates | freq | target | covariates | graph | split | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `pems03` | traffic_flow | node forecasting, 12 → 12 | 358 | 26,208 | 2018-09-01 – 2018-11-30 | 5min | flow | – | binary, symmetric | 60/20/20 of steps, windows within parts | PeMS D3, STSGCN |
| `pems04` | traffic_flow | node forecasting, 12 → 12 | 307 | 16,992 | 2018-01-01 – 2018-02-28 | 5min | flow | occupancy, speed | binary, directed | 60/20/20 of samples | PeMS D4, ASTGCN |
| `pems07` | traffic_flow | node forecasting, 12 → 12 | 883 | 28,224 | 2017-05-01 – 2017-08-06 | 5min | flow | – | binary, symmetric | 60/20/20 of steps, windows within parts | PeMS D7, STSGCN |
| `pems08` | traffic_flow | node forecasting, 12 → 12 | 170 | 17,856 | 2016-07-01 – 2016-08-31 | 5min | flow | occupancy, speed | binary, directed | 60/20/20 of samples | PeMS D8, ASTGCN |

| `largest` | traffic_flow | node forecasting, 12 → 12 | 8,600 | 105,120 | 2019-01-01 – 2019-12-31 | 5min | flow | – | Gaussian kernel of road distance, directed | 60/20/20 of samples, rounded | PeMS, LargeST |

Each follows the paper that introduced it: ASTGCN (Guo et al., 2019) for PEMS04/08, STSGCN
(Song et al., 2020) for PEMS03/07 and LargeST (Liu et al., 2023) for `largest`. Missing
readings are masked by `flow != 0` in all of them, as their sources do. STSGCN states
5/1/2017 – 8/31/2017 for PEMS07, but its release holds 98 days; see `meta.date_note`.
`largest` keeps LargeST's 5-minute release; LargeST's own benchmark averages to 15 minutes.

`largest` is stored once, with each sensor's metadata in `g.node_table` (ID, lat/lng, district,
county, freeway, lanes, direction). Its districts and LargeST's subsets are named node sets,
loadable as datasets:

| name | nodes | selection |
|---|---|---|
| `largest-d3` … `largest-d12` | 211 – 2,352 | one PeMS district each (D3, D4, D5, D6, D7, D8, D10, D11, D12) |
| `largest-sd`, `largest-gba`, `largest-gla` | 716, 2,352, 3,834 | LargeST's San Diego (D11), Bay Area (D4) and Los Angeles (D7, D8, D12) |

## Domains

Every dataset belongs to one domain, and each domain is a collection on the Hub (e.g. "Traffic
flow" in `tgdata-hub`). `tgdata.load_domain(domain)` returns the domain's pretraining pool: the
datasets whose readings no other member contains. Wholes and unions of other members
(`largest`, `largest-sd`, `-gba`, `-gla`) stay out of it, so no reading is seen twice; they
load by name. The traffic-flow pool is PEMS03/04/07/08 and the nine LargeST districts,
10,318 sensors.

`tgdata.list()` gives the live list from the Hub, and `tgdata/registry.json` is the offline copy.

## Format (schema v2)

Each repo holds:

| file | content |
|---|---|
| `arrays/*.npy` | `x [T,N,F]`, `mask [T,N]` or `[T,N,F]`, `covariates [T,C]` or `[T,N,C]`, `timestamps [T]` (int64), `edge_index [2,E]`, `edge_weight [E]`, `edge_ptr [T+1]` (absent = static), `node_features [N,D]`, `node_time [N]` |
| `y/<name>/*.npy` | supervised targets: `values`, and `steps` / `index` when sparse |
| `splits/<name>/*.npy` | node ids per part, for node splits |
| `events.parquet` | `src, dst, t, msg` for continuous-time (TGB) graphs |
| `meta.json` | `schema_version`, identity, target and split specs, `meta` |
| `README.md` | card tagged `domain:*`, `time:*`, `task:*` |

Every field except the identity (`name`, `domain`, `tasks`, `time_mode`) is optional; `x` is
`None` for graphs without a node time series.

**Time.** Data is stored at the source's own sampling interval or snapshots (`meta.freq`),
never resampled or aggregated. Windows and horizons count those steps; a duration such as
`"1h"` is accepted only when it is a whole number of steps.

**Time-varying edges.** The edges of all snapshots are concatenated in `edge_index`, and
snapshot t is `edge_index[:, edge_ptr[t]:edge_ptr[t+1]]`, so any window of snapshots is one
contiguous slice of the memory-mapped array.

**Targets.** `g.y[name]` is a `Target` with `level` (node, edge, graph), `kind` (regression,
class), `values`, and optional `steps`, `index`, `num_classes` and `source`. Forecasting `x`
needs no stored target.

**Splits.** `g.splits[name]` is a `Split` stored as its source defines it: chronological
`fractions` cut over steps or over samples, date `boundaries`, or node sets. It is resolved to
indices only when a task is built, on that task's samples; `"default"` is the source's split.

**Metadata conventions.**

- Values are stored raw. `stats` (per channel) and `stats_node` (per node and channel) cover
  valid entries of the default split's training steps. `normalize="channel" | "node"` applies
  them; pass `transform=` for anything else, or recompute with `tgdata.compute_stats`.
- `x` holds the forecasting target channels only (univariate for traffic flow). Other measured
  channels go in `covariates`, named by `covariate_channels`.
- `edge_weight` is required whenever there are edges. `meta.edge_weight` describes it with
  `kind`, `units`, and `observed`. Weights not directly observed by the collection process
  (`observed: false`, e.g. a kernel of distances) must carry the `formula` that derived them.
- `meta.adjacency` records how the paper that introduced the dataset weights edges
  (`"raw"`, `"binary"`, or `"gaussian"` with `sigma`/`threshold`) with its `formula` and
  `reference`. Tasks and adapters follow it; `adjacency_kind=` overrides it.
- Static topologies have no duplicate edges (`validate` rejects them); time-varying ones
  may repeat edges as genuine multi-edges.
- `meta.default_task` names the task the source defines, with its parameters and reference.

## Tasks

A task is a torch `Dataset` whose samples and targets are all computed when it is built,
deterministically; `__getitem__` only reads them.

| task | sample | collate |
|---|---|---|
| `node_forecasting` | `x` of steps `[t-window, t)`, target of `[t, t+horizon)`, masks, covariates, edges | `collate_pad` |
| `graph_classification` | snapshots `[t-window+1, t]`, restricted to nodes active in the window and relabelled (`node_ids` maps back), graph target at `t` | `collate_concat` |

A sample belongs to a split when its targets lie inside it; its inputs may reach back into the
previous split unless `strict=True`. New task types register with `@tgdata.tasks.register_task`.

## Batching

```python
from functools import partial

from torch.utils.data import ConcatDataset, DataLoader
from tgdata.sampling import MultiDatasetSampler, collate_pad

parts = [tgdata.load(n).task() for n in names]
sampler = MultiDatasetSampler([len(p) for p in parts], batch_size=32, num_batches=10_000,
                              temperature=2.0)
loader = DataLoader(ConcatDataset(parts), batch_sampler=sampler,
                    collate_fn=partial(collate_pad, max_nodes=512))
```

Each batch comes from one dataset, picked with probability ∝ size^(1/temperature).

- `collate_pad` pads nodes to the largest graph in the batch (`node_mask`, `node_ids`) and
  replaces graphs over `max_nodes` by a random induced subgraph; edges stay per sample in
  `batch["edges"]`.
- `collate_concat` concatenates nodes with a `batch` vector, as PyG does, and merges edges step
  by step, so one `edge_ptr [window + 1]` indexes window step k across the whole batch.

## Plotting

```python
from tgdata.plot import plot_forecast, plot_heatmap, plot_signals

plot_signals(g, nodes=[0, 50], start="2016-08-01", end="2016-08-08")  # missing readings are gaps
plot_heatmap(g, start=0, end=288 * 7)                                  # all nodes over a week
sample = g.task(split="test")[0]
plot_forecast(sample, nodes=[0, 50], prediction=model(sample))         # input, target, prediction
```

Each returns its matplotlib axes; pass `ax=` to draw into your own figure.

## Adding a dataset

Write `tgdata/converters/<name>.py` (see `converters/pems.py`): build a `TemporalGraph` from the
source's raw files,
`validate` it, `tgdata.save` it, then push with `tgdata.push(out_dir, name)`, and add its entry
to `tgdata/registry.json`. No registration on the Hub is needed: `push` creates the repo
(private) under `TGDATA_NAMESPACE` (default `tgdata-hub`), and its card tags make it listable.

## Hosting on Hugging Face

One-time setup:

1. Create the organisation `tgdata-hub` at <https://huggingface.co/organizations/new>.
2. Create a token at <https://huggingface.co/settings/tokens>: *write* access to the
   `tgdata-hub` repos to push, *read* access on machines that only load.
3. Log in on each machine with `hf auth login`, or export `HF_TOKEN`.

On a new machine, `tgdata.load(name)` then looks for the dataset under `root`
(`$TGDATA_ROOT`, default `~/.cache/tgdata`), and downloads it from the Hub only when it is
missing. Pass `revision=` to pin a version.
