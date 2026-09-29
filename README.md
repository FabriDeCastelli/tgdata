# tgdata

Temporal-graph datasets in one canonical format, stored as one Hugging Face dataset repo each.

```python
import tgdata

tgdata.list(domain="traffic_flow")                 # from Hub tags, falls back to registry.json
tgdata.info("pems08")                              # meta.json of the repo
g = tgdata.load("pems08", root="/data/tgdata")     # local copy if present, else the Hub; mmap
g.task()                                           # the task the dataset's source defines
g.task("node_forecasting", window="1h", horizon=12, split="test")  # 12 steps of 5 min
tgdata.tasks.available(g)                          # task types this dataset supports
g.with_split("70/10/20")                           # make a named split the default
g.to_pyg(); g.to_tsl()
```

Install with `pip install -e .`, plus `[pyg]`, `[tsl]` or `[tgb]` for the adapters.

## Supported datasets

| name | domain | default task | nodes | steps | freq | target | covariates | graph | source |
|---|---|---|---|---|---|---|---|---|---|
| `pems08` | traffic_flow | node forecasting, 12 → 12 | 170 | 17,856 | 5min | flow | occupancy, speed | static, directed, binary (ASTGCN) | Caltrans PeMS D8, ASTGCN release |

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

## Adding a dataset

Write `tgdata/converters/<name>.py`: build a `TemporalGraph` from the source's raw files,
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
