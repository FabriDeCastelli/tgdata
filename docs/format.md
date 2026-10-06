# Data format

Every dataset is a `TemporalGraph`, stored as one Hugging Face dataset repo. The format is the
same for every source, so code written for one dataset runs on all of them.

## Principles

- **Raw data, as collected.** Values keep the source's sampling interval (`g.meta["freq"]`) and
  are never resampled, aggregated or normalised in storage. Windowing and normalisation happen
  when a task is built.
- **The introducing paper decides.** Graph weighting, split and default task follow the paper
  that introduced the dataset, and the metadata cites where in its code or text.
- **Nothing is lost.** Storage is compact but exact: every value decodes to the float32 the
  source holds.

## Fields

| field | shape | meaning |
|---|---|---|
| `x` | `[T, N, F]` | node signals: T steps, N nodes, F channels (the forecasting targets) |
| `covariates` | `[T, N, C]` or `[T, C]` | other measured channels |
| `mask` | `[T, N]` | `True` where a reading was observed |
| `timestamps` | `[T]` | int64 epoch seconds of each step |
| `edge_index`, `edge_weight` | `[2, E]`, `[E]` | edges and their weights, required together |
| `edge_ptr` | `[T + 1]` | for graphs that change over time: snapshot t is `edge_index[:, edge_ptr[t]:edge_ptr[t + 1]]` |
| `node_features`, `node_time`, `node_table` | `[N, D]`, `[N]`, columns of length N | static node data, when each node appears, per-node metadata |
| `y` | dict of `Target` | supervised targets other than future values of `x` (a label per graph, per node, per edge, or per node pair such as an origin-destination matrix `[T, N, N]`) |
| `splits` | dict of `Split` | train/val/test, as the source defines it; `"default"` is the source's. Fractions cut steps or windows; a `holdout` part (for example MOBINS's last 25% of days) is cut first, and the fractions then split the windows before it |
| `node_sets` | dict of id arrays | named node subsets, loadable as datasets |
| `meta` | dict | everything else: units, sources, statistics, the default task |

`g.select_nodes(...)` returns the subgraph of a node set; `g.with_split(...)` changes the
default split; `tgdata.validate(g)` checks shapes, types and consistency.

## Files

```
arrays/*.npy          node signals, mask, timestamps, edges, node data (memory-mapped on load)
y/<name>/*.npy        targets
splits/<name>/*.npy   node ids of node splits
node_sets/*.npy       named node subsets
nodes.parquet         per-node metadata table
events.parquet        continuous-time graphs only: src, dst, t, msg
meta.json             schema version, splits, targets, encodings, metadata
README.md             Hugging Face card, tagged domain:*, time:*, task:*, role:* (when set)
```

## Compact storage

Each channel is stored as the smallest integer type that holds it exactly, with a number of
decimals per channel: flow as `int16`, occupancy as `int16` with 4 decimals, speed with 1.
Reading `g.x[...]` decodes to float32, so code never sees the encoding. A mask that equals
`x != 0`, as in every traffic dataset, is not stored but recomputed.

Encoded arrays support indexing, slicing, `.shape` and `np.asarray(...)`; use
`np.asarray(g.x)` for whole-array numpy methods such as `.mean()`.

## Metadata

| key | meaning |
|---|---|
| `freq`, `timestamps_tz` | sampling interval, and how timestamps relate to local time |
| `channels`, `covariate_channels`, `units` | channel names and units |
| `mask_rule` | what counts as a missing reading, and the source's reference |
| `edge_weight` | what the stored weights are (`kind`, `units`), and whether they were `observed`; derived weights carry their `formula` |
| `adjacency` | how the introducing paper weights edges (`raw`, `binary` or `gaussian`), with its `formula` and `reference`; tasks follow it |
| `default_task` | the source's task and parameters, with its reference |
| `stats`, `stats_node` | mean and standard deviation of `x` over observed training steps, per channel and per node |
| `source`, `license`, `citation`, `provenance` | where the data comes from, and sha256 hashes of the raw files |
