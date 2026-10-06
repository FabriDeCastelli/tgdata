# Datasets

Every dataset follows the paper that introduced it: its graph, its split and its default task
are that paper's, cited in the dataset's metadata (`g.meta`). All are private to the
[`tgdata-hub`](https://huggingface.co/tgdata-hub) organisation on Hugging Face.

## Traffic flow

Flow of vehicles per 5 minutes on Californian freeways (Caltrans PeMS). Every dataset holds
flow only (the PEMS04 and PEMS08 releases also carry occupancy and speed, which are left out).
A reading of 0 means the sensor did not report and is masked, as all these sources do.

| name | sensors | steps | dates | graph | split | introduced by |
|---|---|---|---|---|---|---|
| `pems03` | 358 | 26,208 | 2018-09-01 – 2018-11-30 | road links, symmetric | 60/20/20 of time, no window crosses | STSGCN (Song et al., 2020) |
| `pems04` | 307 | 16,992 | 2018-01-01 – 2018-02-28 | road links, directed | 60/20/20 of windows | ASTGCN (Guo et al., 2019) |
| `pems07` | 883 | 28,224 | 2017-05-01 – 2017-08-06 | road links, symmetric | 60/20/20 of time, no window crosses | STSGCN (Song et al., 2020) |
| `pems08` | 170 | 17,856 | 2016-07-01 – 2016-08-31 | road links, directed | 60/20/20 of windows | ASTGCN (Guo et al., 2019) |
| `largest` | 8,600 | 105,120 | 2019-01-01 – 2019-12-31 | Gaussian kernel of road distance | 60/20/20 of windows | LargeST (Liu et al., 2023) |

The default task of each is 12 steps (one hour) in, 12 steps out.

Notes:

- STSGCN states 5/1/2017 – 8/31/2017 for PEMS07, but its release holds 98 days; LargeST lists
  the same 5/1 – 8/6 range (`g.meta["date_note"]`).
- `largest` keeps LargeST's 5-minute release. LargeST's own benchmark averages to 15 minutes,
  so its published numbers are not comparable with runs on this data.

### Subsets of LargeST

`largest` is stored once, with each sensor's metadata in `g.node_table` (ID, latitude and
longitude, district, county, freeway, lanes, direction). Its districts and LargeST's own
subsets load like datasets, as views of the stored data:

| name | sensors | selection |
|---|---|---|
| `largest-d3`, `-d4`, `-d5`, `-d6`, `-d7`, `-d8`, `-d10`, `-d11`, `-d12` | 480, 2,352, 211, 484, 1,859, 1,022, 523, 716, 953 | one PeMS district each |
| `largest-sd` | 716 | San Diego (District 11), as in LargeST |
| `largest-gba` | 2,352 | Greater Bay Area (District 4), as in LargeST |
| `largest-gla` | 3,834 | Greater Los Angeles (Districts 7, 8, 12), as in LargeST |

```python
g = tgdata.load("largest-d5")          # or tgdata.load("largest", nodes="d5")
```

## Social

Graphs of interactions between accounts, whose edges change at every snapshot.

| name | accounts | snapshots | interval | features (stored) | target | graph | split | introduced by |
|---|---|---|---|---|---|---|---|---|
| `twittertennis-rg17` | 1,000 | 120 | 1 hour | degree, transitivity | mentions received | mentions, weighted by count | 80/10/10 of snapshots | PyG Temporal (Rozemberczki et al., 2021), from Béres et al. (2018) |
| `twittertennis-uo17` | 1,000 | 112 | 1 hour | degree, transitivity | mentions received | mentions, weighted by count | 80/10/10 of snapshots | PyG Temporal (Rozemberczki et al., 2021), from Béres et al. (2018) |

Twitter mention graphs among the 1,000 most popular accounts of Roland-Garros 2017 and the US
Open 2017. The default task is PyTorch Geometric Temporal's, sample for sample: snapshot t's
16 one-hot features (`features="pygt_encoded"`) and weighted mentions predict `log(1 + y)` of
snapshot t + 1, the last snapshot reusing the final label. Its batches are bit-identical to
PyG Temporal's `TwitterTennisDatasetLoader`. The 70/15/15 split of Gravina and Bacciu (2023) and
PyG Temporal's 80/20 are stored too. The release has no timestamps: which tournament hours the
snapshots are is not recorded.

## Mobility and epidemic (MOBINS)

The six datasets of MOBINS (Na et al., ICWSM 2025), from its Zenodo release. Each has node
series, a spatial network (binary, symmetric, with self-loops, as released) and an
origin-destination matrix of movements between nodes at every step, stored as the pair target
`y["od"]` `[T, N, N]`.

| name | domain | nodes | steps | dates | node channels | train / val / test windows |
|---|---|---|---|---|---|---|
| `mobins-seoul` | `mobility` | 128 | 17,520 hourly | 2022-01-01 – 2023-12-31 | inflow, outflow | 430 / 108 / 172 |
| `mobins-busan` | `mobility` | 60 | 26,280 hourly | 2021-01-01 – 2023-12-31 | inflow, outflow | 649 / 163 / 263 |
| `mobins-daegu` | `mobility` | 61 | 26,280 hourly | 2021-01-01 – 2023-12-31 | inflow, outflow | 649 / 163 / 263 |
| `mobins-nyc` | `mobility` | 5 | 18,960 hourly | 2022-02-01 – 2024-03-31 | ridership | 466 / 117 / 187 |
| `mobins-epi-korea` | `epidemic` | 16 | 1,320 daily | 2020-01-20 – 2023-08-31 | infections | 784 / 196 / 320 |
| `mobins-epi-nyc` | `epidemic` | 5 | 1,401 daily | 2020-03-01 – 2023-12-31 | infections | 832 / 209 / 340 |

The default task is MOBINS's: 4 days in, 7 days out, one window per day, forecasting the node
series and the OD matrix together from both (`batch["y"]` and `batch["y_od"]`). Its batches are
bit-identical to MOBINS's `_make_windowing_and_loader`, before its scaling, at 7, 14 and 30
days. The split is also MOBINS's: the last 25% of days are the test set, with windows cut
inside them, and the windows before split 80/20 into train and val. Licence CC BY-NC-ND 4.0
(the Zenodo record's).

## Transaction networks (MiNT)

The 84 ERC20 token transaction networks of MiNT (Shamsi, Ngo et al., NeurIPS 2025 Datasets and
Benchmarks), from the Zenodo release of their edge lists and Edge Growth/Shrink (Edge GS) labels
(record 15364297). One dataset per token, named `mint-<token>` in lower case (`mint-iotx`,
`mint-doge2.0`). Nodes are Ethereum addresses (`g.node_table["address"]`), 1,448 to 127,780 per
network.

A snapshot is a 7-day window of transfers and the next window starts one day later, so a
transfer appears in up to 7 consecutive snapshots, as in the release. Every row of the edge list is stored
as released: the multi-edges, the token amount as edge weight (a float32 of the release's
float64), and zero or negative amounts and self-loops where the release has them. Snapshot t
is `edge_index[:, edge_ptr[t]:edge_ptr[t+1]]`; each row's `date` column is recoverable from
`meta["first_date"]` and `meta["day_counts"]`. The label `y["edge_gs"]` is the release's: one 0/1
per snapshot, 1 when the transfers of the 7 days starting 10 days after the snapshot's own
start outnumber its own.

| role | networks | snapshots | `g.meta["role"]` |
|---|---|---|---|
| train | 64 | 80 to 2,160 | `train`, with `train_rank` 1 to 64 |
| test | 20 | 100 to 2,080 | `test` |

The 64 train and 20 held-out test networks are the paper's (`dataset_package_64.txt` and
`dataset_package_test.txt` of ScalingTGNs). `train_rank` is the position in the 64-list, whose
first 2, 4, 8, 16 and 32 networks are the paper's scaling subsets. Select by role:

```python
train = tgdata.load_domain("transaction", role="train")
held_out = tgdata.list(domain="transaction", role="test")
```

The default task is MiNT's: classify the label of a snapshot from that snapshot
(`graph_classification`, `target="edge_gs"`, `window=1`). The default split is MiNT's code:
the last `floor(0.15 T)` snapshots are test, the `floor(0.15 T)` before them validation and the rest
train (the paper's 70/15/15; the cuts differ from `0.7 T` by a snapshot for some T). The test
networks also have a `zero-shot` split whose test part is the last 30% of snapshots, which is
what MiNT's `test_foundation_tgc_64.py` scores them on. MiNT's own model collapses each snapshot to an unweighted undirected simple graph and adds
four pooled degree features; neither is stored here. Licence CC BY 4.0 (the Zenodo record's).

## Domains and pretraining pools

Each dataset belongs to one domain, and each domain is a collection on the Hub.
`tgdata.load_domain(domain)` returns the domain's pretraining pool: the datasets whose readings
no other member contains, so no reading is seen twice. `role="train"` or `"test"` keeps one side of a benchmark's networks. `largest`, `largest-sd`, `-gba` and
`-gla` contain districts and stay out of the pool; they load by name.

| domain | pool | sensors |
|---|---|---|
| `traffic_flow` | `pems03`, `pems04`, `pems07`, `pems08`, `largest-d3` … `largest-d12` | 10,318 |
| `social` | `twittertennis-rg17`, `twittertennis-uo17` | 2,000 |
| `mobility` | `mobins-seoul`, `mobins-busan`, `mobins-daegu`, `mobins-nyc` | 254 |
| `epidemic` | `mobins-epi-korea`, `mobins-epi-nyc` | 21 |
| `transaction` | the 84 `mint-*` networks (64 `train`, 20 `test`) | 3,268,430 addresses |

`tgdata.list()` reads the list from the Hub; `tgdata/registry.json` is the offline copy.
