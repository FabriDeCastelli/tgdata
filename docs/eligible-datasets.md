# Datasets by domain, filtered

Part 1 lists the datasets in tgdata. Part 2 lists datasets found online and not in tgdata, with the same filters.

Datasets that are purely discrete-time (no continuous-time graphs), with at least 50 nodes and at least 100 steps. Generated from `tgdata/registry.json`; numbers are the stored datasets', and a subset such as `largest-d5` keeps its parent's steps, frequency, channels and licence. Graph and dates come from [datasets.md](datasets.md).

## traffic_flow

| name | nodes | steps | interval | dates | channels | tasks | graph | licence |
|---|---|---|---|---|---|---|---|---|
| `largest` | 8,600 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | Gaussian kernel of road distance, static | cc-by-nc-4.0 |
| `largest-d10` | 523 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-d11` | 716 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-d12` | 953 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-d3` | 480 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-d4` | 2,352 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-d5` | 211 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-d6` | 484 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-d7` | 1,859 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-d8` | 1,022 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-gba` | 2,352 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-gla` | 3,834 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `largest-sd` | 716 | 105,120 | 5min | 2019-01-01 – 2019-12-31 | flow | node_forecasting | as `largest`, restricted to the selected sensors | cc-by-nc-4.0 |
| `pems03` | 358 | 26,208 | 5min | 2018-09-01 – 2018-11-30 | flow | node_forecasting | road links, symmetric, static | other |
| `pems04` | 307 | 16,992 | 5min | 2018-01-01 – 2018-02-28 | flow | node_forecasting | road links, directed, static | other |
| `pems07` | 883 | 28,224 | 5min | 2017-05-01 – 2017-08-06 | flow | node_forecasting | road links, symmetric, static | other |
| `pems08` | 170 | 17,856 | 5min | 2016-07-01 – 2016-08-31 | flow | node_forecasting | road links, directed, static | other |

## mobility

| name | nodes | steps | interval | dates | channels | tasks | graph | licence |
|---|---|---|---|---|---|---|---|---|
| `mobins-busan` | 60 | 26,280 | 1h | 2021-01-01 – 2023-12-31 | inflow, outflow | node_forecasting | binary spatial network, static; OD matrix per step | cc-by-nc-nd-4.0 |
| `mobins-daegu` | 61 | 26,280 | 1h | 2021-01-01 – 2023-12-31 | inflow, outflow | node_forecasting | binary spatial network, static; OD matrix per step | cc-by-nc-nd-4.0 |
| `mobins-seoul` | 128 | 17,520 | 1h | 2022-01-01 – 2023-12-31 | inflow, outflow | node_forecasting | binary spatial network, static; OD matrix per step | cc-by-nc-nd-4.0 |

## social

| name | nodes | steps | interval | dates | channels | tasks | graph | licence |
|---|---|---|---|---|---|---|---|---|
| `twittertennis-rg17` | 1,000 | 120 | 1h | Roland-Garros 2017 (no timestamps) | degree, transitivity | node_regression, node_forecasting | mentions, weighted by count, edges change every step | unknown |
| `twittertennis-uo17` | 1,000 | 112 | 1h | US Open 2017 (no timestamps) | degree, transitivity | node_regression, node_forecasting | mentions, weighted by count, edges change every step | unknown |

## epidemic

No dataset qualifies.

## transaction

83 of the 84 `mint-*` networks qualify (MiNT, Shamsi, Ngo et al., 2025; see [datasets.md](datasets.md)): 64 `train` and 20 `test` networks, 1,448 to 127,780 addresses, 100 to 2,160 daily-stride snapshots of 7-day windows, token-amount edge weights, graph-level Edge GS label, tasks `graph_classification`, CC BY 4.0, edges change every step. The one that does not is below.

## Discarded from tgdata

| name | domain | reason |
|---|---|---|
| `mobins-epi-korea` | epidemic | 16 nodes |
| `mobins-epi-nyc` | epidemic | 5 nodes |
| `mobins-nyc` | mobility | 5 nodes |
| `mint-mog` | transaction | 80 steps |

# Part 2: candidates online, not in tgdata

Same filters: purely discrete-time, at least 50 nodes, at least 100 steps. "checked" means I read the file or the source page; "derived" means computed from a stated time span; "unchecked" means a search snippet or a paper only. None of these has a converter yet, and each licence must be confirmed before a push.

## traffic_flow

| name | nodes | steps | interval | dates | channels | graph | licence | status |
|---|---|---|---|---|---|---|---|---|
| METR-LA | 207 | 34,272 | 5min | 2012-03-01 – 2012-06-27 | speed | road distance, static (DCRNN) | CC BY 4.0 | checked: CSV on [Zenodo 5146275](https://zenodo.org/records/5146275) |
| PEMS-BAY | 325 | 52,116 | 5min | 2017-01-01 – 2017-06-30 | speed | road distance, static (DCRNN) | CC BY 4.0 | checked: same record |

## mobility

| name | nodes | steps | interval | dates | channels | graph | licence | status |
|---|---|---|---|---|---|---|---|---|
| SHMetro | 288 stations | unchecked | 15min | 2016-07 – 2016-09 | inflow, outflow | metro lines | not found | unchecked (PVCGN) |
| HZMetro | 80 stations | unchecked, about 25 days of operating hours | 15min | 2019-01 | inflow, outflow | metro lines | not found | unchecked (PVCGN) |
| Montevideo bus | 675 stops | 744 | not stated in the JSON | not stated | boardings | bus routes, static | MIT (PyG-Temporal repo) | checked: `montevideo_bus.json` |

## energy

| name | nodes | steps | interval | dates | channels | graph | licence | status |
|---|---|---|---|---|---|---|---|---|
| SDWPF | 134 turbines | about 105,000 | 10min | 2020-01 – 2021-12 | 19 SCADA and weather variables | built from turbine coordinates | not read | derived from a 24-month span; [Figshare](https://figshare.com/articles/dataset/SDWPF_dataset/24798654) |
| EIA-930 | not counted | not counted | 1h | bulk files, current data from 2019 plus a historical file | demand, generation, interchange | interchange between balancing authorities | EIA reuse policy | unchecked node count |

## weather

| name | nodes | steps | interval | dates | channels | graph | licence | status |
|---|---|---|---|---|---|---|---|---|
| Weather2K-R | 1,866 stations | 13,632 | 3h | not read | 13 variables | built from coordinates | non-commercial, agreement form | checked: Hugging Face `BUPT-PRIS-727/Weather2K` |

## hydrology

| name | nodes | steps | interval | dates | channels | graph | licence | status |
|---|---|---|---|---|---|---|---|---|
| LamaH-CE | 859 catchments | at least 12,775 daily or 306,000 hourly | 1D and 1h | 35+ years from 1981 | runoff and meteorology | river network, static | CC BY 4.0 | unchecked: [Zenodo 5153305](https://zenodo.org/records/5153305), steps derived from the span |

## web

| name | nodes | steps | interval | dates | channels | graph | licence | status |
|---|---|---|---|---|---|---|---|---|
| WikiMath (PyG-Temporal `wikivital_mathematics`) | 1,068 | 731 | 1D | not stated | page visits | hyperlinks, static | MIT (PyG-Temporal repo) | checked: JSON. A 2026 audit calls it first-order differenced ([arXiv 2608.20980](https://arxiv.org/pdf/2608.20980)) |

## social and finance (evolving topology)

| name | nodes | steps | interval | dates | channels | graph | licence | status |
|---|---|---|---|---|---|---|---|---|
| ORBITAAL daily snapshots | millions of entities, count not read | about 4,400 | 1D | 2009-01-03 – 2021-01-25 | none (BTC and USD edge weights) | Bitcoin entity transactions, evolving | CC BY 4.0 | [Zenodo 12581515](https://zenodo.org/records/12581515); 24.8 GB for the daily snapshots, 156.9 GB total |

## Candidates discarded

| name | reason |
|---|---|
| EnglandCovid | 61 steps (checked) |
| Chickenpox Hungary | 20 nodes, 521 steps (checked) |
| PedalMe London | 15 nodes (checked) |
| tgbn-trade | 32 yearly steps |
| BACI | about 30 yearly steps (1995 to present) |
| Elliptic, Elliptic++ | 49 steps |
| DBLP-3, Brain | 10 and 12 steps (unchecked) |
| tgbn-genre, tgbn-reddit, tgbn-token | continuous-time event streams (C-TDG) |
| DGB datasets (Wikipedia, Reddit, MOOC, LastFM, Enron, UCI, ...) | continuous-time event streams (C-TDG) |
| SocioPatterns, Copenhagen Networks Study | contact events at 20 s to 5 min resolution; continuous-time sources |

## Not checkable against the filter

Lightning Network snapshots (snapshot count not read), TrafficStream (node count per year not read), Temporal Network Benchmark (26 networks, per-network sizes only inside the zip), CAMELSH (page not retrievable), Wikipedia clickstream (monthly, about 100 months at most, node counts not read).
