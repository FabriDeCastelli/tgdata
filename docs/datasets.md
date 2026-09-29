# Datasets

Every dataset follows the paper that introduced it: its graph, its split and its default task
are that paper's, cited in the dataset's metadata (`g.meta`). All are private to the
[`tgdata-hub`](https://huggingface.co/tgdata-hub) organisation on Hugging Face.

## Traffic flow

Flow of vehicles per 5 minutes on Californian freeways (Caltrans PeMS). The target is flow; a
reading of 0 means the sensor did not report and is masked, as all these sources do.

| name | sensors | steps | dates | covariates | graph | split | introduced by |
|---|---|---|---|---|---|---|---|
| `pems03` | 358 | 26,208 | 2018-09-01 – 2018-11-30 | – | road links, symmetric | 60/20/20 of time, no window crosses | STSGCN (Song et al., 2020) |
| `pems04` | 307 | 16,992 | 2018-01-01 – 2018-02-28 | occupancy, speed | road links, directed | 60/20/20 of windows | ASTGCN (Guo et al., 2019) |
| `pems07` | 883 | 28,224 | 2017-05-01 – 2017-08-06 | – | road links, symmetric | 60/20/20 of time, no window crosses | STSGCN (Song et al., 2020) |
| `pems08` | 170 | 17,856 | 2016-07-01 – 2016-08-31 | occupancy, speed | road links, directed | 60/20/20 of windows | ASTGCN (Guo et al., 2019) |
| `largest` | 8,600 | 105,120 | 2019-01-01 – 2019-12-31 | – | Gaussian kernel of road distance | 60/20/20 of windows | LargeST (Liu et al., 2023) |

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

## Domains and pretraining pools

Each dataset belongs to one domain, and each domain is a collection on the Hub.
`tgdata.load_domain(domain)` returns the domain's pretraining pool: the datasets whose readings
no other member contains, so no reading is seen twice. `largest`, `largest-sd`, `-gba` and
`-gla` contain districts and stay out of the pool; they load by name.

| domain | pool | sensors |
|---|---|---|
| `traffic_flow` | `pems03`, `pems04`, `pems07`, `pems08`, `largest-d3` … `largest-d12` | 10,318 |

`tgdata.list()` reads the list from the Hub; `tgdata/registry.json` is the offline copy.
