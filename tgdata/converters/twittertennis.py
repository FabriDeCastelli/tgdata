"""TwitterTennis (RG17, UO17) as packaged by PyTorch Geometric Temporal.

Hourly mention graphs among the 1,000 most popular accounts of Roland-Garros 2017 and US
Open 2017 (Béres et al., 2018), in the JSON files PyG Temporal's TwitterTennisDatasetLoader
reads. Stored raw: degree and transitivity features, mention counts as edge weights and
targets; PyG's one-hot features and log(1 + y) targets are applied by the default task.

python -m tgdata.converters.twittertennis --event rg17 --out DIR [--raw FILE] [--push]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from .. import hub, io
from ..schema import Split, Target, TemporalGraph, validate

URL = ("https://raw.githubusercontent.com/ferencberes/pytorch_geometric_temporal/developer/"
       "dataset/twitter_tennis_{event}.json")
LOADER = "torch_geometric_temporal/dataset/twitter_tennis.py (TwitterTennisDatasetLoader)"
EVENTS = {"rg17": "Roland-Garros 2017 (May 22 - June 11)",
          "uo17": "US Open 2017 (August 22 - September 10)"}
CITATION = """
@article{beres2018temporal,
  title={Temporal walk based centrality metric for graph streams},
  author={B{\\'e}res, Ferenc and P{\\'a}lovics, R{\\'o}bert and Ol{\\'a}h, Anna and
          Bencz{\\'u}r, Andr{\\'a}s A.},
  journal={Applied Network Science}, volume={3}, pages={32}, year={2018}
}
@inproceedings{rozemberczki2021pytorch,
  title={PyTorch Geometric Temporal: Spatiotemporal Signal Processing with Neural Machine
         Learning Models},
  author={Rozemberczki, Benedek and Scherer, Paul and He, Yixuan and Panagopoulos, George and
          Riedel, Alexander and Astefanoaei, Maria and Kiss, Oliver and Beres, Ferenc and
          L{\\'o}pez, Guzm{\\'a}n and Collignon, Nicolas and Sarkar, Rik},
  booktitle={CIKM}, year={2021}
}
"""


def build(event: str, data: dict[str, Any], provenance: dict[str, str]) -> TemporalGraph:
    T = data["time_periods"]
    snapshots = [data[str(t)] for t in range(T)]
    edges = [np.asarray(s["edges"], dtype=np.int64).reshape(-1, 2) for s in snapshots]
    counts = np.array([len(e) for e in edges])
    accounts = sorted(data["node_ids"], key=data["node_ids"].get)
    g = TemporalGraph(
        name=f"twittertennis-{event}",
        domain="social",
        tasks=["node_regression", "node_forecasting"],
        time_mode="discrete",
        x=np.stack([np.asarray(s["X"], dtype=np.float64) for s in snapshots]),
        edge_index=np.concatenate(edges).T.copy(),
        edge_weight=np.concatenate([s["weights"] for s in snapshots]).astype(np.float32),
        edge_ptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        y={"mentions": Target(
            level="node", kind="regression",
            values=np.stack([np.asarray(s["y"], dtype=np.int64) for s in snapshots]),
            source="shipped: 'y' of each snapshot, the mentions each account received in the "
                   f"original tweet collection ({LOADER} docstring)")},
        node_table={"account": np.array(accounts)},
        splits={
            "default": Split(fractions={"train": 0.8, "val": 0.1, "test": 0.1}, over="samples",
                             reference="80/10/10 chronological split of the snapshots, taken "
                             "as the literature default (project decision)"),
            "70/15/15": Split(fractions={"train": 0.7, "val": 0.15, "test": 0.15},
                              over="samples", reference="Gravina and Bacciu (2023), Deep "
                              "learning for dynamic graphs: models and benchmarks, Sec. V"),
            "80/20": Split(fractions={"train": 0.8, "test": 0.2}, over="samples",
                           reference="torch_geometric_temporal temporal_signal_split "
                           "(train_ratio=0.8)"),
        },
        meta={
            "freq": "1h",
            "time_note": "hourly snapshots (PyG Temporal paper, Table 1); the release does not "
                         f"say which hours of {EVENTS[event]} they are, so there are no "
                         "timestamps",
            "num_nodes": len(accounts),
            "channels": ["degree", "transitivity"],
            "covariate_channels": [],
            "units": {"degree": "accounts", "transitivity": "fraction"},
            "directed": True,
            "edge_weight": {
                "kind": "mention_count",
                "observed": True,
                "units": "mentions per hour",
                "description": "'weights' of each snapshot: how many times one account "
                               "mentioned the other in that hour",
            },
            "adjacency": {"kind": "raw", "reference": f"{LOADER}: edge weights are the "
                          "snapshot 'weights', unchanged"},
            "default_task": {
                "name": "node_regression",
                "params": {"target": "mentions", "window": 1, "offset": 1,
                           "features": "pygt_encoded", "target_transform": "log1p"},
                "reference": f"{LOADER}: feature_mode='encoded', target_offset=1, "
                             "targets log(1 + y), the last snapshot reusing the final label",
            },
            "source": f"Twitter mentions during {EVENTS[event]}, Béres et al. (2018), as "
                      "packaged by PyTorch Geometric Temporal (Rozemberczki et al., 2021)",
            "license": "unknown",
            "citation": CITATION,
            "provenance": provenance,
        },
    )
    validate(g)
    return g


def read_raw(event: str, raw: Path | None) -> tuple[dict[str, Any], dict[str, str]]:
    url = URL.format(event=event)
    content = raw.read_bytes() if raw is not None else urllib.request.urlopen(url).read()
    provenance = {"url": url, "sha256": hashlib.sha256(content).hexdigest()}
    return json.loads(content), provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", choices=sorted(EVENTS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw", type=Path, help="local copy of the JSON; downloaded otherwise")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    g = build(args.event, *read_raw(args.event, args.raw))
    io.save(g, args.out)
    print(f"saved {g.name} to {args.out}")
    if args.push:
        print(hub.push(args.out, g.name))


if __name__ == "__main__":
    main()
