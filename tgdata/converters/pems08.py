"""PEMS08 from the ASTGCN release: PEMS08.npz (data [T, N, 3]) and PEMS08.csv (from, to, cost).

python -m tgdata.converters.pems08 --raw DIR --out DIR [--push]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import hub, io
from ..schema import TemporalGraph, compute_stats, make_splits, validate
from . import Edge, drop_duplicate_edges

NAME = "pems08"
START = datetime(2016, 7, 1, tzinfo=timezone.utc)
END = datetime(2016, 8, 31, 23, 55, tzinfo=timezone.utc)
STEP_SECONDS = 300
NUM_STEPS = 17856
CITATION = """
@inproceedings{guo2019astgcn,
  title={Attention Based Spatial-Temporal Graph Convolutional Networks
         for Traffic Flow Forecasting},
  author={Guo, Shengnan and Lin, Youfang and Feng, Ning and Song, Chao and Wan, Huaiyu},
  booktitle={AAAI},
  year={2019}
}
"""


def build(data: np.ndarray, edges: list[Edge], provenance: dict[str, str]) -> TemporalGraph:
    T, N, _ = data.shape
    edges = drop_duplicate_edges(edges)
    timestamps = int(START.timestamp()) + STEP_SECONDS * np.arange(T, dtype=np.int64)
    edge_index = np.array([(s, d) for s, d, _ in edges], dtype=np.int64).T.reshape(2, -1)
    g = TemporalGraph(
        name=NAME,
        domain="traffic_flow",
        task="forecasting",
        time_mode="discrete",
        x=data[..., :1].astype(np.float32),
        covariates=data[..., 1:].astype(np.float32),
        mask=data[..., 0] != 0,
        timestamps=timestamps,
        edge_index=edge_index,
        edge_weight=np.array([c for _, _, c in edges], dtype=np.float32),
        splits=make_splits(T, 0.2, 0.2),
        meta={
            "freq": "5min",
            "timestamps_tz": "America/Los_Angeles wall time, encoded as UTC epoch seconds",
            "channels": ["flow"],
            "covariate_channels": ["occupancy", "speed"],
            "units": {"flow": "vehicles/5min", "occupancy": "fraction", "speed": "mph"},
            "mask_rule": "flow != 0",
            "directed": True,
            "edge_weight": {
                "kind": "distance",
                "observed": True,
                "units": "unspecified by the source",
                "description": "'cost' column of PEMS08.csv, the road distance between sensors",
            },
            "adjacency": {
                "kind": "binary",
                "formula": "w_ij = 1 for every (i, j) row of PEMS08.csv, 0 otherwise; directed",
                "reference": "ASTGCN lib/utils.py get_adjacency_matrix; "
                "ASTGCN-r-pytorch configurations/PEMS08_astgcn.conf (in_channels = 1)",
            },
            "alt_splits": {
                "60/20/20": make_splits(T, 0.2, 0.2),
                "70/10/20": make_splits(T, 0.1, 0.2),
            },
            "source": "Caltrans PeMS District 8, ASTGCN release (Guo et al., 2019)",
            "license": "other",
            "citation": CITATION,
            "provenance": provenance,
        },
    )
    g.meta["stats"] = compute_stats(g, "train")
    g.meta["stats_node"] = compute_stats(g, "train", per_node=True)
    validate(g)
    return g


def read_raw(raw: Path) -> tuple[np.ndarray, list[Edge], dict[str, str]]:
    npz, csv_path = raw / "PEMS08.npz", raw / "PEMS08.csv"
    data = np.load(npz)["data"]
    with csv_path.open() as f:
        edges = [(int(r["from"]), int(r["to"]), float(r["cost"])) for r in csv.DictReader(f)]
    provenance = {p.name: _sha256(p) for p in (npz, csv_path)}
    return data, edges, provenance


def check_calendar(g: TemporalGraph) -> None:
    assert g.num_steps == NUM_STEPS, g.num_steps
    assert g.timestamps is not None
    first, last = (datetime.fromtimestamp(int(s), timezone.utc) for s in g.timestamps[[0, -1]])
    assert (first, last) == (START, END), (first, last)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    g = build(*read_raw(args.raw))
    check_calendar(g)
    io.save(g, args.out)
    print(f"saved {g.name} to {args.out}")
    if args.push:
        print(hub.push(args.out, NAME))


if __name__ == "__main__":
    main()
