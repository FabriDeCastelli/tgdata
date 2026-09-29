"""LargeST (Liu et al., NeurIPS 2023): 8,600 California sensors, 5-minute flow, one year.

Stored once as CA with per-sensor metadata; districts and LargeST's own subsets (SD, GBA,
GLA) are named node sets, loaded as views. Needs the Kaggle release (liuxu77/largest):
ca_his_raw_<year>.h5, ca_meta.csv and ca_rn_adj.npy (`pip install tgdata[convert]`).

python -m tgdata.converters.largest --raw DIR --out DIR [--year 2019] [--push]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .. import hub, io
from ..schema import Split, TemporalGraph, compute_stats, validate

NAME = "largest"
CITATION = """
@inproceedings{liu2023largest,
  title={LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting},
  author={Liu, Xu and Xia, Yutong and Liang, Yuxuan and Hu, Junfeng and Wang, Yiwei and
          Bai, Lei and Huang, Chao and Liu, Zhenguang and Hooi, Bryan and Zimmermann, Roger},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
"""
# LargeST's subsets, from data/{sd,gba,gla}/generate_*_dataset.ipynb.
SUBSETS = {"sd": (11,), "gba": (4,), "gla": (7, 8, 12)}
TABLE_COLUMNS = {"ID": str, "Lat": float, "Lng": float, "District": int, "County": str,
                 "Fwy": str, "Lanes": int, "Type": str, "Direction": str}


def build(
    flow: np.ndarray,
    timestamps: np.ndarray,
    table: dict[str, np.ndarray],
    adjacency: np.ndarray,
    provenance: dict[str, str],
) -> TemporalGraph:
    """`flow` [T, N] with NaN for missing readings; `adjacency` [N, N] as released."""
    src, dst = np.nonzero(adjacency)
    weights = adjacency[src, dst].astype(np.float32)
    districts = table["district"]
    node_sets = {f"d{d}": np.flatnonzero(districts == d) for d in np.unique(districts)}
    node_sets |= {k: np.flatnonzero(np.isin(districts, v)) for k, v in SUBSETS.items()}
    node_sets = {k: v for k, v in node_sets.items() if len(v)}
    x = np.nan_to_num(flow, nan=0.0).astype(np.float32)[..., None]
    g = TemporalGraph(
        name=NAME,
        domain="traffic_flow",
        tasks=["node_forecasting"],
        time_mode="discrete",
        x=x,
        mask=x[..., 0] != 0,
        timestamps=timestamps.astype(np.int64),
        edge_index=np.stack([src, dst]).astype(np.int64),
        edge_weight=weights,
        node_table=table,
        node_sets=node_sets,
        splits={
            "default": Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2}, over="samples",
                             rounding="round",
                             reference="LargeST data/generate_data_for_training.py: "
                             "num_train = round(n * 0.6), num_val = round(n * 0.2) over samples"),
        },
        meta={
            "freq": "5min",
            "timestamps_tz": "America/Los_Angeles wall time, encoded as UTC epoch seconds",
            "channels": ["flow"],
            "covariate_channels": [],
            "units": {"flow": "vehicles/5min"},
            "mask_rule": "flow is observed (not NaN) and != 0",
            "mask_reference": "LargeST process_ca_his.ipynb fills NaN with 0, and "
                              "src/base/engine.py masks labels equal to 0 in every metric",
            "directed": True,
            "edge_weight": {
                "kind": "gaussian_kernel_similarity",
                "observed": False,
                "formula": "w_ij = exp(-d_ij^2 / sigma^2) if >= r else 0, d_ij the road network "
                           "distance, sigma the std of all distances (LargeST Sec. 2.1)",
                "threshold_lower_bound": float(weights.min()) if len(weights) else None,
                "description": "entries of ca_rn_adj.npy as released; the distances d_ij and "
                               "the threshold r are not published",
            },
            "adjacency": {"kind": "raw", "reference": "LargeST src/utils/dataloader.py "
                          "load_adj_from_numpy('ca_rn_adj.npy')"},
            "node_sets_rule": {
                **{f"d{d}": f"District == {d}" for d in np.unique(districts).tolist()},
                **{k: f"District in {list(v)}" for k, v in SUBSETS.items()},
                "reference": "LargeST data/{sd,gba,gla}/generate_*_dataset.ipynb",
            },
            "default_task": {
                "name": "node_forecasting",
                "params": {"window": 12, "horizon": 12},
                "reference": "LargeST Sec. 5: 12 steps in, 12 out",
                "departure": "LargeST aggregates to 15-minute means before windowing "
                             "(process_ca_his.ipynb); tgdata keeps the 5-minute data as released",
            },
            "source": "Caltrans PeMS, LargeST release (Liu et al., 2023), "
                      "kaggle.com/datasets/liuxu77/largest",
            "license": "cc-by-nc-4.0",
            "citation": CITATION,
            "provenance": provenance,
        },
    )
    g.meta["stats"] = compute_stats(g)
    g.meta["stats_node"] = compute_stats(g, per_node=True)
    validate(g)
    return g


def read_raw(raw: Path, year: int) -> tuple[Any, ...]:
    import h5py

    h5_path, meta_path, adj_path = (raw / f"ca_his_raw_{year}.h5", raw / "ca_meta.csv",
                                    raw / "ca_rn_adj.npy")
    with h5py.File(h5_path, "r") as f:
        ids = [c.decode() for c in f["t/axis0"][:]]
        timestamps = f["t/axis1"][:] // 1_000_000_000
        flow = f["t/block0_values"][:]
    with meta_path.open() as f:
        rows = list(csv.DictReader(f))
    if [r["ID"] for r in rows] != ids or [int(r["ID2"]) for r in rows] != list(range(len(ids))):
        raise ValueError("ca_meta.csv rows do not follow the flow columns")
    table = {k.lower(): np.array([cast(r[k]) for r in rows]) for k, cast in TABLE_COLUMNS.items()}
    provenance = {p.name: _sha256(p) for p in (h5_path, meta_path, adj_path)}
    return flow, timestamps, table, np.load(adj_path), provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    g = build(*read_raw(args.raw, args.year))
    io.save(g, args.out)
    print(f"saved {g.name} to {args.out}")
    if args.push:
        print(hub.push(args.out, NAME))


if __name__ == "__main__":
    main()
