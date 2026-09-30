"""PEMS03/04/07/08 from the releases of the papers that introduced them.

PEMS04 and PEMS08 come from ASTGCN (Guo et al., 2019), PEMS03 and PEMS07 from STSGCN
(Song et al., 2020). Each keeps its introducing paper's graph, split and task.

python -m tgdata.converters.pems --dataset pems04 --raw DIR --out DIR [--push]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import hub, io
from ..encoding import compact
from ..schema import Split, TemporalGraph, compute_stats, validate
from . import Edge, drop_duplicate_edges

STEP_SECONDS = 300
STEPS_PER_DAY = 86400 // STEP_SECONDS

ASTGCN_CITATION = """
@inproceedings{guo2019astgcn,
  title={Attention Based Spatial-Temporal Graph Convolutional Networks
         for Traffic Flow Forecasting},
  author={Guo, Shengnan and Lin, Youfang and Feng, Ning and Song, Chao and Wan, Huaiyu},
  booktitle={AAAI},
  year={2019}
}
"""
STSGCN_CITATION = """
@inproceedings{song2020stsgcn,
  title={Spatial-Temporal Synchronous Graph Convolutional Networks: A New Framework
         for Spatial-Temporal Network Data Forecasting},
  author={Song, Chao and Lin, Youfang and Guo, Shengnan and Wan, Huaiyu},
  booktitle={AAAI},
  year={2020}
}
"""

ASTGCN = {
    "source": "ASTGCN release (Guo et al., 2019)",
    "citation": ASTGCN_CITATION,
    "adjacency": {
        "kind": "binary",
        "formula": "w_ij = 1 for every (i, j) row of the csv, 0 otherwise; directed",
        "reference": "ASTGCN lib/utils.py get_adjacency_matrix",
    },
    "split": Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2}, over="samples",
                   reference="ASTGCN-r-pytorch prepareData.py: int(n * 0.6), int(n * 0.8) "
                   "over samples"),
    "task_reference": "ASTGCN-r-pytorch configurations/{NAME}_astgcn.conf: len_input = 12, "
                      "num_for_predict = 12, in_channels = 1",
}
STSGCN = {
    "source": "STSGCN release (Song et al., 2020)",
    "citation": STSGCN_CITATION,
    "adjacency": {
        "kind": "binary",
        "symmetric": True,
        "formula": "w_ij = w_ji = 1 for every (i, j) row of the csv, 0 otherwise",
        "reference": "STSGCN utils.py get_adjacency_matrix (type_='connectivity')",
    },
    "split": Split(fractions={"train": 0.6, "val": 0.2, "test": 0.2}, strict=True,
                   reference="STSGCN utils.py generate_from_data: series cut at int(T * 0.6), "
                   "int(T * 0.8), windows generated within each part"),
    "task_reference": "STSGCN config/{NAME}/individual_GLU_mask_emb.json: points_per_hour = 12, "
                      "num_for_predict = 12",
}


@dataclass(frozen=True)
class Spec:
    district: int
    start: datetime
    num_steps: int
    left_out: tuple[str, ...]  # channels after flow in the release, not part of the dataset
    lineage: dict[str, Any]
    weight_column: str = "cost"
    uses_sensor_ids: bool = False
    date_note: str | None = None


SPECS = {
    "pems03": Spec(3, datetime(2018, 9, 1, tzinfo=timezone.utc), 26208, (), STSGCN,
                   weight_column="distance", uses_sensor_ids=True),
    "pems04": Spec(4, datetime(2018, 1, 1, tzinfo=timezone.utc), 16992,
                   ("occupancy", "speed"), ASTGCN),
    "pems07": Spec(7, datetime(2017, 5, 1, tzinfo=timezone.utc), 28224, (), STSGCN,
                   date_note="STSGCN Table 1 states 5/1/2017 - 8/31/2017 (123 days) but the "
                   "release holds 98 days; the start is confirmed by the Sunday minimum of mean "
                   "daily flow, so the series ends 2017-08-06 23:55 if contiguous, the range "
                   "LargeST (Liu et al., 2023) Table 1 also lists"),
    "pems08": Spec(8, datetime(2016, 7, 1, tzinfo=timezone.utc), 17856,
                   ("occupancy", "speed"), ASTGCN),
}


def build(
    name: str, data: np.ndarray, edges: list[Edge], provenance: dict[str, str]
) -> TemporalGraph:
    spec = SPECS[name]
    T = data.shape[0]
    edges = drop_duplicate_edges(edges)
    lineage = spec.lineage
    upper = name.upper()
    if data.shape[-1] != 1 + len(spec.left_out):
        raise ValueError(f"{upper} has {data.shape[-1]} channels, expected flow + {spec.left_out}")
    meta: dict[str, Any] = {
        "freq": "5min",
        "timestamps_tz": "America/Los_Angeles wall time, encoded as UTC epoch seconds",
        "channels": ["flow"],
        "covariate_channels": [],
        "units": {"flow": "vehicles/5min"},
        "mask_rule": "flow != 0",
        "directed": True,
        "edge_weight": {
            "kind": "distance",
            "observed": True,
            "units": "unspecified by the source",
            "description": f"'{spec.weight_column}' column of {upper}.csv, the road distance "
                           "between sensors",
        },
        "adjacency": lineage["adjacency"],
        "default_task": {
            "name": "node_forecasting",
            "params": {"window": 12, "horizon": 12},
            "reference": lineage["task_reference"].format(NAME=upper),
        },
        "source": f"Caltrans PeMS District {spec.district}, {lineage['source']}",
        "license": "other",
        "citation": lineage["citation"],
        "provenance": provenance,
    }
    if spec.uses_sensor_ids:
        meta["node_ids"] = "rows follow the PeMS sensor ids listed in PEMS03.txt"
    if spec.date_note:
        meta["date_note"] = spec.date_note
    if spec.left_out:
        meta["left_out_channels"] = {
            "channels": list(spec.left_out),
            "note": f"{upper}.npz also holds these as channels 1-{len(spec.left_out)}; the "
                    "traffic_flow domain keeps flow only (channel 0)",
        }
    g = TemporalGraph(
        name=name,
        domain="traffic_flow",
        tasks=["node_forecasting"],
        time_mode="discrete",
        x=data[..., :1].astype(np.float32),
        mask=data[..., 0] != 0,
        timestamps=int(spec.start.timestamp()) + STEP_SECONDS * np.arange(T, dtype=np.int64),
        edge_index=np.array([(s, d) for s, d, _ in edges], dtype=np.int64).T.reshape(2, -1),
        edge_weight=np.array([w for _, _, w in edges], dtype=np.float32),
        splits={
            "default": lineage["split"],
            "70/10/20": Split(fractions={"train": 0.7, "val": 0.1, "test": 0.2},
                              reference="tsl example configs (val_len=0.1, test_len=0.2), "
                              "over steps"),
        },
        meta=meta,
    )
    g = compact(g)
    g.meta["stats"] = compute_stats(g)
    g.meta["stats_node"] = compute_stats(g, per_node=True)
    validate(g)
    return g


def read_raw(name: str, raw: Path) -> tuple[np.ndarray, list[Edge], dict[str, str]]:
    spec, upper = SPECS[name], name.upper()
    npz, csv_path = raw / f"{upper}.npz", raw / f"{upper}.csv"
    files = [npz, csv_path]
    data = np.load(npz)["data"]
    index = None
    if spec.uses_sensor_ids:
        id_path = raw / f"{upper}.txt"
        files.append(id_path)
        index = {int(s): i for i, s in enumerate(id_path.read_text().split())}
    with csv_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["from"]]
    edges = [(int(r["from"]), int(r["to"]), float(r[spec.weight_column])) for r in rows]
    if index is not None:
        edges = [(index[s], index[d], w) for s, d, w in edges]
    return data, edges, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def check_calendar(g: TemporalGraph) -> None:
    spec = SPECS[g.name]
    assert g.num_steps == spec.num_steps, (g.num_steps, spec.num_steps)
    assert g.num_steps % STEPS_PER_DAY == 0
    assert g.timestamps is not None
    first = datetime.fromtimestamp(int(g.timestamps[0]), timezone.utc)
    last = datetime.fromtimestamp(int(g.timestamps[-1]), timezone.utc)
    days = g.num_steps // STEPS_PER_DAY
    assert first == spec.start
    assert last == spec.start + timedelta(days=days) - timedelta(seconds=STEP_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    g = build(args.dataset, *read_raw(args.dataset, args.raw))
    check_calendar(g)
    io.save(g, args.out)
    print(f"saved {g.name} to {args.out}")
    if args.push:
        print(hub.push(args.out, args.dataset))


if __name__ == "__main__":
    main()
