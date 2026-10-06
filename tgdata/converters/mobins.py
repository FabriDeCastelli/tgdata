"""The six MOBINS datasets (Na et al., 2025), from the Zenodo release of their CSV files.

Each has node series, a static spatial network and an origin-destination (OD) matrix per step.
The spatial network is the graph; the OD matrix is the pair-level target `y["od"]`, since
MOBINS forecasts it together with the node series. Everything is stored as released.

python -m tgdata.converters.mobins --dataset mobins-seoul --raw MOBINS.zip --out DIR [--push]
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc

from .. import hub, io
from ..encoding import compact
from ..schema import Split, Target, TemporalGraph, compute_stats, validate

ZENODO = "https://zenodo.org/records/14590709"
CODE = "github.com/kaist-dmlab/MOBINS"
CITATION = """
@inproceedings{na2025mobins,
  title={Mobility Networked Time-Series Forecasting Benchmark Datasets},
  author={Na, Jihye and Nam, Youngeun and Yoon, Susik and Song, Hwanjun and Lee, Byung Suk and
          Lee, Jae-Gil},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={19}, number={1}, pages={2539--2549}, year={2025},
  doi={10.1609/icwsm.v19i1.35955}
}
"""


@dataclass(frozen=True)
class Spec:
    folder: str
    domain: str
    channels: tuple[str, ...]
    freq: str
    code_name: str  # the dataset's name in MOBINS's code


SPECS = {
    "mobins-seoul": Spec("Transportation-Seoul", "mobility", ("inflow", "outflow"), "1h", "seoul"),
    "mobins-busan": Spec("Transportation-Busan", "mobility", ("inflow", "outflow"), "1h", "busan"),
    "mobins-daegu": Spec("Transportation-Daegu", "mobility", ("inflow", "outflow"), "1h", "daegu"),
    "mobins-nyc": Spec("Transportation-NYC", "mobility", ("ridership",), "1h", "nyc"),
    "mobins-epi-korea": Spec("Epidemic-Korea", "epidemic", ("infection",), "1D", "korea_covid"),
    "mobins-epi-nyc": Spec("Epidemic-NYC", "epidemic", ("infection",), "1D", "nyc_covid"),
}
STEP_SECONDS = {"1h": 3600, "1D": 86400}


def build(name: str, nodes: pa.Table, od: pa.Table, network: pa.Table,
          provenance: dict[str, str]) -> TemporalGraph:
    spec = SPECS[name]
    N, F = network.num_rows, len(spec.channels)
    unit = 86400 // STEP_SECONDS[spec.freq]  # steps per day: MOBINS windows and splits by day
    expected_nodes = [f"N{i}_{c.upper()}" for i in range(N) for c in spec.channels]
    expected_od = [f"N{i}_N{j}" for i in range(N) for j in range(N)]
    if nodes.column_names[1:] != expected_nodes or od.column_names[1:] != expected_od:
        raise ValueError(f"{spec.folder}: unexpected column order")
    if network.column_names != ["INDEX", *(f"N{i}" for i in range(N))]:
        raise ValueError(f"{spec.folder}: unexpected SPATIAL_NETWORK header")
    stamps = _datetimes(nodes)
    if not np.array_equal(stamps, _datetimes(od)):
        raise ValueError(f"{spec.folder}: node and OD files have different datetimes")
    if (np.diff(stamps) != STEP_SECONDS[spec.freq]).any() or len(stamps) % unit:
        raise ValueError(f"{spec.folder}: steps are not a contiguous run of whole days")
    x = _matrix(nodes).reshape(-1, N, F)
    movements = _integers(_matrix(od)).reshape(-1, N, N)
    adj = _matrix(network)
    if not np.isin(adj, (0, 1)).all():
        raise ValueError(f"{spec.folder}: SPATIAL_NETWORK is not binary")
    src, dst = np.nonzero(adj)
    per_day = "hour" if spec.freq == "1h" else "day"
    g = TemporalGraph(
        name=name,
        domain=spec.domain,
        tasks=["node_forecasting"],
        time_mode="discrete",
        x=x,
        timestamps=stamps,
        edge_index=np.stack([src, dst]).astype(np.int64),
        edge_weight=np.ones(len(src), dtype=np.float32),
        y={"od": Target(
            level="pair", kind="regression", values=movements,
            source="OD_MOVEMENTS.csv: od[t, i, j] is column N{i}_N{j}, the movements from "
                   "node i to node j in step t")},
        splits={"default": Split(
            fractions={"train": 0.8, "val": 0.2}, over="samples", holdout={"test": 0.25},
            unit=unit,
            reference=f"{CODE} data_loader.py _make_windowing_and_loader (test_ratio=0.25, "
                      "train_ratio=0.8): test is the last int(0.25 * days) days with windows "
                      "cut inside it; train is the first int(0.8 * n) of the n windows before "
                      "it, val the rest; windows start once a day")},
        meta={
            "freq": spec.freq,
            "timestamps_tz": "the release's 'datetime' column, which names no time zone, "
                             "encoded as UTC epoch seconds",
            "num_nodes": N,
            "channels": list(spec.channels),
            "covariate_channels": [],
            "units": {c: f"count per {per_day}" for c in spec.channels},
            "directed": True,
            "edge_weight": {
                "kind": "binary",
                "observed": True,
                "description": "SPATIAL_NETWORK.csv: w_ij = 1 where row i, column j is 1 "
                               "(symmetric, self-loops included, as released)",
            },
            "adjacency": {"kind": "raw", "reference": f"{CODE} data_loader.py load_datasets: "
                          "the spatial network is used as is (khop=0)"},
            "default_task": {
                "name": "node_forecasting",
                "params": {"window": "4D", "horizon": "7D", "stride": "1D", "pair_target": "od"},
                "reference": f"{CODE} README and Na et al. (2025) Table 4: seq_day=4, "
                             "pred_day=7 (also 14 and 30); baselines.py forecasts "
                             "cat(node features, OD) from the same over the window",
                "alternatives": {"horizon": ["14D", "30D"]},
            },
            "source": f"MOBINS {spec.folder} (Na et al., 2025), Zenodo record 14590709",
            "mobins_name": spec.code_name,
            "license": "cc-by-nc-nd-4.0",
            "license_note": "the Zenodo record's licence; the MOBINS README lists CC BY-NC 4.0 "
                            "for this dataset except Epidemic-Korea. Stored unmodified up to "
                            "format.",
            "citation": CITATION,
            "provenance": provenance,
        },
    )
    g = compact(g)
    g.meta["stats"] = compute_stats(g)
    g.meta["stats_node"] = compute_stats(g, per_node=True)
    validate(g)
    return g


def _datetimes(table: pa.Table) -> np.ndarray:
    values = np.array(table.column("datetime").to_pylist(), dtype="datetime64[s]")
    return values.astype(np.int64)


def _matrix(table: pa.Table) -> np.ndarray:
    columns = [c for c in table.column_names if c not in ("datetime", "INDEX")]
    return np.stack([table.column(c).to_numpy().astype(np.float64) for c in columns], axis=1)


def _integers(values: np.ndarray) -> np.ndarray:
    """Counts as the smallest integer type holding them; float32 of each is exact."""
    if not (values == np.round(values)).all() or values.min() < 0 or values.max() >= 2**24:
        raise ValueError("OD values are not counts float32 represents exactly")
    kind = next(t for t in (np.uint8, np.int16, np.int32) if values.max() <= np.iinfo(t).max)
    return values.astype(kind)


def read_raw(name: str, raw: Path) -> tuple[pa.Table, pa.Table, pa.Table, dict[str, str]]:
    folder = SPECS[name].folder
    provenance = {"url": ZENODO, "MOBINS.zip sha256": _sha256(raw)}
    options = pc.ConvertOptions(column_types={"datetime": pa.string()})
    tables = []
    with zipfile.ZipFile(raw) as archive:
        for file in ("NODE_TIME_SERIES_FEATURES", "OD_MOVEMENTS", "SPATIAL_NETWORK"):
            member = f"{folder}/{file}.csv"
            with archive.open(member) as f:
                tables.append(pc.read_csv(f, convert_options=options))
            provenance[f"{member} crc32"] = f"{archive.getinfo(member).CRC:08x}"
    return tables[0], tables[1], tables[2], provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(2**24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(SPECS), required=True)
    parser.add_argument("--raw", type=Path, required=True, help="MOBINS.zip from Zenodo")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    g = build(args.dataset, *read_raw(args.dataset, args.raw))
    io.save(g, args.out)
    print(f"saved {g.name} to {args.out}")
    if args.push:
        print(hub.push(args.out, g.name))


if __name__ == "__main__":
    main()
