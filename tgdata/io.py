from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .encoding import DerivedMask, Encoded
from .schema import NPY_ARRAYS, SCHEMA_VERSION, Split, Target, TemporalGraph, validate

IDENTITY = ("name", "domain", "tasks", "time_mode")
TARGET_ARRAYS = ("values", "steps", "index")


def save(g: TemporalGraph, path: str | Path) -> Path:
    validate(g)
    path = Path(path)
    (path / "arrays").mkdir(parents=True, exist_ok=True)
    encoding: dict[str, Any] = {}
    for name in NPY_ARRAYS:
        arr = getattr(g, name)
        if isinstance(arr, DerivedMask):
            encoding[name] = {"derived": DerivedMask.rule}
            continue
        if isinstance(arr, Encoded):
            encoding[name], arr = arr.spec(), arr.raw
        if arr is not None:
            np.save(path / "arrays" / f"{name}.npy", np.ascontiguousarray(arr))
    if g.time_mode == "continuous":
        pq.write_table(_events_table(g), path / "events.parquet")
    for name, tgt in g.y.items():
        _save_arrays(path / "y" / name, {k: getattr(tgt, k) for k in TARGET_ARRAYS})
    for name, split in g.splits.items():
        if split.nodes is not None:
            _save_arrays(path / "splits" / name, split.nodes)
    if g.node_sets:
        _save_arrays(path / "node_sets", g.node_sets)
    if g.node_table is not None:
        pq.write_table(pa.table(g.node_table), path / "nodes.parquet")
    meta = {
        "schema_version": SCHEMA_VERSION,
        **{k: getattr(g, k) for k in IDENTITY},
        "encoding": encoding,
        "y": {k: _fields(t, exclude=TARGET_ARRAYS) for k, t in g.y.items()},
        "splits": {k: _fields(s, exclude=("nodes",)) | {"nodes": s.nodes is not None}
                   for k, s in g.splits.items()},
        "meta": g.meta,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2, default=_to_json))
    (path / "README.md").write_text(dataset_card(g))
    return path


def load_dir(path: str | Path, mmap: bool = True) -> TemporalGraph:
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    if meta["schema_version"] > SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema v{meta['schema_version']}; "
            f"this tgdata reads up to v{SCHEMA_VERSION}"
        )
    arrays: dict[str, Any] = {}
    for file in sorted((path / "arrays").glob("*.npy")):
        arrays[file.stem] = np.load(file, mmap_mode="r" if mmap else None)
    if (path / "events.parquet").exists():
        arrays.update(_read_events(path / "events.parquet"))
    for name, spec in meta.get("encoding", {}).items():
        if "decimals" in spec:
            arrays[name] = Encoded(arrays[name], spec["decimals"])
    for name, spec in meta.get("encoding", {}).items():
        if "derived" in spec:
            arrays[name] = DerivedMask(arrays["x"])
    y = {k: Target(**spec, **_load_arrays(path / "y" / k, mmap)) for k, spec in meta["y"].items()}
    splits = {}
    for k, spec in meta["splits"].items():
        nodes = _load_arrays(path / "splits" / k, mmap) if spec.pop("nodes") else None
        if spec["boundaries"] is not None:
            spec["boundaries"] = {s: tuple(b) for s, b in spec["boundaries"].items()}
        splits[k] = Split(**spec, nodes=nodes)
    node_table = None
    if (path / "nodes.parquet").exists():
        table = pq.read_table(path / "nodes.parquet")
        node_table = {k: table[k].to_numpy() for k in table.column_names}
    return TemporalGraph(**{k: meta[k] for k in IDENTITY}, **arrays, y=y, splits=splits,
                         node_table=node_table, node_sets=_load_arrays(path / "node_sets", mmap),
                         meta=meta["meta"])


def _fields(obj: Any, exclude: tuple[str, ...]) -> dict[str, Any]:
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj) if f.name not in exclude}


def _save_arrays(path: Path, arrays: dict[str, np.ndarray | None]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        if arr is not None:
            np.save(path / f"{name}.npy", np.ascontiguousarray(arr))


def _load_arrays(path: Path, mmap: bool) -> dict[str, np.ndarray]:
    return {f.stem: np.load(f, mmap_mode="r" if mmap else None) for f in sorted(path.glob("*.npy"))}


def dataset_card(g: TemporalGraph) -> str:
    tags = [f"domain:{g.domain}", f"time:{g.time_mode}", *(f"task:{t}" for t in g.tasks),
            *([f"role:{g.meta['role']}"] if "role" in g.meta else []), "tgdata"]
    lines = ["---", f"pretty_name: {g.name}"]
    if "license" in g.meta:
        lines.append(f"license: {g.meta['license']}")
    lines += ["tags:", *[f"- {t}" for t in tags], "---", "", f"# {g.name}", ""]
    default = g.splits["default"]
    summary = {"domain": g.domain, "tasks": ", ".join(g.tasks), "time_mode": g.time_mode,
               "num_nodes": g.num_nodes, "num_steps": g.num_steps,
               "targets": ", ".join(g.y) or "future values of x",
               "default split": ({**default.fractions, **default.holdout} if default.holdout
                                 else default.fractions or default.boundaries or "by node"),
               **{k: g.meta[k] for k in ("role", "freq", "source") if k in g.meta}}
    lines += [f"- **{k}**: {v}" for k, v in summary.items()]
    lines += ["", "Load with `tgdata.load(\"" + g.name + "\")`.", ""]
    if "citation" in g.meta:
        lines += ["## Citation", "", "```bibtex", g.meta["citation"].strip(), "```", ""]
    return "\n".join(lines)


def _events_table(g: TemporalGraph) -> pa.Table:
    columns = {name: getattr(g, name) for name in ("src", "dst", "t")}
    table = pa.table({k: pa.array(np.asarray(v)) for k, v in columns.items()})
    if g.msg is not None:
        msg = np.ascontiguousarray(g.msg).reshape(len(g.msg), -1)
        flat = pa.array(msg.reshape(-1))
        table = table.append_column("msg", pa.FixedSizeListArray.from_arrays(flat, msg.shape[1]))
    return table


def _read_events(path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(path)
    out = {k: table[k].to_numpy() for k in ("src", "dst", "t")}
    if "msg" in table.column_names:
        col = table["msg"].combine_chunks()
        out["msg"] = col.values.to_numpy().reshape(len(col), col.type.list_size)
    return out


def _to_json(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")

