from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .encoding import DerivedMask

if TYPE_CHECKING:
    from .tasks.base import Task

SCHEMA_VERSION = 3

TimeMode = Literal["discrete", "continuous"]
Bounds = dict[str, tuple[int, int]]

NPY_ARRAYS = (
    "x", "covariates", "mask", "timestamps",
    "edge_index", "edge_weight", "edge_ptr", "node_features", "node_time",
)
CONTINUOUS_ARRAYS = ("src", "dst", "t", "msg")


@dataclass
class Target:
    """A supervised target other than future values of `x`.

    Layout of `values` by level: node -> [N or K, ...] if static else [S, N or K, ...];
    graph -> [...] if static else [S, ...]; edge -> [E, ...] aligned with `edge_index`.
    S is the number of steps (all of them unless `steps` lists which), K the size of `index`.
    """

    level: Literal["node", "edge", "graph"]
    kind: Literal["regression", "class"]
    values: np.ndarray
    static: bool = False
    steps: np.ndarray | None = None
    index: np.ndarray | None = None
    num_classes: int | None = None
    source: str = "shipped"


@dataclass
class Split:
    """How the source splits the data, resolved to indices only when a task is built.

    `fractions` are chronological and cut `over` steps or over samples (windows);
    `boundaries` are [start, end) in timestamp units, for sources that split by date;
    `nodes` holds node ids per split, for node-split tasks. With `strict`, a sample's inputs
    must lie inside its split too, as when a source cuts windows within each part.
    `rounding` is how the source turns fractions into counts: "floor" cuts at
    int(n * cumulative fraction); "round" sizes each part but the last as round(n * fraction).
    """

    fractions: dict[str, float] | None = None
    over: Literal["steps", "samples"] = "steps"
    strict: bool = False
    rounding: Literal["floor", "round"] = "floor"
    boundaries: Bounds | None = None
    nodes: dict[str, np.ndarray] | None = None
    reference: str | None = None

    def resolve(self, length: int) -> Bounds:
        assert self.fractions is not None
        fractions = list(self.fractions.values())
        if self.rounding == "floor":
            # The epsilon absorbs float sums such as 0.7 + 0.1 = 0.7999999999999999.
            cum = np.cumsum(fractions)[:-1]
            cuts = [0, *(math.floor(c * length + 1e-9) for c in cum), length]
        else:
            # Python's round (half to even), as the sources call it.
            cuts = [0, *np.cumsum([round(length * f) for f in fractions[:-1]]).tolist(), length]
        return {k: (cuts[i], cuts[i + 1]) for i, k in enumerate(self.fractions)}


@dataclass
class TemporalGraph:
    name: str
    domain: str
    tasks: list[str]
    time_mode: TimeMode
    x: np.ndarray | None = None
    covariates: np.ndarray | None = None
    mask: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    edge_index: np.ndarray | None = None
    edge_weight: np.ndarray | None = None
    edge_ptr: np.ndarray | None = None
    node_features: np.ndarray | None = None
    node_time: np.ndarray | None = None
    node_table: dict[str, np.ndarray] | None = None
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    src: np.ndarray | None = None
    dst: np.ndarray | None = None
    t: np.ndarray | None = None
    msg: np.ndarray | None = None
    y: dict[str, Target] = field(default_factory=dict)
    splits: dict[str, Split] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        if self.time_mode == "continuous":
            assert self.t is not None
            return len(self.t)
        if self.x is not None:
            return self.x.shape[0]
        if self.edge_ptr is not None:
            return len(self.edge_ptr) - 1
        assert self.timestamps is not None
        return len(self.timestamps)

    @property
    def num_nodes(self) -> int:
        if self.x is not None:
            return self.x.shape[1]
        if self.node_features is not None:
            return self.node_features.shape[0]
        return int(self.meta["num_nodes"])

    @property
    def is_static(self) -> bool:
        return self.edge_ptr is None

    def edges_at(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        assert self.edge_index is not None and self.edge_weight is not None
        if self.edge_ptr is None:
            return self.edge_index, self.edge_weight
        lo, hi = int(self.edge_ptr[step]), int(self.edge_ptr[step + 1])
        return self.edge_index[:, lo:hi], self.edge_weight[lo:hi]

    def with_split(
        self,
        name: str | None = None,
        *,
        val_fraction: float | None = None,
        test_fraction: float | None = None,
    ) -> TemporalGraph:
        """Make `name` (or new chronological fractions) the default split."""
        if name is not None:
            split = self.splits[name]
        else:
            if val_fraction is None or test_fraction is None:
                raise ValueError("pass a split name or both val_fraction and test_fraction")
            train = 1.0 - val_fraction - test_fraction
            split = Split(fractions={"train": train, "val": val_fraction, "test": test_fraction})
        return dataclasses.replace(self, splits={**self.splits, "default": split})

    def select_nodes(self, nodes: str | np.ndarray) -> TemporalGraph:
        return select_nodes(self, nodes)

    def task(self, name: str = "default", **params: Any) -> Task:
        from .tasks import make_task

        return make_task(self, name, **params)

    def window(self, window: int | str, horizon: int | str, split: str = "train",
               **params: Any) -> Task:
        return self.task("node_forecasting", window=window, horizon=horizon, split=split,
                         **params)

    def to_pyg(self, **kwargs: Any) -> Any:
        from .adapters.pyg import to_pyg

        return to_pyg(self, **kwargs)

    def to_tsl(self, **kwargs: Any) -> Any:
        from .adapters.tsl import to_tsl

        return to_tsl(self, **kwargs)


def adjacency(g: TemporalGraph, kind: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Edges weighted as `meta.adjacency` prescribes (the source paper's setting) unless `kind`
    overrides it: "raw" keeps stored weights, "binary" sets them to 1, "gaussian" maps distances
    to exp(-(d / sigma)^2) (sigma defaults to the std of d) and drops edges below `threshold`.
    A binary spec with `symmetric` adds each edge's reverse, as sources building A[j, i] = A[i, j]
    do."""
    assert g.edge_index is not None and g.edge_weight is not None
    spec = dict(g.meta.get("adjacency", {"kind": "raw"}))
    if kind is not None and kind != spec["kind"]:
        spec = {"kind": kind}
    if spec["kind"] == "raw":
        return g.edge_index, g.edge_weight
    if spec["kind"] == "binary":
        ei = np.asarray(g.edge_index)
        if spec.get("symmetric"):
            if not g.is_static:
                raise ValueError("symmetric adjacency needs a static graph")
            ei = np.unique(np.concatenate([ei, ei[::-1]], axis=1), axis=1)
        return ei, np.ones(ei.shape[1], dtype=np.float32)
    if spec["kind"] != "gaussian":
        raise ValueError(f"unknown adjacency kind {spec['kind']!r}")
    if g.meta["edge_weight"]["kind"] != "distance" or not g.is_static:
        raise ValueError("gaussian adjacency needs static distance-weighted edges")
    dist = np.asarray(g.edge_weight, dtype=np.float64)
    sim = np.exp(-np.square(dist / spec.get("sigma", dist.std())))
    keep = sim >= spec.get("threshold", 0.0)
    return np.asarray(g.edge_index)[:, keep], sim[keep].astype(np.float32)


def select_nodes(g: TemporalGraph, nodes: str | np.ndarray) -> TemporalGraph:
    """The subgraph induced by a named node set or sorted node ids.

    A contiguous id range stays a view of the stored (memory-mapped) arrays; other sets copy
    only their own columns. Channel stats are recombined exactly from the per-node stats.
    Graph-level targets and node splits describe the full graph and are dropped.
    """
    ids = np.asarray(g.node_sets[nodes] if isinstance(nodes, str) else nodes, dtype=np.int64)
    if len(ids) == 0 or (np.diff(ids) <= 0).any():
        raise ValueError("node ids must be non-empty, sorted and unique")
    contiguous = ids[-1] - ids[0] + 1 == len(ids)
    at: slice | np.ndarray = slice(int(ids[0]), int(ids[-1]) + 1) if contiguous else ids
    relabel = np.full(g.num_nodes, -1, dtype=np.int64)
    relabel[ids] = np.arange(len(ids))

    def rows(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else a[at]

    def cols(a: Any) -> Any:
        if a is None or hasattr(a, "take_nodes"):
            return None if a is None else a.take_nodes(at)
        return a[:, at]

    x = cols(g.x)
    changes: dict[str, Any] = {
        "x": x, "mask": DerivedMask(x) if isinstance(g.mask, DerivedMask) else cols(g.mask),
        "covariates": cols(g.covariates) if g.covariates is not None and g.covariates.ndim == 3
        else g.covariates,
        "node_features": rows(g.node_features), "node_time": rows(g.node_time),
        "node_table": None if g.node_table is None
        else {k: v[at] for k, v in g.node_table.items()},
        "node_sets": {},
        "splits": {k: s for k, s in g.splits.items() if s.nodes is None},
        "y": {},
    }
    keep = None
    if g.edge_index is not None:
        ei = relabel[np.asarray(g.edge_index)]
        keep = (ei >= 0).all(axis=0)
        changes["edge_index"] = ei[:, keep]
        changes["edge_weight"] = np.asarray(g.edge_weight)[keep]
        if g.edge_ptr is not None:
            ptr = np.asarray(g.edge_ptr)
            step = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
            counts = np.bincount(step[keep], minlength=len(ptr) - 1)
            changes["edge_ptr"] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    for key, tgt in g.y.items():
        if tgt.level == "node" and tgt.index is None:
            values = tgt.values[at] if tgt.static else tgt.values[:, at]
            changes["y"][key] = dataclasses.replace(tgt, values=values)
        elif tgt.level == "node":
            inside = relabel[tgt.index] >= 0
            values = tgt.values[inside] if tgt.static else tgt.values[:, inside]
            changes["y"][key] = dataclasses.replace(tgt, values=values,
                                                    index=relabel[tgt.index][inside])
        elif tgt.level == "edge" and keep is not None:
            changes["y"][key] = dataclasses.replace(tgt, values=tgt.values[keep])
    meta = {**g.meta, "num_nodes": len(ids),
            "selected_from": {"dataset": g.name,
                              "nodes": nodes if isinstance(nodes, str) else "ids"}}
    if "stats_node" in g.meta:
        node_stats = {k: np.asarray(v)[at] for k, v in g.meta["stats_node"].items()
                      if k != "steps"}
        meta["stats_node"] = {**{k: v.tolist() for k, v in node_stats.items()},
                              "steps": g.meta["stats_node"]["steps"]}
        meta["stats"] = _pool_stats(node_stats, g.meta["stats_node"]["steps"])
    name = f"{g.name}-{nodes}" if isinstance(nodes, str) else g.name
    return dataclasses.replace(g, name=name, meta=meta, **changes)


def _pool_stats(node: dict[str, np.ndarray], steps: list[int]) -> dict[str, list]:
    """Channel mean/std over nodes from per-node (count, mean, std), exactly."""
    count, mean, std = (np.asarray(node[k], dtype=np.float64) for k in ("count", "mean", "std"))
    total = count.sum(axis=0)
    pooled = (count * mean).sum(axis=0) / np.maximum(total, 1)
    var = (count * (std**2 + (mean - pooled) ** 2)).sum(axis=0) / np.maximum(total, 1)
    return {"mean": pooled.tolist(), "std": np.sqrt(var).tolist(), "count": total.tolist(),
            "steps": steps}


def compute_stats(
    g: TemporalGraph, split: str = "default", per_node: bool = False
) -> dict[str, list]:
    """Mean/std of `x` over valid entries of the training steps of `split`.

    Streams over time in chunks, so it works on memory-mapped series larger than RAM.
    """
    assert g.x is not None
    start, end = g.splits[split].resolve(g.num_steps)["train"]
    N, F = g.x.shape[1:]
    count, mean, m2 = np.zeros((N, F)), np.zeros((N, F)), np.zeros((N, F))
    chunk = max(1, 2**26 // (N * F))
    for lo in range(start, end, chunk):
        hi = min(lo + chunk, end)
        x = np.asarray(g.x[lo:hi], dtype=np.float64)
        valid = (np.ones(x.shape, dtype=bool) if g.mask is None
                 else np.broadcast_to(_mask_as_x(np.asarray(g.mask[lo:hi]), x), x.shape))
        c = valid.sum(axis=0)
        m = np.where(valid, x, 0.0).sum(axis=0) / np.maximum(c, 1)
        v = np.where(valid, (x - m) ** 2, 0.0).sum(axis=0)
        # Chan et al.: merge (count, mean, M2) of two disjoint sets exactly.
        total = count + c
        delta = m - mean
        mean = mean + delta * c / np.maximum(total, 1)
        m2 = m2 + v + delta**2 * count * c / np.maximum(total, 1)
        count = total
    std = np.sqrt(m2 / np.maximum(count, 1))
    node = {"count": count, "mean": mean, "std": std}
    if per_node:
        return {k: v.tolist() for k, v in node.items()} | {"steps": [start, end]}
    return _pool_stats(node, [start, end])


def _mask_as_x(mask: np.ndarray, x: np.ndarray) -> np.ndarray:
    return mask[..., None] if mask.ndim == x.ndim - 1 else mask


def validate(g: TemporalGraph) -> None:
    errors = [f"{f} is empty" for f in ("name", "domain", "tasks") if not getattr(g, f)]
    if g.time_mode == "discrete":
        errors += _discrete_errors(g)
    elif g.time_mode == "continuous":
        errors += _continuous_errors(g)
    else:
        errors.append(f"unknown time_mode {g.time_mode!r}")
    if not errors:
        errors += _split_errors(g) + _target_errors(g)
    if errors:
        raise ValueError(f"invalid TemporalGraph {g.name!r}:\n  " + "\n  ".join(errors))


def _discrete_errors(g: TemporalGraph) -> list[str]:
    errors = [f"{n} is set on a discrete graph" for n in CONTINUOUS_ARRAYS
              if getattr(g, n) is not None]
    if "freq" not in g.meta:
        errors.append("meta.freq (the source's sampling interval) is required")
    if g.x is None and g.edge_index is None:
        return errors + ["a discrete graph needs x or edges"]
    if g.x is None and g.edge_ptr is None and g.timestamps is None:
        return errors + ["without x, the steps come from edge_ptr or timestamps"]
    if g.x is None and g.node_features is None and "num_nodes" not in g.meta:
        return errors + ["without x or node_features, meta.num_nodes is required"]
    if g.x is not None and g.x.ndim != 3:
        return errors + ["x must be a [T, N, F] array"]
    T, N = g.num_steps, g.num_nodes
    if g.x is not None:
        if not np.issubdtype(g.x.dtype, np.floating):
            errors.append(f"x must be floating, got {g.x.dtype}")
        F = g.x.shape[2]
        if g.mask is not None:
            if g.mask.dtype != np.bool_:
                errors.append(f"mask must be bool, got {g.mask.dtype}")
            if g.mask.shape not in ((T, N), (T, N, F)):
                errors.append(f"mask shape {g.mask.shape} is neither {(T, N)} nor {(T, N, F)}")
    elif g.mask is not None:
        errors.append("mask is set without x")
    if g.covariates is not None and (
        g.covariates.shape[0] != T
        or g.covariates.ndim not in (2, 3)
        or (g.covariates.ndim == 3 and g.covariates.shape[1] != N)
    ):
        errors.append(f"covariates shape {g.covariates.shape} is neither [T, C] nor [T, N, C]")
    if g.timestamps is not None:
        if g.timestamps.shape != (T,) or g.timestamps.dtype != np.int64:
            errors.append(f"timestamps must be int64 of shape {(T,)}")
        elif T > 1 and not (np.diff(g.timestamps) > 0).all():
            errors.append("timestamps are not strictly increasing")
    if g.node_features is not None and g.node_features.shape[0] != N:
        errors.append(f"node_features has {g.node_features.shape[0]} rows, expected {N}")
    if g.node_time is not None and g.node_time.shape != (N,):
        errors.append(f"node_time must have shape {(N,)}")
    for col, values in (g.node_table or {}).items():
        if len(values) != N:
            errors.append(f"node_table[{col}] has {len(values)} rows, expected {N}")
    for key, ids in g.node_sets.items():
        if len(ids) == 0 or (np.diff(ids) <= 0).any() or ids[0] < 0 or ids[-1] >= N:
            errors.append(f"node_sets[{key}] must be sorted unique ids in [0, {N})")
    return errors + _edge_errors(g, T, N)


def _edge_errors(g: TemporalGraph, T: int, N: int) -> list[str]:
    if g.edge_index is None:
        present = [n for n in ("edge_weight", "edge_ptr") if getattr(g, n) is not None]
        return [f"{n} is set without edge_index" for n in present]
    errors: list[str] = []
    ei = g.edge_index
    if ei.ndim != 2 or ei.shape[0] != 2 or not np.issubdtype(ei.dtype, np.integer):
        return ["edge_index must be an integer [2, E] array"]
    E = ei.shape[1]
    if E and (ei.min() < 0 or ei.max() >= N):
        errors.append(f"edge_index values outside [0, {N})")
    if g.is_static and E and len(np.unique(ei, axis=1).T) != E:
        errors.append("static edge_index has duplicate edges")
    if g.edge_weight is None or g.edge_weight.shape != (E,):
        errors.append(f"edge_weight must be given with shape {(E,)}")
    errors += _edge_meta_errors(g.meta)
    if g.edge_ptr is not None:
        ptr = g.edge_ptr
        if ptr.shape != (T + 1,) or not np.issubdtype(ptr.dtype, np.integer):
            errors.append(f"edge_ptr must be an integer [{T + 1}] array")
        elif ptr[0] != 0 or ptr[-1] != E or (np.diff(ptr) < 0).any():
            errors.append("edge_ptr must start at 0, end at E, and be non-decreasing")
    return errors


def _edge_meta_errors(meta: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    weight = meta.get("edge_weight")
    if not isinstance(weight, dict) or not {"kind", "observed"} <= weight.keys():
        errors.append("meta.edge_weight must describe the weights with 'kind' and 'observed'")
    elif not weight["observed"] and "formula" not in weight:
        errors.append("meta.edge_weight is derived (observed=false) but has no 'formula'")
    adj = meta.get("adjacency", {"kind": "raw"})
    if adj.get("kind") != "raw" and "formula" not in adj:
        errors.append(f"meta.adjacency kind {adj.get('kind')!r} needs a 'formula'")
    return errors


def _continuous_errors(g: TemporalGraph) -> list[str]:
    if g.src is None or g.dst is None or g.t is None:
        return ["continuous graphs need src, dst and t"]
    errors: list[str] = []
    n = len(g.t)
    for name in ("src", "dst", "t"):
        arr = getattr(g, name)
        if arr.shape != (n,) or not np.issubdtype(arr.dtype, np.integer):
            errors.append(f"{name} must be an integer [{n}] array")
    if n > 1 and (np.diff(g.t) < 0).any():
        errors.append("t is not sorted")
    if g.msg is not None and g.msg.shape[0] != n:
        errors.append(f"msg has {g.msg.shape[0]} rows, expected {n}")
    if g.x is not None:
        errors.append("x is set on a continuous graph; use node_features")
    return errors


def _split_errors(g: TemporalGraph) -> list[str]:
    if "default" not in g.splits:
        return ["splits needs a 'default' entry"]
    errors: list[str] = []
    for name, split in g.splits.items():
        label = f"splits[{name}]"
        given = [s for s in (split.fractions, split.boundaries, split.nodes) if s is not None]
        if len(given) != 1:
            errors.append(f"{label} must set exactly one of fractions, boundaries, nodes")
        elif split.fractions is not None:
            fr = split.fractions
            if min(fr.values()) <= 0 or not math.isclose(sum(fr.values()), 1.0):
                errors.append(f"{label} fractions must be positive and sum to 1, got {fr}")
        elif split.boundaries is not None:
            bounds = sorted((int(s), int(e), k) for k, (s, e) in split.boundaries.items())
            if any(s >= e for s, e, _ in bounds) or any(
                a[1] > b[0] for a, b in zip(bounds, bounds[1:], strict=False)
            ):
                errors.append(f"{label} boundaries must be non-empty and non-overlapping")
        else:
            assert split.nodes is not None
            ids = np.concatenate([np.asarray(v) for v in split.nodes.values()])
            if len(ids) and (ids.min() < 0 or ids.max() >= g.num_nodes):
                errors.append(f"{label} node ids outside [0, {g.num_nodes})")
            if len(np.unique(ids)) != len(ids):
                errors.append(f"{label} node sets overlap")
    return errors


def _target_errors(g: TemporalGraph) -> list[str]:
    errors: list[str] = []
    for name, tgt in g.y.items():
        label = f"y[{name}]"
        v = tgt.values
        if tgt.kind == "class" and (tgt.num_classes is None
                                    or not np.issubdtype(v.dtype, np.integer)):
            errors.append(f"{label} class targets need integer values and num_classes")
        if tgt.level == "edge":
            if g.edge_index is None or v.shape[:1] != (g.edge_index.shape[1],):
                errors.append(f"{label} edge values must align with edge_index")
            continue
        if not tgt.static:
            steps = g.num_steps if tgt.steps is None else len(tgt.steps)
            if v.shape[:1] != (steps,):
                errors.append(f"{label} leading dim {v.shape[:1]} != {(steps,)} steps")
            if tgt.steps is not None and len(tgt.steps) and (
                tgt.steps.min() < 0 or tgt.steps.max() >= g.num_steps
            ):
                errors.append(f"{label} steps outside [0, {g.num_steps})")
        if tgt.level == "node":
            axis = 0 if tgt.static else 1
            n = g.num_nodes if tgt.index is None else len(tgt.index)
            if v.ndim <= axis or v.shape[axis] != n:
                errors.append(f"{label} node axis has {v.shape[axis:axis + 1]}, expected {n}")
    return errors
