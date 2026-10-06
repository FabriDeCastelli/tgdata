from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import tasks
from .hub import download, push
from .io import load_dir, save
from .registry import domains, info, list, resolve
from .schema import (
    SCHEMA_VERSION,
    Split,
    Target,
    TemporalGraph,
    adjacency,
    compute_stats,
    validate,
)
from .tasks import ConcatTasks

try:
    __version__ = version("tgdata")
except PackageNotFoundError:  # imported from a source checkout that is not installed
    __version__ = "0+unknown"


def load(
    name: str,
    root: str | Path | None = None,
    revision: str | None = None,
    nodes: str | None = None,
) -> TemporalGraph:
    """Local copy under `root` if present, else the Hub; `nodes` selects a named node set."""
    stored, nodes = resolve(name, nodes)
    g = load_dir(download(stored, root=root, revision=revision))
    return g if nodes is None else g.select_nodes(nodes)


def load_domain(domain: str, root: str | Path | None = None,
                role: str | None = None) -> dict[str, TemporalGraph]:
    """The pretraining pool of `domain`: every dataset whose nodes no other member contains.

    Wholes and unions of other members (LargeST's CA, SD, GBA, GLA) are left out, so no
    reading appears twice; load them by name. `role` keeps one side of a benchmark's train and
    held-out test networks (MiNT).
    """
    return {name: load(name, root=root) for name in list(domain=domain, pool=True, role=role)}


__all__ = [
    "SCHEMA_VERSION", "ConcatTasks", "Split", "Target", "TemporalGraph", "__version__", "adjacency",
    "compute_stats", "domains", "download", "info", "list", "load", "load_dir", "load_domain",
    "push", "save", "tasks", "validate",
]
