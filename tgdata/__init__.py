from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

from . import tasks
from .hub import download, push
from .io import load_dir, save
from .registry import info, list
from .schema import (
    SCHEMA_VERSION,
    Split,
    Target,
    TemporalGraph,
    adjacency,
    compute_stats,
    validate,
)

__version__ = version("tgdata")


def load(name: str, root: str | Path | None = None, revision: str | None = None) -> TemporalGraph:
    return load_dir(download(name, root=root, revision=revision))


__all__ = [
    "SCHEMA_VERSION", "Split", "Target", "TemporalGraph", "__version__", "adjacency",
    "compute_stats", "download", "info", "list", "load", "load_dir", "push", "save", "tasks",
    "validate",
]
