from __future__ import annotations

import builtins
import json
from importlib import resources
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from .hub import namespace, repo_id

TAGS = {"domain": "domain", "task": "tasks", "time": "time_mode", "role": "role"}


def packaged_registry() -> dict[str, dict[str, Any]]:
    return json.loads(resources.files("tgdata").joinpath("registry.json").read_text())


def catalog(offline: bool = False) -> dict[str, dict[str, Any]]:
    """name -> {domain, tasks, time_mode}; from Hub tags, else the packaged registry.

    Aliases (named node subsets of a stored dataset) come from the packaged registry and are
    listed when the dataset they select from is.
    """
    registry = packaged_registry()
    entries = {k: v for k, v in registry.items() if "alias_of" not in v}
    if not offline:
        try:
            entries = {d.id.split("/", 1)[1]: _from_tags(d.tags or [])
                       for d in HfApi().list_datasets(author=namespace())}
        except Exception:
            pass
    aliases = {k: v for k, v in registry.items() if v.get("alias_of") in entries}
    return {**entries, **aliases}


def list(
    domain: str | None = None,
    task: str | None = None,
    time_mode: str | None = None,
    pool: bool = False,
    role: str | None = None,
    offline: bool = False,
) -> builtins.list[str]:
    """Dataset names; with `pool`, only those that belong to their domain's pretraining pool.

    `role` is "train" or "test" for datasets that come in a benchmark's train and held-out test
    networks (MiNT); the others have none.
    """
    registry = packaged_registry()
    return sorted(
        name
        for name, entry in catalog(offline).items()
        if (domain is None or entry["domain"] == domain)
        and (task is None or task in entry["tasks"])
        and (time_mode is None or entry["time_mode"] == time_mode)
        and (role is None or entry.get("role") == role)
        and (not pool or registry.get(name, {}).get("pool", True))
    )


def domains(offline: bool = False) -> dict[str, builtins.list[str]]:
    grouped: dict[str, builtins.list[str]] = {}
    for name, entry in sorted(catalog(offline).items()):
        grouped.setdefault(entry["domain"], []).append(name)
    return grouped


def info(name: str, revision: str | None = None, offline: bool = False) -> dict[str, Any]:
    registry = packaged_registry()
    if "alias_of" in registry.get(name, {}):
        return registry[name]
    if not offline:
        try:
            path = hf_hub_download(repo_id(name), "meta.json", repo_type="dataset",
                                   revision=revision)
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    if name not in registry:
        raise KeyError(f"unknown dataset {name!r}; known offline: {sorted(registry)}")
    return registry[name]


def resolve(name: str, nodes: str | None) -> tuple[str, str | None]:
    """The stored dataset and node set behind `name`, which may be an alias."""
    entry = packaged_registry().get(name, {})
    if "alias_of" not in entry:
        return name, nodes
    if nodes is not None:
        raise ValueError(f"{name!r} already selects nodes {entry['nodes']!r}")
    return entry["alias_of"], entry["nodes"]


def _from_tags(tags: builtins.list[str]) -> dict[str, Any]:
    entry: dict[str, Any] = {"domain": None, "tasks": [], "time_mode": None}
    for tag in tags:
        key, _, value = tag.partition(":")
        if key == "task":
            entry["tasks"].append(value)
        elif key in TAGS:
            entry[TAGS[key]] = value
    return entry
