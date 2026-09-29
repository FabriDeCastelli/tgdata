from __future__ import annotations

import json
from importlib import resources
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from .hub import namespace, repo_id


def packaged_registry() -> dict[str, dict[str, Any]]:
    return json.loads(resources.files("tgdata").joinpath("registry.json").read_text())


def list(
    domain: str | None = None,
    task: str | None = None,
    time_mode: str | None = None,
    offline: bool = False,
) -> list[str]:
    wanted = {"domain": domain, "task": task, "time": time_mode}
    if not offline:
        tags = [f"{k}:{v}" for k, v in wanted.items() if v is not None]
        try:
            found = HfApi().list_datasets(author=namespace(), tags=tags or None)
            return sorted(d.id.split("/", 1)[1] for d in found)
        except Exception:
            pass
    return sorted(
        name
        for name, entry in packaged_registry().items()
        if (domain is None or entry["domain"] == domain)
        and (task is None or task in entry["tasks"])
        and (time_mode is None or entry["time_mode"] == time_mode)
    )


def info(name: str, revision: str | None = None, offline: bool = False) -> dict[str, Any]:
    if not offline:
        try:
            path = hf_hub_download(repo_id(name), "meta.json", repo_type="dataset",
                                   revision=revision)
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    registry = packaged_registry()
    if name not in registry:
        raise KeyError(f"unknown dataset {name!r}; known offline: {sorted(registry)}")
    return registry[name]
