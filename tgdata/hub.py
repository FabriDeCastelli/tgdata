from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from .io import load_dir
from .schema import SCHEMA_VERSION, validate

DEFAULT_NAMESPACE = "tgdata-hub"


def namespace() -> str:
    return os.environ.get("TGDATA_NAMESPACE", DEFAULT_NAMESPACE)


def default_root() -> Path:
    return Path(os.environ.get("TGDATA_ROOT", Path.home() / ".cache" / "tgdata"))


def repo_id(name: str) -> str:
    return name if "/" in name else f"{namespace()}/{name}"


def download(name: str, root: str | Path | None = None, revision: str | None = None) -> Path:
    kwargs = dict(repo_id=repo_id(name), repo_type="dataset", revision=revision,
                  cache_dir=root or default_root())
    try:
        return Path(snapshot_download(**kwargs, local_files_only=True))
    except LocalEntryNotFoundError:
        return Path(snapshot_download(**kwargs))


def push(
    path: str | Path,
    name: str,
    private: bool = True,
    commit_message: str | None = None,
    revision: str | None = None,
) -> str:
    path = Path(path)
    g = load_dir(path)
    validate(g)
    api = HfApi()
    rid = repo_id(name)
    api.create_repo(rid, repo_type="dataset", private=private, exist_ok=True)
    info = api.upload_folder(
        repo_id=rid,
        repo_type="dataset",
        folder_path=path,
        revision=revision,
        commit_message=commit_message or f"Upload {g.name} (schema v{SCHEMA_VERSION})",
    )
    add_to_domain_collection(api, rid, g.domain, private=private)
    return info.commit_url


def add_to_domain_collection(api: HfApi, rid: str, domain: str, private: bool = True) -> str:
    """One Hub collection per domain, titled like "Traffic flow", so the org page groups them."""
    title = domain.replace("_", " ").capitalize()
    collection = api.create_collection(title, namespace=namespace(), private=private,
                                       description=f"tgdata datasets of the {domain} domain",
                                       exists_ok=True)
    api.add_collection_item(collection.slug, rid, "dataset", exists_ok=True)
    return collection.slug
