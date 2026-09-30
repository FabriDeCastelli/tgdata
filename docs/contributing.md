# Contributing

## Installing

tgdata never needs a GPU; the install only decides which torch build you download.

| goal | command |
|---|---|
| use tgdata in a uv project, default torch | `uv add "tgdata @ git+ssh://git@github.com/FabriDeCastelli/tgdata.git" --tag v0.2.1` |
| ... with CUDA 12.8 drivers | `uv add "tgdata[cu128] @ git+ssh://..." --tag v0.2.1` |
| ... on a CPU-only machine (about 1 GB smaller) | `uv add "tgdata[cpu] @ git+ssh://..." --tag v0.2.1` |
| use tgdata with pip | `pip install "tgdata @ git+ssh://git@github.com/FabriDeCastelli/tgdata.git@v0.2.1"` |
| develop tgdata, CPU | `uv sync --extra cpu`, then `uv run --extra cpu pytest` |
| develop tgdata, CUDA 12.8 driver | `uv sync --extra cu128`, then `uv run --extra cu128 pytest` |

Extras: `plot` (matplotlib), `pyg` and `tsl` (adapters `g.to_pyg()`, `g.to_tsl()`), `convert`
(h5py, for converters). When developing, pass the same `--extra` to every `uv run`: without it,
uv switches back to PyPI's default torch.

Datasets are private to the `tgdata-hub` organisation. Log in with `hf auth login` (browser
login), or use a token with read access to `tgdata-hub` repos.

## Adding a dataset

1. Write `tgdata/converters/<name>.py`, following `converters/pems.py`: read the source's raw
   files and build a `TemporalGraph` with the introducing paper's graph, split and default task,
   citing where each comes from.
2. Call `compact(g)` so channels are stored exactly and compactly, then `tgdata.validate(g)`.
3. Save and upload: `python -m tgdata.converters.<name> --raw DIR --out DIR --push`. The upload
   creates the private repo `tgdata-hub/<name>` and adds it to its domain's collection.
4. Add the dataset to `tgdata/registry.json` and to [datasets.md](datasets.md).

## Hosting

Datasets live under the Hugging Face organisation `tgdata-hub` (set `TGDATA_NAMESPACE` to use
another). Uploading needs a token with write access to it, including "write to collections".
`tgdata.load(name)` reads a dataset from `root` (`$TGDATA_ROOT`, default `~/.cache/tgdata`) and
downloads it only when missing; `revision=` pins a version.

## Checks

```bash
uv run --extra cpu ruff check .
uv run --extra cpu pytest
```

CI runs both on every push. `benchmarks/loader.py` measures loading speed on a GPU against tsl.
