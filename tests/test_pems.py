from pathlib import Path

import numpy as np
import pytest

import tgdata
from tgdata.converters import pems

RAW = Path("/raid/f.decastelli/data/raw/traffic")

# Train/val/test sizes the introducing papers' own code produces:
# ASTGCN cuts its T - 23 samples at int(n * 0.6), int(n * 0.8);
# STSGCN cuts the series at int(T * 0.6), int(T * 0.8) and windows each part (T_part - 23).
EXPECTED = {
    "pems03": ((26208, 358, 1), 547, 1093, [15701, 5219, 5219]),
    "pems04": ((16992, 307, 1), 340, 340, [10181, 3394, 3394]),
    "pems07": ((28224, 883, 1), 866, 1732, [16911, 5622, 5622]),
    "pems08": ((17856, 170, 1), 277, 277, [10699, 3567, 3567]),
}


def synthetic(name, T=100, N=5):
    rng = np.random.default_rng(0)
    data = rng.uniform(1, 100, size=(T, N, 1 + len(pems.SPECS[name].covariates)))
    data[3, 2, 0] = 0
    return data


def test_astgcn_lineage(tmp_path):
    g = pems.build("pems08", synthetic("pems08"), [(0, 1, 10.0), (1, 2, 20.0), (0, 1, 10.0)],
                   {"PEMS08.npz": "x"})
    assert g.edge_index.tolist() == [[0, 1], [1, 2]]
    assert g.x.shape == (100, 5, 1) and g.covariates.shape == (100, 5, 2)
    np.testing.assert_array_equal(g.edge_weight, [10.0, 20.0])
    assert tgdata.adjacency(g)[1].tolist() == [1.0, 1.0]
    assert not g.mask[3, 2] and g.mask.sum() == 100 * 5 - 1
    split = g.splits["default"]
    assert split.over == "samples" and not split.strict
    s = g.task()[0]
    assert s["y"].shape == (12, 5, 1) and s["covariates"].shape == (24, 5, 2)
    tgdata.save(g, tmp_path)
    tgdata.validate(tgdata.load_dir(tmp_path))


def test_stsgcn_lineage():
    g = pems.build("pems07", synthetic("pems07", T=200), [(0, 1, 1.0), (1, 2, 2.0), (2, 1, 2.0)],
                   {"PEMS07.npz": "x"})
    assert g.covariates is None
    ei, w = tgdata.adjacency(g)
    assert sorted(map(tuple, ei.T.tolist())) == [(0, 1), (1, 0), (1, 2), (2, 1)]
    assert (w == 1).all()
    assert g.splits["default"].strict
    for part in ("train", "val", "test"):
        start, end = g.splits["default"].resolve(200)[part]
        anchors = g.task(split=part).anchors
        assert anchors[0] == start + 12 and anchors[-1] == end - 12
    assert "date_note" in g.meta


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_real_release(name):
    raw = RAW / name.upper()
    if not raw.exists():
        pytest.skip(f"raw {name} not available")
    g = pems.build(name, *pems.read_raw(name, raw))
    pems.check_calendar(g)
    shape, stored, adjacency, sizes = EXPECTED[name]
    assert g.x.shape == shape
    assert g.edge_index.shape[1] == stored
    assert tgdata.adjacency(g)[0].shape[1] == adjacency
    assert [len(g.task(split=s)) for s in ("train", "val", "test")] == sizes
