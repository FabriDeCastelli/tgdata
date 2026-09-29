from pathlib import Path

import numpy as np
import pytest

import tgdata
from tgdata.converters import pems08

RAW = Path("/raid/f.decastelli/data/raw/traffic/PEMS08")


def test_build_from_synthetic_raw(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.uniform(1, 100, size=(100, 5, 3))
    data[3, 2, :2] = 0
    g = pems08.build(data, [(0, 1, 10.0), (1, 2, 20.0), (0, 1, 10.0)], {"PEMS08.npz": "x"})
    assert g.edge_index.tolist() == [[0, 1], [1, 2]]
    assert g.meta["adjacency"]["kind"] == "binary"
    assert g.x.shape == (100, 5, 1) and g.covariates.shape == (100, 5, 2)
    np.testing.assert_array_equal(g.edge_weight, [10.0, 20.0])
    assert tgdata.adjacency(g)[1].tolist() == [1.0, 1.0]
    assert not g.mask[3, 2] and g.mask.sum() == 100 * 5 - 1
    assert g.splits["default"].over == "samples"
    assert g.splits["70/10/20"].resolve(100)["train"] == (0, 70)
    assert np.asarray(g.meta["stats_node"]["mean"]).shape == (5, 1)
    s = g.task()[0]
    assert s["y"].shape == (12, 5, 1) and s["covariates"].shape == (24, 5, 2)
    tgdata.save(g, tmp_path)
    tgdata.validate(tgdata.load_dir(tmp_path))


@pytest.mark.skipif(not RAW.exists(), reason="raw PEMS08 not available")
def test_real_pems08_calendar():
    g = pems08.build(*pems08.read_raw(RAW))
    pems08.check_calendar(g)
    assert g.x.shape == (17856, 170, 1)
    assert g.edge_index.shape == (2, 277)
    sizes = [len(g.task(split=s)) for s in ("train", "val", "test")]
    assert sizes == [10699, 3567, 3567]  # ASTGCN: 17833 samples cut at int(n*0.6), int(n*0.8)
