import numpy as np
import pytest

from tgdata.encoding import DerivedMask, Encoded, encode, mask_is_derivable


def test_integer_flows_become_int16_exactly():
    x = np.random.default_rng(0).integers(0, 1900, size=(50, 7, 1)).astype(np.float64)
    enc = encode(x)
    assert enc.raw.dtype == np.int16 and enc.decimals == [0]
    np.testing.assert_array_equal(enc[:], x.astype(np.float32))
    assert enc[3:9, 2:4, 0].shape == (6, 2) and enc.dtype == np.float32


def test_decimal_channels_decode_to_the_same_float32():
    rng = np.random.default_rng(1)
    occupancy = np.round(rng.uniform(0, 0.9, size=(40, 5)), 4)
    speed = np.round(rng.uniform(0, 85, size=(40, 5)), 1)
    values = np.stack([occupancy, speed], -1)
    enc = encode(values)
    assert enc.decimals == [4, 1] and enc.raw.dtype == np.int16
    decoded = enc[:]
    assert decoded.dtype == np.float32
    expected = values.astype(np.float32)
    np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(enc[5:7, :, 1], values[5:7, :, 1].astype(np.float32))


def test_negative_values_and_refusal():
    assert encode(np.array([[[-3.0]], [[2.0]]])).raw.dtype == np.int8
    assert encode(np.array([[[0.1234567]]])) is None
    assert encode(np.array([[[1e12]]])) is None


def test_derived_mask_and_node_views():
    x = np.array([[[0.0], [4.0], [7.0]], [[5.0], [0.0], [2.0]]])
    enc = encode(x)
    mask = DerivedMask(enc)
    np.testing.assert_array_equal(mask[:], x[..., 0] != 0)
    assert mask_is_derivable(x[..., 0] != 0, enc)
    assert not mask_is_derivable(np.ones((2, 3), bool), enc)
    view = enc.take_nodes(slice(1, 3))
    assert np.shares_memory(view.raw, enc.raw)
    np.testing.assert_array_equal(mask.take_nodes(slice(1, 3))[:], x[:, 1:3, 0] != 0)


def test_encoded_rejects_ambiguous_indexing():
    enc = Encoded(np.zeros((2, 3, 1), np.uint8), [0])
    with pytest.raises(IndexError):
        enc[..., 0]


def test_compact_graph_through_the_pipeline(static_graph, tmp_path):
    import tgdata
    from tgdata.encoding import compact

    g = static_graph
    g.x = np.round(np.abs(g.x) * 100).astype(np.float32)
    g.mask = g.x[..., 0] != 0
    g.covariates = np.round(g.covariates, 2)
    c = compact(g)
    assert isinstance(c.x, Encoded) and isinstance(c.mask, DerivedMask)
    assert isinstance(c.covariates, Encoded) and c.covariates.decimals == [2, 2, 2]
    tgdata.save(c, tmp_path)
    assert not (tmp_path / "arrays" / "mask.npy").exists()
    h = tgdata.load_dir(tmp_path)
    assert h.x.raw.dtype == np.int16
    np.testing.assert_array_equal(np.asarray(h.x), g.x)
    np.testing.assert_array_equal(np.asarray(h.mask), g.mask)
    np.testing.assert_array_equal(np.asarray(h.covariates), g.covariates.astype(np.float32))
    assert tgdata.compute_stats(h) == tgdata.compute_stats(g)
    a, b = h.window(4, 3)[5], g.window(4, 3)[5]
    for key in a:
        if hasattr(a[key], "shape"):
            np.testing.assert_array_equal(a[key], b[key])
    sub = h.select_nodes(np.array([1, 2, 3]))
    assert np.shares_memory(sub.x.raw, h.x.raw)
    np.testing.assert_array_equal(np.asarray(sub.mask), g.mask[:, 1:4])
