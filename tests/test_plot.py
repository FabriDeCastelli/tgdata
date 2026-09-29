import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from tgdata.plot import plot_forecast, plot_heatmap, plot_signals  # noqa: E402


def test_signals_leave_gaps_at_missing_readings(static_graph):
    ax = plot_signals(static_graph, nodes=[0, 3], start=5, end=40)
    lines = ax.get_lines()
    assert len(lines) == 2
    y = lines[0].get_ydata()
    assert len(y) == 35
    np.testing.assert_array_equal(np.isnan(y), ~static_graph.mask[5:40, 0])


def test_signals_accept_dates(static_graph):
    ax = plot_signals(static_graph, nodes=[1], start="1970-01-01T00:10:00",
                      end="1970-01-01T01:00:00")
    assert len(ax.get_lines()[0].get_ydata()) == 10


def test_covariates_and_heatmap(static_graph):
    static_graph.covariates = np.ones((50, 6, 1), np.float32)
    assert plot_signals(static_graph, nodes=[0], field="covariates").get_lines()
    image = plot_heatmap(static_graph, 0, 20).get_images()[0]
    assert image.get_array().shape == (6, 20)


def test_forecast(static_graph):
    sample = static_graph.task()[0]
    prediction = (sample["y"] * 0).requires_grad_()
    axes = plot_forecast(sample, nodes=[0, 2], prediction=prediction)
    assert len(axes) == 2 and len(axes[0].get_lines()) == 4


def test_all_missing_keeps_the_requested_time_range(static_graph):
    static_graph.mask[:, 4] = False
    ax = plot_signals(static_graph, nodes=[4], start=10, end=20)
    assert np.isnan(ax.get_lines()[0].get_ydata()).all()
    lo, hi = ax.get_xlim()
    assert lo < hi and ax.get_lines()[0].get_xdata()[0] == np.datetime64(3000, "s")
