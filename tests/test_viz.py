"""Smoke tests for plotting helpers (Agg backend, no display)."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from heataxis import viz  # noqa: E402


def test_schematic_axes_and_save(tmp_path):
    viz.setup_figure()
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    viz.schematic_axes(ax, "predictor", "response")
    viz.save_figure(fig, "smoke", tmp_path)
    plt.close(fig)
    assert (tmp_path / "smoke.svg").exists()
    assert (tmp_path / "smoke.png").exists()
