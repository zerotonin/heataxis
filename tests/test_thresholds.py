"""Recovery checks for the implemented threshold methods."""

from __future__ import annotations

import numpy as np

from heataxis import thresholds


def _sharp(psi=6.0, n=200):
    x = np.linspace(0.0, 10.0, n)
    y = np.where(x <= psi, 1.0, 1.0 + 0.9 * (x - psi))
    return x, y


def test_broken_stick_recovers_breakpoint():
    x, y = _sharp(psi=6.0)
    fit = thresholds.broken_stick(x, y)
    assert fit.converged
    assert abs(fit.threshold - 6.0) < 0.6


def test_hill_fit_recovers_ec50():
    x = np.linspace(0.1, 10.0, 200)
    y = 1.0 + 5.0 / (1.0 + (5.0 / x) ** 4)
    fit = thresholds.hill_fit(x, y)
    assert fit.converged
    assert abs(fit.params["ec50"] - 5.0) < 0.5


def test_stub_raises():
    import pytest
    with pytest.raises(NotImplementedError):
        thresholds.davies_test(None, None)
