"""Numeric checks for the thermal-index implementations."""

from __future__ import annotations

import numpy as np

from heataxis import indices


def test_thi_nrc_known_value():
    # (1.8*28+32) - (0.55 - 0.0055*60)*(1.8*28 - 26) = 77.032
    assert np.isclose(float(indices.thi_nrc(28.0, 60.0)), 77.032, atol=1e-3)


def test_hli_piecewise_branches():
    hot = float(indices.hli(30.0, 50.0, 1.0))     # Tbg >= 25 branch
    cool = float(indices.hli(20.0, 50.0, 1.0))    # Tbg < 25 branch
    assert np.isclose(cool, 10.66 + 0.28 * 50 + 1.30 * 20 - 1.0)
    assert hot > cool


def test_vectorised_io():
    ta = np.array([20.0, 30.0])
    rh = np.array([50.0, 60.0])
    out = indices.thi_nrc(ta, rh)
    assert out.shape == (2,)


def test_psychrometrics_sane():
    # dew point <= air temp; wet bulb between dew point and air temp.
    ta, rh = 25.0, 60.0
    tdp = float(indices.dew_point(ta, rh))
    tw = float(indices.wet_bulb(ta, rh))
    assert tdp <= tw <= ta


def test_etic_finite():
    assert np.isfinite(float(indices.etic(30.0, 50.0, 1.0, 400.0)))


def test_unimplemented_raise():
    import pytest
    with pytest.raises(NotImplementedError):
        indices.cci(30.0, 50.0, 1.0, 400.0)
