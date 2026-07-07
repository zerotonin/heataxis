"""Numeric checks for the thermal-index implementations."""

from __future__ import annotations

import numpy as np

from heataxis import indices


def test_thi_nrc_known_value():
    # (1.8*28+32) - (0.55 - 0.0055*60)*(1.8*28 - 26) = 77.032
    assert np.isclose(float(indices.thi_nrc(28.0, 60.0)), 77.032, atol=1e-3)


def test_primary_source_aliases_are_same_callable():
    # Descriptive names must be the exact same functions (single source of truth).
    assert indices.thi_nrc1971 is indices.thi_nrc
    assert indices.thi_adj_mader is indices.thi_adj
    assert indices.eti_baeta is indices.eti
    assert indices.hli_gaughan is indices.hli


def test_tbg_estimate_physics():
    # No solar -> globe equals air temperature exactly.
    assert np.isclose(float(indices.tbg_estimate(30.0, 0.0, 1.0)), 30.0)
    # Solar load raises the globe above air temperature...
    assert float(indices.tbg_estimate(30.0, 800.0, 0.5)) > 30.0
    # ...and more wind cools it back towards air temperature.
    hot_still = float(indices.tbg_estimate(30.0, 800.0, 0.2))
    hot_windy = float(indices.tbg_estimate(30.0, 800.0, 4.0))
    assert 30.0 < hot_windy < hot_still


def test_thi_variants_set():
    # The NRC-1971 variant must match the standalone function exactly.
    v0 = indices.thi_variants(32.0, 50.0)
    assert np.isclose(v0["NRC1971_RH"], float(indices.thi_nrc1971(32.0, 50.0)))
    # All variants finite; at a hot-humid point they genuinely disagree
    # (a band, not a line) — the point of the THI-variant overlay.
    v = indices.thi_variants(36.0, 85.0)
    vals = np.array(list(v.values()), dtype=float)
    assert np.all(np.isfinite(vals))
    assert vals.max() - vals.min() > 1.0


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


def test_cci_worked_example():
    # Source worked example: Ta 30, RH 50, u 1, SR 500 -> 37.9 degC.
    assert np.isclose(float(indices.cci(30.0, 50.0, 1.0, 500.0)), 37.9, atol=0.1)


def test_eti_wind_is_cooling():
    # Full Baeta form: at a warm baseline, more wind lowers ETI.
    assert float(indices.eti(24.0, 60.0, 3.0)) < float(indices.eti(24.0, 60.0, 0.0))


def test_esi_worked_value_and_solar_monotone():
    # 0.63*35 - 0.03*50 + 0.002*800 + 0.0054*35*50 - 0.073/(0.1+800) = 31.60
    assert np.isclose(float(indices.esi(35.0, 50.0, 800.0)), 31.60, atol=0.01)
    assert float(indices.esi(30.0, 50.0, 800.0)) > float(indices.esi(30.0, 50.0, 100.0))


def test_itsc_finite_and_requires_inputs():
    import pytest
    val = float(indices.itsc(32.0, 1.5, rh=60.0, sr=700.0, t_rm=44.0))
    assert np.isfinite(val)
    with pytest.raises(ValueError):
        indices.itsc(32.0, 1.5)  # no pv/rh, no erhl/(sr,t_rm)
