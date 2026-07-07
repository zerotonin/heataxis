"""Checks for the exposure-history transforms."""

from __future__ import annotations

import numpy as np

from heataxis import history


def test_heat_load_above_rectifies():
    x = history.heat_load_above([70.0, 72.0, 75.0], 72.0)
    assert np.allclose(x, [0.0, 0.0, 3.0])


def test_leaky_integrator_saturates_at_tau_times_load():
    # Constant load above baseline -> steady state L_ss = tau * x.
    dt, tau, load = 1.0, 10.0, 4.0
    index = np.full(400, 72.0 + load)      # long constant forcing
    out = history.leaky_integrate(index, dt, tau, baseline=72.0)
    assert np.isclose(out[-1], tau * load, rtol=1e-3)
    # Rises monotonically towards the bound and never overshoots it.
    assert np.all(np.diff(out) >= -1e-9)
    assert out[-1] <= tau * load + 1e-6


def test_leaky_integrator_recovers_when_load_stops():
    dt, tau = 1.0, 6.0
    index = np.concatenate([np.full(50, 80.0), np.full(50, 60.0)])  # heat then cool
    out = history.leaky_integrate(index, dt, tau, baseline=72.0)
    assert out[49] > 0.0                    # accumulated during heat
    assert out[-1] < out[49] * 0.05         # discharged during the cool spell


def test_cumsum_unbounded_vs_leaky_bounded():
    dt, load = 1.0, 3.0
    index = np.full(500, 72.0 + load)
    cum = history.cumulative_load(index, dt, baseline=72.0)
    leaky = history.leaky_integrate(index, dt, 10.0, baseline=72.0)
    # Cum-sum grows without bound; leaky saturates far below it.
    assert cum[-1] > 10.0 * leaky[-1]
    assert np.isclose(cum[-1], load * dt * 500, rtol=1e-6)


def test_ahl_accumulates_and_dissipates_and_floors():
    dt = 1.0
    hli = np.concatenate([np.full(10, 96.0), np.full(40, 60.0)])  # hot then cold
    ahl = history.accumulated_heat_load(hli, dt, upper=86.0, lower=77.0)
    assert ahl[9] > 0.0                     # (96-86)*10 accumulated
    assert np.isclose(ahl[9], (96.0 - 86.0) * 10.0)
    assert ahl[-1] == 0.0                   # dissipated and floored at zero


def test_tsd_and_tsl_count_duration_and_load():
    dt = 1.0
    index = np.array([70.0, 74.0, 76.0, 71.0])   # 2 hours above 72
    tsd = history.thermal_stress_duration(index, dt, 72.0)
    tsl = history.thermal_stress_load(index, dt, 72.0)
    assert np.isclose(tsd[-1], 2.0)               # cumulative hours above
    assert np.isclose(tsl[-1], (74 - 72) + (76 - 72))  # 2 + 4 degree-hours
    # Non-cumulative TSD reports the current consecutive bout and resets.
    bout = history.thermal_stress_duration(index, dt, 72.0, cumulative=False)
    assert np.allclose(bout, [0.0, 1.0, 2.0, 0.0])
