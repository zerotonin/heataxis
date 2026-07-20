"""Checks for the exposure-history transforms."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_leaky_integrator_variable_dt_matches_scalar_and_forgets_over_gaps():
    tau = 8.0
    index = np.full(200, 78.0)                 # constant load above 72
    scalar = history.leaky_integrate(index, 1.0, tau, baseline=72.0)
    array = history.leaky_integrate(index, np.full(200, 1.0), tau, baseline=72.0)
    assert np.allclose(scalar, array)          # array dt == scalar dt
    # A huge gap before the last sample must reset the integrator to tau*x.
    dt = np.full(200, 1.0)
    dt[-1] = 1000.0                             # ~125*tau gap -> full forgetting
    gapped = history.leaky_integrate(index, dt, tau, baseline=72.0)
    assert np.isclose(gapped[-1], tau * 6.0, rtol=1e-3)  # x = 78-72 = 6
    with pytest.raises(ValueError):
        history.leaky_integrate(index, np.ones(5), tau)


def test_leaky_integrator_rectify_flag_gates_discharge_below_baseline():
    # Rectified: a driver below the baseline contributes nothing, so the state
    # only ever decays towards zero and never goes negative.
    dt, tau = 1.0, 6.0
    index = np.full(200, 65.0)                       # 7 below the baseline
    rect = history.leaky_integrate(index, dt, tau, baseline=72.0, rectify=True)
    assert np.allclose(rect, 0.0)
    # Signed: the same driver drives the state to the negative steady state
    # tau * x, i.e. the animal sheds heat towards a cool equilibrium.
    signed = history.leaky_integrate(index, dt, tau, baseline=72.0, rectify=False)
    assert np.isclose(signed[-1], tau * -7.0, rtol=1e-3)
    assert np.all(signed <= 0.0)


def test_low_pass_tracks_the_driver_and_is_scale_invariant_in_tau():
    # Defining contrast with leaky_integrate: the steady state is the driver
    # itself, so it does NOT scale with tau.
    dt, level = 1.0, 78.0
    index = np.full(600, level)
    for tau in (1.0, 8.0, 24.0):
        out = history.low_pass(index, dt, tau)
        assert np.isclose(out[-1], level, rtol=1e-6)
    # The accumulator, on the same input, does scale with tau.
    acc = history.leaky_integrate(index, dt, 8.0, baseline=72.0)
    assert np.isclose(acc[-1], 8.0 * 6.0, rtol=1e-3)


def test_low_pass_step_response_reaches_one_minus_1_over_e_at_tau():
    # Textbook first-order step response: 63.2 % of the step after one tau.
    dt, tau = 0.01, 5.0
    index = np.concatenate([np.zeros(1), np.ones(100_000)])
    out = history.low_pass(index, dt, tau, initial=0.0)
    at_tau = out[1 + int(round(tau / dt)) - 1]
    assert np.isclose(at_tau, 1.0 - np.exp(-1.0), atol=1e-3)
    assert np.isclose(out[-1], 1.0, rtol=1e-6)       # settles at the step value


def test_low_pass_defaults_to_no_warm_up_transient():
    index = np.full(50, 42.0)
    assert history.low_pass(index, 1.0, 10.0)[0] == 42.0     # starts settled
    assert history.low_pass(index, 1.0, 10.0, initial=0.0)[0] < 42.0


def test_low_pass_variable_dt_matches_scalar_and_forgets_over_gaps():
    tau = 8.0
    index = np.full(200, 78.0)
    scalar = history.low_pass(index, 1.0, tau)
    array = history.low_pass(index, np.full(200, 1.0), tau)
    assert np.allclose(scalar, array)
    # A gap far longer than tau restarts the state at the current sample.
    step = np.concatenate([np.full(100, 60.0), np.full(100, 80.0)])
    dt = np.full(200, 1.0)
    dt[100] = 1000.0
    gapped = history.low_pass(step, dt, tau)
    assert np.isclose(gapped[100], 80.0, rtol=1e-3)
    with pytest.raises(ValueError):
        history.low_pass(index, np.ones(5), tau)
    with pytest.raises(ValueError):
        history.low_pass([], 1.0, tau)


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
