# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — history                                              ║
# ║  « the x-axis needs a memory — exposure-history transforms »     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Turn an index TIME SERIES into a load history.  Where indices.py ║
# ║  maps the present environment to a number, these map a history of ║
# ║  load to an accumulated / recovering state — the Rung-0 and       ║
# ║  Rung-1 tools of the exposure-history model (see the DigiMuh      ║
# ║  concept notes).                                                  ║
# ║                                                                  ║
# ║  Inputs are 1-D series at a fixed step ``dt_h`` (hours).          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Exposure-history transforms: an index time series -> a load history.

These operate on a *time series* of a thermal index, not a single reading:
the accumulated / recovering heat load is a function of the history of load,
not the instantaneous value.  All series share a fixed time step ``dt_h`` in
hours.
"""

from __future__ import annotations

import numpy as np

from heataxis.constants import ArrayLike

__all__ = [
    "heat_load_above", "leaky_integrate", "cumulative_load",
    "accumulated_heat_load", "thermal_stress_duration", "thermal_stress_load",
]


def heat_load_above(index: ArrayLike, baseline: float) -> np.ndarray:
    """Rectified load above a baseline: ``max(index - baseline, 0)``.

    Args:
        index: Thermal-index time series.
        baseline: Load-onset baseline (index units).

    Returns:
        The non-negative excess of the index over the baseline.
    """
    return np.clip(np.asarray(index, dtype=float) - baseline, 0.0, None)


def leaky_integrate(index: ArrayLike, dt_h: float, tau_h: float, *,
                    baseline: float = 0.0) -> np.ndarray:
    """Leaky integral of the load above ``baseline`` (low-pass with recovery).

    Solves ``dL/dt = -L/tau + x(t)`` with ``x = max(index - baseline, 0)`` using
    the exact update for a piecewise-constant input, so it is stable at any step.
    Unlike a pure cum-sum it **forgets**: a cool spell discharges the accumulated
    load.  Under a sustained load it **saturates** at ``L_ss = tau * x`` (a bounded
    state), and reaches ~95 % of that in ~3*tau.  This is the Rung-1 model of the
    exposure-history ladder.

    Args:
        index: Thermal-index time series.
        dt_h: Time step (hours).
        tau_h: Memory / recovery time constant (hours).
        baseline: Load-onset baseline (index units).

    Returns:
        The leaky-integrated load (index-hours), same length as ``index``.
    """
    x = heat_load_above(index, baseline)
    decay = float(np.exp(-dt_h / tau_h))
    gain = tau_h * (1.0 - decay)          # exact step response over dt_h
    out = np.empty_like(x)
    acc = 0.0
    for i in range(x.size):
        acc = acc * decay + x[i] * gain
        out[i] = acc
    return out


def cumulative_load(index: ArrayLike, dt_h: float, *,
                    baseline: float = 0.0) -> np.ndarray:
    """Pure cumulative load above ``baseline`` (cum-sum; no recovery).

    The ``tau -> infinity`` limit of :func:`leaky_integrate`: it grows without
    bound because it never forgets, the crude extreme against which the leaky
    integrator is the physically-motivated middle.

    Args:
        index: Thermal-index time series.
        dt_h: Time step (hours).
        baseline: Load-onset baseline (index units).

    Returns:
        The cumulative load (index-hours).
    """
    return np.cumsum(heat_load_above(index, baseline)) * dt_h


def accumulated_heat_load(hli: ArrayLike, dt_h: float, *,
                          upper: float = 86.0, lower: float = 77.0) -> np.ndarray:
    """Accumulated heat load (AHL) from a heat-load-index series (Gaughan, 2008).

    Heat load accumulates while HLI is above ``upper`` and dissipates while below
    ``lower`` (balanced in between), floored at zero.  This is the field's
    fixed-formula precursor of the leaky integrator: a threshold with a night
    recovery term, but not fitted per animal.  ``upper``/``lower`` are
    genotype- and management-dependent; the defaults (86 / 77) are for unshaded
    *Bos taurus*.

    Args:
        hli: Heat-load-index (HLI) time series.
        dt_h: Time step (hours).
        upper: Accumulation threshold (HLI units).
        lower: Dissipation threshold (HLI units).

    Returns:
        Accumulated heat load (HLI-hours above threshold), floored at 0.

    References:
        Gaughan, J. B., Mader, T. L., Holt, S. M., & Lisle, A. (2008). A new
        heat load index for feedlot cattle. Journal of Animal Science, 86(1),
        226-234. https://doi.org/10.2527/jas.2007-0305
    """
    hli = np.asarray(hli, dtype=float)
    out = np.empty_like(hli)
    acc = 0.0
    for i in range(hli.size):
        h = hli[i]
        if h > upper:
            rate = h - upper
        elif h < lower:
            rate = h - lower          # negative -> dissipation
        else:
            rate = 0.0
        acc = max(0.0, acc + rate * dt_h)
        out[i] = acc
    return out


def thermal_stress_duration(index: ArrayLike, dt_h: float, threshold: float, *,
                            cumulative: bool = True) -> np.ndarray:
    """Thermal stress duration (TSD): time the index spends above ``threshold``.

    Cumulative running total (hours) by default; with ``cumulative=False`` it
    returns the length of the *current* consecutive bout, resetting to 0 whenever
    the index drops below the threshold.

    Args:
        index: Thermal-index time series.
        dt_h: Time step (hours).
        threshold: Stress-onset threshold (index units).
        cumulative: Running total (True) or current bout length (False).

    Returns:
        Duration above threshold (hours).

    References:
        History-based thermal-stress indices, Neira et al. (2026).
        # TODO verify the exact TSD definition (window, threshold) against the
        # primary source before using in the paper.
    """
    over = np.asarray(index, dtype=float) > threshold
    if cumulative:
        return np.cumsum(over) * dt_h
    out = np.empty(over.size, dtype=float)
    bout = 0.0
    for i, is_over in enumerate(over):
        bout = bout + dt_h if is_over else 0.0
        out[i] = bout
    return out


def thermal_stress_load(index: ArrayLike, dt_h: float,
                        threshold: float) -> np.ndarray:
    """Thermal stress load (TSL): cumulative load above ``threshold`` (degree-hours).

    The accumulated excess ``sum (index - threshold)+ * dt``.  Equivalent to
    :func:`cumulative_load` anchored at the stress threshold, named here as the
    established exposure-*load* comparator.

    Args:
        index: Thermal-index time series.
        dt_h: Time step (hours).
        threshold: Stress-onset threshold (index units).

    Returns:
        Cumulative load above threshold (index-hours).

    References:
        History-based thermal-stress indices, Neira et al. (2026).
        # TODO verify the exact TSL definition against the primary source.
    """
    return cumulative_load(index, dt_h, baseline=threshold)
