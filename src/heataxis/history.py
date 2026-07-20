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
    "heat_load_above", "leaky_integrate", "low_pass", "cumulative_load",
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


def leaky_integrate(index: ArrayLike, dt_h: float | np.ndarray, tau_h: float, *,
                    baseline: float = 0.0, rectify: bool = True) -> np.ndarray:
    """Leaky integral of the load relative to ``baseline`` (low-pass with recovery).

    Solves ``dL/dt = -L/tau + x(t)`` using the exact update for a piecewise-constant
    input, so it is stable at any step. The driver is ``x = index - baseline``,
    **rectified to its positive part when ``rectify`` is True** (load only above the
    baseline) or **signed when ``rectify`` is False** (the state also *discharges*
    below the baseline — the physically correct choice when ``index`` is a body-heat
    equilibrium that the animal sheds heat towards, not a threshold-gated load). The
    rectified form saturates at ``L_ss = tau * x`` and, because it clamps at zero,
    is heavily zero-inflated whenever ``index`` sits below ``baseline``.

    ``dt_h`` may be a scalar (regular sampling) or a per-sample array giving the
    gap from the previous sample to each one — the latter is the correct choice
    for **irregular / gappy** real sensor series: a large gap makes ``exp(-dt/tau)``
    vanish, so the integrator forgets the past and restarts, exactly as it should.

    Args:
        index: Thermal-index (or equilibrium) time series.
        dt_h: Time step (hours); scalar or one value per sample.
        tau_h: Memory / recovery time constant (hours).
        baseline: Reference the load is measured from.
        rectify: Clamp the driver at zero (threshold load) if True; keep it signed
            (continuous heat balance) if False.

    Returns:
        The leaky-integrated load, same length as ``index``.

    Raises:
        ValueError: If ``dt_h`` is an array whose length differs from ``index``.
    """
    if rectify:
        x = heat_load_above(index, baseline)
    else:
        x = np.asarray(index, dtype=float) - baseline
    dt = np.asarray(dt_h, dtype=float)
    if dt.ndim == 0:
        decay = np.full(x.size, float(np.exp(-dt / tau_h)))
    elif dt.size == x.size:
        decay = np.exp(-dt / tau_h)
    else:
        raise ValueError("dt_h array must match the length of index")
    out = np.empty_like(x)
    acc = 0.0
    for i in range(x.size):
        d = decay[i]
        acc = acc * d + x[i] * tau_h * (1.0 - d)   # exact step response over dt
        out[i] = acc
    return out


def low_pass(index: ArrayLike, dt_h: float | np.ndarray, tau_h: float, *,
             initial: float | None = None) -> np.ndarray:
    """First-order low-pass (Newton cooling) of an index series.

    Solves ``dL/dt = (x - L)/tau`` with the exact update for a piecewise-constant
    input.  Unlike :func:`leaky_integrate` this **tracks** rather than
    accumulates: the steady state is the driver itself (``L_ss = x``) instead of
    ``tau * x``, so the output keeps the units of ``index`` and stays comparable
    across ``tau``.  That is what makes it the right transform for asking *how
    much memory* a predictor needs — varying ``tau`` changes the smoothing only,
    not the scale, so the resulting fits are commensurable.

    It is the exponential moving average in continuous time; the discrete update
    is ``L_i = L_{i-1} * exp(-dt/tau) + x_i * (1 - exp(-dt/tau))``.

    ``dt_h`` may be a scalar (regular sampling) or one value per sample; a gap
    much larger than ``tau`` sends the decay to zero, so the state restarts from
    the current sample.

    Args:
        index: Thermal-index (or any driver) time series.
        dt_h: Time step (hours); scalar or one value per sample.
        tau_h: Memory time constant (hours).
        initial: Starting state.  Defaults to ``index[0]``, which begins the
            series already in equilibrium and so avoids a warm-up transient
            that would otherwise contaminate the first few ``tau``.

    Returns:
        The low-passed series, same length and units as ``index``.

    Raises:
        ValueError: If ``index`` is empty, or ``dt_h`` is an array whose length
            differs from ``index``.
    """
    x = np.asarray(index, dtype=float)
    if x.size == 0:
        raise ValueError("index must be non-empty")
    dt = np.asarray(dt_h, dtype=float)
    if dt.ndim == 0:
        decay = np.full(x.size, float(np.exp(-dt / tau_h)))
    elif dt.size == x.size:
        decay = np.exp(-dt / tau_h)
    else:
        raise ValueError("dt_h array must match the length of index")
    out = np.empty_like(x)
    acc = float(x[0]) if initial is None else float(initial)
    for i in range(x.size):
        d = decay[i]
        acc = acc * d + x[i] * (1.0 - d)
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
