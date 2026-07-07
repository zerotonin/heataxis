# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — thresholds                                           ║
# ║  « the y-axis — physiological threshold detection (Part II) »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  A consistent API over threshold-detection methods.  Each fit    ║
# ║  returns a ThresholdFit with the estimated threshold and fit stats. ║
# ║                                                                  ║
# ║  Implemented: broken_stick, hill_fit, derivative_exceedance.     ║
# ║  Stubs (Part II work): davies_test, gam_threshold, bayesian_changepoint. ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Threshold-detection methods with a consistent interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ThresholdFit",
    "broken_stick", "hill_fit", "derivative_exceedance",
    "davies_test", "gam_threshold", "bayesian_changepoint",
]


@dataclass
class ThresholdFit:
    """Result of a threshold fit.

    Attributes:
        method:    Name of the method.
        threshold: Estimated threshold on the predictor axis (NaN if none).
        converged: Whether the fit produced a finite threshold.
        params:    Method-specific parameters (slopes, EC50, Hill n, ...).
        stats:     Goodness-of-fit statistics (r2, aic, ...).
    """

    method: str
    threshold: float
    converged: bool
    params: dict[str, float] = field(default_factory=dict)
    stats: dict[str, float] = field(default_factory=dict)


def _clean(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok]


def broken_stick(x: np.ndarray, y: np.ndarray, *, n_grid: int = 100) -> ThresholdFit:
    """Two-segment continuous piecewise-linear fit by grid search on the breakpoint."""
    x, y = _clean(x, y)
    if x.size < 4:
        return ThresholdFit("broken_stick", float("nan"), False)
    lo, hi = np.quantile(x, [0.1, 0.9])
    best = None
    for psi in np.linspace(lo, hi, n_grid):
        hinge = np.clip(x - psi, 0.0, None)
        design = np.column_stack([np.ones_like(x), x, hinge])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        sse = float(resid @ resid)
        if best is None or sse < best[0]:
            best = (sse, psi, coef)
    sse, psi, coef = best
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return ThresholdFit(
        "broken_stick", float(psi), True,
        params={"intercept": float(coef[0]), "slope1": float(coef[1]),
                "slope2": float(coef[1] + coef[2])},
        stats={"r2": r2, "sse": sse},
    )


def hill_fit(x: np.ndarray, y: np.ndarray) -> ThresholdFit:
    """Four-parameter logistic (Hill) fit; threshold is EC50.

    Requires scipy (install the ``stats`` extra).
    """
    from scipy.optimize import curve_fit

    x, y = _clean(x, y)
    if x.size < 5:
        return ThresholdFit("hill_fit", float("nan"), False)

    def four_pl(xx, ymin, ymax, ec50, n):
        return ymin + (ymax - ymin) / (1.0 + (ec50 / np.clip(xx, 1e-9, None)) ** n)

    p0 = [float(np.min(y)), float(np.max(y)), float(np.median(x)), 2.0]
    try:
        popt, _ = curve_fit(four_pl, x, y, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError):
        return ThresholdFit("hill_fit", float("nan"), False)
    resid = y - four_pl(x, *popt)
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return ThresholdFit(
        "hill_fit", float(popt[2]), True,
        params={"ymin": float(popt[0]), "ymax": float(popt[1]),
                "ec50": float(popt[2]), "hill_n": float(popt[3])},
        stats={"r2": r2, "sse": sse},
    )


def derivative_exceedance(x: np.ndarray, y: np.ndarray, *, n_bins: int = 20,
                          k_sd: float = 2.0) -> ThresholdFit:
    """Bin the predictor, then flag the first bin whose slope exceeds baseline + k·SD."""
    x, y = _clean(x, y)
    if x.size < 8:
        return ThresholdFit("derivative_exceedance", float("nan"), False)
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    means = np.array([y[idx == b].mean() if np.any(idx == b) else np.nan
                      for b in range(n_bins)])
    valid = np.isfinite(means)
    if valid.sum() < 4:
        return ThresholdFit("derivative_exceedance", float("nan"), False)
    slope = np.gradient(means[valid], centres[valid])
    base = slope[: max(2, slope.size // 4)]
    cutoff = base.mean() + k_sd * base.std()
    over = np.where(slope > cutoff)[0]
    if over.size == 0:
        return ThresholdFit("derivative_exceedance", float("nan"), False)
    thr = float(centres[valid][over[0]])
    return ThresholdFit("derivative_exceedance", thr, True,
                        stats={"cutoff_slope": float(cutoff)})


def davies_test(*_args, **_kwargs) -> ThresholdFit:
    """Davies / pseudo-Score existence test (Part II). Not yet implemented."""
    raise NotImplementedError("davies_test is part of the Part II work.")


def gam_threshold(*_args, **_kwargs) -> ThresholdFit:
    """GAM second-derivative threshold (Part II). Not yet implemented."""
    raise NotImplementedError("gam_threshold is part of the Part II work.")


def bayesian_changepoint(*_args, **_kwargs) -> ThresholdFit:
    """Bayesian change-point detection (Part II). Not yet implemented."""
    raise NotImplementedError("bayesian_changepoint is part of the Part II work.")
