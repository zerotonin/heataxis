# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — indices                                              ║
# ║  « the x-axis — cattle thermal / heat-load indices (Part I) »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Vectorised, unit-tested implementations of the cattle thermal   ║
# ║  indices, grouped by heat-exchange pathway (see Part I).  Inputs ║
# ║  follow constants.UNITS (Ta °C, RH %, u m s⁻¹, SR W m⁻²).        ║
# ║                                                                  ║
# ║  CCI and ITSC are intentionally left unimplemented until their   ║
# ║  piecewise coefficient sets are transcribed from the primary papers. ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Cattle thermal / heat-load indices, grouped by heat-exchange pathway."""

from __future__ import annotations

import numpy as np

from heataxis.constants import ArrayLike

__all__ = [
    "dew_point", "wet_bulb",
    "thi_nrc", "thi_wetbulb", "thi_dewpoint",
    "bghi", "hli", "eti", "thi_adj", "etic",
    "cci", "itsc",
]


# ─────────────────────────────────────────────────────────────
#  Psychrometrics « derive Tw / Tdp from Ta and RH »
# ─────────────────────────────────────────────────────────────

def dew_point(ta: ArrayLike, rh: ArrayLike) -> np.ndarray:
    """Dew-point temperature (°C) via the Magnus–Tetens approximation."""
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    a, b = 17.27, 237.7
    gamma = (a * ta) / (b + ta) + np.log(rh / 100.0)
    return (b * gamma) / (a - gamma)


def wet_bulb(ta: ArrayLike, rh: ArrayLike) -> np.ndarray:
    """Wet-bulb temperature (°C) via the Stull (2011) approximation.

    Valid roughly for 5–99 % RH at sea level; adequate for barn climate.
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    return (ta * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(ta + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh)
            - 4.686035)


# ─────────────────────────────────────────────────────────────
#  Class 1 « temperature–humidity indices »
# ─────────────────────────────────────────────────────────────

def thi_nrc(ta: ArrayLike, rh: ArrayLike) -> np.ndarray:
    """THI, relative-humidity form (NRC, 1971)."""
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    return (1.8 * ta + 32) - (0.55 - 0.0055 * rh) * (1.8 * ta - 26)


def thi_wetbulb(ta: ArrayLike, tw: ArrayLike | None = None, *,
                rh: ArrayLike | None = None) -> np.ndarray:
    """THI, wet-bulb form (Thom, 1959; NRC, 1971).  Derives Tw from RH if needed."""
    ta = np.asarray(ta, dtype=float)
    if tw is None:
        if rh is None:
            raise ValueError("provide either tw or rh")
        tw = wet_bulb(ta, rh)
    tw = np.asarray(tw, dtype=float)
    return 0.72 * (ta + tw) + 40.6


def thi_dewpoint(ta: ArrayLike, tdp: ArrayLike | None = None, *,
                 rh: ArrayLike | None = None) -> np.ndarray:
    """THI, dew-point form.  Derives Tdp from RH if needed."""
    ta = np.asarray(ta, dtype=float)
    if tdp is None:
        if rh is None:
            raise ValueError("provide either tdp or rh")
        tdp = dew_point(ta, rh)
    tdp = np.asarray(tdp, dtype=float)
    return ta + 0.36 * tdp + 41.5


# ─────────────────────────────────────────────────────────────
#  Class 2 « radiation-augmented (black-globe) indices »
# ─────────────────────────────────────────────────────────────

def bghi(tbg: ArrayLike, tdp: ArrayLike | None = None, *,
         ta: ArrayLike | None = None, rh: ArrayLike | None = None) -> np.ndarray:
    """Black-globe–humidity index (Buffington et al., 1981)."""
    tbg = np.asarray(tbg, dtype=float)
    if tdp is None:
        if ta is None or rh is None:
            raise ValueError("provide either tdp or both ta and rh")
        tdp = dew_point(ta, rh)
    tdp = np.asarray(tdp, dtype=float)
    return tbg + 0.36 * tdp + 41.5


def hli(tbg: ArrayLike, rh: ArrayLike, u: ArrayLike) -> np.ndarray:
    """Heat load index, feedlot (Gaughan et al., 2008); piecewise in Tbg."""
    tbg = np.asarray(tbg, dtype=float)
    rh = np.asarray(rh, dtype=float)
    u = np.asarray(u, dtype=float)
    hot = 8.62 + 0.38 * rh + 1.55 * tbg + np.exp(2.4 - u) - 0.5 * u
    cool = 10.66 + 0.28 * rh + 1.30 * tbg - u
    return np.where(tbg >= 25.0, hot, cool)


# ─────────────────────────────────────────────────────────────
#  Class 3 « wind-augmented index »
# ─────────────────────────────────────────────────────────────

def eti(ta: ArrayLike, rh: ArrayLike, u: ArrayLike) -> np.ndarray:
    """Equivalent temperature index (Baeta et al., 1987).

    Note: verify the full term set against the primary source.
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    u = np.asarray(u, dtype=float)
    return (27.88 - 0.456 * ta + 0.010754 * ta ** 2
            - 0.4905 * rh + 0.00088 * rh ** 2 + 1.1507 * u)


# ─────────────────────────────────────────────────────────────
#  Class 4a « full microclimate, empirical adjustment »
# ─────────────────────────────────────────────────────────────

def thi_adj(ta: ArrayLike, rh: ArrayLike, u: ArrayLike, sr: ArrayLike) -> np.ndarray:
    """Wind/solar-adjusted THI (Mader et al., 2006)."""
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    u = np.asarray(u, dtype=float)
    sr = np.asarray(sr, dtype=float)
    base = 0.8 * ta + (rh / 100.0) * (ta - 14.4) + 46.4
    return base + 4.51 - 1.992 * u + 0.0068 * sr


# ─────────────────────────────────────────────────────────────
#  Class 4b « full microclimate, biophysically derived »
# ─────────────────────────────────────────────────────────────

def etic(ta: ArrayLike, rh: ArrayLike, u: ArrayLike, sr: ArrayLike) -> np.ndarray:
    """Equivalent temperature index for cattle (Wang et al., 2018).

    Note: verify the wind exponent (0.707) and coefficients against the primary.
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    u = np.asarray(u, dtype=float)
    sr = np.asarray(sr, dtype=float)
    return (ta - 0.0038 * ta * (100 - rh)
            - 0.1173 * np.power(np.clip(u, 0.0, None), 0.707) * (39.2 - ta)
            + 1.86e-4 * ta * sr)


def cci(*_args, **_kwargs) -> np.ndarray:
    """Comprehensive climate index (Mader et al., 2010). Not yet implemented.

    The three correction functions (RH, wind, radiation) are piecewise with
    many coefficients; transcribe them verbatim from Mader et al. (2010) before
    enabling, and unit-test against their published values.
    """
    raise NotImplementedError(
        "CCI awaits transcription of its component equations from Mader et al. (2010)."
    )


def itsc(*_args, **_kwargs) -> np.ndarray:
    """Index of thermal stress for cows (da Silva et al., 2015). Not yet implemented."""
    raise NotImplementedError(
        "ITSC awaits transcription of its heat-balance terms from da Silva et al. (2015)."
    )
