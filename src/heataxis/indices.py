# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — indices                                              ║
# ║  « the x-axis — cattle thermal / heat-load indices (Part I) »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Vectorised, unit-tested cattle thermal / heat-load indices,     ║
# ║  grouped by heat-exchange pathway (see Part I).  Inputs follow   ║
# ║  constants.UNITS (Ta °C, RH %, u m/s, SR W/m²).                  ║
# ║                                                                  ║
# ║  CCI and ITSC are stubbed until their piecewise coefficient      ║
# ║  sets are transcribed from the primary papers.                   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Cattle thermal / heat-load indices, grouped by heat-exchange pathway.

Each function documents the paper in which the index was first introduced
(see its ``References`` section).  Inputs use the unit conventions in
:data:`heataxis.constants.UNITS`: air temperature in degrees Celsius,
relative humidity as a percentage (0-100), wind speed in metres per second,
and global solar radiation in watts per square metre.
"""

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
    """Dew-point temperature via the Magnus-Tetens approximation.

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).

    Returns:
        Dew-point temperature (°C), broadcast to the shape of the inputs.

    References:
        Alduchov, O. A., & Eskridge, R. E. (1996). Improved Magnus form
        approximation of saturation vapor pressure. Journal of Applied
        Meteorology, 35(4), 601-609.
        https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    a, b = 17.27, 237.7
    gamma = (a * ta) / (b + ta) + np.log(rh / 100.0)
    return (b * gamma) / (a - gamma)


def wet_bulb(ta: ArrayLike, rh: ArrayLike) -> np.ndarray:
    """Wet-bulb temperature via the Stull (2011) empirical approximation.

    Valid for roughly 5-99 % RH near sea level, which covers barn climate.

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).

    Returns:
        Wet-bulb temperature (°C).

    References:
        Stull, R. (2011). Wet-bulb temperature from relative humidity and air
        temperature. Journal of Applied Meteorology and Climatology, 50(11),
        2267-2269. https://doi.org/10.1175/JAMC-D-11-0143.1
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
    """Temperature-humidity index, relative-humidity form.

    Class 1 (temperature-humidity): represents the dry-sensible and
    evaporative-limitation pathways while ignoring wind and radiation.

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).

    Returns:
        THI (dimensionless).

    References:
        National Research Council. (1971). A guide to environmental research
        on animals. National Academy of Sciences, Washington, DC.
        Original index: Thom, E. C. (1959). The discomfort index. Weatherwise,
        12(2), 57-61. https://doi.org/10.1080/00431672.1959.9926960
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    return (1.8 * ta + 32) - (0.55 - 0.0055 * rh) * (1.8 * ta - 26)


def thi_wetbulb(ta: ArrayLike, tw: ArrayLike | None = None, *,
                rh: ArrayLike | None = None) -> np.ndarray:
    """Temperature-humidity index, wet-bulb form.

    Derives wet-bulb temperature from ``rh`` (see :func:`wet_bulb`) when
    ``tw`` is not supplied.

    Args:
        ta: Air temperature (°C).
        tw: Wet-bulb temperature (°C). If None, derived from ``rh``.
        rh: Relative humidity (%, 0-100); required only when ``tw`` is None.

    Returns:
        THI (dimensionless).

    Raises:
        ValueError: If neither ``tw`` nor ``rh`` is provided.

    References:
        Thom, E. C. (1959). The discomfort index. Weatherwise, 12(2), 57-61.
        https://doi.org/10.1080/00431672.1959.9926960
        National Research Council. (1971). A guide to environmental research
        on animals. National Academy of Sciences, Washington, DC.
    """
    ta = np.asarray(ta, dtype=float)
    if tw is None:
        if rh is None:
            raise ValueError("provide either tw or rh")
        tw = wet_bulb(ta, rh)
    tw = np.asarray(tw, dtype=float)
    return 0.72 * (ta + tw) + 40.6


def thi_dewpoint(ta: ArrayLike, tdp: ArrayLike | None = None, *,
                 rh: ArrayLike | None = None) -> np.ndarray:
    """Temperature-humidity index, dew-point form.

    Derives dew-point temperature from ``rh`` (see :func:`dew_point`) when
    ``tdp`` is not supplied.

    Args:
        ta: Air temperature (°C).
        tdp: Dew-point temperature (°C). If None, derived from ``rh``.
        rh: Relative humidity (%, 0-100); required only when ``tdp`` is None.

    Returns:
        THI (dimensionless).

    Raises:
        ValueError: If neither ``tdp`` nor ``rh`` is provided.

    References:
        Yousef, M. K. (1985). Stress physiology in livestock (Vol. 1). CRC
        Press, Boca Raton, FL.
        Original index: Thom, E. C. (1959). The discomfort index. Weatherwise,
        12(2), 57-61. https://doi.org/10.1080/00431672.1959.9926960
    """
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
    """Black-globe-humidity index (BGHI).

    Class 2 (radiation-augmented): the black globe integrates solar, thermal,
    and air contributions, adding the radiant pathway a THI misses. Derives
    dew point from ``ta`` and ``rh`` when ``tdp`` is not supplied.

    Args:
        tbg: Black-globe temperature (°C).
        tdp: Dew-point temperature (°C). If None, derived from ``ta``/``rh``.
        ta: Air temperature (°C); used only when ``tdp`` is None.
        rh: Relative humidity (%, 0-100); used only when ``tdp`` is None.

    Returns:
        BGHI (dimensionless).

    Raises:
        ValueError: If ``tdp`` is None and ``ta``/``rh`` are not both given.

    References:
        Buffington, D. E., Collazo-Arocho, A., Canton, G. H., Pitt, D.,
        Thatcher, W. W., & Collier, R. J. (1981). Black globe-humidity index
        (BGHI) as comfort equation for dairy cows. Transactions of the ASAE,
        24(3), 711-714. https://doi.org/10.13031/2013.34325
    """
    tbg = np.asarray(tbg, dtype=float)
    if tdp is None:
        if ta is None or rh is None:
            raise ValueError("provide either tdp or both ta and rh")
        tdp = dew_point(ta, rh)
    tdp = np.asarray(tdp, dtype=float)
    return tbg + 0.36 * tdp + 41.5


def hli(tbg: ArrayLike, rh: ArrayLike, u: ArrayLike) -> np.ndarray:
    """Heat load index (HLI) for feedlot cattle.

    Class 2 (radiation + wind): a piecewise function of black-globe
    temperature with separate hot (Tbg >= 25 °C) and cool (Tbg < 25 °C)
    branches.

    Args:
        tbg: Black-globe temperature (°C).
        rh: Relative humidity (%, 0-100).
        u: Wind speed (m s⁻¹).

    Returns:
        HLI (dimensionless).

    References:
        Gaughan, J. B., Mader, T. L., Holt, S. M., & Lisle, A. (2008). A new
        heat load index for feedlot cattle. Journal of Animal Science, 86(1),
        226-234. https://doi.org/10.2527/jas.2007-0305
    """
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
    """Equivalent temperature index (ETI).

    Class 3 (wind-augmented): adds the convective pathway through wind speed.

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).
        u: Wind speed (m s⁻¹).

    Returns:
        ETI as an equivalent temperature (°C).

    Note:
        Verify the full term set (any Ta·RH or u² terms) against the primary;
        this implementation follows widely used secondary reproductions.

    References:
        Baeta, F. C., Meador, N. F., Shanklin, M. D., & Johnson, H. D. (1987).
        Equivalent temperature index at temperatures above the thermoneutral
        for lactating dairy cows (ASAE Paper No. 87-4015). American Society of
        Agricultural Engineers, St. Joseph, MI.
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
    """Wind- and solar-adjusted temperature-humidity index (THI_adj).

    Class 4a (full microclimate, empirical adjustment): corrects a base THI
    toward an apparent temperature using wind speed and solar radiation.

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).
        u: Wind speed (m s⁻¹).
        sr: Global solar radiation (W m⁻²).

    Returns:
        Adjusted THI (dimensionless).

    References:
        Mader, T. L., Davis, M. S., & Brown-Brandl, T. (2006). Environmental
        factors influencing heat stress in feedlot cattle. Journal of Animal
        Science, 84(3), 712-719. https://doi.org/10.2527/2006.843712x
    """
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
    """Equivalent temperature index for cattle (ETIC).

    Class 4b (full microclimate, biophysically derived): a heat-balance index
    combining temperature, humidity, wind, and solar radiation, expressed as
    an equivalent temperature.

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).
        u: Wind speed (m s⁻¹).
        sr: Global solar radiation (W m⁻²).

    Returns:
        ETIC as an equivalent temperature (°C).

    Note:
        Verify the wind exponent (0.707) and coefficients, and confirm the
        exact volume/pages/DOI, against the primary before manuscript use.

    References:
        Wang, X., Gao, H., Gebremedhin, K. G., Bjerg, B. S., Van Os, J.,
        Choi, C. Y., & Zhang, G. (2018). A predictive model of equivalent
        temperature index for dairy cattle (ETIC). Biosystems Engineering,
        173, 11-21 (verify volume, pages, and DOI before citing).
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    u = np.asarray(u, dtype=float)
    sr = np.asarray(sr, dtype=float)
    return (ta - 0.0038 * ta * (100 - rh)
            - 0.1173 * np.power(np.clip(u, 0.0, None), 0.707) * (39.2 - ta)
            + 1.86e-4 * ta * sr)


def cci(*_args, **_kwargs) -> np.ndarray:
    """Comprehensive climate index (CCI). Not yet implemented.

    Class 4a (full microclimate). The three correction functions (relative
    humidity, wind, radiation) are piecewise with many coefficients and span
    cold as well as heat (-30 to +45 °C). Transcribe them verbatim from the
    primary and unit-test against its published values before enabling.

    Raises:
        NotImplementedError: Always, until the component equations are added.

    References:
        Mader, T. L., Johnson, L. J., & Gaughan, J. B. (2010). A comprehensive
        index for assessing environmental stress in animals. Journal of Animal
        Science, 88(6), 2153-2165. https://doi.org/10.2527/jas.2009-2586
    """
    raise NotImplementedError(
        "CCI awaits transcription of its component equations from Mader et al. (2010)."
    )


def itsc(*_args, **_kwargs) -> np.ndarray:
    """Index of thermal stress for cows (ITSC). Not yet implemented.

    Class 4b (full microclimate, biophysically derived): a radiation-driven
    heat-balance index built for high-solar tropical environments.

    Raises:
        NotImplementedError: Always, until the heat-balance terms are added.

    References:
        da Silva, R. G., Maia, A. S. C., & de Macêdo Costa, L. L. (2015).
        Index of thermal stress for cows (ITSC) under high solar radiation in
        tropical environments. International Journal of Biometeorology, 59(5),
        551-559. https://doi.org/10.1007/s00484-014-0868-7
    """
    raise NotImplementedError(
        "ITSC awaits transcription of its heat-balance terms from da Silva et al. (2015)."
    )
