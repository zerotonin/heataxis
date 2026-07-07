# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — indices                                              ║
# ║  « the x-axis — cattle thermal / heat-load indices (Part I) »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Vectorised, unit-tested cattle thermal / heat-load indices,     ║
# ║  grouped by heat-exchange pathway (see Part I).  Inputs follow   ║
# ║  constants.UNITS (Ta °C, RH %, u m/s, SR W/m²).                  ║
# ║                                                                  ║
# ║  All equations verified against their primary papers; CCI and    ║
# ║  ITSC carry their full piecewise / heat-balance forms.           ║
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
    "dew_point", "wet_bulb", "partial_vapour_pressure", "tbg_estimate",
    "thi_nrc", "thi_nrc1971", "thi_wetbulb", "thi_dewpoint", "thi_variants",
    "bghi", "hli", "hli_gaughan", "eti", "eti_baeta",
    "thi_adj", "thi_adj_mader", "etic",
    "cci", "itsc", "esi",
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


def partial_vapour_pressure(ta: ArrayLike, rh: ArrayLike) -> np.ndarray:
    """Air partial vapour pressure (kPa) from Ta and RH via the Tetens formula.

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).

    Returns:
        Partial vapour pressure (kPa).
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    es = 0.6108 * np.exp(17.27 * ta / (ta + 237.3))
    return (rh / 100.0) * es


def tbg_estimate(ta: ArrayLike, sr: ArrayLike, u: ArrayLike) -> np.ndarray:
    """Black-globe temperature from air temperature, solar radiation, and wind.

    Linearised steady-state energy balance of a standard 150 mm black globe:
    absorbed solar gain is balanced by convective and long-wave-radiative loss
    to surroundings assumed to sit at air temperature, so

        Tbg = Ta + alpha·(SR/4) / (h_c + 4·eps·sigma·Ta_K³)

    with globe absorptivity/emissivity ``alpha = eps = 0.95``, Stefan-Boltzmann
    ``sigma``, and the convective coefficient ``h_c = 6.3·u^0.6`` W m⁻² K⁻¹ for a
    150 mm globe.  The ``SR/4`` factor is the diffuse-averaged solar flux over the
    sphere surface (projected/total area = 1/4).

    Assumptions (state in any figure caption using it): standard 150 mm globe;
    surroundings at ``Ta`` (no cold-sky term — appropriate indoors / enclosed);
    long-wave loss linearised about ``Ta``; diffuse solar loading. **Indoors with
    no direct beam (``sr = 0``) it returns ``Ta`` exactly.** It is an engineering
    approximation for constructing index comparisons, not a calibrated sensor
    model.

    Args:
        ta: Air temperature (°C).
        sr: Global solar radiation (W m⁻²).
        u: Wind speed (m s⁻¹).

    Returns:
        Estimated black-globe temperature (°C).

    References:
        Kuehn, L. A., Stubbs, R. A., & Weaver, R. S. (1970). Theory of the
        globe thermometer. Journal of Applied Physiology, 29(5), 750-757.
        ISO 7726 (1998). Ergonomics of the thermal environment — instruments
        for measuring physical quantities.
        Dimiceli, V. E., Piltz, S. F., & Amburn, S. A. (2011). Estimation of
        black globe temperature for calculation of the WBGT index. Proc. World
        Congress on Engineering and Computer Science, II.
    """
    ta = np.asarray(ta, dtype=float)
    sr = np.asarray(sr, dtype=float)
    u = np.asarray(u, dtype=float)
    sigma = 5.670374419e-8            # Stefan-Boltzmann (W m⁻² K⁻⁴)
    alpha, eps = 0.95, 0.95           # globe solar absorptivity / LW emissivity
    # 150 mm-globe convective coefficient. # TODO verify the 6.3 constant against
    # ISO 7726 / Kuehn (1970) for the exact globe diameter in use.
    h_c = 6.3 * np.power(np.clip(u, 0.0, None), 0.6)
    ta_k = ta + 273.15
    denom = h_c + 4.0 * eps * sigma * ta_k ** 3
    return ta + alpha * (sr / 4.0) / denom


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


def thi_variants(ta: ArrayLike, rh: ArrayLike) -> dict[str, np.ndarray]:
    """A set of published THI coefficient variants, all as functions of Ta, RH.

    Every variant reads only temperature and humidity (wet-bulb / dew-point are
    derived internally), so they can be evaluated on a common ``(Ta, RH)`` grid.
    Their spread for identical weather is the point: "THI" is a family, not one
    number, and the min-max envelope of this set is the grey THI band drawn in
    the Part I construction figure.

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).

    Returns:
        Mapping ``variant name -> THI value`` (all dimensionless).

    References:
        Bohmanova, J., Misztal, I., & Cole, J. B. (2007). Temperature-humidity
        indices as indicators of milk production losses due to heat stress.
        Journal of Dairy Science, 90(4), 1947-1956. Catalogues the variants.
        Primaries: NRC (1971); Thom (1959); Yousef (1985); the NRC-1971 form as
        used by Dikmen & Hansen (2009) J. Dairy Sci. 92:109-116.
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    tw = wet_bulb(ta, rh)
    tdp = dew_point(ta, rh)
    return {
        "NRC1971_RH": thi_nrc1971(ta, rh),
        "Thom1959_wetbulb": 0.72 * (ta + tw) + 40.6,
        "Yousef1985_dewpoint": ta + 0.36 * tdp + 41.5,
        "Dikmen2009_NRC": 0.8 * ta + (rh / 100.0) * (ta - 14.4) + 46.4,
    }


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
        Full nine-term polynomial in Ta, RH and u, including the u², Ta·RH and
        Ta·u interactions; wind is net cooling at realistic speeds. The index
        was proposed by Baeta et al. (1987); the coefficients implemented here
        are those printed by Yan et al. (2021), verified against that paper.

    References:
        Baeta, F. C., Meador, N. F., Shanklin, M. D., & Johnson, H. D. (1987).
        Equivalent temperature index at temperatures above the thermoneutral
        for lactating dairy cows (ASAE Paper No. 87-4015). American Society of
        Agricultural Engineers, St. Joseph, MI.

        Yan, G., Liu, H., Shi, Z., & Li, H. (2021). Evaluation of thermal
        indices as the indicators of heat stress in dairy cows in a temperate
        climate. Animals, 11(8), 2459. https://doi.org/10.3390/ani11082459
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    u = np.asarray(u, dtype=float)
    return (27.88 - 0.456 * ta + 0.010754 * ta ** 2
            - 0.4905 * rh + 0.00088 * rh ** 2 + 1.15 * u
            - 0.12644 * u ** 2 + 0.019876 * ta * rh - 0.046313 * ta * u)


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
        Verified against Eq. 15 of the source. Indoors (no solar) set ``sr=0``.

    References:
        Wang, X., Gao, H., Gebremedhin, K. G., Bjerg, B. S., Van Os, J.,
        Tucker, C. B., & Zhang, G. (2018). A predictive model of equivalent
        temperature index for dairy cattle (ETIC). Journal of Thermal Biology,
        76, 165-170. https://doi.org/10.1016/j.jtherbio.2018.07.013
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    u = np.asarray(u, dtype=float)
    sr = np.asarray(sr, dtype=float)
    return (ta - 0.0038 * ta * (100 - rh)
            - 0.1173 * np.power(np.clip(u, 0.0, None), 0.707) * (39.2 - ta)
            + 1.86e-4 * ta * sr)


def cci(ta: ArrayLike, rh: ArrayLike, u: ArrayLike, sr: ArrayLike) -> np.ndarray:
    """Comprehensive climate index (CCI).

    Class 4a (full microclimate): an apparent temperature ``CCI = Ta + RH-corr
    + WS-corr + RAD-corr`` valid from about -30 to +45 °C, combining the three
    correction terms (Eqs. 1-3 of the source).

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).
        u: Wind speed (m s⁻¹).
        sr: Global solar radiation (W m⁻²).

    Returns:
        CCI apparent temperature (°C).

    Note:
        Reproduces the worked example in the source: Ta 30 °C, RH 50 %,
        u 1 m/s, SR 500 W/m² -> 37.9 °C.

    References:
        Mader, T. L., Johnson, L. J., & Gaughan, J. B. (2010). A comprehensive
        index for assessing environmental stress in animals. Journal of Animal
        Science, 88(6), 2153-2165. https://doi.org/10.2527/jas.2009-2586
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    u = np.asarray(u, dtype=float)
    sr = np.asarray(sr, dtype=float)
    rh_corr = (np.exp(0.00182 * rh + 1.8e-5 * ta * rh)
               * (0.000054 * ta ** 2 + 0.00192 * ta - 0.0246) * (rh - 30))
    ws_inner = (2.9 + 1.14e-6 * u ** 2.5
                - np.log(np.power(2.26 * u + 0.33, -2)) / np.log(0.3))
    ws_corr = (-6.56 / np.exp(np.power(1.0 / (2.26 * u + 0.23), 0.45) * ws_inner)
               - 0.00566 * u ** 2 + 3.33)
    rad_corr = (0.0076 * sr - 0.00002 * sr * ta
                + 0.00005 * ta ** 2 * np.sqrt(sr) + 0.1 * ta - 2)
    return ta + rh_corr + ws_corr + rad_corr


def itsc(ta: ArrayLike, u: ArrayLike, *,
         rh: ArrayLike | None = None, pv: ArrayLike | None = None,
         sr: ArrayLike | None = None, t_rm: ArrayLike | None = None,
         erhl: ArrayLike | None = None) -> np.ndarray:
    """Index of thermal stress for cows (ITSC).

    Class 4b (full microclimate, biophysically derived): a regression on air
    temperature, wind, air partial vapour pressure, and the effective radiant
    heat load (Eq. 11 of the source).

    Partial vapour pressure ``pv`` is derived from ``rh`` when not supplied (see
    :func:`partial_vapour_pressure`).  The effective radiant heat load ``erhl``
    (W m⁻²) is derived from solar radiation and mean radiant temperature as
    ``erhl = 0.5*sr + sigma*(t_rm+273.15)**4`` when not supplied; ``t_rm`` (°C)
    may be approximated by the black-globe temperature.

    Args:
        ta: Air temperature (°C).
        u: Wind speed (m s⁻¹).
        rh: Relative humidity (%, 0-100); used only when ``pv`` is None.
        pv: Air partial vapour pressure (kPa).
        sr: Global solar radiation (W m⁻²); used only when ``erhl`` is None.
        t_rm: Mean radiant temperature (°C); used only when ``erhl`` is None.
        erhl: Effective radiant heat load (W m⁻²).

    Returns:
        ITSC (dimensionless index).

    Raises:
        ValueError: If ``pv``/``rh`` or ``erhl``/(``sr``,``t_rm``) are missing.

    References:
        da Silva, R. G., Maia, A. S. C., & de Macêdo Costa, L. L. (2015).
        Index of thermal stress for cows (ITSC) under high solar radiation in
        tropical environments. International Journal of Biometeorology, 59(5),
        551-559. https://doi.org/10.1007/s00484-014-0868-7
    """
    ta = np.asarray(ta, dtype=float)
    u = np.asarray(u, dtype=float)
    if pv is None:
        if rh is None:
            raise ValueError("provide either pv or rh")
        pv = partial_vapour_pressure(ta, rh)
    pv = np.asarray(pv, dtype=float)
    if erhl is None:
        if sr is None or t_rm is None:
            raise ValueError("provide either erhl or both sr and t_rm")
        erhl = (0.5 * np.asarray(sr, dtype=float)
                + 5.67e-8 * (np.asarray(t_rm, dtype=float) + 273.15) ** 4)
    erhl = np.asarray(erhl, dtype=float)
    return (77.1747 + 4.8327 * ta - 34.8189 * u + 1.111 * u ** 2
            + 118.6981 * pv - 14.7956 * pv ** 2 - 0.1059 * erhl)


# ─────────────────────────────────────────────────────────────
#  Adjacent « human-origin index applied to cattle (WBGT family) »
# ─────────────────────────────────────────────────────────────

def esi(ta: ArrayLike, rh: ArrayLike, sr: ArrayLike) -> np.ndarray:
    """Environmental stress index (ESI).

    A human-origin index proposed as a substitute for the wet-bulb globe
    temperature, using air temperature, humidity, and solar radiation (no wind
    or globe sensor).  Applied to cattle by da Silva et al. (2023), with risk
    bands ESI < 25 (low) and 25-33 (moderate-to-high).

    Args:
        ta: Air temperature (°C).
        rh: Relative humidity (%, 0-100).
        sr: Global solar radiation (W m⁻²).

    Returns:
        ESI (°C-equivalent stress index).

    Note:
        The final ``0.073*(0.1+SR)⁻¹`` term is negligible at full sun but holds
        the index finite as ``sr`` approaches zero. Indoors (no solar) set
        ``sr=0``.

    References:
        Moran, D. S., Pandolf, K. B., Shapiro, Y., Heled, Y., Shani, Y.,
        Mathew, W. T., & Gonzalez, R. R. (2001). An environmental stress index
        (ESI) as a substitute for the wet bulb globe temperature (WBGT).
        Journal of Thermal Biology, 26(4-5), 427-431.
        https://doi.org/10.1016/S0306-4565(01)00055-9

        da Silva, W. C., et al. (2023). Characterization of thermal patterns
        using infrared thermography and thermolytic responses of cattle reared
        in three different systems during the transition period in the eastern
        Amazon. Animals, 13(17), 2735. https://doi.org/10.3390/ani13172735
    """
    ta = np.asarray(ta, dtype=float)
    rh = np.asarray(rh, dtype=float)
    sr = np.asarray(sr, dtype=float)
    return (0.63 * ta - 0.03 * rh + 0.002 * sr
            + 0.0054 * ta * rh - 0.073 / (0.1 + sr))


# ─────────────────────────────────────────────────────────────
#  Primary-source name aliases « same function, cited-source name »
# ─────────────────────────────────────────────────────────────
# Descriptive names used by the Part I construction figure; each is the exact
# same callable as its short name (single source of truth, no duplicated maths).
thi_nrc1971 = thi_nrc          # NRC (1971)
thi_adj_mader = thi_adj        # Mader, Davis & Brown-Brandl (2006)
eti_baeta = eti                # Baeta et al. (1987) / Yan et al. (2021)
hli_gaughan = hli              # Gaughan et al. (2008)
