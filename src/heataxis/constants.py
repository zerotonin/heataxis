# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — constants                                            ║
# ║  « one source of truth for colours, units, and figure rules »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Wong (2011) colourblind-safe palette, input-unit conventions,   ║
# ║  and default figure settings.  Import from here — never hardcode ║
# ║  a colour, unit, or magic number in a function body.             ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Shared palette, unit conventions, and figure defaults."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

# ┌────────────────────────────────────────────────────────────┐
# │ Wong (2011) palette  « colourblind-safe base colours »     │
# └────────────────────────────────────────────────────────────┘
WONG: dict[str, str] = {
    "black":          "#000000",
    "orange":         "#E69F00",
    "sky_blue":       "#56B4E9",
    "bluish_green":   "#009E73",
    "yellow":         "#F0E442",
    "blue":           "#0072B2",
    "vermilion":      "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey":           "#999999",
}

# Figure defaults.
FIGURE_DPI: int = 200
RCPARAMS: dict[str, object] = {
    "svg.fonttype": "none",     # editable <text> in SVG (Inkscape-friendly)
    "figure.dpi": 120,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 11,
    "axes.titlesize": 12,
}

# ┌────────────────────────────────────────────────────────────┐
# │ Input units  « the single most common source of index bugs » │
# └────────────────────────────────────────────────────────────┘
# Ta   air temperature            °C
# RH   relative humidity          %        (45, not 0.45)
# Tw   wet-bulb temperature       °C
# Tdp  dew-point temperature      °C
# Tbg  black-globe temperature    °C
# u    wind speed                 m s⁻¹   (NB: several papers tabulate km h⁻¹)
# SR   global solar radiation     W m⁻²
UNITS: dict[str, str] = {
    "Ta": "degC", "RH": "percent", "Tw": "degC", "Tdp": "degC",
    "Tbg": "degC", "u": "m/s", "SR": "W/m2",
}

ArrayLike: TypeAlias = "float | np.ndarray"
