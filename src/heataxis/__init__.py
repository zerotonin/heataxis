# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — __init__                                             ║
# ║  « thermal heat-load indices + physiological threshold detection » ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Companion-paper toolkit for dairy-cattle heat-stress analysis:  ║
# ║    heataxis.indices    — the x-axis (thermal indices, Part I)    ║
# ║    heataxis.thresholds — the y-axis (threshold methods, Part II) ║
# ║    heataxis.viz        — shared, publication-ready plotting      ║
# ║  One install, one tool: pick an index, then detect the threshold. ║
# ╚══════════════════════════════════════════════════════════════════╝
"""heataxis: thermal indices and physiological threshold detection for cattle."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("heataxis")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0+dev"

from heataxis import constants, indices, thresholds  # noqa: E402,F401

__all__ = ["constants", "indices", "thresholds", "__version__"]
