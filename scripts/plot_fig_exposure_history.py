# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — plot_fig_exposure_history (CLI)                      ║
# ║  « the x-axis needs a memory: same weather, different history »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Thin command-line wrapper over                                  ║
# ║  heataxis.viz.plot_exposure_history(); engine lives in           ║
# ║  heataxis._history_figure. Runs standalone from anywhere.        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Build the exposure-history figure (temporal counterpart to Fig 1).

A fully simulated construction figure (no measured data): one synthetic
heat-wave drives real THI/HLI, then the history transforms diverge. Outputs
SVG + PNG (≥300 dpi) + a CSV.

Run (standalone, from anywhere)::

    python plot_fig_exposure_history.py
    python plot_fig_exposure_history.py --out /path/to/dir
    python plot_fig_exposure_history.py --days 14
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from heataxis.viz import plot_exposure_history  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_SRC.parent / "results",
                        help="output directory (default: <repo>/results)")
    parser.add_argument("--days", type=int, default=12,
                        help="length of the synthetic heat-wave (default: 12)")
    args = parser.parse_args(argv)

    out = plot_exposure_history(args.out, days=args.days)
    print(f"wrote fig_exposure_history (SVG/PNG) + CSV to {out}")


if __name__ == "__main__":
    main()
