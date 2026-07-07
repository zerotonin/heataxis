# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — plot_fig1_index_construction (CLI)                   ║
# ║  « Part I, Fig 1: the thermal index is not one thing »           ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Thin command-line wrapper over                                  ║
# ║  heataxis.viz.plot_index_construction(); the figure engine lives ║
# ║  in heataxis._index_construction. Runs standalone from anywhere. ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Build Part I Figure 1, "Indices differ by construction".

A fully simulated construction figure (no measured data): sensitivity matrix
plus wind and solar fans. Outputs SVG + PNG (≥300 dpi) + one CSV per panel.

Run (standalone, from anywhere)::

    python plot_fig1_index_construction.py
    python plot_fig1_index_construction.py --out /path/to/dir
    python plot_fig1_index_construction.py --norm threshold   # alt normalisation
    python plot_fig1_index_construction.py --itsc             # overlay ITSC
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from heataxis.viz import plot_index_construction  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_SRC.parent / "results",
                        help="output directory (default: <repo>/results)")
    parser.add_argument("--norm", choices=("zscore", "threshold"),
                        default="zscore",
                        help="fan normalisation (default: zscore)")
    parser.add_argument("--itsc", action="store_true",
                        help="overlay the optional ITSC index in the fans")
    args = parser.parse_args(argv)

    out = plot_index_construction(args.out, norm=args.norm,
                                  include_itsc=args.itsc)
    print(f"wrote fig1_index_construction (SVG/PNG) + 3 panel CSVs to {out}")


if __name__ == "__main__":
    main()
