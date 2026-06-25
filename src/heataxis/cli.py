# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — cli                                                  ║
# ║  « a thin command-line entry point »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Print the version, or compute a single THI from Ta and RH.      ║
# ║  Kept minimal; the library API is the primary interface.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Minimal command-line interface for heataxis."""

from __future__ import annotations

import argparse

from heataxis import __version__
from heataxis.indices import thi_nrc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="heataxis",
                                     description="heataxis command-line tools")
    parser.add_argument("--version", action="version",
                        version=f"heataxis {__version__}")
    sub = parser.add_subparsers(dest="command")
    p_thi = sub.add_parser("thi", help="compute the NRC temperature-humidity index")
    p_thi.add_argument("--ta", type=float, required=True, help="air temperature (degC)")
    p_thi.add_argument("--rh", type=float, required=True, help="relative humidity (%)")
    args = parser.parse_args(argv)
    if args.command == "thi":
        print(f"{float(thi_nrc(args.ta, args.rh)):.2f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
