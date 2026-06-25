# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — viz                                                  ║
# ║  « shared, publication-ready plotting helpers »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Figure setup, SVG+PNG export, and a serious schematic register  ║
# ║  (arrowed tickless axes) for pure-maths / idealised-response panels. ║
# ║  matplotlib is imported lazily — install the 'viz' extra to use this. ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Shared plotting setup, figure I/O, and schematic-axis helpers."""

from __future__ import annotations

from pathlib import Path

from heataxis.constants import RCPARAMS, WONG


def setup_figure() -> None:
    """Apply the house matplotlib rcParams (Wong palette, editable SVG text)."""
    import matplotlib as mpl
    mpl.rcParams.update(RCPARAMS)


def save_figure(fig, stem: str, out_dir: Path) -> None:
    """Export a figure as SVG + PNG into ``out_dir`` (created on demand)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")


def schematic_axes(ax, xlabel: str, ylabel: str) -> None:
    """Turn an Axes into an arrowed, tickless schematic frame ('not data')."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    arrow = dict(arrowstyle="-|>", color=WONG["black"], lw=1.3)
    ax.annotate("", xy=(1.0, 0.0), xytext=(0.0, 0.0),
                xycoords="axes fraction", arrowprops=arrow)
    ax.annotate("", xy=(0.0, 1.0), xytext=(0.0, 0.0),
                xycoords="axes fraction", arrowprops=arrow)
    ax.text(1.0, -0.05, xlabel, transform=ax.transAxes, ha="right", va="top")
    ax.text(-0.04, 1.0, ylabel, transform=ax.transAxes, ha="right", va="top")
    ax.margins(x=0.02, y=0.08)
