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


def save_figure(fig, stem: str, out_dir: Path, *, dpi: int | None = None) -> None:
    """Export a figure as SVG + PNG into ``out_dir`` (created on demand).

    Args:
        fig:     Matplotlib figure to save.
        stem:    Filename stem (no extension).
        out_dir: Target directory (created if needed).
        dpi:     PNG resolution; ``None`` uses the house ``savefig.dpi`` rcParam.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight",
                    dpi=dpi if ext == "png" else None)


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


def plot_index_construction(out_dir, *, norm: str = "zscore",
                            include_itsc: bool = False):
    """Build Part I Figure 1, "Indices differ by construction".

    A fully simulated, data-free systems-analysis figure: a sensitivity matrix
    plus wind and solar fans showing that classic cattle heat indices diverge
    precisely along the inputs a temperature-humidity index cannot see. Writes
    SVG + PNG (≥300 dpi) + one CSV per panel into ``out_dir``.

    Args:
        out_dir:      Output directory (created on demand).
        norm:         Fan normalisation, ``"zscore"`` (default) or
                      ``"threshold"``.
        include_itsc: Overlay the optional ITSC index in the fans.

    Returns:
        The output directory (``pathlib.Path``).
    """
    from heataxis import _index_construction as impl
    return impl.build(out_dir, norm=norm, include_itsc=include_itsc)


def plot_exposure_history(out_dir, *, days: int = 12):
    """Build the exposure-history figure (temporal counterpart to Fig 1).

    A fully simulated figure: one synthetic heat-wave drives real THI/HLI, and
    the history transforms then diverge — memoryless tracks the input, cum-sum
    never forgets, the leaky integrator accumulates and recovers, and TSD/TSL
    count duration and load. Writes SVG + PNG (≥300 dpi) + a CSV into ``out_dir``.

    Args:
        out_dir: Output directory (created on demand).
        days:    Length of the synthetic heat-wave (days).

    Returns:
        The output directory (``pathlib.Path``).
    """
    from heataxis import _history_figure as impl
    return impl.build(out_dir, days=days)
