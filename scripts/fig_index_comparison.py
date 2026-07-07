# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — fig_index_comparison                                 ║
# ║  « theoretical, data-free comparison of thermal indices »        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Two figures, no measured data, straight from the index formulae: ║
# ║    index_coverage   — the climate box and which modalities each  ║
# ║                       index reads, grouped by heat-exchange pathway ║
# ║    index_comparison — magnitude sensitivity heatmap (blue-grey-red) ║
# ║                       plus one sweep per driver (vary one, hold rest) ║
# ║                                                                  ║
# ║  Outputs SVG + PNG + CSV into the gitignored results/ folder.    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Theoretical comparison of cattle thermal indices (Part I, x-axis).

These figures use no measured data: every value comes from the closed-form
index definitions in :mod:`heataxis.indices`.  They explain the theoretical
differences between the indices before any real data are introduced.

``index_coverage`` shows the multimodal climate box and which modalities each
index reads, grouped by heat-exchange pathway.  ``index_comparison`` then
quantifies the response: a local-sensitivity heatmap (blue = max negative,
grey = 0, red = max positive, normalised within each input) and one sweep per
independent driver (air temperature, humidity, wind, solar), each varying that
driver across its full range while the others are held at a standard value.
The black globe tracks air temperature and solar, since it is derived, not
independent.

Run:
    python scripts/fig_index_comparison.py
    python scripts/fig_index_comparison.py --out /path/to/dir
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np

# Allow running from a source checkout without installing the package.
_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from heataxis import indices as idx  # noqa: E402
from heataxis.constants import WONG  # noqa: E402
from heataxis.viz import save_figure, setup_figure  # noqa: E402

State = namedtuple("State", "ta rh u sr tbg")

# Black-globe model: globe temperature rises with the solar load it absorbs.
RADIATIVE_GAIN = 6.0   # degC added to the globe at full sun (1000 W/m2)

# ── The climate box: the realistic range of barn weather. It defines the
# multimodal input space and the envelope over which indices are z-scored.
ENVELOPE = {"Ta": (15.0, 35.0), "RH": (30.0, 90.0),
            "u": (0.0, 6.0), "SR": (0.0, 1000.0)}
RANGE_LABEL = {"Ta": "15–35 °C", "RH": "30–90 %", "u": "0–6 m s$^{-1}$",
               "SR": "0–1000 W m$^{-2}$", "Tbg": "derived (Ta, SR)"}

# Heat-exchange pathway each environmental modality feeds.
PATHWAY = {"Ta": "sensible", "u": "sensible", "RH": "evaporative",
           "SR": "radiant", "Tbg": "radiant"}
PATHWAY_COLOUR = {"sensible": WONG["orange"], "evaporative": WONG["blue"],
                  "radiant": WONG["vermilion"]}
COVER_COLS = ("Ta", "u", "RH", "SR", "Tbg")   # grouped by pathway, left to right

# Matrix columns (all readable inputs) and finite-difference steps.
COLUMNS = ("Ta", "RH", "u", "SR", "Tbg")
_FIELD = {"Ta": "ta", "RH": "rh", "u": "u", "SR": "sr", "Tbg": "tbg"}
_STEP = {"Ta": 0.5, "RH": 1.0, "u": 0.1, "SR": 10.0, "Tbg": 0.5}

# Sweep panels: the four independent drivers, each over its full plausible range.
SWEEP_VARS = ("Ta", "RH", "u", "SR")
SWEEP_RANGE = {"Ta": (0.0, 42.0), "RH": (1.0, 100.0),
               "u": (0.0, 8.0), "SR": (0.0, 1000.0)}
SWEEP_XLABEL = {"Ta": "air temperature (°C)", "RH": "relative humidity (%)",
                "u": "wind speed (m s$^{-1}$)", "SR": "solar radiation (W m$^{-2}$)"}


def tbg_from(ta, sr):
    """Simple black-globe temperature: air temperature plus a radiative gain."""
    return np.asarray(ta, dtype=float) + RADIATIVE_GAIN * (np.asarray(sr, dtype=float) / 1000.0)


# Index registry: name -> (callable on a State, declared inputs).
REGISTRY = {
    "THI (NRC)": (lambda s: idx.thi_nrc(s.ta, s.rh), ("Ta", "RH")),
    "BGHI": (lambda s: idx.bghi(s.tbg, ta=s.ta, rh=s.rh), ("Tbg", "Ta", "RH")),
    "ETI": (lambda s: idx.eti(s.ta, s.rh, s.u), ("Ta", "RH", "u")),
    "HLI": (lambda s: idx.hli(s.tbg, s.rh, s.u), ("Tbg", "RH", "u")),
    "THI_adj": (lambda s: idx.thi_adj(s.ta, s.rh, s.u, s.sr), ("Ta", "RH", "u", "SR")),
    "CCI": (lambda s: idx.cci(s.ta, s.rh, s.u, s.sr), ("Ta", "RH", "u", "SR")),
    "ETIC": (lambda s: idx.etic(s.ta, s.rh, s.u, s.sr), ("Ta", "RH", "u", "SR")),
    "ITSC": (lambda s: idx.itsc(s.ta, s.u, rh=s.rh, sr=s.sr, t_rm=s.tbg),
             ("Ta", "RH", "u", "SR", "Tbg")),
    "ESI": (lambda s: idx.esi(s.ta, s.rh, s.sr), ("Ta", "RH", "SR")),
}

_COLOURS = (WONG["blue"], WONG["vermilion"], WONG["bluish_green"], WONG["orange"],
            WONG["reddish_purple"], WONG["sky_blue"], WONG["yellow"], WONG["black"],
            WONG["grey"])

# Adjacent, human-origin indices are dashed to mark them as conceptual outsiders.
_DASHED = {"ESI"}

# Standard barn condition: the held value for inputs not being swept.
STANDARD = State(ta=24.0, rh=60.0, u=1.0, sr=400.0, tbg=float(tbg_from(24.0, 400.0)))


# ─────────────────────────────────────────────────────────────
#  Climate box + index coverage (the conceptual figure)
# ─────────────────────────────────────────────────────────────

def _pathway_groups():
    """Consecutive COVER_COLS sharing a pathway -> (pathway, start, end) spans."""
    groups = []
    start = 0
    for j in range(1, len(COVER_COLS) + 1):
        if j == len(COVER_COLS) or PATHWAY[COVER_COLS[j]] != PATHWAY[COVER_COLS[start]]:
            groups.append((PATHWAY[COVER_COLS[start]], start, j - 1))
            start = j
    return groups


def plot_coverage(out_dir) -> None:
    """Draw the climate box and which modalities each index can read."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    setup_figure()

    names = list(REGISTRY)
    ncol, nrow = len(COVER_COLS), len(names)
    fig, ax = plt.subplots(figsize=(9.5, 0.62 * nrow + 3.0))
    ax.set_xlim(-0.7, ncol - 0.3)
    ax.set_ylim(nrow - 0.4, -2.4)   # high-then-low => first index at the top

    for pathway, a, b in _pathway_groups():
        ax.add_patch(Rectangle((a - 0.42, -2.15), (b - a) + 0.84, 0.5,
                               color=PATHWAY_COLOUR[pathway], alpha=0.22,
                               lw=0, clip_on=False))
        ax.text((a + b) / 2.0, -1.9, pathway, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=PATHWAY_COLOUR[pathway])

    for j, col in enumerate(COVER_COLS):
        ax.text(j, -1.25, col, ha="center", va="center", fontsize=10.5, fontweight="bold")
        ax.text(j, -0.92, RANGE_LABEL[col], ha="center", va="center",
                fontsize=7.5, color="#555555")

    for i, name in enumerate(names):
        used = REGISTRY[name][1]
        for j, col in enumerate(COVER_COLS):
            if col in used:
                ax.scatter(j, i, s=270, color=PATHWAY_COLOUR[PATHWAY[col]],
                           edgecolor="white", linewidth=1.2, zorder=3)
            else:
                ax.scatter(j, i, s=46, color="#dddddd", zorder=2)

    ax.set_yticks(range(nrow))
    ax.set_yticklabels(names)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("What each index reads from the barn climate box\n"
                 "filled = the index uses this modality   ·   grey = blind to it",
                 fontsize=12, pad=44)
    save_figure(fig, "index_coverage", out_dir)
    plt.close(fig)


def write_coverage_csv(out_dir) -> None:
    """1/0 coverage of each modality, plus per-pathway coverage flags."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pathways = ("sensible", "evaporative", "radiant")
    with open(out_dir / "index_coverage.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", *COVER_COLS, *pathways])
        for name, (_, used) in REGISTRY.items():
            cols = [1 if c in used else 0 for c in COVER_COLS]
            covers = [int(any(PATHWAY[c] == p for c in used)) for p in pathways]
            writer.writerow([name, *cols, *covers])


# ─────────────────────────────────────────────────────────────
#  Sensitivity matrix + per-driver sweeps (the quantitative figure)
# ─────────────────────────────────────────────────────────────

def _perturb(state: State, field: str, delta: float) -> State:
    d = state._asdict()
    d[field] = d[field] + delta
    return State(**d)


def sensitivity_matrix() -> tuple[list[str], np.ndarray]:
    """Finite-difference d(index)/d(input) at STANDARD; NaN where input unused."""
    names = list(REGISTRY)
    mat = np.full((len(names), len(COLUMNS)), np.nan)
    for i, name in enumerate(names):
        fn, used = REGISTRY[name]
        for j, col in enumerate(COLUMNS):
            if col not in used:
                continue
            field = _FIELD[col]
            h = _STEP[col]
            hi = float(fn(_perturb(STANDARD, field, h)))
            lo = float(fn(_perturb(STANDARD, field, -h)))
            mat[i, j] = (hi - lo) / (2.0 * h)
    return names, mat


def envelope_stats() -> dict[str, tuple[float, float]]:
    """Mean and SD of each index over the climate box (for z-scoring)."""
    ta = np.linspace(*ENVELOPE["Ta"], 11)
    rh = np.linspace(*ENVELOPE["RH"], 7)
    u = np.linspace(*ENVELOPE["u"], 7)
    sr = np.linspace(*ENVELOPE["SR"], 6)
    grids = np.meshgrid(ta, rh, u, sr, indexing="ij")
    ta_g, rh_g, u_g, sr_g = (g.ravel() for g in grids)
    state = State(ta_g, rh_g, u_g, sr_g, tbg_from(ta_g, sr_g))
    stats = {}
    for name, (fn, _) in REGISTRY.items():
        values = np.asarray(fn(state), dtype=float)
        stats[name] = (float(np.nanmean(values)), float(np.nanstd(values)))
    return stats


def sweep_variable(stats: dict[str, tuple[float, float]], var: str):
    """z-scored response as one driver varies and the rest stay at STANDARD."""
    n = 80
    swept = np.linspace(*SWEEP_RANGE[var], n)
    base = STANDARD._asdict()
    local = {"ta": np.full(n, base["ta"]), "rh": np.full(n, base["rh"]),
             "u": np.full(n, base["u"]), "sr": np.full(n, base["sr"])}
    local[_FIELD[var]] = swept
    state = State(local["ta"], local["rh"], local["u"], local["sr"],
                  tbg_from(local["ta"], local["sr"]))
    out = {}
    for name, (fn, _) in REGISTRY.items():
        mean, sd = stats[name]
        values = np.asarray(fn(state), dtype=float)
        out[name] = (values - mean) / sd if sd > 0 else np.zeros_like(values)
    return swept, out


def _draw_matrix(fig, ax, names, mat) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "blue_gray_red", [WONG["blue"], "#dddddd", WONG["vermilion"]])
    cmap.set_bad("#f5f5f5")
    norm = np.full_like(mat, np.nan)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        peak = np.nanmax(np.abs(col)) if np.any(np.isfinite(col)) else np.nan
        if peak and np.isfinite(peak):
            norm[:, j] = col / peak
    image = ax.imshow(np.ma.masked_invalid(norm), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(COLUMNS)))
    ax.set_xticklabels(COLUMNS)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title("Local sensitivity d(index)/d(input) at the standard condition\n"
                 "colour normalised within each input  (red +, grey 0, blue −)")
    for i in range(len(names)):
        for j in range(len(COLUMNS)):
            value = mat[i, j]
            if np.isnan(value):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#999999")
                continue
            strong = abs(norm[i, j]) > 0.55
            ax.text(j, i, f"{value:+.3g}", ha="center", va="center", fontsize=8,
                    color="white" if strong else "#333333")
    ax.set_xticks(np.arange(-0.5, len(COLUMNS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names)), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.012, ticks=[-1, 0, 1])
    cbar.ax.set_yticklabels(["−max", "0", "+max"])


def _draw_sweep_panel(ax, stats, var, with_ylabel):
    x, curves = sweep_variable(stats, var)
    lines = []
    for (name, z), colour in zip(curves.items(), _COLOURS, strict=False):
        ls = "--" if name in _DASHED else "-"
        line, = ax.plot(x, z, color=colour, lw=1.8, ls=ls, label=name)
        lines.append(line)
    ax.axhline(0.0, color="#cccccc", lw=0.8, ls="--")
    ax.axvline(STANDARD._asdict()[_FIELD[var]], color="#bbbbbb", lw=0.9, ls=":")
    ax.set_xlabel(SWEEP_XLABEL[var])
    if with_ylabel:
        ax.set_ylabel("heat load (z over climate box)")
    ax.set_title(f"Vary {var}, others at standard")
    return lines


def plot_comparison(names, mat, stats, out_dir) -> None:
    """Render the sensitivity heatmap + per-driver sweeps and save SVG + PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_figure()

    fig = plt.figure(figsize=(10.5, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0], hspace=0.5, wspace=0.26)
    _draw_matrix(fig, fig.add_subplot(gs[0, :]), names, mat)

    cells = {"Ta": gs[1, 0], "RH": gs[1, 1], "u": gs[2, 0], "SR": gs[2, 1]}
    handles = None
    for var in SWEEP_VARS:
        handles = _draw_sweep_panel(fig.add_subplot(cells[var]), stats, var,
                                    with_ylabel=var in ("Ta", "u"))
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle("Thermal indices: theoretical comparison   "
                 f"(standard Ta {STANDARD.ta:.0f} °C, RH {STANDARD.rh:.0f} %, "
                 f"u {STANDARD.u:.0f} m/s, SR {STANDARD.sr:.0f} W/m²)",
                 fontsize=12, y=0.995)
    save_figure(fig, "index_comparison", out_dir)
    plt.close(fig)


def write_comparison_csvs(names, mat, stats, out_dir) -> None:
    """Write CSV companions for the matrix and each per-driver sweep."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "index_comparison_matrix.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", *COLUMNS])
        for name, row in zip(names, mat, strict=False):
            cells = ["" if np.isnan(v) else f"{v:.6g}" for v in row]
            writer.writerow([name, *cells])
    for var in SWEEP_VARS:
        x, curves = sweep_variable(stats, var)
        with open(out_dir / f"index_comparison_{var}.csv", "w", newline="",
                  encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([var, *curves])
            for k in range(len(x)):
                writer.writerow([f"{x[k]:.4g}", *(f"{curves[name][k]:.6g}" for name in curves)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Theoretical thermal-index figures (no measured data).")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent.parent / "results",
                        help="output directory (default: <repo>/results)")
    args = parser.parse_args(argv)

    plot_coverage(args.out)
    write_coverage_csv(args.out)

    names, mat = sensitivity_matrix()
    stats = envelope_stats()
    plot_comparison(names, mat, stats, args.out)
    write_comparison_csvs(names, mat, stats, args.out)
    print(f"wrote index_coverage + index_comparison (SVG/PNG/CSV) to {args.out}")


if __name__ == "__main__":
    main()
