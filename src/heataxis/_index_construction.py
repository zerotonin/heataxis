# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — _index_construction                                  ║
# ║  « Part I, Fig 1 engine: the thermal index is not one thing »    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  100 % simulated systems-analysis figure — no measured data.     ║
# ║  Public entry point is heataxis.viz.plot_index_construction();   ║
# ║  scripts/plot_fig1_index_construction.py is a thin CLI over it.  ║
# ║                                                                  ║
# ║    Panel A — row-normalised sensitivity matrix (∂index/∂input)   ║
# ║    Panel B — wind fan (vary u, SR = 0)                           ║
# ║    Panel C — solar fan (vary SR, u = 0.2)                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Engine for Part I Figure 1, "Indices differ by construction".

Everything comes from the closed-form index equations in
:mod:`heataxis.indices` evaluated over environmental-input grids; no measured
data.  It is an illustrative construction figure, not a claim about measured
barn heat load.  Call it through :func:`heataxis.viz.plot_index_construction`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from heataxis import indices as idx
from heataxis.constants import STRESS_ONSET as THRESHOLD
from heataxis.constants import WONG
from heataxis.viz import save_figure, setup_figure

# ─────────────────────────────────────────────────────────────
#  Baseline, envelope, and sweep configuration
# ─────────────────────────────────────────────────────────────
# Hot baseline for the sensitivity matrix and the fan hold-points.
BASE_TA, BASE_RH = 32.0, 50.0
# Matrix baseline also needs u and SR; a moderate sunlit point keeps every
# finite difference well defined (CCI's sqrt(SR) term is singular at SR = 0).
BASE_U, BASE_SR = 0.2, 500.0

# The climate envelope for z-scoring (a common yardstick across indices).
ENVELOPE = {"Ta": (15.0, 40.0), "RH": (20.0, 90.0),
            "u": (0.0, 6.0), "SR": (0.0, 1000.0)}
N_ENVELOPE = 10_000
ENVELOPE_SEED = 20260707

U_FLOOR = 0.1        # convective terms diverge at u = 0
N_SWEEP = 160

# Panel A matrix: columns and central-difference steps.
COLUMNS = ("Ta", "RH", "Tw", "Tdp", "Tbg", "u", "SR")
_STEP = {"Ta": 0.5, "RH": 1.0, "Tw": 0.5, "Tdp": 0.5,
         "Tbg": 0.5, "u": 0.1, "SR": 10.0}

# Semantic Wong colour per index (lab convention).
COLOUR = {
    "THI": WONG["black"], "THI_adj": WONG["orange"], "ETI": WONG["sky_blue"],
    "BGHI": WONG["bluish_green"], "HLI": WONG["yellow"], "CCI": WONG["blue"],
    "ETIC": WONG["vermilion"], "ITSC": WONG["reddish_purple"],
}
BAND_COLOUR = WONG["grey"]
LEGEND_ORDER = ["THI", "BGHI", "THI_adj", "ETI", "HLI", "CCI", "ETIC"]

# Heat-stress onset thresholds for the alternative fan normalisation; the
# values live in constants.STRESS_ONSET so the scripts and the library cannot
# drift apart on what counts as stress.


# ─────────────────────────────────────────────────────────────
#  Index evaluation — physical mode (fans, envelope)
# ─────────────────────────────────────────────────────────────
# Physical mode: Tw, Tdp, Tbg are DERIVED from (Ta, RH, u, SR), i.e. real
# barn conditions.  Used for the fans and the z-scoring envelope.

def _tbg(ta, sr, u):
    return idx.tbg_estimate(ta, sr, u)


INDEX_PHYS = {
    "THI":     lambda ta, rh, u, sr: idx.thi_nrc1971(ta, rh),
    "BGHI":    lambda ta, rh, u, sr: idx.bghi(_tbg(ta, sr, u),
                                              tdp=idx.dew_point(ta, rh)),
    "THI_adj": lambda ta, rh, u, sr: idx.thi_adj_mader(ta, rh, u, sr),
    "ETI":     lambda ta, rh, u, sr: idx.eti_baeta(ta, rh, u),
    "HLI":     lambda ta, rh, u, sr: idx.hli_gaughan(_tbg(ta, sr, u), rh, u),
    "CCI":     lambda ta, rh, u, sr: idx.cci(ta, rh, u, sr),
    "ETIC":    lambda ta, rh, u, sr: idx.etic(ta, rh, u, sr),
}


def _itsc_phys(ta, rh, u, sr):
    return idx.itsc(ta, u, rh=rh, sr=sr, t_rm=_tbg(ta, sr, u))


def _physical_registry(include_itsc: bool) -> dict:
    reg = dict(INDEX_PHYS)
    if include_itsc:
        reg["ITSC"] = _itsc_phys
    return reg


# ─────────────────────────────────────────────────────────────
#  Index evaluation — structural mode (Panel A matrix)
# ─────────────────────────────────────────────────────────────
# Structural mode: Ta, RH, Tw, Tdp, Tbg, u, SR are treated as INDEPENDENT
# inputs, so ∂index/∂input reveals which inputs an index structurally reads.
# An index that ignores an input yields exactly 0 for that column.
# Only the 7 core indices form the matrix rows (ITSC is a fan-only overlay).

INDEX_STRUCT = {
    "THI":     (lambda e: idx.thi_nrc1971(e["Ta"], e["RH"]), ("Ta", "RH")),
    "BGHI":    (lambda e: idx.bghi(e["Tbg"], e["Tdp"]), ("Tbg", "Tdp")),
    "THI_adj": (lambda e: idx.thi_adj_mader(e["Ta"], e["RH"], e["u"], e["SR"]),
                ("Ta", "RH", "u", "SR")),
    "ETI":     (lambda e: idx.eti_baeta(e["Ta"], e["RH"], e["u"]),
                ("Ta", "RH", "u")),
    "HLI":     (lambda e: idx.hli_gaughan(e["Tbg"], e["RH"], e["u"]),
                ("Tbg", "RH", "u")),
    "CCI":     (lambda e: idx.cci(e["Ta"], e["RH"], e["u"], e["SR"]),
                ("Ta", "RH", "u", "SR")),
    "ETIC":    (lambda e: idx.etic(e["Ta"], e["RH"], e["u"], e["SR"]),
                ("Ta", "RH", "u", "SR")),
}


def _baseline_env() -> dict:
    """The 7-input baseline for the sensitivity matrix."""
    return {
        "Ta": BASE_TA, "RH": BASE_RH,
        "Tw": float(idx.wet_bulb(BASE_TA, BASE_RH)),
        "Tdp": float(idx.dew_point(BASE_TA, BASE_RH)),
        "Tbg": float(idx.tbg_estimate(BASE_TA, BASE_SR, BASE_U)),
        "u": BASE_U, "SR": BASE_SR,
    }


def sensitivity_matrix() -> tuple[list[str], np.ndarray]:
    """Row-normalised local sensitivity ∂index/∂input at the baseline."""
    names = list(INDEX_STRUCT)
    mat = np.zeros((len(names), len(COLUMNS)))
    env0 = _baseline_env()
    for i, name in enumerate(names):
        fn, used = INDEX_STRUCT[name]
        for j, col in enumerate(COLUMNS):
            if col not in used:
                continue
            hi, lo = dict(env0), dict(env0)
            hi[col] = env0[col] + _STEP[col]
            lo[col] = env0[col] - _STEP[col]
            mat[i, j] = (float(fn(hi)) - float(fn(lo))) / (2.0 * _STEP[col])
        peak = np.max(np.abs(mat[i]))
        if peak > 0:
            mat[i] /= peak
    return names, mat


# ─────────────────────────────────────────────────────────────
#  Climate-envelope statistics (for normalisation)
# ─────────────────────────────────────────────────────────────

def _latin_hypercube(n: int, rng) -> dict:
    """Stratified Latin-hypercube sample over the four environmental drivers."""
    out = {}
    for key, (lo, hi) in ENVELOPE.items():
        strata = (rng.permutation(n) + rng.random(n)) / n
        out[key] = lo + strata * (hi - lo)
    return out


def _finite_stats(values: np.ndarray) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)), float(np.std(v))


def envelope_stats(registry: dict, rng) -> tuple[dict, dict]:
    """Mean/SD of each index and each THI variant over the climate envelope."""
    s = _latin_hypercube(N_ENVELOPE, rng)
    ta, rh, u, sr = s["Ta"], s["RH"], s["u"], s["SR"]
    index_stats = {name: _finite_stats(fn(ta, rh, u, sr))
                   for name, fn in registry.items()}
    variant_stats = {vname: _finite_stats(vv)
                     for vname, vv in idx.thi_variants(ta, rh).items()}
    return index_stats, variant_stats


def _normalise(name: str, values: np.ndarray, stats: dict, norm: str) -> np.ndarray:
    mean, sd = stats[name]
    if sd <= 0:
        return np.zeros_like(values)
    centre = THRESHOLD[name] if norm == "threshold" else mean
    return (values - centre) / sd


# ─────────────────────────────────────────────────────────────
#  Fans (Panels B and C)
# ─────────────────────────────────────────────────────────────

def _variant_band(ta: np.ndarray, rh: np.ndarray,
                  variant_stats: dict, norm: str) -> tuple[np.ndarray, np.ndarray]:
    """Min-max envelope across the z-scored THI coefficient variants."""
    curves = []
    for vname, vv in idx.thi_variants(ta, rh).items():
        mean, sd = variant_stats[vname]
        centre = THRESHOLD["THI"] if norm == "threshold" else mean
        curves.append((vv - centre) / sd if sd > 0 else np.zeros_like(vv))
    stack = np.vstack(curves)
    return stack.min(axis=0), stack.max(axis=0)


def wind_fan(registry, stats, variant_stats, norm):
    """Vary wind 0.1->6 m/s at the hot baseline, SR = 0."""
    u = np.linspace(U_FLOOR, 6.0, N_SWEEP)
    ta = np.full_like(u, BASE_TA)
    rh = np.full_like(u, BASE_RH)
    sr = np.zeros_like(u)
    curves = {name: _normalise(name, fn(ta, rh, u, sr), stats, norm)
              for name, fn in registry.items()}
    band = _variant_band(ta, rh, variant_stats, norm)
    return u, curves, band


def solar_fan(registry, stats, variant_stats, norm):
    """Vary solar 0->1000 W/m² at the hot baseline, u = 0.2 m/s."""
    sr = np.linspace(0.0, 1000.0, N_SWEEP)
    ta = np.full_like(sr, BASE_TA)
    rh = np.full_like(sr, BASE_RH)
    u = np.full_like(sr, BASE_U)
    curves = {name: _normalise(name, fn(ta, rh, u, sr), stats, norm)
              for name, fn in registry.items()}
    band = _variant_band(ta, rh, variant_stats, norm)
    return sr, curves, band


# ─────────────────────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────────────────────

def _draw_matrix(fig, ax, names, mat) -> None:
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(COLUMNS)), COLUMNS)
    ax.set_yticks(range(len(names)), names)
    ax.set_title("A  ·  which inputs each index reads "
                 f"(∂index/∂input at Ta {BASE_TA:.0f} °C, RH {BASE_RH:.0f} %)",
                 fontsize=10, loc="left")
    for i in range(len(names)):
        for j in range(len(COLUMNS)):
            if mat[i, j] != 0:
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if abs(mat[i, j]) > 0.6 else "#222222")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("relative sensitivity (row-normalised)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


def _draw_fan(ax, x, curves, band, xlabel, title, ylabel, legend_order):
    lo, hi = band
    ax.fill_between(x, lo, hi, color=BAND_COLOUR, alpha=0.30, lw=0,
                    label="THI variant band")
    lines = []
    for name in legend_order:
        if name not in curves:
            continue
        is_ref = name == "THI"     # THI is the flat reference line
        line, = ax.plot(x, curves[name], color=COLOUR[name],
                        lw=2.4 if is_ref else 1.7,
                        zorder=3 if is_ref else 2, label=name)
        lines.append(line)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    ax.axhline(0.0, color="#cccccc", lw=0.8, ls="--", zorder=1)
    return lines


def _plot_figure(names, mat, wind, solar, ylabel, out_dir, legend_order) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    setup_figure()

    fig = plt.figure(figsize=(7.09, 6.4))   # ~180 mm wide (2-column)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                          hspace=0.42, wspace=0.26)
    _draw_matrix(fig, fig.add_subplot(gs[0, :]), names, mat)

    ux, ucur, uband = wind
    sx, scur, sband = solar
    _draw_fan(fig.add_subplot(gs[1, 0]), ux, ucur, uband,
              "wind speed  u  (m s$^{-1}$)", "B  ·  wind fan (SR = 0)",
              ylabel, legend_order)
    handles = _draw_fan(fig.add_subplot(gs[1, 1]), sx, scur, sband,
                        "solar radiation  SR  (W m$^{-2}$)",
                        f"C  ·  solar fan (u = {BASE_U} m s$^{{-1}}$)",
                        ylabel, legend_order)

    band_handle = Patch(facecolor=BAND_COLOUR, alpha=0.30,
                        label="THI variant band")
    fig.legend([*handles, band_handle],
               [h.get_label() for h in handles] + ["THI variant band"],
               loc="lower center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("The thermal index is not one thing "
                 "(construction figure, simulated — not measured barn values)",
                 fontsize=11, y=0.98)
    save_figure(fig, "fig1_index_construction", out_dir, dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  CSV companions (one per panel)
# ─────────────────────────────────────────────────────────────

def _write_csv(path: Path, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_panel_csvs(names, mat, wind, solar, out_dir, legend_order) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(out_dir / "fig1_panelA_matrix.csv", ["index", *COLUMNS],
               [[name, *[f"{v:.6g}" for v in mat[i]]]
                for i, name in enumerate(names)])

    for stem, (x, curves, band), xname in (
        ("fig1_panelB_wind.csv", wind, "u"),
        ("fig1_panelC_solar.csv", solar, "SR"),
    ):
        cols = [n for n in legend_order if n in curves]
        header = [xname, *cols, "THI_band_min", "THI_band_max"]
        rows = []
        for k in range(len(x)):
            rows.append([f"{x[k]:.6g}",
                         *[f"{curves[c][k]:.6g}" for c in cols],
                         f"{band[0][k]:.6g}", f"{band[1][k]:.6g}"])
        _write_csv(out_dir / stem, header, rows)


# ─────────────────────────────────────────────────────────────
#  Public build
# ─────────────────────────────────────────────────────────────

def build(out_dir: Path, *, norm: str = "zscore",
          include_itsc: bool = False, seed: int = ENVELOPE_SEED) -> Path:
    """Compute and render Part I Figure 1, writing SVG + PNG + 3 panel CSVs.

    Args:
        out_dir:      Output directory (created on demand).
        norm:         Fan normalisation, ``"zscore"`` (default) or
                      ``"threshold"`` (deviation from each index's published
                      heat-stress threshold).
        include_itsc: Overlay the optional ITSC index in the fans.
        seed:         RNG seed for the Latin-hypercube climate envelope.

    Returns:
        The output directory.
    """
    out_dir = Path(out_dir)
    registry = _physical_registry(include_itsc)
    legend_order = [*LEGEND_ORDER, "ITSC"] if include_itsc else list(LEGEND_ORDER)

    rng = np.random.default_rng(seed)
    index_stats, variant_stats = envelope_stats(registry, rng)

    names, mat = sensitivity_matrix()
    wind = wind_fan(registry, index_stats, variant_stats, norm)
    solar = solar_fan(registry, index_stats, variant_stats, norm)

    ylabel = ("apparent heat load\n(SD within climate envelope)"
              if norm == "zscore"
              else "deviation from threshold\n(SD within climate envelope)")

    _plot_figure(names, mat, wind, solar, ylabel, out_dir, legend_order)
    _write_panel_csvs(names, mat, wind, solar, out_dir, legend_order)
    return out_dir
