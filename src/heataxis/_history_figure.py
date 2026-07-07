# ╔══════════════════════════════════════════════════════════════════╗
# ║  heataxis — _history_figure                                      ║
# ║  « the x-axis needs a memory: same weather, different history »  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Temporal counterpart to the Fig 1 construction figure. One      ║
# ║  synthetic heat-wave drives real THI / HLI; the history          ║
# ║  transforms then diverge — memoryless tracks the input, cum-sum  ║
# ║  never forgets, the leaky integrator accumulates AND recovers,   ║
# ║  and TSD / TSL count duration and load.                          ║
# ║                                                                  ║
# ║  100 % simulated — no measured data. Public entry point is       ║
# ║  heataxis.viz.plot_exposure_history().                           ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Engine for the exposure-history figure (temporal counterpart to Fig 1).

A single synthetic heat-wave (diurnal cycle + a multi-day event with hot,
poorly-recovering nights at its core) drives real THI and HLI series via
:mod:`heataxis.indices`; :mod:`heataxis.history` then turns them into load
histories.  It is an illustrative construction figure, not measured data.
Call it through :func:`heataxis.viz.plot_exposure_history`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from heataxis import history
from heataxis import indices as idx
from heataxis.constants import WONG
from heataxis.viz import save_figure, setup_figure

# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────
DAYS = 12
DT_H = 1.0                 # hourly resolution
THI_THRESHOLD = 72.0       # stress onset for THI-based history
TAU_FAST, TAU_SLOW = 6.0, 48.0     # leaky-integrator memory constants (hours)
AHL_UPPER, AHL_LOWER = 86.0, 77.0  # Gaughan AHL thresholds (Bos taurus)

COLOUR = {
    "THI": WONG["black"],
    "memoryless": WONG["grey"],
    "leaky_fast": WONG["sky_blue"],
    "leaky_slow": WONG["blue"],
    "cumsum": WONG["bluish_green"],
    "AHL": WONG["vermilion"],
    "TSD": WONG["bluish_green"],
    "TSL": WONG["reddish_purple"],
}


# ─────────────────────────────────────────────────────────────
#  Synthetic weather -> real indices -> history transforms
# ─────────────────────────────────────────────────────────────

def synthetic_weather(days: int = DAYS, dt_h: float = DT_H) -> dict:
    """One synthetic heat-wave: diurnal cycle under a multi-day event.

    The event core (days 3-9) raises the daily mean and *shrinks* the diurnal
    swing, so its nights stay hot and recover poorly — the contrast that makes
    memory visible.
    """
    t = np.arange(0.0, days * 24.0, dt_h)
    hod = t % 24.0
    day = t / 24.0
    bump = np.where((day >= 3) & (day <= 9),
                    0.5 * (1.0 - np.cos(2.0 * np.pi * (day - 3) / 6.0)), 0.0)
    diurnal = np.cos(2.0 * np.pi * (hod - 15.0) / 24.0)   # peak ~15:00
    # Event core: higher daily mean, smaller swing -> nights stay above the
    # stress threshold (poor recovery). Off-core days stay below it.
    ta = 19.0 + 12.0 * bump + (5.0 - 2.5 * bump) * diurnal
    rh = np.clip(65.0 - 20.0 * bump - 6.0 * diurnal, 30.0, 95.0)
    u = np.full_like(t, 0.3)                              # still indoor air
    daylight = np.where((hod >= 6.0) & (hod <= 18.0),
                        np.sin(np.pi * (hod - 6.0) / 12.0), 0.0)
    # Modest indoor/diffuse solar (naturally ventilated barn, no direct beam).
    sr = np.clip(320.0 * daylight * (0.7 + 0.3 * bump), 0.0, None)
    return {"t": t, "day": day, "hod": hod, "ta": ta, "rh": rh, "u": u, "sr": sr}


def compute_series(weather: dict) -> dict:
    """Real THI and HLI from the weather, then all history transforms."""
    ta, rh, u, sr = weather["ta"], weather["rh"], weather["u"], weather["sr"]
    thi = idx.thi_nrc1971(ta, rh)
    tbg = idx.tbg_estimate(ta, sr, u)
    hli = idx.hli_gaughan(tbg, rh, u)
    return {
        "THI": thi,
        "HLI": hli,
        "memoryless": history.heat_load_above(thi, THI_THRESHOLD),
        "leaky_fast": history.leaky_integrate(thi, DT_H, TAU_FAST,
                                              baseline=THI_THRESHOLD),
        "leaky_slow": history.leaky_integrate(thi, DT_H, TAU_SLOW,
                                              baseline=THI_THRESHOLD),
        "cumsum": history.cumulative_load(thi, DT_H, baseline=THI_THRESHOLD),
        "AHL": history.accumulated_heat_load(hli, DT_H,
                                             upper=AHL_UPPER, lower=AHL_LOWER),
        "TSD": history.thermal_stress_duration(thi, DT_H, THI_THRESHOLD),
        "TSL": history.thermal_stress_load(thi, DT_H, THI_THRESHOLD),
    }


def _norm(a: np.ndarray) -> np.ndarray:
    peak = float(np.max(a))
    return a / peak if peak > 0 else a


# ─────────────────────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────────────────────

def _shade_nights(ax, days: int) -> None:
    for i in range(days):
        ax.axvspan(i + 0.75, i + 1.25, color=WONG["black"], alpha=0.045, lw=0)


def _draw(weather, series, out_dir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_figure()

    day = weather["day"]
    n_days = int(np.ceil(day[-1]))
    fig, (axA, axB, axC) = plt.subplots(
        3, 1, figsize=(7.09, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.05, 0.9], "hspace": 0.16})

    # Panel A — the driver.
    _shade_nights(axA, n_days)
    thi = series["THI"]
    axA.plot(day, thi, color=COLOUR["THI"], lw=1.6, label="THI")
    axA.axhline(THI_THRESHOLD, color=WONG["grey"], lw=0.9, ls="--",
                label=f"stress threshold ({THI_THRESHOLD:.0f})")
    axA.fill_between(day, THI_THRESHOLD, thi, where=thi > THI_THRESHOLD,
                     color=WONG["vermilion"], alpha=0.18, lw=0)
    axA.set_ylabel("THI")
    axA.set_title("A  ·  the weather — one synthetic heat-wave "
                  "(hot, poorly-recovering nights at its core)",
                  fontsize=10, loc="left")
    axA.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)

    # Panel B — same weather, different memory (each scaled to its own max).
    _shade_nights(axB, n_days)
    order = [("memoryless", "memoryless  (THI−72)₊"),
             ("leaky_fast", f"leaky τ={TAU_FAST:.0f} h"),
             ("leaky_slow", f"leaky τ={TAU_SLOW:.0f} h"),
             ("AHL", "AHL (Gaughan, from HLI)"),
             ("cumsum", "cum-sum (no recovery)")]
    for key, label in order:
        axB.plot(day, _norm(series[key]), color=COLOUR[key], lw=1.7, label=label)
    axB.set_ylabel("load\n(each scaled to its own max)")
    axB.set_title("B  ·  same weather, different memory — "
                  "cum-sum never forgets; the leaky integrator recovers & saturates",
                  fontsize=10, loc="left")
    axB.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)

    # Panel C — established duration / load comparators (twin axes).
    _shade_nights(axC, n_days)
    axC.plot(day, series["TSD"], color=COLOUR["TSD"], lw=1.8, label="TSD")
    axC.set_ylabel("TSD  (h above threshold)", color=COLOUR["TSD"])
    axC.tick_params(axis="y", labelcolor=COLOUR["TSD"])
    axC2 = axC.twinx()
    axC2.plot(day, series["TSL"], color=COLOUR["TSL"], lw=1.8, ls="-", label="TSL")
    axC2.set_ylabel("TSL  (°C·h above threshold)", color=COLOUR["TSL"])
    axC2.tick_params(axis="y", labelcolor=COLOUR["TSL"])
    axC.set_title("C  ·  established duration (TSD) & load (TSL) comparators — "
                  "monotone counters",
                  fontsize=10, loc="left")
    axC.set_xlabel("time (days)")
    axC.set_xlim(day[0], day[-1])

    fig.suptitle("The x-axis needs a memory "
                 "(construction figure, simulated — not measured barn values)",
                 fontsize=11, y=0.995)
    save_figure(fig, "fig_exposure_history", out_dir, dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  CSV companion (one shared time series)
# ─────────────────────────────────────────────────────────────

def _write_csv(weather, series, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["day", "THI", "HLI", "memoryless", "leaky_fast", "leaky_slow",
            "cumsum", "AHL", "TSD_h", "TSL_degh"]
    keymap = {"TSD_h": "TSD", "TSL_degh": "TSL"}
    day = weather["day"]
    with open(out_dir / "fig_exposure_history.csv", "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(cols)
        for k in range(day.size):
            row = [f"{day[k]:.4g}"]
            for c in cols[1:]:
                row.append(f"{series[keymap.get(c, c)][k]:.6g}")
            writer.writerow(row)


# ─────────────────────────────────────────────────────────────
#  Public build
# ─────────────────────────────────────────────────────────────

def build(out_dir: Path, *, days: int = DAYS) -> Path:
    """Compute and render the exposure-history figure (SVG + PNG + CSV).

    Args:
        out_dir: Output directory (created on demand).
        days:    Length of the synthetic heat-wave (days).

    Returns:
        The output directory.
    """
    out_dir = Path(out_dir)
    weather = synthetic_weather(days=days, dt_h=DT_H)
    series = compute_series(weather)
    _draw(weather, series, out_dir)
    _write_csv(weather, series, out_dir)
    return out_dir
