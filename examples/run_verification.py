"""Generate the published verification artifacts for Thermal Streams.

Run (from anywhere; outputs land in ``docs/results/``)::

    python examples/run_verification.py

Outputs
-------
- ``effectiveness_vs_ntu.png``          cover figure: eps vs NTU for several C_r
- ``counter_flow_temperature_profiles.png``  position-resolved profiles,
  documented case (see below)
- ``counterflow_profiles_ntu.gif``      48-frame animation: counter-flow
  profiles as the exchanger grows, NTU = 0.25 -> 12
- ``heat_balance_table.csv``            energy-balance table over the sweep
- ``run_summary.json``                  runtime versions + computed metrics

Documented profile case
-----------------------
Counter-flow water/water case: U = 800 W/(m^2 K), A = 6 m^2,
C_h = 1200 W/K, C_c = 2000 W/K, T_h,in = 90 degC, T_c,in = 25 degC
=> C_min = 1200 W/K, C_r = 0.60, NTU = 4.0. The plotted profiles are the
analytic exponential solutions of the 1-D steady-state model documented in
``docs/model.md`` (uniform U, constant capacity rates, no axial conduction,
no ambient loss), evaluated on ``s in [0, 1]``, the fractional transfer area
from the hot inlet (s = 0: hot inlet / cold outlet; s = 1: hot outlet /
cold inlet).
"""
from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from thermal_streams import (  # noqa: E402
    FLOWS,
    ExchangerConfig,
    counter_flow_profile,
    effectiveness,
    lmtd_duty_W,
    solve_exchanger,
)

RESULTS = REPO / "docs" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

HOT_COLOR = "#9E2B25"    # deep brick red
COLD_COLOR = "#175676"   # deep steel blue
GAP_COLOR = "#B9B9B9"
CR_COLORS = {
    0.0: "#1F4E79",
    0.25: "#2E7D6B",
    0.5: "#B3282D",
    0.75: "#8C6D31",
    1.0: "#5B5B5B",
}

# Documented counter-flow case (see module docstring).
CASE = ExchangerConfig(
    flow="counter",
    U_W_m2K=800.0,
    A_m2=6.0,
    C_h_W_K=1200.0,
    C_c_W_K=2000.0,
    T_h_in_K=363.15,
    T_c_in_K=298.15,
)


def build_config(flow, ntu, c_r, hot_is_cmin, c_min=1200.0, d_t=65.0,
                 t_c_in=298.15, u=800.0):
    """Build a config with the requested NTU and C_r (C_r=0 -> C_max=1e12 W/K)."""
    c_max = c_min / c_r if c_r > 0.0 else 1.0e12
    c_h, c_c = (c_min, c_max) if hot_is_cmin else (c_max, c_min)
    return ExchangerConfig(
        flow=flow,
        U_W_m2K=u,
        A_m2=ntu * c_min / u,
        C_h_W_K=c_h,
        C_c_W_K=c_c,
        T_h_in_K=t_c_in + d_t,
        T_c_in_K=t_c_in,
    )


# ---------------------------------------------------------------------------
# Figure 1: effectiveness vs NTU cover figure
# ---------------------------------------------------------------------------

def effectiveness_cover_figure() -> Path:
    ntu = np.linspace(0.01, 8.0, 321)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for c_r, color in CR_COLORS.items():
        ax.plot(ntu, effectiveness(ntu, c_r, "counter"), color=color, lw=1.9,
                label=f"counter, $C_r$ = {c_r:.2f}")
        ax.plot(ntu, effectiveness(ntu, c_r, "parallel"), color=color, lw=1.3,
                ls="--")
    # Parallel-flow asymptotes eps_max = 1 / (1 + C_r) for two levels.
    for c_r, color in [(0.5, "#6E6E6E"), (1.0, "#9A9A9A")]:
        ax.axhline(1.0 / (1.0 + c_r), color=color, lw=0.8, ls=":")
    ax.annotate("parallel-flow limit $1/(1+C_r)$", xy=(6.4, 1.0 / 1.5),
                xytext=(5.0, 0.87), fontsize=8, color="#555555",
                arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=0.7))
    ax.annotate("", xy=(6.4, 0.5), xytext=(5.6, 0.62),
                arrowprops=dict(arrowstyle="-", color="#9A9A9A", lw=0.7))
    ax.set_xlabel(r"NTU $= UA/C_{min}$  (dimensionless)")
    ax.set_ylabel(r"Effectiveness $\varepsilon$  (dimensionless)")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlim(0.0, 8.0)
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)
    fig.tight_layout()
    path = RESULTS / "effectiveness_vs_ntu.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: documented counter-flow temperature profile
# ---------------------------------------------------------------------------

def profile_figure() -> Path:
    prof = counter_flow_profile(CASE, 201)
    res = prof.result
    s = prof.s
    t_h = prof.T_h_K - 273.15
    t_c = prof.T_c_K - 273.15

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(s, t_c, t_h, color=GAP_COLOR, alpha=0.18, lw=0)
    ax.plot(s, t_h, color=HOT_COLOR, lw=2.0,
            label=r"hot stream ($C_h$ = 1200 W/K)")
    ax.plot(s, t_c, color=COLD_COLOR, lw=2.0,
            label=r"cold stream ($C_c$ = 2000 W/K)")

    ax.plot(0.0, t_h[0], marker="o", ms=4.5, color=HOT_COLOR)
    ax.plot(0.0, t_c[0], marker="o", ms=4.5, color=COLD_COLOR)
    ax.plot(1.0, t_h[-1], marker="o", ms=4.5, color=HOT_COLOR)
    ax.plot(1.0, t_c[-1], marker="o", ms=4.5, color=COLD_COLOR)
    ax.annotate(f"hot inlet\n{t_h[0]:.1f} °C", xy=(0.0, t_h[0]),
                xytext=(0.04, t_h[0] - 1.0), fontsize=8, color=HOT_COLOR)
    ax.annotate(f"cold outlet\n{t_c[0]:.1f} °C", xy=(0.0, t_c[0]),
                xytext=(0.04, t_c[0] + 0.5), fontsize=8, color=COLD_COLOR)
    ax.annotate(f"hot outlet\n{t_h[-1]:.1f} °C", xy=(1.0, t_h[-1]),
                xytext=(0.82, t_h[-1] + 3.0), fontsize=8, color=HOT_COLOR)
    ax.annotate(f"cold inlet\n{t_c[-1]:.1f} °C", xy=(1.0, t_c[-1]),
                xytext=(0.82, t_c[-1] + 3.0), fontsize=8, color=COLD_COLOR)

    ax.text(0.5, 0.985,
            "s = 0: hot inlet / cold outlet      s = 1: hot outlet / cold inlet",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#555555")
    ax.set_xlabel(r"fractional transfer area $s = A(s)/A_{tot}$  (dimensionless)")
    ax.set_ylabel(f"Stream temperature (°C)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(15.0, 100.0)
    ax.set_title(
        f"Counter-flow temperature profile: NTU = {res.ntu:.2f}, "
        f"$C_r$ = {res.c_r:.2f}, $\\varepsilon$ = {res.effectiveness:.4f}"
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    fig.tight_layout()
    path = RESULTS / "counter_flow_temperature_profiles.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3 (animated): profiles as the exchanger grows, NTU 0.25 -> 12
# ---------------------------------------------------------------------------

def animation_gif():
    ntu_frames = np.geomspace(0.25, 12.0, 48)
    c_min = min(CASE.C_h_W_K, CASE.C_c_W_K)

    fig, ax = plt.subplots(figsize=(10.0, 6.0), dpi=100)

    def draw(i: int):
        ntu = float(ntu_frames[i])
        cfg = replace(CASE, A_m2=ntu * c_min / CASE.U_W_m2K)
        prof = counter_flow_profile(cfg, 241)
        res = prof.result
        t_h = prof.T_h_K - 273.15
        t_c = prof.T_c_K - 273.15

        ax.clear()
        ax.fill_between(prof.s, t_c, t_h, color=GAP_COLOR, alpha=0.15, lw=0)
        ax.plot(prof.s, t_h, color=HOT_COLOR, lw=2.2,
                label=r"hot stream, $C_h$ = 1200 W/K")
        ax.plot(prof.s, t_c, color=COLD_COLOR, lw=2.2,
                label=r"cold stream, $C_c$ = 2000 W/K")
        ax.plot(0.0, t_h[0], marker="o", ms=4.5, color=HOT_COLOR)
        ax.plot(1.0, t_c[-1], marker="o", ms=4.5, color=COLD_COLOR)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(15.0, 100.0)
        ax.set_xlabel(
            r"fractional transfer area $s = A(s)/A_{tot}$  (dimensionless)"
        )
        ax.set_ylabel(f"Stream temperature (°C)")
        ax.set_title(
            f"Counter-flow exchanger grown to NTU = {ntu:.2f}   "
            f"($C_r$ = {res.c_r:.2f})"
        )
        ax.text(
            0.5, 0.985,
            "s = 0: hot inlet / cold outlet      s = 1: hot outlet / cold inlet",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            color="#555555",
        )
        info = (
            f"$\\varepsilon$ = {res.effectiveness:.3f}\n"
            f"$Q$ = {res.duty_W / 1e3:.1f} kW\n"
            f"$T_{{h,out}}$ = {res.T_h_out_K - 273.15:.1f} "
            f"°C\n"
            f"$T_{{c,out}}$ = {res.T_c_out_K - 273.15:.1f} "
            f"°C"
        )
        ax.text(0.02, 0.04, info, transform=ax.transAxes, fontsize=9.5,
                va="bottom",
                bbox=dict(facecolor="white", edgecolor="#999999", lw=0.7,
                          boxstyle="round,pad=0.35"))
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    anim = FuncAnimation(fig, draw, frames=len(ntu_frames), interval=125)
    path = RESULTS / "counterflow_profiles_ntu.gif"
    anim.save(path, writer=PillowWriter(fps=8),
              savefig_kwargs={"facecolor": "white"})
    plt.close(fig)
    with Image.open(path) as img:
        return path, int(img.n_frames), img.size


# ---------------------------------------------------------------------------
# Heat-balance table CSV
# ---------------------------------------------------------------------------

def heat_balance_table():
    path = RESULTS / "heat_balance_table.csv"
    fields = [
        "flow", "ntu", "c_r", "C_min_W_per_K", "C_max_W_per_K",
        "hot_side_is_C_min", "T_h_in_degC", "T_c_in_degC", "effectiveness",
        "duty_W", "T_h_out_degC", "T_c_out_degC", "Q_hot_loss_W",
        "Q_cold_gain_W", "energy_balance_residual_W",
    ]
    n_rows = 0
    max_residual = 0.0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for flow in FLOWS:
            for c_r in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for ntu in [0.5, 1.0, 2.0, 4.0, 8.0]:
                    for hot_is_cmin in (True, False):
                        cfg = build_config(flow, ntu, c_r, hot_is_cmin)
                        res = solve_exchanger(cfg)
                        q_hot = cfg.C_h_W_K * (cfg.T_h_in_K - res.T_h_out_K)
                        q_cold = cfg.C_c_W_K * (res.T_c_out_K - cfg.T_c_in_K)
                        residual = abs(q_hot - q_cold)
                        max_residual = max(max_residual, residual)
                        writer.writerow([
                            flow, f"{ntu:g}", f"{res.c_r:.12g}",
                            f"{res.c_min_W_K:.10g}", f"{res.c_max_W_K:.10g}",
                            str(hot_is_cmin),
                            f"{cfg.T_h_in_K - 273.15:.4g}",
                            f"{cfg.T_c_in_K - 273.15:.4g}",
                            f"{res.effectiveness:.12g}",
                            f"{res.duty_W:.10g}",
                            f"{res.T_h_out_K - 273.15:.8g}",
                            f"{res.T_c_out_K - 273.15:.8g}",
                            f"{q_hot:.10g}", f"{q_cold:.10g}",
                            f"{residual:.6e}",
                        ])
                        n_rows += 1
    return path, n_rows, max_residual


# ---------------------------------------------------------------------------
# Metrics sweeps
# ---------------------------------------------------------------------------

def compute_metrics():
    # Energy-balance closure across every arrangement.
    max_residual = 0.0
    max_residual_rel = 0.0
    for flow in FLOWS:
        for c_r in np.linspace(0.1, 1.0, 10):
            for ntu in np.geomspace(0.1, 100.0, 30):
                for hot_is_cmin in (True, False):
                    res = solve_exchanger(build_config(flow, ntu, float(c_r),
                                                       hot_is_cmin))
                    residual = res.energy_balance_residual_W
                    max_residual = max(max_residual, residual)
                    max_residual_rel = max(max_residual_rel,
                                           residual / res.duty_W)

    # eps-NTU vs LMTD duty agreement (counter flow: the required strong check;
    # parallel flow: restricted to the well-conditioned range NTU <= 5,
    # see docs/verification.md).
    max_lmtd_counter = 0.0
    max_lmtd_parallel = 0.0
    for c_r in [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.99, 1.0]:
        for hot_is_cmin in (True, False):
            for ntu in np.geomspace(0.1, 15.8, 24):
                res = solve_exchanger(build_config("counter", ntu, c_r,
                                                   hot_is_cmin))
                dev = abs(lmtd_duty_W(res) - res.duty_W) / res.duty_W
                max_lmtd_counter = max(max_lmtd_counter, dev)
            for ntu in np.geomspace(0.1, 5.0, 20):
                res = solve_exchanger(build_config("parallel", ntu, c_r,
                                                   hot_is_cmin))
                dev = abs(lmtd_duty_W(res) - res.duty_W) / res.duty_W
                max_lmtd_parallel = max(max_lmtd_parallel, dev)

    # C_r = 0 anchor: solve() with a 1e12 W/K opposite stream vs the closed
    # form 1 - exp(-NTU).
    max_anchor = 0.0
    for flow in FLOWS:
        for ntu in np.geomspace(0.1, 50.0, 60):
            res = solve_exchanger(build_config(flow, float(ntu), 0.0, True))
            anchor = -(np.expm1(-res.ntu))
            max_anchor = max(max_anchor, abs(res.effectiveness - float(anchor)))

    return {
        "energy_balance_sweep_cases": 5 * 10 * 30 * 2,
        "max_energy_balance_residual_W": max_residual,
        "max_energy_balance_residual_relative": max_residual_rel,
        "lmtd_counter_sweep_cases": 11 * 2 * 24,
        "max_relative_lmtd_deviation_counter": max_lmtd_counter,
        "lmtd_parallel_sweep_cases": 11 * 2 * 20,
        "max_relative_lmtd_deviation_parallel": max_lmtd_parallel,
        "cr_zero_anchor_sweep_cases": 5 * 60,
        "cr_zero_anchor_max_abs_deviation": max_anchor,
    }


def run_test_suite():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(REPO), "-q", "--tb=no",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    summary_line = lines[-1] if lines else ""
    match = re.search(r"(\d+) passed", summary_line)
    return {
        "exit_code": proc.returncode,
        "summary_line": summary_line,
        "passed": int(match.group(1)) if match else None,
    }


def main():
    t0 = time.perf_counter()
    print("Thermal Streams - verification artifacts")
    print(f"repo root: {REPO}")
    print(f"output dir: {RESULTS}")

    cover = effectiveness_cover_figure()
    print(f"[1/5] cover figure            {cover.name}")
    prof_fig = profile_figure()
    print(f"[2/5] profile figure          {prof_fig.name}")
    gif_path, gif_frames, gif_size = animation_gif()
    print(f"[3/5] animation               {gif_path.name} "
          f"({gif_frames} frames, {gif_size[0]}x{gif_size[1]} px)")
    csv_path, csv_rows, csv_max_residual = heat_balance_table()
    print(f"[4/5] heat-balance table      {csv_path.name} ({csv_rows} rows)")
    metrics = compute_metrics()
    tests = run_test_suite()
    print(f"[5/5] pytest                  {tests['summary_line']}")

    artifacts = {}
    for path in [cover, prof_fig, gif_path, csv_path]:
        artifacts[path.name] = {"bytes": path.stat().st_size}
    artifacts[gif_path.name]["frames"] = gif_frames
    artifacts[gif_path.name]["pixel_size"] = list(gif_size)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "packages": {
                "numpy": metadata.version("numpy"),
                "matplotlib": metadata.version("matplotlib"),
                "pillow": metadata.version("pillow"),
                "pytest": metadata.version("pytest"),
            },
        },
        "tests": tests,
        "metrics": metrics,
        "documented_case": {
            "description": "counter-flow water/water case, "
                           "U = 800 W/(m^2 K), A = 6 m^2, C_h = 1200 W/K, "
                           "C_c = 2000 W/K, T_h_in = 90 degC, T_c_in = 25 degC",
            "ntu": None,
            "c_r": None,
            "effectiveness": None,
            "duty_W": None,
            "energy_balance_residual_W": None,
            "lmtd_duty_W": None,
        },
        "artifacts": artifacts,
        "wall_time_s": round(time.perf_counter() - t0, 3),
    }
    # Fill the documented case with real solved numbers (no hard-coded values).
    res = solve_exchanger(CASE)
    summary["documented_case"].update({
        "ntu": res.ntu,
        "c_r": res.c_r,
        "effectiveness": res.effectiveness,
        "duty_W": res.duty_W,
        "T_h_out_degC": res.T_h_out_K - 273.15,
        "T_c_out_degC": res.T_c_out_K - 273.15,
        "energy_balance_residual_W": res.energy_balance_residual_W,
        "lmtd_duty_W": lmtd_duty_W(res),
    })

    summary_path = RESULTS / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"      run summary             {summary_path.name}")

    print()
    print("Metrics (all computed in this run):")
    print(f"  max energy-balance residual      "
          f"{metrics['max_energy_balance_residual_W']:.3e} W "
          f"(relative {metrics['max_energy_balance_residual_relative']:.3e})")
    print(f"  max rel |Q_epsNTU - Q_LMTD|      counter "
          f"{metrics['max_relative_lmtd_deviation_counter']:.3e}, "
          f"parallel {metrics['max_relative_lmtd_deviation_parallel']:.3e}")
    print(f"  C_r = 0 anchor max deviation     "
          f"{metrics['cr_zero_anchor_max_abs_deviation']:.3e}")
    print(f"  wall time {summary['wall_time_s']:.1f} s")


if __name__ == "__main__":
    main()
