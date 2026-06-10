#!/usr/bin/env python
"""Intensity-scaling confirmation: is the small-Δ rotation χ⁵ (θ∝I²) or residual χ³ (∝I)?

At Δ=0.006 (small — sidebands inside the probe mode) vs Δ=0.0234 (native), sweep the balanced
σ⁺σ⁻ pump intensity and fit |θ| = A·I^p in log-log. p≈2 ⇒ χ⁵ (the target cascade); p≈1 ⇒
residual χ³. Beat-resolved (T = 18/Δ). Separate-entity; reuses the existing CLI via subprocess
and the shared config from run_delta_scan.py. Run under micromamba env meep-mpi.
"""
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from run_delta_scan import F_MEAN, GEOM, MODES, MATERIAL_FLAGS, MEEP, HERE

OUTROOT = HERE / "intensity_scaling" / "sic_L3p2um"
DELTAS = [0.006, 0.0234]
INTENSITIES = [5e10, 1e11, 2e11, 4e11, 8e11]
BEATS = 18
RES = 60


def ilabel(intensity: float) -> str:
    return f"{intensity:.1e}".replace("+", "").replace(".", "p")


def run_one(delta: float, intensity: float) -> tuple:
    f1 = F_MEAN + 0.5 * delta
    f2 = F_MEAN - 0.5 * delta
    outdir = OUTROOT / f"d{delta:.4f}_I{ilabel(intensity)}"
    outdir.mkdir(parents=True, exist_ok=True)
    if (outdir / "faraday_summary.json").exists():
        return delta, intensity, "cached"
    until = BEATS / delta  # resolve the pump-sideband beat (T >> 1/Δ)
    cmd = [
        sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *MATERIAL_FLAGS,
        "--geometry-file", GEOM, "--cavity-modes-file", MODES, "--resolution", str(RES),
        "--until-time", f"{until:.3f}", "--pump-intensity", str(intensity),
        "--pump1-frequency", f"{f1:.9f}", "--pump2-frequency", f"{f2:.9f}",
        "--output-dir", str(outdir),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    with open(outdir / "run.log", "w") as lf:
        r = subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return delta, intensity, ("ok" if r.returncode == 0 else f"FAIL({r.returncode})")


def theta_of(delta: float, intensity: float):
    p = OUTROOT / f"d{delta:.4f}_I{ilabel(intensity)}" / "faraday_summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return {
        "theta": d["probe_rotation_deg"]["final_relative_deg"],
        "dolp": d["probe_rotation_deg"]["coherent_window_estimate"]["dolp"],
        "p2p1": d["pump_monitor_metrics"]["coherent_reference"]["ratio_p2_over_p1"]["tail_weighted"],
    }


def main() -> None:
    OUTROOT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, I) for d in DELTAS for I in INTENSITIES]
    print(f"[iscan] {len(jobs)} runs (Δ×I), beats={BEATS}, res={RES}", flush=True)
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(run_one, d, I) for d, I in jobs]
        for f in as_completed(futs):
            d, I, s = f.result()
            print(f"  Δ={d:.4f} I={I:.1e} -> {s}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    colors = {0.006: "tab:red", 0.0234: "tab:blue"}
    for d in DELTAS:
        Iv, th = [], []
        for Ival in INTENSITIES:
            r = theta_of(d, Ival)
            if r and r["theta"] is not None:
                Iv.append(Ival)
                th.append(abs(float(r["theta"])))
                rows.append({"delta": d, "I": Ival, "abs_theta": abs(float(r["theta"])),
                             "dolp": r["dolp"], "p2p1": r["p2p1"]})
        Iv = np.array(Iv)
        th = np.array(th)
        m = (th > 1e-9) & np.isfinite(th)
        c = colors.get(d, None)
        if m.sum() >= 2:
            p, logA = np.polyfit(np.log10(Iv[m]), np.log10(th[m]), 1)
            ax.plot(Iv, th, "o-", color=c, label=f"Δ={d:.4f} (~{d/0.0042:.0f}·Γ_s): slope p={p:.2f}")
            ax.plot(Iv[m], 10 ** logA * Iv[m] ** p, "--", color=c, alpha=0.4)
        else:
            ax.plot(Iv, th, "o-", color=c, label=f"Δ={d:.4f}")
    # reference slopes
    xref = np.array([INTENSITIES[0], INTENSITIES[-1]], dtype=float)
    for pslope, ls in ((1.0, ":"), (2.0, "-.")):
        ax.plot(xref, 0.02 * (xref / xref[0]) ** pslope, ls, color="0.6", lw=1.0,
                label=f"slope {pslope:.0f} ({'χ³' if pslope == 1 else 'χ⁵'})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pump intensity (W/cm²)")
    ax.set_ylabel("|θ| (deg)")
    ax.set_title("Intensity scaling at small vs native Δ — χ⁵ (p≈2) vs χ³ (p≈1)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTROOT / "intensity_scaling.png", dpi=130, bbox_inches="tight")

    if rows:
        with open(OUTROOT / "intensity_scaling_points.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[iscan] -> {OUTROOT}")
    for r in rows:
        print(f"  Δ={r['delta']:.4f} I={r['I']:.1e} |θ|={r['abs_theta']:.4f} "
              f"DoLP={r['dolp']:.4f} p2/p1={r['p2p1']:.3f}")


if __name__ == "__main__":
    main()
