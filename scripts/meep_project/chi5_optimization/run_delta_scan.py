#!/usr/bin/env python
"""1D Delta-scan — verify the real-system optimal pump beat Delta.

Holds the probe frequency and the pump *mean* frequency fixed and varies only
Delta = f1 - f2 (sidebands at f_probe +/- Delta), at moderate *balanced* sigma+sigma-
intensity, reading the three-theta rotation. Tests the design-plan prediction
(design_optimization_plan.md, sec 2b): with fixed ~100 fs sources, the dispersive
optimum is Delta_opt ~ source bandwidth (~0.046 /um), NOT the much smaller cavity
linewidth (~0.004 /um). If |theta|(Delta) peaks near the source bandwidth -> confirmed;
if it falls monotonically from the smallest Delta -> the naive cavity-linewidth CMT.

Separate-entity experiment: reuses faraday_meep_fp_circ.py via subprocess; does NOT
modify the existing pipeline. Run under micromamba env meep-mpi.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent  # scripts/meep_project
OUTROOT = HERE / "delta_scan" / "sic_L3p2um"

# --- Fixed operating point (SiC L=3.2um cavity) ---------------------------------
GEOM = "SiC_optimizations/sic_L3p2um/geometry.json"
MODES = "SiC_optimizations/sic_L3p2um/cavity_modes.json"
F_PROBE = 1.213338942106731            # 1/um (held fixed via the modes file)
F_MEAN = 0.5919899874843556            # (f_pump1 + f_pump2)/2 from the modes file
SRC_BW = 0.0462                        # Meep pump source fwidth (1/um), predicted Delta_opt
KAPPA_S_HALF = F_PROBE / 144.54 / 2.0  # cavity half-linewidth (~0.0042 /um), naive-CMT optimum

MATERIAL_FLAGS = [
    "--materials", "fit", "--sin-fit", "sic.csv", "--sio2-fit", "sio2.csv",
    "--fit-poles", "3", "--fit-window", "600", "2000", "--high-index-material", "sic",
]

# Delta grid (1/um), bracketing the predicted ~0.046 peak; includes the native 0.0234.
DELTAS = [0.006, 0.010, 0.016, 0.0234, 0.032, 0.046, 0.063, 0.086, 0.115, 0.150]


def run_one(delta: float, intensity: float, resolution: int, decay: float,
            beats: float | None = None) -> tuple:
    f1 = F_MEAN + 0.5 * delta
    f2 = F_MEAN - 0.5 * delta
    outdir = OUTROOT / f"delta_{delta:.4f}"
    outdir.mkdir(parents=True, exist_ok=True)
    summ = outdir / "faraday_summary.json"
    if summ.exists():
        return delta, "cached"
    # CRITICAL: the pump-sideband beat period is 1/delta. The measurement must capture
    # many beats (T >> 1/delta) or the f_probe DFT bin leaks the unresolved sidebands ->
    # spurious rotation that blows up as delta->0. So scale the run length to the beat
    # period when --beats is set; otherwise fall back to the (delta-blind) decay threshold.
    if beats is not None:
        until = beats / max(delta, 1e-9)
        stop_flags = ["--until-time", f"{until:.3f}"]
    else:
        stop_flags = ["--decay-threshold", str(decay)]
    cmd = [
        sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full",
        *MATERIAL_FLAGS,
        "--geometry-file", GEOM, "--cavity-modes-file", MODES,
        "--resolution", str(resolution), *stop_flags,
        "--pump-intensity", str(intensity),
        "--pump1-frequency", f"{f1:.9f}", "--pump2-frequency", f"{f2:.9f}",
        "--output-dir", str(outdir),
    ]
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    with open(outdir / "run.log", "w") as lf:
        r = subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return delta, ("ok" if r.returncode == 0 else f"FAIL({r.returncode})")


def aggregate() -> list:
    rows = []
    for delta in DELTAS:
        p = OUTROOT / f"delta_{delta:.4f}" / "faraday_summary.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        pr = d["probe_rotation_deg"]
        fi = d["probe_stokes_dft"]["tail_weighted"]
        ts = d.get("probe_stokes_total", {}).get("tail_weighted", {})
        pm = d.get("pump_monitor_metrics", {}).get("coherent_reference", {})
        f = d["frequencies_inv_um"]
        rows.append({
            "delta_inv_um": delta,
            "pump_separation_nm": 1e3 * (1.0 / f["pump2"] - 1.0 / f["pump1"]),
            "f1": f["pump1"], "f2": f["pump2"],
            "theta_fwd_coh_deg": pr.get("final_relative_deg"),
            "theta_fwd_incoh_deg": fi.get("theta_relative_deg"),
            "theta_total_deg": ts.get("theta_relative_deg"),
            "dolp_coh": pr.get("coherent_window_estimate", {}).get("dolp"),
            "s0_rel_max": fi.get("S0_rel_max"),
            "p2_over_p1": pm.get("ratio_p2_over_p1", {}).get("tail_weighted"),
        })
    return rows


def write_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(rows: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.array([r["delta_inv_um"] for r in rows])
    coh = np.array([r["theta_fwd_coh_deg"] for r in rows], dtype=float)
    inc = np.array([r["theta_fwd_incoh_deg"] for r in rows], dtype=float)
    tot = np.array([r["theta_total_deg"] for r in rows], dtype=float)
    bal = np.array([r["p2_over_p1"] for r in rows], dtype=float)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.4), sharex=True)

    ax1.plot(d, coh, "o-", color="tab:blue", label="forward-coherent (optimizer metric)")
    ax1.plot(d, inc, "s--", color="tab:green", lw=1.3, label="forward-incoherent")
    ax1.plot(d, tot, "^:", color="tab:red", lw=1.3, label="total-field")
    ax1.axhline(0, color="0.6", lw=0.8)
    ax1.set_ylabel("probe rotation θ (deg, signed)")
    ax1.set_title("1D Δ-scan (SiC L=3.2µm): probe rotation vs pump beat Δ")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="best")

    ax2.plot(d, np.abs(coh), "o-", color="tab:blue", label="|θ| forward-coherent")
    ax2.axvline(SRC_BW, color="tab:purple", ls="--", lw=1.4,
                label=f"source bandwidth ≈ {SRC_BW:.3f} (predicted Δ_opt)")
    ax2.axvline(KAPPA_S_HALF, color="tab:gray", ls=":", lw=1.4,
                label=f"cavity ½-linewidth ≈ {KAPPA_S_HALF:.3f} (naive-CMT Δ_opt)")
    ax2.set_xlabel("Δ = f₁ − f₂  (1/µm)")
    ax2.set_ylabel("|θ| (deg)")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=8, loc="best")

    # annotate pump balance to confirm the runs stay balanced across Δ
    txt = "p2/p1: " + ", ".join(f"{b:.2f}" for b in bal)
    ax2.text(0.01, 0.02, txt, transform=ax2.transAxes, fontsize=7, va="bottom",
             bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def main() -> None:
    global DELTAS, OUTROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intensity", type=float, default=2e11, help="Balanced pump intensity (W/cm^2).")
    ap.add_argument("--resolution", type=int, default=100)
    ap.add_argument("--decay-threshold", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=min(len(DELTAS), os.cpu_count() or 1),
                    help="Parallel runs; defaults to #sweep-points capped at core count (all points at once).")
    ap.add_argument("--aggregate-only", action="store_true", help="Skip runs; just aggregate + plot.")
    ap.add_argument("--deltas", type=str, default=None,
                    help="Comma-separated Δ values (1/µm) to override the default grid.")
    ap.add_argument("--label", type=str, default="sic_L3p2um",
                    help="Output subdir under delta_scan/ (use a new label for a separate scan).")
    ap.add_argument("--beats", type=float, default=None,
                    help="Run length = beats/Δ per point (resolves the sideband beat). "
                         "Overrides --decay-threshold. Use ≳20 to avoid DFT leakage at small Δ.")
    args = ap.parse_args()

    if args.deltas:
        DELTAS = [float(x) for x in args.deltas.split(",") if x.strip()]
    OUTROOT = HERE / "delta_scan" / args.label
    OUTROOT.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        print(f"[delta-scan] {len(DELTAS)} runs, I={args.intensity:.1e}, res={args.resolution}, "
              f"decay={args.decay_threshold}, workers={args.workers}")
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_one, dl, args.intensity, args.resolution, args.decay_threshold,
                              args.beats): dl
                    for dl in DELTAS}
            for fut in as_completed(futs):
                dl, status = fut.result()
                print(f"[delta-scan] Δ={dl:.4f} -> {status}", flush=True)

    rows = aggregate()
    write_csv(rows, OUTROOT / "delta_scan_points.csv")
    if rows:
        plot(rows, OUTROOT / "delta_scan.png")
    print(f"[delta-scan] aggregated {len(rows)}/{len(DELTAS)} points -> {OUTROOT}")
    for r in rows:
        print(f"  Δ={r['delta_inv_um']:.4f}  |θ_coh|={abs(r['theta_fwd_coh_deg'] or float('nan')):.4f}  "
              f"p2/p1={r['p2_over_p1']:.3f}  DoLP={r['dolp_coh']:.4f}  S0rel={r['s0_rel_max']:.3f}")


if __name__ == "__main__":
    main()
