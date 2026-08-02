#!/usr/bin/env python
"""3D pump-probe delay scan on the SiN best_absolute cavity (MPI, one sim per invocation).

Why 3D at all: in 1D this design gives 0.137 deg but in 3D 1.991 deg -- a 14.5x enhancement
(pipeline_cluster_20260325_114410, 24 ranks, res 30, 7604 s). The delay study so far is 1D,
so the open question is whether the delay DEPENDENCE (and that enhancement) survives in 3D.

Cost control -- two deliberate choices:
  * tau >= 0 ONLY. For non-negative delays the "common pad" and the legacy convention are
    IDENTICAL (pump1 starts at tau, pump2+probe at 0), so no start-time pad is needed and
    every run is as short as it can be. The pad exists only to make tau<0 physical, and a
    negative-delay 3D branch would cost ~35% more per run for the half of the axis we can
    already characterise in 1D.
  * 4 carrier sub-samples per delay (T1/4 apart). Delaying pump1 multiplies its field by
    exp(i w1 tau), so a term with n powers of E1 and m of E1* carries exp(i(n-m) w1 tau);
    averaging the Stokes vector over one T1 kills every n != m term and leaves the physical
    chi3/chi5 response. N=4 removes harmonics 1,2,3 (the measured fringe has a fundamental
    plus a ~26% second harmonic).

Same geometry, modes, materials and n2 as the 1D study and as the 1.991 deg reference run,
so 3D/1D ratios are apples-to-apples.

  python chi5_optimization/delay_3d.py --index 7          # run job 7 (under mpirun)
  python chi5_optimization/delay_3d.py --aggregate        # collect results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
C0_UM_FS = 0.299792458

BASE_DIR = MEEP / "SiN_optimizations" / "best_absolute"
FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
         "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
PUMP_INTENSITY = 1e12
PROBE_INTENSITY = 5e7
LAM_PUMP1_UM = 1.5214626391096977
T_CARRIER_FS = LAM_PUMP1_UM / C0_UM_FS

TAUS_FS = [0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0]
SUBSAMPLES = 4
OUTROOT = HERE / "delay_3d"


def jobs():
    return [(t, s) for t in TAUS_FS for s in range(SUBSAMPLES)]


def tag(tau, sub):
    return "t{:+07.1f}_s{:d}".format(tau, sub)


def print_args(k, res, decay, dim):
    """Emit the faraday_meep_fp_circ.py argument list for job k, or nothing if it is done.

    The 3D runs are MPI. Only ONE program may run under mpirun: if this driver spawned the
    simulation as a subprocess, each of the 24 ranks would fork its own non-MPI child, whose
    `import meep` then calls MPI_Init outside the job and dies with
    "PMI_Get_appnum returned -1". So the sbatch runs faraday_meep_fp_circ.py directly under
    mpirun and uses this (serial, rank-free) call only to build its arguments.
    """
    tau, sub = jobs()[k]
    out = OUTROOT / tag(tau, sub)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "faraday_summary.json").exists():
        return  # already done -> empty output -> the sbatch skips it
    tau_eff = tau + sub * T_CARRIER_FS / SUBSAMPLES
    args = ["--dim", str(dim), "--mode", "full", *FLAGS,
            "--geometry-file", str(BASE_DIR / "geometry.json"),
            "--cavity-modes-file", str(BASE_DIR / "cavity_modes.json"),
            "--resolution", str(res), "--decay-threshold", str(decay),
            "--pump-intensity", str(PUMP_INTENSITY), "--probe-intensity", str(PROBE_INTENSITY),
            "--pump1-delay-fs", "{:.6f}".format(tau_eff),
            "--probe-azimuth-deg", "45.0", "--probe-ellipticity-deg", "0.0",
            "--pump-imbalance", "1.0", "--output-dir", str(out)]
    print(" ".join(args))


def stokes_to_angles(S0, S1, S2, S3):
    theta = 0.5 * np.degrees(np.arctan2(S2, S1)) - 45.0
    theta = (theta + 90.0) % 180.0 - 90.0
    lin = np.sqrt(S1 ** 2 + S2 ** 2)
    return theta, 0.5 * np.degrees(np.arctan2(S3, lin)), lin / max(S0, 1e-30), -S1 / max(S0, 1e-30)


def aggregate():
    rows = []
    for tau in TAUS_FS:
        recs = []
        for s in range(SUBSAMPLES):
            p = OUTROOT / tag(tau, s) / "faraday_summary.json"
            if not p.exists():
                continue
            d = json.load(open(p))
            pi = d.get("probe_pulse_integrated") or {}
            if pi:
                recs.append((pi, (d.get("probe_rotation_deg") or {}).get("final_relative_deg")))
        if len(recs) < SUBSAMPLES:
            print("  tau {:+.1f}: only {}/{} sub-samples -- skipped".format(tau, len(recs), SUBSAMPLES))
            continue
        S = {k: float(np.mean([r[0][k] for r in recs])) for k in ("S0", "S1", "S2", "S3")}
        th, ch, dolp, vmh = stokes_to_angles(S["S0"], S["S1"], S["S2"], S["S3"])
        leg = np.array([r[1] for r in recs], dtype=float)
        sub_th = [stokes_to_angles(r[0]["S0"], r[0]["S1"], r[0]["S2"], r[0]["S3"])[0] for r in recs]
        rows.append({"tau_fs": tau, "theta_pulse_deg": th, "chi_pulse_deg": ch,
                     "vmh_norm": vmh, "dolp": dolp,
                     "theta_pulse_fringe_ptp_deg": float(np.ptp(sub_th)),
                     "theta_legacy_deg": float(np.mean(leg)),
                     "theta_legacy_fringe_ptp_deg": float(np.ptp(leg))})
    out = {"design": "SiN_optimizations/best_absolute", "dim": 3,
           "subsamples": SUBSAMPLES, "carrier_period_fs": T_CARRIER_FS,
           "pump_intensity_w_cm2": PUMP_INTENSITY, "rows": rows}
    json.dump(out, open(OUTROOT / "delay_3d_result.json", "w"), indent=2)
    print("\n  {:>9s} {:>14s} {:>14s} {:>10s} {:>9s}".format(
        "tau (fs)", "theta_pulse", "theta_legacy", "chi", "DoLP"))
    for r in rows:
        print("  {:+9.1f} {:+14.6f} {:+14.6f} {:+10.5f} {:9.5f}".format(
            r["tau_fs"], r["theta_pulse_deg"], r["theta_legacy_deg"], r["chi_pulse_deg"], r["dolp"]))
    print("\n-> {}".format(OUTROOT / "delay_3d_result.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--print-args", action="store_true",
                    help="emit the simulator arguments for --index (empty if already done)")
    ap.add_argument("--res", type=int, default=30)
    ap.add_argument("--decay", default="1e-4")
    ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    if args.list:
        for k, (t, s) in enumerate(jobs()):
            print(k, tag(t, s))
        return
    if args.aggregate:
        aggregate()
        return
    if args.index is None:
        raise SystemExit("pass --index k (or --aggregate / --list)")
    print_args(args.index, args.res, args.decay, args.dim)


if __name__ == "__main__":
    main()
