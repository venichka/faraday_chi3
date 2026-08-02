#!/usr/bin/env python
"""Physically-consistent pump-probe delay study on the SiN best_absolute cavity.

This is a rebuild of `delay_scan.py` after a from-scratch audit. Three things changed, all
of them for physics reasons; the old driver is left untouched for reproducibility.

1. CONSISTENT DELAY CONVENTION.  In the lab only pump1 moves; pump2 and the probe stay
   locked. The old driver kept the relative TIMING but, for tau<0, shifted pump2+probe
   instead of pump1 -- which gives them phases exp(i w2 |tau|), exp(i ws |tau|) rather than
   pump1 picking up exp(-i w1 |tau|). The two halves of the scan were therefore different
   experiments. Here every run uses a FIXED common start-time pad (`--delay-pad-fs`), so
   pump1 is the only source whose timing changes, for both signs of tau. A common offset is
   harmless: it shifts all fields in time, and Stokes parameters are built from
   same-frequency products, so a global phase cancels.

2. CARRIER AVERAGING.  Delaying pump1 by tau multiplies its field by exp(i w1 tau). Any
   observable is a sum of products of fields; a term with n powers of E1 and m of E1* carries
   exp(i(n-m) w1 tau). So:
       n == m  ->  no carrier phase: the genuine chi3 / chi5 response (envelope overlap only)
       n-m = +-1 -> a fringe at T1 = 5.075 fs
       n-m = +-2 -> a fringe at T1/2
   Averaging the Stokes vector uniformly over ONE pump1 optical period therefore KILLS every
   n != m term and leaves exactly the physical rotation. That is also what an experiment
   whose delay line is not interferometrically phase-stable measures. `--subsamples N`
   averages N runs spaced by T1/N (N=4 removes harmonics 1,2,3).
   NOTE: sampling at whole multiples of T1 (what the old stage 1 did) does NOT average --
   it freezes the fringe at one phase, i.e. it measures the fringe maximum.

3. BOTH ESTIMATORS REPORTED.  `legacy` = tail/final-window polarization azimuth (the
   settled-state number that reproduces the published 0.137 deg) and `pulse` =
   pulse-energy-integrated Stokes (the balanced-detector observable). Averaging is done on
   the Stokes components for `pulse` (the correct incoherent average) and on the angle for
   `legacy` (which has no detector analogue anyway).

Delay grid is set by the cavity, not by convenience: the pump-band mode spacings of this
geometry are 152.8 / 234.6 / 92.6 fs, so the step must resolve ~92 fs (25 fs = 3.7 pts) and
the span must outlast the ~35-40 fs energy ring-down (+-400 fs is ~10 lifetimes).

  python chi5_optimization/delay_physics.py --workers 26 --slice 0/8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
T_CARRIER_FS = LAM_PUMP1_UM / C0_UM_FS          # 5.0751 fs

TAU_MAX_FS = 400.0
TAU_STEP_FINE = 25.0     # chi0 = 0 : resolves the 92.6 fs mode beat with 3.7 points
TAU_STEP_COARSE = 50.0   # ellipticity families
PAD_FS = 500.0           # fixed, > TAU_MAX_FS, identical for every run in the study
ELLIPTICITIES = [0.0, 5.0, 10.0, 20.0]


def job_tag(tau_fs, sub, ellip):
    return "t{:+08.2f}_s{:d}_el{:04.1f}".format(tau_fs, sub, ellip)


def jobs_all(subsamples):
    """(tau, subsample index, ellipticity) for the whole study."""
    fine = np.round(np.arange(-TAU_MAX_FS, TAU_MAX_FS + 1e-9, TAU_STEP_FINE), 4)
    coarse = np.round(np.arange(-TAU_MAX_FS, TAU_MAX_FS + 1e-9, TAU_STEP_COARSE), 4)
    out = []
    for el in ELLIPTICITIES:
        taus = fine if el == 0.0 else coarse
        for t in taus:
            for s in range(subsamples):
                out.append((float(t), int(s), float(el)))
    return out


def run_one(tau_fs, sub, ellip, subsamples, res, decay, outroot):
    """One 1D-FDTD run. The sub-sample offset is added to tau, so the carrier phase
    w1*(tau + s*T1/N) is swept over one period while the envelope barely moves
    (T1 = 5.08 fs vs a ~120 fs pulse)."""
    out = Path(outroot) / job_tag(tau_fs, sub, ellip)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "faraday_summary.json").exists():
        tau_eff = tau_fs + sub * T_CARRIER_FS / subsamples
        cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *FLAGS,
               "--geometry-file", str(BASE_DIR / "geometry.json"),
               "--cavity-modes-file", str(BASE_DIR / "cavity_modes.json"),
               "--resolution", str(res), "--decay-threshold", str(decay),
               "--pump-intensity", str(PUMP_INTENSITY), "--probe-intensity", str(PROBE_INTENSITY),
               "--pump1-delay-fs", "{:.6f}".format(tau_eff),
               "--delay-pad-fs", "{:.6f}".format(PAD_FS),
               "--probe-azimuth-deg", "45.0",
               "--probe-ellipticity-deg", "{:.6f}".format(ellip),
               "--pump-imbalance", "1.0",
               "--output-dir", str(out)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
        with open(out / "run.log", "w") as lf:
            subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return tau_fs, sub, ellip


def read_one(tau_fs, sub, ellip, outroot):
    p = Path(outroot) / job_tag(tau_fs, sub, ellip) / "faraday_summary.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    pi = d.get("probe_pulse_integrated") or {}
    if not pi:
        return None
    return {"S0": pi.get("S0"), "S1": pi.get("S1"), "S2": pi.get("S2"), "S3": pi.get("S3"),
            "legacy": (d.get("probe_rotation_deg") or {}).get("final_relative_deg"),
            "dolp": pi.get("dolp")}


def stokes_to_angles(S0, S1, S2, S3):
    """theta relative to the 45 deg launch azimuth, ellipticity chi, DoLP, and V-H."""
    theta = 0.5 * np.degrees(np.arctan2(S2, S1)) - 45.0
    theta = (theta + 90.0) % 180.0 - 90.0
    lin = np.sqrt(S1 ** 2 + S2 ** 2)
    chi = 0.5 * np.degrees(np.arctan2(S3, lin))
    return theta, chi, lin / np.maximum(S0, 1e-30), -S1 / np.maximum(S0, 1e-30)


def aggregate(outroot, subsamples):
    """Carrier-average the Stokes vector at each (tau, ellipticity)."""
    res = {}
    for el in ELLIPTICITIES:
        taus = sorted({t for t, s, e in jobs_all(subsamples) if e == el})
        rows = []
        for t in taus:
            recs = [read_one(t, s, el, outroot) for s in range(subsamples)]
            recs = [r for r in recs if r]
            if len(recs) < subsamples:
                continue
            S = {k: float(np.mean([r[k] for r in recs])) for k in ("S0", "S1", "S2", "S3")}
            th, ch, dolp, vmh = stokes_to_angles(S["S0"], S["S1"], S["S2"], S["S3"])
            # Carrier-fringe amplitude. The sub-samples sit at uniform phases
            # phi_k = 2 pi k / N over one T1, so the FUNDAMENTAL is obtained exactly by a
            # discrete Fourier projection -- unbiased, unlike peak-to-peak, which for N=4
            # ranges over sqrt(2)A..2A depending on where the samples fall relative to the
            # extrema (a ~30% jitter that swamps the weak large-|tau| fringe).
            per_sub = [stokes_to_angles(r["S0"], r["S1"], r["S2"], r["S3"]) for r in recs]
            th_sub = np.array([p[0] for p in per_sub])
            vmh_sub = np.array([p[3] for p in per_sub])
            leg = np.array([r["legacy"] for r in recs], dtype=float)
            phi = 2.0 * np.pi * np.arange(subsamples) / subsamples

            def fundamental(y):
                return float(2.0 * np.abs(np.sum(np.asarray(y, float) * np.exp(-1j * phi)))
                             / subsamples)
            rows.append({"tau_fs": float(t),
                         "theta_pulse_deg": float(th), "chi_pulse_deg": float(ch),
                         "vmh_norm": float(vmh), "dolp": float(dolp),
                         "theta_pulse_fringe_ptp_deg": float(np.ptp(th_sub)),
                         "vmh_fringe_ptp": float(np.ptp(vmh_sub)),
                         "theta_legacy_deg": float(np.mean(leg)),
                         "theta_legacy_fringe_ptp_deg": float(np.ptp(leg)),
                         # unbiased fundamental amplitudes (preferred over the ptp above)
                         "theta_pulse_fringe_amp_deg": fundamental(th_sub),
                         "vmh_fringe_amp": fundamental(vmh_sub),
                         "theta_legacy_fringe_amp_deg": fundamental(leg),
                         "S0": S["S0"], "S1": S["S1"], "S2": S["S2"], "S3": S["S3"]})
        res["el{:g}".format(el)] = rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--res", type=int, default=80)
    ap.add_argument("--decay", default="1e-4")
    ap.add_argument("--subsamples", type=int, default=4,
                    help="carrier sub-samples per delay (uniform over one T1)")
    ap.add_argument("--slice", default=None,
                    help="i/n -- run only slice i of n, so the study can be farmed across "
                         "nodes. Output dirs are disjoint per job and the runner is "
                         "idempotent, so slices may overlap and be re-run safely.")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    outroot = Path(args.out) if args.out else HERE / "delay_physics"
    outroot.mkdir(parents=True, exist_ok=True)
    jobs = jobs_all(args.subsamples)

    if not args.aggregate_only:
        mine = jobs
        if args.slice:
            i, n = (int(x) for x in args.slice.split("/"))
            mine = [j for k, j in enumerate(jobs) if k % n == i]
        print("=== delay_physics | SiN best_absolute | pad {:.0f} fs | {} carrier sub-samples ==="
              .format(PAD_FS, args.subsamples))
        print("  tau in [{:+.0f}, {:+.0f}] fs; step {:.0f} fs (chi0=0) / {:.0f} fs (ellipticity)"
              .format(-TAU_MAX_FS, TAU_MAX_FS, TAU_STEP_FINE, TAU_STEP_COARSE))
        print("  {} of {} sims in this slice, {} workers".format(len(mine), len(jobs), args.workers))
        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, t, s, e, args.subsamples, args.res, args.decay, outroot)
                    for t, s, e in mine]
            for _ in as_completed(futs):
                done += 1
                if done % 10 == 0 or done == len(mine):
                    print("  {}/{} ({:.0f}s)".format(done, len(mine), time.time() - t0), flush=True)
        print("  slice done in {:.0f}s".format(time.time() - t0))

    have = sum(1 for t, s, e in jobs
               if (outroot / job_tag(t, s, e) / "faraday_summary.json").exists())
    print("\n  {}/{} sims present".format(have, len(jobs)))
    res = {"design": "SiN_optimizations/best_absolute", "pad_fs": PAD_FS,
           "subsamples": args.subsamples, "carrier_period_fs": T_CARRIER_FS,
           "pump_intensity_w_cm2": PUMP_INTENSITY, "probe_intensity_w_cm2": PROBE_INTENSITY,
           "resolution": args.res, "decay_threshold": args.decay,
           "families": aggregate(outroot, args.subsamples)}
    path = outroot / "delay_physics_result.json"
    json.dump(res, open(path, "w"), indent=2)
    for k, rows in res["families"].items():
        if rows:
            i0 = int(np.argmin([abs(r["tau_fs"]) for r in rows]))
            print("  {:6s} n={:3d}  theta_pulse(0)={:+.5f} deg  theta_legacy(0)={:+.5f} deg  "
                  "chi(0)={:+.4f} deg".format(k, len(rows), rows[i0]["theta_pulse_deg"],
                                              rows[i0]["theta_legacy_deg"], rows[i0]["chi_pulse_deg"]))
    print("\n-> {}".format(path))


if __name__ == "__main__":
    main()
