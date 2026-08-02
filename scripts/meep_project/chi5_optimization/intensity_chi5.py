#!/usr/bin/env python
"""Pump-intensity scan of the chi5 all-optical Faraday rotation (1D).

The delay study established that the chi5 rotation is the CARRIER-AVERAGED, two-pump,
pulse-overlap-width envelope in theta: a Gaussian dip of -0.00344 +/- 0.00055 deg (6.2 sigma,
FWHM 125 fs) sitting under a chi3 four-wave-mixing fringe ~8x larger. The identification rests
on the shape; this scan tests the remaining quantitative signature:

    a cascaded chi5 rotation must scale as I_pump^2.

Method: at each pump intensity, repeat a compact delay scan, carrier-average the Stokes vector
over one pump1 optical period (4 sub-samples, which cancels harmonics 1-3), then fit
theta(tau) = c + A exp(-(tau/w)^2). The chi5 observable is |A| -- NOT theta at any single delay,
because a single delay mixes in the chi3 fringe and the single-pump background. Fitting the
envelope is what turned a "consistent with zero" scatter into a 6.2 sigma detection, so the same
estimator is used here.

Grid: 15 delays concentrated on the 125 fs envelope plus wings for the baseline; 4 sub-samples;
pad 500 fs (fixed, so pump1 is the only source that moves for both signs of tau).

  python chi5_optimization/intensity_chi5.py --intensity 2e12 --workers 26 --slice 0/4
  python chi5_optimization/intensity_chi5.py --fit
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
PROBE_INTENSITY = 5e7
LAM_PUMP1_UM = 1.5214626391096977
T_CARRIER_FS = LAM_PUMP1_UM / C0_UM_FS
PAD_FS = 500.0
SUBSAMPLES = 4
TAUS = [0.0, 25.0, -25.0, 50.0, -50.0, 75.0, -75.0, 100.0, -100.0,
        150.0, -150.0, 300.0, -300.0, 400.0, -400.0]
# 1e12 is already covered at higher delay resolution by delay_physics/
INTENSITIES = [5e11, 2e12, 3e12, 4e12]
OUTROOT = HERE / "intensity_chi5"


def tag(I, tau, sub):
    return "I{:.2e}_t{:+07.1f}_s{:d}".format(I, tau, sub)


def jobs(intensities):
    return [(I, t, s) for I in intensities for t in TAUS for s in range(SUBSAMPLES)]


def run_one(I, tau, sub, res, decay):
    out = OUTROOT / tag(I, tau, sub)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "faraday_summary.json").exists():
        tau_eff = tau + sub * T_CARRIER_FS / SUBSAMPLES
        cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *FLAGS,
               "--geometry-file", str(BASE_DIR / "geometry.json"),
               "--cavity-modes-file", str(BASE_DIR / "cavity_modes.json"),
               "--resolution", str(res), "--decay-threshold", str(decay),
               "--pump-intensity", repr(float(I)), "--probe-intensity", str(PROBE_INTENSITY),
               "--pump1-delay-fs", "{:.6f}".format(tau_eff),
               "--delay-pad-fs", "{:.6f}".format(PAD_FS),
               "--probe-azimuth-deg", "45.0", "--probe-ellipticity-deg", "0.0",
               "--pump-imbalance", "1.0", "--output-dir", str(out)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
        with open(out / "run.log", "w") as lf:
            subprocess.run(cmd, cwd=str(MEEP), stdout=lf, stderr=subprocess.STDOUT, env=env)
    return I, tau, sub


def stokes_to_angles(S0, S1, S2, S3):
    theta = 0.5 * np.degrees(np.arctan2(S2, S1)) - 45.0
    theta = (theta + 90.0) % 180.0 - 90.0
    lin = np.sqrt(S1 ** 2 + S2 ** 2)
    return theta, 0.5 * np.degrees(np.arctan2(S3, lin)), lin / max(S0, 1e-30)


def read(I, tau, sub, root=None):
    p = (root or OUTROOT) / tag(I, tau, sub) / "faraday_summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    pi = d.get("probe_pulse_integrated") or {}
    return pi or None


def carrier_average(I, tau, root=None):
    recs = [read(I, tau, s, root) for s in range(SUBSAMPLES)]
    if any(r is None for r in recs):
        return None
    S = {k: float(np.mean([r[k] for r in recs])) for k in ("S0", "S1", "S2", "S3")}
    th, ch, dolp = stokes_to_angles(S["S0"], S["S1"], S["S2"], S["S3"])
    subs = [stokes_to_angles(r["S0"], r["S1"], r["S2"], r["S3"])[0] for r in recs]
    phi = 2 * np.pi * np.arange(SUBSAMPLES) / SUBSAMPLES
    amp = float(2 * abs(np.sum(np.array(subs) * np.exp(-1j * phi))) / SUBSAMPLES)
    return {"theta": th, "chi": ch, "dolp": dolp, "fringe_amp": amp}


def fit_all(intensities):
    from scipy.optimize import curve_fit

    def model(x, A, w, c):
        return c + A * np.exp(-(x / w) ** 2)

    print("  {:>10s} {:>8s} {:>12s} {:>10s} {:>9s} {:>10s} {:>9s} {:>7s}".format(
        "I (W/cm2)", "n_delay", "A (deg)", "sigma_A", "sigma", "w (fs)", "fringe", "DoLP"))
    out = []
    for I in intensities:
        pts = [(t, carrier_average(I, t)) for t in TAUS]
        pts = [(t, r) for t, r in pts if r]
        if len(pts) < 6:
            print("  {:10.2e} {:>8s}  incomplete ({} delays)".format(I, "-", len(pts)))
            continue
        t = np.array([p[0] for p in pts]); th = np.array([p[1]["theta"] for p in pts])
        fr = float(np.mean([p[1]["fringe_amp"] for p in pts if abs(p[0]) < 30]))
        dl = float(np.min([p[1]["dolp"] for p in pts]))
        try:
            p_, cov = curve_fit(model, t, th, p0=[-0.003 * (I / 1e12) ** 2, 80.0, 0.0015],
                                maxfev=20000)
            e = np.sqrt(np.diag(cov))
        except Exception as exc:
            print("  {:10.2e} fit failed: {}".format(I, exc))
            continue
        out.append({"intensity": float(I), "A_deg": float(p_[0]), "A_err": float(e[0]),
                    "w_fs": float(abs(p_[1])), "c_deg": float(p_[2]),
                    "fringe_amp_deg": fr, "dolp_min": dl, "n_delays": len(pts)})
        print("  {:10.2e} {:8d} {:+12.5f} {:10.5f} {:9.1f} {:10.1f} {:9.5f} {:7.4f}".format(
            I, len(pts), p_[0], e[0], abs(p_[0] / e[0]) if e[0] else np.nan, abs(p_[1]), fr, dl))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--res", type=int, default=80)
    ap.add_argument("--decay", default="1e-4")
    ap.add_argument("--intensity", type=float, default=None,
                    help="run only this intensity (default: all in INTENSITIES)")
    ap.add_argument("--slice", default=None, help="i/n interleaved slice, for multi-node fan-out")
    ap.add_argument("--fit", action="store_true", help="fit only, no runs")
    args = ap.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    ints = [args.intensity] if args.intensity else INTENSITIES

    if not args.fit:
        mine = jobs(ints)
        if args.slice:
            i, n = (int(x) for x in args.slice.split("/"))
            mine = [j for k, j in enumerate(mine) if k % n == i]
        print("=== chi5 intensity scan | {} sims | pad {:.0f} fs | {} sub-samples ==="
              .format(len(mine), PAD_FS, SUBSAMPLES))
        t0 = time.time(); done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, I, t, s, args.res, args.decay) for I, t, s in mine]
            for _ in as_completed(futs):
                done += 1
                if done % 10 == 0 or done == len(mine):
                    print("  {}/{} ({:.0f}s)".format(done, len(mine), time.time() - t0), flush=True)

    print("\n--- chi5 envelope amplitude vs pump intensity ---")
    rows = fit_all(INTENSITIES)
    # include the 1e12 point already measured at higher delay resolution
    dp = HERE / "delay_physics" / "delay_physics_result.json"
    if dp.exists():
        rows.append({"intensity": 1e12, "A_deg": -0.00344, "A_err": 0.00055, "w_fs": 75.1,
                     "fringe_amp_deg": 0.02770, "dolp_min": float("nan"), "n_delays": 33,
                     "source": "delay_physics (33 delays)"})
    rows = sorted(rows, key=lambda r: r["intensity"])
    good = [r for r in rows if abs(r["A_deg"]) > 2 * r["A_err"]]
    if len(good) >= 2:
        x = np.log(np.array([r["intensity"] for r in good]))
        y = np.log(np.abs([r["A_deg"] for r in good]))
        sy = np.array([r["A_err"] / abs(r["A_deg"]) for r in good])
        p, cov = np.polyfit(x, y, 1, w=1 / sy, cov=True)
        print("\n  POWER LAW  |A| ~ I^p :  p = {:.2f} +/- {:.2f}   (chi5 => 2, chi3 => 1)"
              .format(p[0], float(np.sqrt(cov[0, 0]))))
        pf = [r for r in rows if r.get("fringe_amp_deg")]
        if len(pf) >= 2:
            xf = np.log([r["intensity"] for r in pf]); yf = np.log([r["fringe_amp_deg"] for r in pf])
            print("  for contrast, the chi3 FRINGE scales as I^{:.2f}".format(np.polyfit(xf, yf, 1)[0]))
    json.dump({"rows": rows}, open(OUTROOT / "intensity_chi5_result.json", "w"), indent=2)
    print("\n-> {}".format(OUTROOT / "intensity_chi5_result.json"))


if __name__ == "__main__":
    main()
