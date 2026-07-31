#!/usr/bin/env python
"""1D resolution-convergence sweep of cand07 at the cross-check operating point.

The 3D cross-check runs at res 30 (the practical 3D limit), but the res-30 1D companion gave
|theta|=7.99 deg vs the refine baseline's 2.39 deg at res 80 -- a 3.3x discrepancy. The probe
(lambda=0.8um in n=2.07 -> ~11 px/wavelength at res 30) is under-resolved. This sweep maps |theta|
and DoLP vs resolution so we know (a) the converged value, (b) what the res-30 3D number means, and
(c) whether courant 0.25 vs 0.5 matters (refine used 0.5; the cross-check uses 0.25).

Reuses the prepped cross3d/cand07 inputs (geometry + cavity_modes at center 0.69013, Delta 0.006).
1D Meep is single-core, so we run the resolutions in parallel (one core each).

  python chi5_optimization/resconv.py --workers 8
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

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent

SIN_FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
             "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
INTENSITY = 1e12
F1, F2 = 0.693130, 0.687130   # cand07 center 0.69013 +- Delta/2 (Delta=0.006)
RES = [30, 40, 50, 60, 80, 100, 120]


def _ilist(s):
    return [int(x) for x in s.split(",")]


def run_one(cand: Path, res: int, courant: str, decay: str, outroot: Path):
    out = outroot / "res{}_c{}".format(res, courant.replace(".", "p"))
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "faraday_summary.json").exists():
        cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *SIN_FLAGS,
               "--geometry-file", str(cand / "geometry.json"),
               "--cavity-modes-file", str(cand / "cavity_modes.json"),
               "--resolution", str(res), "--courant", courant, "--decay-threshold", decay,
               "--pump-intensity", str(INTENSITY),
               "--pump1-frequency", "{:.9f}".format(F1), "--pump2-frequency", "{:.9f}".format(F2),
               "--output-dir", str(out)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
        with open(out / "run.log", "w") as lf:
            subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return res, courant


def theta(out: Path):
    p = out / "faraday_summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))["probe_rotation_deg"]
    return abs(d["final_relative_deg"]), d.get("coherent_window_estimate", {}).get("dolp", float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--res", type=_ilist, default=RES)
    ap.add_argument("--decay", default="1e-4")
    ap.add_argument("--out", default=str(HERE / "cross3d" / "cand07" / "resconv"))
    args = ap.parse_args()
    cand = HERE / "cross3d" / "cand07"
    outroot = Path(args.out)
    outroot.mkdir(parents=True, exist_ok=True)

    # courant 0.25 across the sweep (isolates spatial resolution) + one res-80 c0.5 to check courant.
    jobs = [(cand, r, "0.25", args.decay, outroot) for r in args.res]
    jobs.append((cand, 80, "0.5", args.decay, outroot))
    print("=== cand07 1D resolution convergence | res={} (c0.25) + res80 c0.5 ===".format(args.res), flush=True)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for _ in as_completed([ex.submit(run_one, *j) for j in jobs]):
            done += 1
            print("  {}/{} ({:.0f}s)".format(done, len(jobs), time.time() - t0), flush=True)
    print("sims in {:.0f}s".format(time.time() - t0), flush=True)

    print("\n{:>6s} {:>8s} {:>9s} {:>7s}".format("res", "courant", "|theta|", "DoLP"))
    grid = []
    for (cand_, r, c, _, _) in jobs:
        out = outroot / "res{}_c{}".format(r, c.replace(".", "p"))
        t = theta(out)
        if t:
            grid.append({"res": r, "courant": float(c), "theta": t[0], "dolp": t[1]})
            print("{:6d} {:>8s} {:9.4f} {:7.3f}".format(r, c, t[0], t[1]))
    json.dump({"operating_point": {"center": 0.69013, "delta": 0.006, "f1": F1, "f2": F2},
               "grid": grid}, open(outroot / "resconv_result.json", "w"), indent=2)
    print("-> {}".format(outroot / "resconv_result.json"))


if __name__ == "__main__":
    main()
