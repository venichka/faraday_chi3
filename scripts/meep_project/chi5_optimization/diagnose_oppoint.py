#!/usr/bin/env python
"""Diagnostic: 1D-FDTD operating-point landscape for a FIXED geometry (SiN best_absolute).
Probe fixed (from the modes file); sweep pump (center, Delta). Reveals what actually maximizes
|theta| -> tests whether the proxy's FWM-sum criterion (2*center ~ f_probe) is the right one.
Run under meep-mpi on the head node (sims are 1D single-core, run in parallel).
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
GEOM = MEEP / "SiN_optimizations/best_absolute/geometry.json"
MODES = MEEP / "SiN_optimizations/best_absolute/cavity_modes.json"   # sets the probe (~803 nm)
FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
         "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
INTENSITY = 1e12
RES = 80
DECAY = "1e-4"
F_S = 1.2461        # probe frequency (FWM-sum reference: octave match is 2*center == F_S)
OUT = HERE / "diagnose" / "oppoint_sin"

# grid of pump (center, Delta); plus the ORIGINAL best_absolute operating point as a reference
CENTERS = [0.605, 0.620, 0.635, 0.650, 0.665]
DELTAS = [0.015, 0.025, 0.040]
EXTRA = [(0.64635, 0.02210)]   # original pumps 1521/1574 nm (f 0.6574/0.6353)


def run_one(center, delta, tag):
    f1, f2 = center + 0.5 * delta, center - 0.5 * delta
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "faraday_summary.json").exists():
        return tag, "cached"
    cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *FLAGS,
           "--geometry-file", str(GEOM), "--cavity-modes-file", str(MODES),
           "--resolution", str(RES), "--decay-threshold", DECAY,
           "--pump-intensity", str(INTENSITY),
           "--pump1-frequency", "{:.9f}".format(f1), "--pump2-frequency", "{:.9f}".format(f2),
           "--output-dir", str(out)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    with open(out / "run.log", "w") as lf:
        r = subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return tag, ("ok" if r.returncode == 0 else "FAIL")


def read(tag):
    p = OUT / tag / "faraday_summary.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    pr = d["probe_rotation_deg"]
    cr = d.get("pump_monitor_metrics", {}).get("coherent_reference", {})
    return {"theta": pr["final_relative_deg"],
            "dolp": pr.get("coherent_window_estimate", {}).get("dolp", float("nan")),
            "p2p1": cr.get("ratio_p2_over_p1", {}).get("tail_weighted", float("nan")),
            "pbuild": cr.get("tail_weighted_abs", float("nan"))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, d, "c{:.3f}_d{:.3f}".format(c, d)) for c in CENTERS for d in DELTAS]
    jobs += [(c, d, "orig_c{:.3f}_d{:.3f}".format(c, d)) for c, d in EXTRA]
    print("running {} operating-point sims (parallel)...".format(len(jobs)), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for f in as_completed([ex.submit(run_one, *j) for j in jobs]):
            print("  ", f.result(), flush=True)
    print("done in {:.0f}s".format(time.time() - t0))

    print("\n=== SiN best_absolute operating-point landscape (probe fixed ~803nm, I=1e12) ===")
    print("{:>7s} {:>7s} {:>9s} {:>9s} {:>9s} {:>7s} {:>7s}".format(
        "center", "Delta", "2*c-F_S", "|theta|", "pbuild", "DoLP", "p2/p1"))
    rows = []
    for c, d, tag in jobs:
        r = read(tag)
        if r is None:
            continue
        rows.append((c, d, abs(r["theta"]), r))
        print("{:7.3f} {:7.3f} {:9.4f} {:9.4f} {:9.3f} {:7.3f} {:7.3f}".format(
            c, d, 2 * c - F_S, abs(r["theta"]), r["pbuild"], r["dolp"], r["p2p1"]))
    if rows:
        best = max(rows, key=lambda x: x[2])
        print("\nMAX |theta| = {:.4f} at center={:.3f} Delta={:.3f} (2c-F_S={:.4f})".format(
            best[2], best[0], best[1], 2 * best[0] - F_S))


if __name__ == "__main__":
    main()
