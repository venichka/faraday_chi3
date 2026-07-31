#!/usr/bin/env python
"""CLEAN L-sweep: evaluate each geometry at its FDTD-OPTIMAL operating point, to remove the
operating-point realization noise (TMM-freq imprecision) that corrupted the earlier calibration.
For each cavity L (best_absolute mirror fixed, N=3), sweep the pump (center x Delta) finely in
1D FDTD with the probe fixed at its TMM mode, take MAX |theta| = the geometry's true capability.
Then |theta|_max(L) tells us whether the proxy's ~1/L^2 scaling is real or was pure noise.

decay-threshold 1e-5 (well-decayed). Designed for a free 96-core node (90 parallel 1D sims).
"""
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
sys.path.insert(0, str(HERE))
import tmm          # noqa: E402
import optimize as O  # noqa: E402

BASE = json.load(open(MEEP / "SiN_optimizations/best_absolute/geometry.json"))
HI = BASE["cavity"]["mat"]
T_HI = float(np.mean([l["thk_um"] for l in BASE["mirrors"]["left"] if l["mat"] == HI]))
T_LO = float(np.mean([l["thk_um"] for l in BASE["mirrors"]["left"] if l["mat"] != HI]))

FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
         "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
INTENSITY = 1e12
RES = 80
DECAY = "1e-5"
N_PAIRS = 3
LS = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
CENTERS = np.round(np.linspace(0.600, 0.665, 10), 5)
DELTAS = [0.015, 0.022, 0.030]
WORKERS = 90
OUT = HERE / "clean_lsweep"


def probe_freq(geom):
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    ms = tmm.find_modes_in_band(layers, idx, 1.0 / 0.810, 1.0 / 0.790)
    if not ms:
        return None
    return max(ms, key=lambda m: m["Q"])["freq"]


def run_one(L, fprobe, geom, center, delta):
    f1, f2 = center + 0.5 * delta, center - 0.5 * delta
    tag = "L{:.2f}/c{:.4f}_d{:.3f}".format(L, center, delta)
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "faraday_summary.json").exists():
        return tag, "cached"
    json.dump(geom, open(out / "geometry.json", "w"))
    modes = {"probe": {"frequency": fprobe, "lambda_um": 1.0 / fprobe},
             "pump1": {"frequency": f1, "lambda_um": 1.0 / f1},
             "pump2": {"frequency": f2, "lambda_um": 1.0 / f2},
             "sidebands": {"frequency_plus": fprobe + delta, "frequency_minus": fprobe - delta,
                           "delta_frequency": delta, "pump_separation_um": abs(1.0 / f2 - 1.0 / f1)}}
    json.dump(modes, open(out / "cavity_modes.json", "w"))
    cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *FLAGS,
           "--geometry-file", str(out / "geometry.json"), "--cavity-modes-file", str(out / "cavity_modes.json"),
           "--resolution", str(RES), "--decay-threshold", DECAY, "--pump-intensity", str(INTENSITY),
           "--pump1-frequency", "{:.9f}".format(f1), "--pump2-frequency", "{:.9f}".format(f2),
           "--output-dir", str(out)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    with open(out / "run.log", "w") as lf:
        r = subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return tag, ("ok" if r.returncode == 0 else "FAIL")


def theta(tag):
    p = OUT / tag / "faraday_summary.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
        return abs(d["probe_rotation_deg"]["final_relative_deg"]), \
            d["probe_rotation_deg"].get("coherent_window_estimate", {}).get("dolp", float("nan"))
    except Exception:
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geoms, probes = {}, {}
    jobs = []
    for L in LS:
        g = O.build_geometry(BASE, N_PAIRS, T_HI, T_LO, L)
        fp = probe_freq(g)
        if fp is None:
            print("L={}: no probe mode".format(L)); continue
        geoms[L], probes[L] = g, fp
        for c in CENTERS:
            for d in DELTAS:
                jobs.append((L, fp, g, float(c), d))
    print("running {} FDTD sims (decay {}, {} workers)...".format(len(jobs), DECAY, WORKERS), flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fu in as_completed([ex.submit(run_one, *j) for j in jobs]):
            done += 1
            if done % 30 == 0:
                print("  {}/{} ({:.0f}s)".format(done, len(jobs), time.time() - t0), flush=True)
    print("all sims in {:.0f}s\n".format(time.time() - t0))

    print("=== clean L-sweep: MAX |theta| over operating points per L (probe ~800nm) ===")
    print("{:>5s} {:>9s} {:>8s} {:>8s} {:>6s} {:>10s}".format("L", "max|th|", "@center", "@Delta", "DoLP", "probe_nm"))
    summary = []
    for L in LS:
        if L not in geoms:
            continue
        best = None
        for c in CENTERS:
            for d in DELTAS:
                r = theta("L{:.2f}/c{:.4f}_d{:.3f}".format(L, c, d))
                if r is None:
                    continue
                if best is None or r[0] > best[0]:
                    best = (r[0], c, d, r[1])
        if best is None:
            print("{:5.2f}  (no results)".format(L)); continue
        summary.append({"L": L, "max_theta": best[0], "center": best[1], "delta": best[2],
                        "dolp": best[3], "probe_nm": 1000.0 / probes[L]})
        print("{:5.2f} {:9.4f} {:8.4f} {:8.3f} {:6.3f} {:10.1f}".format(
            L, best[0], best[1], best[2], best[3], 1000.0 / probes[L]))
    json.dump(summary, open(OUT / "lsweep_summary.json", "w"), indent=2)
    if len(summary) >= 3:
        Ls = np.array([s["L"] for s in summary]); th = np.array([s["max_theta"] for s in summary])
        slope = np.polyfit(np.log(Ls), np.log(th), 1)[0]
        print("\nmax|theta| ~ L^{:.2f}   (proxy predicts ~ L^-2; flat=0)".format(slope))
    print("-> {}".format(OUT / "lsweep_summary.json"))


if __name__ == "__main__":
    main()
