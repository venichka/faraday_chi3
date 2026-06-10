#!/usr/bin/env python
"""Phase-1 validation gate, stage 1: 1D FDTD ground truth.

Runs faraday_meep_fp_circ (1D, three-theta) on the BASELINE geometry + the top chi5-refined
candidates, each at its proxy-selected operating point, and compares the real |theta|. Same
(perturbative) pump intensity per material so candidate-vs-baseline is apples-to-apples.
Run under meep-mpi.  python chi5_optimization/validate_phase1.py [--cands 2]
"""
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

MAT = {
    "sin": {"flags": ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
                      "--fit-poles", "2", "--fit-window", "600", "2000",
                      "--high-index-material", "sin"],
            "intensity": 1e12, "base_dir": "SiN_optimizations/best_absolute"},
    "sic": {"flags": ["--materials", "fit", "--sin-fit", "sic.csv", "--sio2-fit", "sio2.csv",
                      "--fit-poles", "3", "--fit-window", "600", "2000",
                      "--high-index-material", "sic"],
            "intensity": 2e11, "base_dir": "SiC_optimizations/sic_L3p2um"},
}
RES = 80
DECAY = "1e-4"


def vdir(mat, label):
    d = HERE / "phase1" / mat / "validate" / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_one(mat, label, geom, modes, f1, f2):
    cfg = MAT[mat]
    out = vdir(mat, label)
    cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *cfg["flags"],
           "--geometry-file", str(geom), "--cavity-modes-file", str(modes),
           "--resolution", str(RES), "--decay-threshold", DECAY,
           "--pump-intensity", str(cfg["intensity"]),
           "--pump1-frequency", "{:.9f}".format(f1), "--pump2-frequency", "{:.9f}".format(f2),
           "--output-dir", str(out)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    with open(out / "run.log", "w") as lf:
        r = subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return mat, label, r.returncode


def theta(mat, label):
    p = vdir(mat, label) / "faraday_summary.json"
    if not p.exists():
        return None
    return json.load(open(p))["probe_rotation_deg"]["final_relative_deg"]


def baseline_modes(mat):
    summ = json.load(open(HERE / "phase1" / mat / "phase1_summary.json"))
    bf = summ["baseline"]["freqs"]
    d = vdir(mat, "baseline")
    json.dump({"probe": {"frequency": bf["probe"], "lambda_um": 1.0 / bf["probe"]},
               "pump1": {"frequency": bf["pump1"], "lambda_um": 1.0 / bf["pump1"]},
               "pump2": {"frequency": bf["pump2"], "lambda_um": 1.0 / bf["pump2"]},
               "sidebands": {"frequency_plus": bf["sb_plus"], "frequency_minus": bf["sb_minus"],
                             "delta_frequency": bf["pump1"] - bf["pump2"],
                             "pump_separation_um": abs(1.0 / bf["pump2"] - 1.0 / bf["pump1"])}},
              open(d / "cavity_modes.json", "w"), indent=2)
    return bf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=int, default=2)
    args = ap.parse_args()

    jobs = []
    for mat in ("sin", "sic"):
        bf = baseline_modes(mat)
        jobs.append((mat, "baseline", MEEP / MAT[mat]["base_dir"] / "geometry.json",
                     vdir(mat, "baseline") / "cavity_modes.json", bf["pump1"], bf["pump2"]))
        for k in range(args.cands):
            cd = HERE / "phase1" / mat / "cand{:02d}".format(k)
            cm = json.load(open(cd / "cavity_modes.json"))
            jobs.append((mat, "cand{:02d}".format(k), cd / "geometry.json",
                         cd / "cavity_modes.json", cm["pump1"]["frequency"], cm["pump2"]["frequency"]))

    print("running {} 1D FDTD validations (res {}, parallel)...".format(len(jobs), RES), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for f in as_completed([ex.submit(run_one, *j) for j in jobs]):
            print("  done:", f.result(), flush=True)
    print("all sims in {:.0f}s".format(time.time() - t0))

    print("\n=== Phase-1 1D FDTD validation (final_relative_deg) ===")
    for mat in ("sin", "sic"):
        bt = theta(mat, "baseline")
        if bt is None:
            print("{}: baseline FAILED".format(mat)); continue
        print("{}: baseline |theta| = {:.4f} deg   (I={:.0e})".format(mat, abs(bt), MAT[mat]["intensity"]))
        for k in range(args.cands):
            t = theta(mat, "cand{:02d}".format(k))
            if t is None:
                print("   cand{:02d}: FAILED".format(k))
            else:
                print("   cand{:02d}: |theta| = {:.4f} deg   ({:.2f}x baseline)".format(k, abs(t), abs(t) / abs(bt)))


if __name__ == "__main__":
    main()
