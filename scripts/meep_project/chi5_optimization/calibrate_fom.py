#!/usr/bin/env python
"""Calibrate/diagnose the chi5 FoM geometry-scaling against FDTD. Clean sweep: fix the
best_absolute mirror (thicknesses), vary cavity L and pair count N. For each geometry,
compute the proxy FoM + components (Q_s, |eta*zeta|, V_mode) at the TMM-selected operating
point, and the real 1D-FDTD |theta| at that operating point. Tabulate so the FoM's L/Q/V
dependence can be re-fit (the over-rewarding of short L / high Q is the suspected flaw).
Run under meep-mpi.
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
import objective    # noqa: E402
import optimize as O  # noqa: E402

BASE = json.load(open(MEEP / "SiN_optimizations/best_absolute/geometry.json"))
EX = json.load(open(MEEP / "SiN_optimizations/best_absolute/tcmt_derivation_analysis/tcmt_extracted_params_derivation.json"))
CHI = float(EX["material_constants"]["chi_iso_meep"])
HI = BASE["cavity"]["mat"]
T_HI = float(np.mean([l["thk_um"] for l in BASE["mirrors"]["left"] if l["mat"] == HI]))
T_LO = float(np.mean([l["thk_um"] for l in BASE["mirrors"]["left"] if l["mat"] != HI]))

GRID = [(3, 4.0), (3, 5.0), (3, 5.894), (3, 7.0), (4, 5.0), (4, 5.894), (4, 7.0)]
FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
         "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
INTENSITY = 1e12
RES = 80
DECAY = "1e-4"
OUT = HERE / "calibrate"


def proxy_eval(geom):
    s, f = O.select_operating_point(geom, CHI)
    if f is None:
        return None
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    mpb = tmm.find_mode(layers, idx, f["probe"])
    z, E, H, eps = tmm.field_profile(layers, idx, mpb["freq"])
    V = tmm.mode_volume(z, E, eps)
    raw, _ = objective.counter_coefficients(geom, f, CHI)
    ez = abs(raw["eta_minus"] * raw["zeta_minus"]) + abs(raw["eta_plus"] * raw["zeta_plus"])
    return {"fom": s["fom_rotation"], "freqs": f, "Qs": mpb["Q"], "V": V, "etazeta": ez,
            "Bs": s["Bs"], "B1": s["B1"], "B2": s["B2"]}


def run_fdtd(tag, geom, f):
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    json.dump(geom, open(d / "geometry.json", "w"), indent=2)
    modes = {"probe": {"frequency": f["probe"], "lambda_um": 1.0 / f["probe"]},
             "pump1": {"frequency": f["pump1"], "lambda_um": 1.0 / f["pump1"]},
             "pump2": {"frequency": f["pump2"], "lambda_um": 1.0 / f["pump2"]},
             "sidebands": {"frequency_plus": f["sb_plus"], "frequency_minus": f["sb_minus"],
                           "delta_frequency": f["pump1"] - f["pump2"],
                           "pump_separation_um": abs(1.0 / f["pump2"] - 1.0 / f["pump1"])}}
    json.dump(modes, open(d / "cavity_modes.json", "w"), indent=2)
    if (d / "faraday_summary.json").exists():
        return tag, "cached"
    cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *FLAGS,
           "--geometry-file", str(d / "geometry.json"), "--cavity-modes-file", str(d / "cavity_modes.json"),
           "--resolution", str(RES), "--decay-threshold", DECAY, "--pump-intensity", str(INTENSITY),
           "--pump1-frequency", "{:.9f}".format(f["pump1"]), "--pump2-frequency", "{:.9f}".format(f["pump2"]),
           "--output-dir", str(d)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    with open(d / "run.log", "w") as lf:
        r = subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return tag, ("ok" if r.returncode == 0 else "FAIL")


def fdtd_theta(tag):
    p = OUT / tag / "faraday_summary.json"
    if not p.exists():
        return None
    return abs(json.load(open(p))["probe_rotation_deg"]["final_relative_deg"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    jobs = []
    for n, L in GRID:
        geom = O.build_geometry(BASE, n, T_HI, T_LO, L)
        pe = proxy_eval(geom)
        if pe is None:
            print("N={} L={}: no operating point".format(n, L)); continue
        tag = "N{}_L{:.2f}".format(n, L)
        rows.append({"tag": tag, "N": n, "L": L, **{k: pe[k] for k in
                    ("fom", "Qs", "V", "etazeta", "Bs", "B1", "B2")}, "freqs": pe["freqs"]})
        jobs.append((tag, geom, pe["freqs"]))
    print("running {} FDTD calibration sims...".format(len(jobs)), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for fu in as_completed([ex.submit(run_fdtd, *j) for j in jobs]):
            print("  ", fu.result(), flush=True)
    print("done in {:.0f}s\n".format(time.time() - t0))

    print("{:>10s} {:>2s} {:>6s} {:>10s} {:>10s} {:>6s} {:>9s} {:>5s} {:>5s}".format(
        "tag", "N", "L", "FoM", "FDTD|th|", "Qs", "etazeta", "Vmode", "Bs"))
    for r in rows:
        th = fdtd_theta(r["tag"])
        r["fdtd"] = th
        print("{:>10s} {:>2d} {:6.2f} {:10.3e} {:10.4f} {:6.0f} {:9.2e} {:5.2f} {:5.1f}".format(
            r["tag"], r["N"], r["L"], r["fom"], th if th is not None else float("nan"),
            r["Qs"], r["etazeta"], r["V"], r["Bs"]))
    # save for offline fitting
    for r in rows:
        r.pop("freqs", None)
    json.dump(rows, open(OUT / "calibration.json", "w"), indent=2)
    print("\n-> {}".format(OUT / "calibration.json"))


if __name__ == "__main__":
    main()
