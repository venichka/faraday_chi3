#!/usr/bin/env python
"""Stage C: 3D + TCMT cross-check of the SiN hybrid winner cand07.

Takes the 1D-FDTD winner cand07 at its defensible clean operating point and:
  (a) confirms the rotation survives full 3D vector geometry. The SiC L=3.2um study showed NO 3D
      enhancement (accumulated theta 3D/1D = 0.99x) because its FWM-matched pumps were NON-resonant
      -> the response was bulk-chi3 (dimension-independent). cand07's pumps sit ON resonant pump
      modes (that is how the hybrid Stage-B centers were chosen), so we expect the SiN-best_absolute-
      like resonant enhancement to SURVIVE into 3D.
  (b) extracts reduced-order TCMT params (frequencies, kappa split, eta overlaps) for FaradayJL.

Operating point (cand07, from refine_sin_lowd):
  probe = geometry's fixed TMM probe mode  f=1.24889  (lambda=0.8007um)
  center = 0.69013  Delta = 0.006  ->  1D res80 theta = 2.39 deg, DoLP 0.975 (the CLEAN defensible
  point; the raw max was theta=2.70 deg at Delta=0.004, but Delta=0.004 sits ~11x BELOW the 100fs
  pump bandwidth (fwidth~0.046) where the two-pump/discrete-sideband picture loses meaning, so we
  cross-check at Delta=0.006).
  f1 = center + Delta/2 = 0.69313   f2 = center - Delta/2 = 0.68713

Same material/intensity config as the 1D hybrid/refine (fit SiN/SiO2, I=1e12) so 3D-vs-1D is
apples-to-apples; the 1D res-30 companion (--run-aux) removes the resolution confound (3D is res 30).

  python chi5_optimization/cross3d.py --prep        # write inputs (json only; head-node ok)
  python chi5_optimization/cross3d.py --run-aux     # 1D res-30 companion + TCMT extraction (1D, serial)
The heavy 3D run is launched from cross3d.sbatch (mpirun -np 96 ... faraday_meep_fp_circ.py --dim 3).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent

WINNER = "cand07"
CENTER = 0.69013
DELTA = 0.006

# SiN fit config (identical to hybrid.MAT["sin"]).
SIN_FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
             "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
INTENSITY = 1e12
# Constant indices for the (linear) TCMT extraction = the geometry's stored SiN/SiO2 indices.
NH, NL, KH = 2.06687952693004, 1.456812738340945, 0.0
N2_SIN = 5.0e-19   # nonlinear_materials "sin" preset (m^2/W)


def prep(out: Path, center: float, delta: float):
    """Write geometry.json + cavity_modes.json for cand07 at the chosen operating point."""
    geom = json.load(open(HERE / "hybrid" / "sin" / "winners" / WINNER / "geometry.json"))
    fprobe = json.load(open(HERE / "hybrid" / "sin" / "winners" / WINNER / "cavity_modes.json"))["probe"]["frequency"]
    f1, f2 = center + 0.5 * delta, center - 0.5 * delta
    cand = out / WINNER
    cand.mkdir(parents=True, exist_ok=True)
    json.dump(geom, open(cand / "geometry.json", "w"), indent=2)
    json.dump({"probe": {"frequency": fprobe, "lambda_um": 1.0 / fprobe},
               "pump1": {"frequency": f1, "lambda_um": 1.0 / f1},
               "pump2": {"frequency": f2, "lambda_um": 1.0 / f2},
               "sidebands": {"frequency_plus": fprobe + delta, "frequency_minus": fprobe - delta,
                             "delta_frequency": delta, "pump_separation_um": abs(1.0 / f2 - 1.0 / f1)}},
              open(cand / "cavity_modes.json", "w"), indent=2)
    print("prepped {}  probe f={:.6f} (lam={:.4f}um)  center={:.5f} Delta={:.4f}  f1={:.6f} f2={:.6f}".format(
        cand, fprobe, 1.0 / fprobe, center, delta, f1, f2), flush=True)
    return cand, fprobe


def _run(cmd, log):
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    print("  -> rc={} (log {})".format(rc, log), flush=True)
    return rc


def run_aux(out: Path, center: float, delta: float, res1d: int, decay: str):
    """1D res-30 companion (matched-resolution dimensionality baseline) + TCMT extraction."""
    cand, _ = prep(out, center, delta)
    f1, f2 = center + 0.5 * delta, center - 0.5 * delta
    geo, mod = cand / "geometry.json", cand / "cavity_modes.json"

    comp = cand / "run1d_res{}".format(res1d)
    comp.mkdir(parents=True, exist_ok=True)
    print("== 1D res-{} companion ==".format(res1d), flush=True)
    _run([sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *SIN_FLAGS,
          "--geometry-file", str(geo), "--cavity-modes-file", str(mod),
          "--resolution", str(res1d), "--courant", "0.25", "--decay-threshold", decay,
          "--pump-intensity", str(INTENSITY),
          "--pump1-frequency", "{:.9f}".format(f1), "--pump2-frequency", "{:.9f}".format(f2),
          "--output-dir", str(comp)], comp / "run.log")

    print("== TCMT extraction (FaradayJL params) ==", flush=True)
    _run([sys.executable, "extract_tcmt_params.py", "--geometry-file", str(geo), "--modes-file", str(mod),
          "--nH", str(NH), "--kH", str(KH), "--nL", str(NL), "--n2", str(N2_SIN),
          "--output-json", str(cand / "tcmt_extracted_params.json")], cand / "tcmt.log")

    # Summaries.
    cs = comp / "faraday_summary.json"
    if cs.exists():
        d = json.load(open(cs))["probe_rotation_deg"]
        print("1D res-{}: |theta|={:.4f} deg  DoLP={}".format(
            res1d, abs(d["final_relative_deg"]),
            d.get("coherent_window_estimate", {}).get("dolp", "n/a")), flush=True)
    tp = cand / "tcmt_extracted_params.json"
    if tp.exists():
        print("TCMT params -> {}".format(tp), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", type=float, default=CENTER)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--res1d", type=int, default=30, help="resolution for the 1D companion (match 3D)")
    ap.add_argument("--decay", default="1e-4")
    ap.add_argument("--out", default=str(HERE / "cross3d"))
    ap.add_argument("--prep", action="store_true", help="write operating-point inputs only")
    ap.add_argument("--run-aux", action="store_true", help="1D res-30 companion + TCMT extraction")
    args = ap.parse_args()
    out = Path(args.out)
    if args.run_aux:
        run_aux(out, args.center, args.delta, args.res1d, args.decay)
    else:
        prep(out, args.center, args.delta)


if __name__ == "__main__":
    main()
