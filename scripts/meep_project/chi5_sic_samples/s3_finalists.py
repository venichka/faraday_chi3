#!/usr/bin/env python
"""Stage 3 -- validate the finalist operating points of each SiC sample.

Two things the 1D operating-point map cannot settle on its own:

  --what delay   the predicted lab trace: carrier-averaged effect AND fringe amplitude vs
                 envelope delay, so we can say whether the chi5 signal is legible in a raw
                 delay scan or whether the dithering procedure is required.
  --what 3d      1D is a plane-wave idealisation; the SiN campaign found 3D enhancements of
                 4.5-9.3x, so the quoted rotation must come from 3D.

⚠️ THE MPI RECIPE (learned from failed submissions, 2026-08-01 -- see run-environments):
  1. sbatch with --ntasks=1 --cpus-per-task=24, NOT --ntasks=24. With --ntasks=N the MPICH
     Hydra launcher bootstraps through SLURM's PMI and dies with "PMI_Get_appnum returned -1".
  2. Never launch the simulator by subprocess from a script running under mpirun -- each rank
     forks a non-MPI child whose `import meep` calls MPI_Init outside the job and aborts.
     So `--what 3d --print-args --index k` emits an argument list SERIALLY and the sbatch runs
     `mpirun -np 24 python faraday_meep_fp_circ.py $ARGS` directly.

The finalist list is FROZEN to runs/s3_finalists/finalists.json on first use: in the SiN
campaign a trailing scan task rewrote the ranking and swapped a finalist after its 3D jobs had
already been submitted.

  python chi5_sic_samples/s3_finalists.py --what freeze
  python chi5_sic_samples/s3_finalists.py --what delay --workers 45
  python chi5_sic_samples/s3_finalists.py --what 3d --print-args --index 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_sic as S  # noqa: E402
import common as C      # noqa: E402

SCAN = HERE / "runs" / "s1_scan"
OUT = HERE / "runs" / "s3_finalists"
FROZEN = OUT / "finalists.json"

I_REF = 1e11              # chosen by s0_intensity.py; see README
# +-150 fs, not +-300: the effect and fringe envelopes are both ~100 fs FWHM (they are the
# same pump-pump overlap integral), so beyond ~150 fs both channels sit on the N=4 residual
# floor and add no information -- only wall time. Step stays 25 fs = 4 points per FWHM.
TAU_MAX_FS = 150.0
TAU_STEP_FS = 25.0
PAD_DELAY_FS = 350.0      # fixed, so run length does not vary with tau
RES_3D = 30
DECAY_3D = "1e-3"


def contrast(r):
    return abs(r["theta_chi5_deg"]) / max(r["theta_fringe_amp_deg"], 1e-12)


def freeze():
    """Pick, and permanently record, the finalists.

    ⚠️ NOT "max contrast". Contrast is a RATIO, so maximising it rewards operating points where
    the fringe vanishes faster than the effect -- i.e. points with almost no signal. Measured on
    the phase-A data: at probe 850.2 nm on L=3.2, max-contrast picks Delta=0.0230 (contrast 1.46,
    theta 0.00082 deg) over Delta=0.0100 (contrast 1.44, theta 0.00486 deg) -- trading 6x the
    signal for 0.02 of contrast. On L=4.8 it picks a 969.9 nm probe that the lab cannot even
    reach. The physically useful question is "the most signal I can get while the effect is still
    legible", so contrast enters as a THRESHOLD and |theta| is maximised subject to it.

    Three kinds per sample, de-duplicated:
      maxsignal  max |theta| over every scanned probe (the extended-probe scenario)
      nowbest    max |theta| restricted to probes reachable today
      legible    max |theta| subject to contrast >= 1; if nothing reaches 1, the best |theta|
                 among the top-quartile-contrast points, and the label records that it fell back
    """
    OUT.mkdir(parents=True, exist_ok=True)
    if FROZEN.exists():
        return json.load(open(FROZEN))
    fin = []
    for s in S.CAVITY_LENGTHS_UM:
        p = SCAN / s / "result.json"
        if not p.exists():
            continue
        rows = json.load(open(p))["rows"]
        if not rows:
            continue
        # deterministic tie-break so an exact tie cannot reorder between runs
        def by_theta(rs):
            return max(rs, key=lambda r: (abs(r["theta_chi5_deg"]), -r["op"]["delta"]))

        picks = [("maxsignal", by_theta(rows), True)]

        now = [r for r in rows if S.in_windows(r["op"]["probe_nm"] / 1000.0,
                                               S.PROBE_WINDOWS_NOW)]
        if now:
            picks.append(("nowbest", by_theta(now), True))

        legible = [r for r in rows if contrast(r) >= 1.0]
        if legible:
            picks.append(("legible", by_theta(legible), True))
        else:
            cs = sorted((contrast(r) for r in rows), reverse=True)
            cut = cs[max(0, len(cs) // 4 - 1)]
            top = [r for r in rows if contrast(r) >= cut]
            if top:
                picks.append(("legible_fallback", by_theta(top), False))

        for kind, r, reached in picks:
            if any(S.tag(f["op"]) == S.tag(r["op"]) and f["sample"] == s for f in fin):
                continue        # one point can win several criteria -- carry it once
            fin.append({"label": "{}_{}".format(s, kind), "sample": s, "kind": kind,
                        "op": r["op"], "theta_1d_deg": r["theta_chi5_deg"],
                        "fringe_1d_deg": r["theta_fringe_amp_deg"],
                        "contrast_1d": contrast(r), "dolp_1d": r["dolp"],
                        "lab_accessible": S.in_windows(r["op"]["probe_nm"] / 1000.0,
                                                       S.PROBE_WINDOWS_NOW),
                        "contrast_threshold_met": bool(reached)})
    json.dump(fin, open(FROZEN, "w"), indent=2)
    print("froze {} finalists -> {}".format(len(fin), FROZEN))
    for f in fin:
        print("  {:22s} probe {:.1f} nm  pumps {:.1f}/{:.1f} nm  D {:.4f}  "
              "theta {:.5f}  contrast {:.2f}".format(
                  f["label"], f["op"]["probe_nm"], f["op"]["pump1_nm"], f["op"]["pump2_nm"],
                  f["op"]["delta"], abs(f["theta_1d_deg"]), f["contrast_1d"]))
    return fin


def taus():
    n = int(round(2 * TAU_MAX_FS / TAU_STEP_FS)) + 1
    return list(np.linspace(-TAU_MAX_FS, TAU_MAX_FS, n))


def delay_jobs(fin):
    js = []
    for f in fin:
        geom = S.sic_geometry(S.CAVITY_LENGTHS_UM[f["sample"]])
        for t0 in taus():
            for sub, dt in enumerate(C.subsample_taus(f["op"]["pump1"])):
                d = OUT / "delay" / f["label"] / "t{:+08.2f}".format(t0) / "s{}".format(sub)
                js.append((d, geom, f["op"], sub, t0 + dt))
    return js


def run_delay(workers, res, decay):
    fin = freeze()
    js = delay_jobs(fin)
    print("{} delay sims ({} finalists x {} delays x {} sub-samples)".format(
        len(js), len(fin), len(taus()), C.SUBSAMPLES))
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(S.run_case, d, g, o, s, t, res, decay, PAD_DELAY_FS, 1, I_REF)
                for (d, g, o, s, t) in js]
        for _ in as_completed(futs):
            done += 1
            if done % 25 == 0 or done == len(js):
                print("  {}/{} ({:.0f}s)".format(done, len(js), time.time() - t0), flush=True)
    aggregate_delay(fin)


def aggregate_delay(fin):
    out = {}
    for f in fin:
        rows = []
        for t0 in taus():
            recs = [C.read_case(OUT / "delay" / f["label"] / "t{:+08.2f}".format(t0) /
                                "s{}".format(s)) for s in range(C.SUBSAMPLES)]
            if any(r is None for r in recs):
                continue
            a = C.carrier_average(recs)
            rows.append({"tau_fs": t0, "theta_avg_deg": a["theta_chi5_deg"],
                         "fringe_amp_deg": a["theta_fringe_amp_deg"],
                         "theta_single_deg": a["theta_single_phase_deg"], "dolp": a["dolp"]})
        out[f["label"]] = {"op": f["op"], "rows": rows,
                           "carrier_period_fs": C.carrier_period_fs(f["op"]["pump1"])}
        if rows:
            near = [r for r in rows if abs(r["tau_fs"]) <= 50]
            c = np.median([abs(r["theta_avg_deg"]) / max(abs(r["fringe_amp_deg"]), 1e-12)
                           for r in near]) if near else float("nan")
            print("  {:22s} contrast (|tau|<=50 fs) = {:.2f}".format(f["label"], c))
    json.dump({"config": {"tau_max_fs": TAU_MAX_FS, "tau_step_fs": TAU_STEP_FS,
                          "pad_fs": PAD_DELAY_FS, "intensity": I_REF},
               "finalists": out}, open(OUT / "delay_result.json", "w"), indent=2)
    print("-> {}".format(OUT / "delay_result.json"))


def jobs_3d(fin):
    js = []
    for f in fin:
        geom = S.sic_geometry(S.CAVITY_LENGTHS_UM[f["sample"]])
        for sub in range(C.SUBSAMPLES):
            js.append((f["label"], geom, f["op"], sub))
    return js


def print_args_3d(index, res, decay):
    """Serial, rank-free: emit one 3D job's argument list, or nothing if it is already done."""
    js = jobs_3d(freeze())
    if not (0 <= index < len(js)):
        return
    label, geom, op, sub = js[index]
    out = OUT / "3d" / label / "s{}".format(sub)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "faraday_summary.json").exists():
        return
    json.dump(geom, open(out / "geometry.json", "w"))
    json.dump(C.modes_json(op["probe"], op["pump1"], op["pump2"]),
              open(out / "cavity_modes.json", "w"))
    tau = C.subsample_taus(op["pump1"])[sub]
    cmd = C.fdtd_cmd(out, res, decay, tau, C.PAD_FS, 0.0, 3,
                     pump_intensity=I_REF, extra=S.FDTD_FLAGS_SIC)
    print(" ".join(cmd[2:]))          # drop [python, script]; the sbatch supplies those


def aggregate_3d():
    fin = freeze()
    rows = {}
    print("\n=== 3D vs 1D (carrier-averaged chi5 channel, I = {:.0e}) ===".format(I_REF))
    print("  {:22s} {:>11s} {:>11s} {:>7s} {:>10s} {:>9s} {:>7s}".format(
        "finalist", "theta_3D", "theta_1D", "ratio", "fringe_3D", "contrast", "DoLP"))
    for f in fin:
        recs = [C.read_case(OUT / "3d" / f["label"] / "s{}".format(s))
                for s in range(C.SUBSAMPLES)]
        if any(r is None for r in recs):
            continue
        a = C.carrier_average(recs)
        t3, t1 = abs(a["theta_chi5_deg"]), abs(f["theta_1d_deg"])
        rows[f["label"]] = {"theta_3d_deg": a["theta_chi5_deg"], "theta_1d_deg": f["theta_1d_deg"],
                            "ratio": t3 / max(t1, 1e-30),
                            "fringe_3d_deg": a["theta_fringe_amp_deg"],
                            "contrast_3d": t3 / max(a["theta_fringe_amp_deg"], 1e-12),
                            "dolp_3d": a["dolp"], "op": f["op"]}
        r = rows[f["label"]]
        print("  {:22s} {:11.5f} {:11.5f} {:7.2f} {:10.5f} {:9.2f} {:7.4f}".format(
            f["label"], t3, t1, r["ratio"], r["fringe_3d_deg"], r["contrast_3d"], r["dolp_3d"]))
    json.dump(rows, open(OUT / "3d_result.json", "w"), indent=2)
    print("-> {}".format(OUT / "3d_result.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["freeze", "delay", "3d"], default="freeze")
    ap.add_argument("--workers", type=int, default=45)
    ap.add_argument("--res", type=int, default=C.RES_1D)
    ap.add_argument("--decay", default=C.DECAY_1D)
    ap.add_argument("--res-3d", type=int, default=RES_3D)
    ap.add_argument("--decay-3d", default=DECAY_3D)
    ap.add_argument("--print-args", action="store_true")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()

    if a.what == "freeze":
        freeze()
    elif a.what == "delay":
        if a.aggregate:
            aggregate_delay(freeze())
        else:
            run_delay(a.workers, a.res, a.decay)
    else:
        if a.print_args:
            print_args_3d(a.index, a.res_3d, a.decay_3d)
        else:
            aggregate_3d()


if __name__ == "__main__":
    main()
