#!/usr/bin/env python
"""Stage 3 -- validate the Stage-2 finalists. Three independent checks, each with a clear
pass/fail meaning; a design that fails any of them is not a deliverable.

  --what intensity   Does the winner actually scale as chi5?  Sweep I_pump and fit the LOCAL
                     log-log slope of |theta_chi5| vs I.  chi5 => 2, chi3 => 1.  Read the LOCAL
                     slope, not a global fit: the response is a chi3 -> chi5 CROSSOVER, and a
                     single power-law fit averages across it (the SiC study measured global
                     1.43-1.61 where the local slope reached 2.14).  Also watch DoLP: it sagging
                     with intensity means the rotation is turning into ellipticity.

  --what tolerance   Is the design fabricable?  Perturb every layer thickness by independent
                     Gaussian errors (sigma = 3 and 5 nm) and re-measure.  Reports the median
                     and the 10th percentile of |theta| retained.  A razor-sharp optimum that
                     loses most of its signal at +-3 nm is not worth fabricating, however well
                     it scores nominally.  (User chose: report sensitivity on finalists rather
                     than build it into the objective.)

  --what 3d          Does the enhancement survive in 3D?  best_absolute gained 16.2x on the
                     chi5 channel going 1D -> 3D, but the SiC design gained nothing, so this is
                     design-specific and must be measured.  MPI; see the launch recipe below.

MPI recipe for 3D (learned the hard way -- two failed submissions, 2026-08-01):
  1. Request --ntasks=1 --cpus-per-task=24, NOT --ntasks=24. With --ntasks=N the MPICH Hydra
     launcher bootstraps through SLURM's PMI and dies with "PMI_Get_appnum returned -1".
  2. NEVER launch the simulation via subprocess from a script running under mpirun: each rank
     would fork its own non-MPI child whose `import meep` calls MPI_Init outside the job.
     So `--what 3d --print-args --index k` emits an argument list (serial, rank-free) and the
     sbatch runs `mpirun -np 24 python faraday_meep_fp_circ.py $ARGS` directly.

  python chi5_dbr_design/s3_validate.py --what intensity --workers 24
  python chi5_dbr_design/s3_validate.py --what tolerance --workers 24
  python chi5_dbr_design/s3_validate.py --what 3d --print-args --index 0
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT = HERE / "runs" / "s3_validate"
S2 = HERE / "runs" / "s2_fdtd" / "s2_result.json"

INTENSITIES = [2.5e11, 5e11, 1e12, 2e12, 4e12]
TOL_SIGMAS_NM = [3.0, 5.0]
TOL_DRAWS = 12
N_FINALISTS = 3


# ------------------------------------------------------------------------ finalists --- #
def load_finalists(s2_path: Path, n=N_FINALISTS):
    """[(label, geom, op)] -- the Stage-2 finalists plus the baseline for reference.

    Selected on BOTH axes, because Stage 2 showed they rank almost oppositely:
      * |theta_chi5|          -- how much rotation the design produces;
      * contrast = |theta| / fringe amplitude -- whether that rotation can be SEEN, i.e.
        whether it sits above or below the coherent carrier fringe it must be extracted from.
    The raw-signal winner (cand07, 11.4x baseline) has contrast 0.10, no better than the
    fabricated baseline's 0.09 -- it is 11x more effect buried under 11x more fringe. Two other
    designs put the effect ABOVE the fringe (contrast > 1), which for a lab that cannot phase-
    stabilise its delay line is the more valuable property. Both families go to Stage 3.

    FROZEN. The selection is written to runs/s3_validate/finalists.json the first time it is
    computed and read back thereafter. Without this the list is not stable: Stage 2's trailing
    array tasks each rewrite s2_result.json as they finish, and two candidates tie exactly on
    contrast (cand15 and cand19 at 1.32), so an unstable tie-break can silently swap a finalist
    *after* its 3D jobs have been submitted -- leaving the aggregation looking for results that
    were never computed. Delete the file to re-select deliberately.
    """
    res = json.load(open(s2_path))
    base = C.load_base_geometry()
    out, seen = [], set()
    ranking = [r for r in res["ranking"] if r["label"] != "baseline"]

    def contrast(r):
        return abs(r["best"]["theta_chi5_deg"]) / max(r["best"]["theta_fringe_amp_deg"], 1e-12)

    frozen_path = OUT / "finalists.json"
    if frozen_path.exists():
        chosen = json.load(open(frozen_path))["labels"]
        picks = [r for lab in chosen for r in res["ranking"] if r["label"] == lab]
    else:
        # ties broken by rotation, so equal-contrast designs are ordered by how much signal
        # they actually deliver
        by_theta = sorted(ranking, key=lambda r: -abs(r["best"]["theta_chi5_deg"]))
        by_contrast = sorted(ranking, key=lambda r: (-contrast(r),
                                                     -abs(r["best"]["theta_chi5_deg"])))
        picks = by_theta[:2] + by_contrast[:2]
        picks += [r for r in res["ranking"] if r["label"] == "baseline"]
    for r in picks:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        p = r["params"]
        g = C.build_geometry(base, int(p["n_left"]), int(p["n_right"]),
                             float(p["t_hi"]), float(p["t_lo"]), float(p["L_cav"]))
        out.append((r["label"], g, r["best"]["op"]))
    if not frozen_path.exists():
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"labels": [lab for lab, _, _ in out],
                   "note": "frozen at first Stage-3 launch; delete to re-select"},
                  open(frozen_path, "w"), indent=2)
    return out


def freqs_of(op):
    return {"probe": op["probe"], "pump1": op["pump1"], "pump2": op["pump2"]}


def collect(d: Path, n_sub=C.SUBSAMPLES):
    recs = [C.read_case(d / "s{}".format(s)) for s in range(n_sub)]
    if any(r is None for r in recs):
        return None
    return C.carrier_average(recs)


def execute(jobs, workers, label="", slice_spec=None):
    """Run `jobs` concurrently. `slice_spec` = "i/n" keeps only every n-th job so the study can
    be farmed across nodes; the runner is idempotent, so slices may overlap and be re-run."""
    if slice_spec:
        i, n = (int(x) for x in slice_spec.split("/"))
        jobs = [j for k, j in enumerate(jobs) if k % n == i]
    if not jobs:
        return
    print("  {}: {} sims on {} workers...".format(label, len(jobs), workers), flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(*j) for j in jobs]
        for _ in as_completed(futs):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print("    {}/{} ({:.0f}s)".format(done, len(jobs), time.time() - t0), flush=True)


# ------------------------------------------------------------------ intensity scaling --- #
def run_intensity(fin, args):
    root = OUT / "intensity"
    jobs = []
    for label, geom, op in fin:
        for I in INTENSITIES:
            for s, tau in enumerate(C.subsample_taus(op["pump1"])):
                d = root / label / "I{:.2e}".format(I) / "s{}".format(s)
                jobs.append((C.run_case, d, geom, freqs_of(op), tau, args.res, args.decay,
                             C.PAD_FS, 0.0, I))
    if not args.aggregate_only:
        execute(jobs, args.workers, "intensity sweep", args.slice)

    print("\n=== intensity scaling: |theta_chi5| vs I_pump (chi5 => slope 2) ===")
    out = {}
    for label, geom, op in fin:
        rows = []
        for I in INTENSITIES:
            r = collect(root / label / "I{:.2e}".format(I))
            if r:
                rows.append({"I": I, "theta": abs(r["theta_chi5_deg"]), "dolp": r["dolp"],
                             "fringe": r["theta_fringe_amp_deg"]})
        if len(rows) < 2:
            continue
        x = np.log10([r["I"] for r in rows])
        y = np.log10([max(r["theta"], 1e-12) for r in rows])
        local = [(rows[i]["I"], rows[i + 1]["I"], float((y[i + 1] - y[i]) / (x[i + 1] - x[i])))
                 for i in range(len(rows) - 1)]
        glob = float(np.polyfit(x, y, 1)[0])
        out[label] = {"rows": rows, "local_slopes": local, "global_slope": glob}
        print("\n  {}".format(label))
        print("    {:>10s} {:>12s} {:>8s} {:>10s}".format("I (W/cm2)", "|theta| deg", "DoLP", "slope"))
        for i, r in enumerate(rows):
            sl = "{:8.2f}".format(local[i][2]) if i < len(local) else "     -- "
            print("    {:>10.2e} {:>12.5f} {:>8.4f} {:>10s}".format(r["I"], r["theta"], r["dolp"], sl))
        print("    global fit slope = {:.2f}   (read the LOCAL slopes: the response is a "
              "chi3->chi5 crossover)".format(glob))
    json.dump(out, open(OUT / "intensity_result.json", "w"), indent=2)
    print("\n-> {}".format(OUT / "intensity_result.json"))


# ---------------------------------------------------------------- fabrication tolerance --- #
def perturb(geom, sigma_um, rng):
    """Independent Gaussian thickness error on every deposited layer, including the cavity --
    a PECVD run gets each layer slightly wrong, independently."""
    g = copy.deepcopy(geom)
    for side in ("left", "right"):
        for l in g["mirrors"][side]:
            l["thk_um"] = float(max(0.005, l["thk_um"] + rng.normal(0.0, sigma_um)))
    g["cavity"]["L_um"] = float(max(0.05, g["cavity"]["L_um"] + rng.normal(0.0, sigma_um)))
    return g


def run_tolerance(fin, args):
    root = OUT / "tolerance"
    jobs, plan = [], []
    for label, geom, op in fin:
        for sig_nm in TOL_SIGMAS_NM:
            for k in range(TOL_DRAWS):
                rng = np.random.default_rng(abs(hash((label, sig_nm, k))) % (2 ** 31))
                gp = perturb(geom, sig_nm / 1000.0, rng)
                plan.append((label, sig_nm, k, gp))
                for s, tau in enumerate(C.subsample_taus(op["pump1"])):
                    d = root / label / "sig{:.0f}".format(sig_nm) / "k{:02d}".format(k) / "s{}".format(s)
                    jobs.append((C.run_case, d, gp, freqs_of(op), tau, args.res, args.decay))
    if not args.aggregate_only:
        execute(jobs, args.workers, "tolerance draws", args.slice)

    print("\n=== fabrication tolerance: |theta_chi5| retained under layer-thickness error ===")
    print("  (operating point held FIXED at the nominal design point -- i.e. the lab does not")
    print("   re-tune after fabrication. Re-tuning would recover part of the loss.)")
    out = {}
    for label, geom, op in fin:
        nominal = collect(OUT / "intensity" / label / "I{:.2e}".format(1e12))
        nom = abs(nominal["theta_chi5_deg"]) if nominal else None
        out[label] = {"nominal": nom, "sigmas": {}}
        print("\n  {}  (nominal |theta| = {})".format(
            label, "{:.5f} deg".format(nom) if nom else "not available"))
        for sig_nm in TOL_SIGMAS_NM:
            vals, dolps = [], []
            for k in range(TOL_DRAWS):
                r = collect(root / label / "sig{:.0f}".format(sig_nm) / "k{:02d}".format(k))
                if r:
                    vals.append(abs(r["theta_chi5_deg"]))
                    dolps.append(r["dolp"])
            if not vals:
                continue
            v = np.array(vals)
            rel = v / nom if nom else v / np.median(v)
            out[label]["sigmas"][str(sig_nm)] = {
                "n": len(vals), "median_rel": float(np.median(rel)),
                "p10_rel": float(np.percentile(rel, 10)), "min_rel": float(rel.min()),
                "median_dolp": float(np.median(dolps)), "values": vals}
            print("    sigma {:>3.0f} nm (n={:2d}): median {:5.1f}% of nominal, "
                  "10th pct {:5.1f}%, worst {:5.1f}%, DoLP {:.3f}".format(
                      sig_nm, len(vals), 100 * np.median(rel), 100 * np.percentile(rel, 10),
                      100 * rel.min(), float(np.median(dolps))))
    json.dump(out, open(OUT / "tolerance_result.json", "w"), indent=2)
    print("\n-> {}".format(OUT / "tolerance_result.json"))


# ---------------------------------------------------------------------------- 3D --- #
def jobs_3d(fin):
    return [(label, geom, op, s)
            for label, geom, op in fin
            for s in range(C.SUBSAMPLES)]


def print_args_3d(fin, index, res, decay):
    """Emit the simulator argument list for one 3D job (see the MPI recipe in the header).
    Prints nothing if the job is already done, so the sbatch skips it."""
    js = jobs_3d(fin)
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
    cmd = C.fdtd_cmd(out, res, decay, tau, C.PAD_FS, dim=3)
    print(" ".join(cmd[2:]))          # drop [python, script]; the sbatch supplies those


def aggregate_3d(fin):
    print("\n=== 3D vs 1D on the chi5 channel ===")
    print("  {:>10s} {:>14s} {:>14s} {:>8s} {:>8s}".format(
        "label", "theta_3D", "theta_1D", "ratio", "DoLP"))
    out = {}
    for label, geom, op in fin:
        r3 = collect(OUT / "3d" / label)
        r1 = collect(OUT / "intensity" / label / "I{:.2e}".format(1e12))
        if not r3:
            print("  {:>10s}   (3D incomplete)".format(label))
            continue
        t3 = abs(r3["theta_chi5_deg"])
        t1 = abs(r1["theta_chi5_deg"]) if r1 else float("nan")
        out[label] = {"theta_3d": t3, "theta_1d": t1, "ratio": t3 / t1 if t1 else None,
                      "dolp_3d": r3["dolp"], "fringe_3d": r3["theta_fringe_amp_deg"]}
        print("  {:>10s} {:>14.5f} {:>14.5f} {:>8.2f} {:>8.4f}".format(
            label, t3, t1, t3 / t1 if t1 else float("nan"), r3["dolp"]))
    json.dump(out, open(OUT / "3d_result.json", "w"), indent=2)
    print("\n-> {}".format(OUT / "3d_result.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["intensity", "tolerance", "3d"], required=True)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--res", type=int, default=C.RES_1D)
    ap.add_argument("--res-3d", type=int, default=30)
    ap.add_argument("--decay", default=C.DECAY_1D)
    ap.add_argument("--decay-3d", default="1e-3")
    ap.add_argument("--s2", default=str(S2))
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--print-args", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--slice", default=None, help="i/n -- run only slice i of n")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fin = load_finalists(Path(args.s2))

    if args.what == "3d":
        if args.list:
            for k, (label, _g, _op, s) in enumerate(jobs_3d(fin)):
                print(k, label, "s{}".format(s))
            return
        if args.print_args:
            print_args_3d(fin, args.index, args.res_3d, args.decay_3d)
            return
        aggregate_3d(fin)
        return

    print("=== Stage 3 ({}) | finalists: {} ===".format(
        args.what, ", ".join(l for l, _, _ in fin)))
    if args.what == "intensity":
        run_intensity(fin, args)
    else:
        run_tolerance(fin, args)


if __name__ == "__main__":
    main()
