#!/usr/bin/env python
"""Stage 0 -- validate the measurement harness and choose the screening estimator.

Nothing in this campaign can be trusted until two questions are answered, and both are about
MEASUREMENT rather than design.  Stage 0 answers them with ~56 cheap 1D sims.

PART A -- harness validation + the pulse-duration correction.
  Run the fabricated SiN best_absolute cavity at its own design operating point, tau = 0, with
  4 pump1-carrier sub-samples, at BOTH pulse settings:
     * label 100.0 fs  = 120.1 fs intensity FWHM -- what every historical number used;
     * label 83.2555 fs = 100.0 fs intensity FWHM -- what the lab actually has.
  The first must reproduce the committed record (carrier-averaged pulse-integrated theta at
  tau = 0 approx -0.0019 deg, legacy fringe-max approx 0.138-0.143 deg).  If it does, the readout,
  the carrier averaging and the delay convention are all wired correctly, and the second run
  measures what the 20% pulse-length correction does to the effect.

PART B -- which estimator may we screen with?
  The objective is the carrier-averaged, pulse-integrated rotation, and it costs 4 FDTD runs
  per operating point.  If a 1-run estimator ranked geometries the same way, the whole campaign
  would be 4x cheaper.  Three candidates are compared against the objective over a spread of
  (geometry, operating point) pairs:
     theta_single_phase  -- pulse-integrated, ONE carrier phase   (1 sim)
     theta_legacy        -- tail-window azimuth, the OLD objective that produced best_absolute
     fringe amplitude    -- the coherent artifact, as a channel in its own right
  Rank correlation vs theta_chi5 decides. A high correlation buys a 4x cheaper screen; a low one
  proves the old objective was optimizing a different physical quantity, which is itself the
  central claim of the experimental audit and worth establishing on this design space.

  python chi5_dbr_design/s0_harness.py --workers 40
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
import common as C  # noqa: E402

OUT = HERE / "runs" / "s0_harness"

# Part A: the two pulse settings, as (tag, label_fs, intensity_FWHM_fs)
PULSE_SETTINGS = [("legacy120", 100.0, 120.1),
                  ("true100", C.PULSE_LABEL_FS, 100.0)]

# Part B: geometry spread.  Cavity length is the one lever with a validated FDTD trend
# (max|theta| ~ L^+1.2 at fixed mirrors), so an L family gives a well-separated signal ladder
# to correlate estimators against, plus one 4-pair variant to break the single-family degeneracy.
L_FAMILY = [4.0, 5.0, 5.893719385144864, 6.5, 7.5, 8.5]
PART_B_DELTAS = [0.014, 0.023]


def case_dir(tag: str) -> Path:
    return OUT / tag


def part_a_jobs():
    """(tag, geom, freqs, tau, pulse_label) for the baseline validation."""
    g = C.load_base_geometry()
    m = C.load_base_modes()
    freqs = {"probe": m["probe"]["frequency"],
             "pump1": m["pump1"]["frequency"], "pump2": m["pump2"]["frequency"]}
    jobs = []
    for tag, label, _ in PULSE_SETTINGS:
        for s, tau in enumerate(C.subsample_taus(freqs["pump1"])):
            jobs.append(("A/{}/s{}".format(tag, s), g, freqs, tau, label))
    return jobs, freqs


def part_b_cases():
    """(label, geom, freqs) for the estimator-correlation spread, all at the true 100 fs pulse."""
    base = C.load_base_geometry()
    p = C.geometry_params(base)
    cases = []
    variants = [("n3_L{:.2f}".format(L), 3, 3, p["t_hi"], p["t_lo"], L) for L in L_FAMILY]
    variants += [("n4_L{:.2f}".format(L), 4, 4, p["t_hi"], p["t_lo"], L) for L in (5.0, 6.5)]
    for name, nl, nr, t_hi, t_lo, L in variants:
        g = C.build_geometry(base, nl, nr, t_hi, t_lo, L)
        bad = C.fab_violations(g)
        if bad:
            print("  skip {}: {}".format(name, "; ".join(bad)))
            continue
        # One probe mode (the best) and one pump center per geometry: Part B varies GEOMETRY
        # and Delta to build a signal ladder for the estimator correlation, so extra operating
        # points would only add cost.  max_probes=1 also keeps the case label unique -- the
        # label carries geometry and Delta but not the probe frequency.
        ops = C.operating_points(g, max_centers=1, max_probes=1, deltas=PART_B_DELTAS)
        if not ops:
            print("  skip {}: no TMM probe/pump modes".format(name))
            continue
        for op in ops:
            cases.append(("B/{}_d{:.3f}".format(name, op["delta"]), g,
                          {"probe": op["probe"], "pump1": op["pump1"], "pump2": op["pump2"]},
                          op))
    return cases


def part_b_jobs(cases):
    jobs = []
    for label, g, freqs, _op in cases:
        for s, tau in enumerate(C.subsample_taus(freqs["pump1"])):
            jobs.append(("{}/s{}".format(label, s), g, freqs, tau, C.PULSE_LABEL_FS))
    return jobs


def run_job(tag, geom, freqs, tau, pulse_label, res, decay):
    C.run_case(case_dir(tag), geom, freqs, tau_fs=tau, res=res, decay=decay,
               pulse_label_fs=pulse_label)
    return tag


def collect(prefix, n_sub=C.SUBSAMPLES):
    recs = [C.read_case(case_dir("{}/s{}".format(prefix, s))) for s in range(n_sub)]
    if any(r is None for r in recs):
        return None
    return C.carrier_average(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--res", type=int, default=C.RES_1D)
    ap.add_argument("--decay", default=C.DECAY_1D)
    ap.add_argument("--slice", default=None, help="i/n -- run only slice i of n")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    a_jobs, base_freqs = part_a_jobs()
    b_cases = part_b_cases()
    b_jobs = part_b_jobs(b_cases)
    jobs = a_jobs + b_jobs

    print("=== Stage 0 | harness validation + estimator study ===")
    print("  Part A: {} sims (baseline x {} pulse settings x {} carrier sub-samples)".format(
        len(a_jobs), len(PULSE_SETTINGS), C.SUBSAMPLES))
    print("  Part B: {} sims ({} geometry/op-point cases x {} sub-samples)".format(
        len(b_jobs), len(b_cases), C.SUBSAMPLES))
    print("  res {}, decay {}, pad {:.0f} fs, I_pump {:.0e} W/cm2".format(
        args.res, args.decay, C.PAD_FS, C.PUMP_INTENSITY))

    if not args.aggregate_only:
        mine = jobs
        if args.slice:
            i, n = (int(x) for x in args.slice.split("/"))
            mine = [j for k, j in enumerate(jobs) if k % n == i]
        print("  running {} of {} sims on {} workers".format(len(mine), len(jobs), args.workers))
        t0, done = time.time(), 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_job, *j, args.res, args.decay) for j in mine]
            for _ in as_completed(futs):
                done += 1
                if done % 5 == 0 or done == len(mine):
                    print("    {}/{} ({:.0f}s)".format(done, len(mine), time.time() - t0),
                          flush=True)
        print("  sims done in {:.0f}s".format(time.time() - t0))

    # ------------------------------------------------------------------ Part A report --- #
    res = {"config": {"res": args.res, "decay": args.decay, "pad_fs": C.PAD_FS,
                      "subsamples": C.SUBSAMPLES, "pump_intensity": C.PUMP_INTENSITY,
                      "probe_intensity": C.PROBE_INTENSITY,
                      "base_freqs": base_freqs}, "part_a": {}, "part_b": []}
    print("\n--- Part A: baseline SiN best_absolute at its design operating point, tau=0 ---")
    print("{:>10s} {:>10s} {:>14s} {:>14s} {:>13s} {:>8s}".format(
        "pulse", "FWHM_fs", "theta_chi5_deg", "fringe_amp_deg", "legacy_deg", "DoLP"))
    for tag, label, fwhm in PULSE_SETTINGS:
        r = collect("A/{}".format(tag))
        if r is None:
            print("{:>10s}   (incomplete)".format(tag))
            continue
        r["pulse_label_fs"] = label
        r["intensity_fwhm_fs"] = fwhm
        res["part_a"][tag] = r
        print("{:>10s} {:>10.1f} {:>14.5f} {:>14.5f} {:>13.5f} {:>8.4f}".format(
            tag, fwhm, r["theta_chi5_deg"], r["theta_fringe_amp_deg"],
            r["theta_legacy_deg"], r["dolp"]))
    ref = res["part_a"].get("legacy120")
    if ref:
        print("  reference (committed record, legacy pulse): theta_chi5(tau=0) ~ -0.00189 deg,"
              " legacy fringe-max ~ 0.1383-0.1430 deg")

    # ------------------------------------------------------------------ Part B report --- #
    print("\n--- Part B: estimator comparison across geometries (true 100 fs pulse) ---")
    print("{:>18s} {:>7s} {:>6s} {:>13s} {:>13s} {:>13s} {:>12s} {:>7s}".format(
        "case", "L_um", "Delta", "theta_chi5", "single_phase", "fringe_amp", "legacy", "DoLP"))
    rows = []
    for label, g, freqs, op in b_cases:
        r = collect(label)
        if r is None:
            print("{:>18s}   (incomplete)".format(label.split("/")[-1]))
            continue
        p = C.geometry_params(g)
        row = {"case": label, "params": p, "op": op, **r}
        rows.append(row)
        print("{:>18s} {:>7.3f} {:>6.3f} {:>13.5f} {:>13.5f} {:>13.5f} {:>12.5f} {:>7.4f}".format(
            label.split("/")[-1], p["L_cav"], op["delta"], r["theta_chi5_deg"],
            r["theta_single_phase_deg"], r["theta_fringe_amp_deg"], r["theta_legacy_deg"],
            r["dolp"]))
    res["part_b"] = rows

    if len(rows) >= 3:
        obj = [abs(r["theta_chi5_deg"]) for r in rows]
        comp = {"single_phase": [abs(r["theta_single_phase_deg"]) for r in rows],
                "legacy": [abs(r["theta_legacy_deg"]) for r in rows],
                "fringe_amp": [r["theta_fringe_amp_deg"] for r in rows]}
        print("\n  Spearman rank correlation vs the objective |theta_chi5| (n={}):".format(len(rows)))
        res["correlations"] = {}
        for k, v in comp.items():
            rho = C.spearman(obj, v)
            res["correlations"][k] = rho
            verdict = ("usable screen" if rho > 0.9 else
                       "weak -- do not screen with this" if rho > 0.5 else
                       "DIFFERENT QUANTITY")
            print("    {:>14s} : rho = {:+.3f}   ({})".format(k, rho, verdict))
        res["correlations"]["fringe_over_effect_median"] = float(np.median(
            [r["theta_fringe_amp_deg"] / max(abs(r["theta_chi5_deg"]), 1e-12) for r in rows]))
        print("    median fringe/effect ratio = {:.1f}x".format(
            res["correlations"]["fringe_over_effect_median"]))

    path = OUT / "s0_result.json"
    json.dump(res, open(path, "w"), indent=2)
    print("\n-> {}".format(path))


if __name__ == "__main__":
    main()
