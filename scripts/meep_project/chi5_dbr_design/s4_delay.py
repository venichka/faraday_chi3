#!/usr/bin/env python
"""Stage 4 -- predict the trace the lab would actually record, for the new designs.

Stages 0-3 evaluate everything at tau = 0 (pulses coincident). The experiment instead scans the
pump1 delay and records V - H = -S1. The delay study established that on the FABRICATED design
that trace is dominated by a coherent carrier fringe ~12x larger than the rectified chi5
envelope, which is why the lab most likely measures the fringe (delay_scan-study). Stage 2 found
designs whose contrast is > 1, i.e. where the effect should DOMINATE the trace instead.

This stage tests that end-to-end: run a full delay scan on a high-contrast design and on the
fabricated baseline, and plot what each detector would see, in two readouts:

  * carrier-averaged  -- 4 sub-samples per delay, i.e. a delay line that is NOT phase stable
                         (or is deliberately dithered by >= 5.1 fs). This is the chi5 envelope.
  * single phase      -- one run per delay, i.e. a phase-stable line. This is fringe + envelope,
                         and is what the current experiment appears to be recording.

The prediction to check: on the baseline the two look completely different (the fringe swamps
the envelope); on a contrast > 1 design they should look similar, because the envelope is the
larger term. If that holds, the design removes the need for the delay-dither prescription.

Grid: the delay step must resolve the pump-band mode beats (~90-150 fs on these cavities) and
the span must outlast the energy ring-down. +-300 fs at 25 fs is 25 points.

  python chi5_dbr_design/s4_delay.py --workers 26 --slice 0/6
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
import s3_validate as S3  # noqa: E402

OUT = HERE / "runs" / "s4_delay"
S2 = HERE / "runs" / "s2_fdtd" / "s2_result.json"

TAU_MAX_FS = 300.0
TAU_STEP_FS = 25.0
# Pad must exceed the largest |negative delay| and be held FIXED across the whole scan, so that
# pump1 is the only source whose timing changes for both signs of tau (the delay-convention bug).
PAD_FS = 350.0
# Which designs to trace: the high-contrast winner, the all-rounder, and the fabricated control.
DESIGNS = ["cand16", "cand13", "baseline"]


def taus():
    return list(np.round(np.arange(-TAU_MAX_FS, TAU_MAX_FS + 1e-9, TAU_STEP_FS), 4))


def case_dir(label, tau, sub):
    return OUT / label / "t{:+08.2f}".format(tau) / "s{}".format(sub)


def build_jobs(fin):
    jobs = []
    for label, geom, op in fin:
        if label not in DESIGNS:
            continue
        T1 = C.carrier_period_fs(op["pump1"])
        for tau in taus():
            for s in range(C.SUBSAMPLES):
                tau_eff = tau + s * T1 / C.SUBSAMPLES
                jobs.append((label, geom, op, tau, s, tau_eff))
    return jobs


def run_job(label, geom, op, tau, sub, tau_eff, res, decay):
    C.run_case(case_dir(label, tau, sub), geom,
               {"probe": op["probe"], "pump1": op["pump1"], "pump2": op["pump2"]},
               tau_fs=tau_eff, res=res, decay=decay, pad_fs=PAD_FS)
    return label, tau, sub


def aggregate(fin):
    out = {}
    for label, geom, op in fin:
        if label not in DESIGNS:
            continue
        rows = []
        for tau in taus():
            recs = [C.read_case(case_dir(label, tau, s)) for s in range(C.SUBSAMPLES)]
            if any(r is None for r in recs):
                continue
            r = C.carrier_average(recs)
            rows.append({"tau_fs": tau,
                         # what a NON-phase-stable (or dithered) line records: the chi5 envelope
                         "theta_avg_deg": r["theta_chi5_deg"], "vmh_avg": r["vmh_chi5"],
                         # what a phase-STABLE line records at one phase: fringe + envelope
                         "theta_single_deg": r["theta_single_phase_deg"],
                         "vmh_single": r["vmh_sub"][0],
                         "fringe_amp_deg": r["theta_fringe_amp_deg"],
                         "dolp": r["dolp"]})
        if rows:
            out[label] = {"rows": rows, "op": op,
                          "params": C.geometry_params(geom),
                          "carrier_period_fs": C.carrier_period_fs(op["pump1"])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--res", type=int, default=C.RES_1D)
    ap.add_argument("--decay", default=C.DECAY_1D)
    ap.add_argument("--slice", default=None, help="i/n -- run only slice i of n")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fin = S3.load_finalists(S2)
    jobs = build_jobs(fin)
    print("=== Stage 4 | predicted lab trace | designs: {} ===".format(", ".join(DESIGNS)))
    print("  tau in [{:+.0f}, {:+.0f}] fs step {:.0f} ({} points) x {} carrier sub-samples"
          .format(-TAU_MAX_FS, TAU_MAX_FS, TAU_STEP_FS, len(taus()), C.SUBSAMPLES))
    print("  {} sims total, pad {:.0f} fs (fixed), res {}, decay {}".format(
        len(jobs), PAD_FS, args.res, args.decay))

    if not args.aggregate_only:
        mine = jobs
        if args.slice:
            i, n = (int(x) for x in args.slice.split("/"))
            mine = [j for k, j in enumerate(jobs) if k % n == i]
        print("  running {} of {} on {} workers".format(len(mine), len(jobs), args.workers))
        t0, done = time.time(), 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_job, *j, args.res, args.decay) for j in mine]
            for _ in as_completed(futs):
                done += 1
                if done % 20 == 0 or done == len(mine):
                    print("    {}/{} ({:.0f}s)".format(done, len(mine), time.time() - t0),
                          flush=True)

    res = aggregate(fin)
    print("\n=== predicted traces ===")
    for label, d in res.items():
        rows = d["rows"]
        env = np.array([abs(r["theta_avg_deg"]) for r in rows])
        fr = np.array([r["fringe_amp_deg"] for r in rows])
        i0 = int(np.argmin([abs(r["tau_fs"]) for r in rows]))
        print("  {:>9s} n={:2d}  theta_avg(0)={:+.5f}°  peak|env|={:.5f}°  "
              "median fringe/env={:.2f}".format(
                  label, len(rows), rows[i0]["theta_avg_deg"], env.max(),
                  float(np.median(fr / np.maximum(env, 1e-12)))))
    path = OUT / "s4_result.json"
    json.dump({"config": {"tau_max_fs": TAU_MAX_FS, "tau_step_fs": TAU_STEP_FS,
                          "pad_fs": PAD_FS, "subsamples": C.SUBSAMPLES,
                          "res": args.res, "decay": args.decay,
                          "pump_intensity": C.PUMP_INTENSITY},
               "designs": res}, open(path, "w"), indent=2)
    print("\n-> {}".format(path))


if __name__ == "__main__":
    main()
