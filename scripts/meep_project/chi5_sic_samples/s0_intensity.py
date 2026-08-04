#!/usr/bin/env python
"""Stage 0 -- pre-flight: pick the reference pump intensity for the SiC samples.

The SiC cavity has ~10x the n2 of SiN, so the intensity that keeps the SiN sample comfortably
perturbative (1e12 W/cm^2) does NOT here: a single-phase probe at 1e12 comes back with
DoLP 0.72 and ~0.9 deg of rotation, i.e. deep in the large-signal regime. That is the same trap
the 2026-06 SiC study fell into ("the probe spins ~370 deg during the pulse then freezes").

A chi5 analysis is only meaningful where the response is perturbative:
  * DoLP stays ~1 (the probe stays linearly polarized -- an azimuth is only defined if it does),
  * the local log-log slope d ln|theta| / d ln I is ~2 (chi5), not rolling over.

This runs a carrier-averaged intensity ladder at one FWM-matched operating point per cavity and
reports both, so the main scan runs at an intensity where the numbers mean what we say they do.

  python chi5_sic_samples/s0_intensity.py --workers 24
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

OUT = HERE / "runs" / "s0_intensity"
INTENSITIES = [1.0e10, 2.5e10, 5.0e10, 1.0e11, 2.5e11, 5.0e11, 1.0e12]


def ref_op(geom, probe_window=(0.790, 0.810)):
    """One representative operating point: the ~800 nm probe mode (reachable today),
    FWM-matched pumps, mid-range Delta."""
    ops = S.operating_points(geom, [probe_window], center_offsets=(0.0,), deltas=(0.014,))
    if not ops:
        raise RuntimeError("no operating point in {}".format(probe_window))
    return ops[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--res", type=int, default=C.RES_1D)
    ap.add_argument("--decay", default=C.DECAY_1D)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    jobs, plan = [], {}
    for name, L in S.CAVITY_LENGTHS_UM.items():
        geom = S.sic_geometry(L)
        op = ref_op(geom)
        plan[name] = {"geom": geom, "op": op}
        print("{}: probe {:.1f} nm, pumps {:.1f}/{:.1f} nm, Delta {:.3f}".format(
            name, op["probe_nm"], op["pump1_nm"], op["pump2_nm"], op["delta"]))
        for I in INTENSITIES:
            for sub, tau in enumerate(C.subsample_taus(op["pump1"])):
                d = OUT / name / "I{:.2e}".format(I) / "s{}".format(sub)
                jobs.append((d, geom, op, sub, tau, I))
    print("{} sims".format(len(jobs)))

    if not args.aggregate_only:
        t0, done = time.time(), 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(S.run_case, d, g, o, s, t, args.res, args.decay,
                              C.PAD_FS, 1, I) for (d, g, o, s, t, I) in jobs]
            for _ in as_completed(futs):
                done += 1
                if done % 8 == 0 or done == len(jobs):
                    print("  {}/{} ({:.0f}s)".format(done, len(jobs), time.time() - t0),
                          flush=True)

    report = {}
    for name in plan:
        rows = []
        for I in INTENSITIES:
            recs = [C.read_case(OUT / name / "I{:.2e}".format(I) / "s{}".format(s))
                    for s in range(C.SUBSAMPLES)]
            if any(r is None for r in recs):
                continue
            a = C.carrier_average(recs)
            rows.append({"I": I, "theta": abs(a["theta_chi5_deg"]),
                         "fringe": a["theta_fringe_amp_deg"], "dolp": a["dolp"],
                         "contrast": abs(a["theta_chi5_deg"]) /
                                     max(a["theta_fringe_amp_deg"], 1e-12)})
        report[name] = {"op": plan[name]["op"], "rows": rows}
        print("\n=== {} (SiC L={} um), probe {:.1f} nm ===".format(
            name, S.CAVITY_LENGTHS_UM[name], plan[name]["op"]["probe_nm"]))
        print("  {:>10s} {:>11s} {:>10s} {:>8s} {:>9s} {:>12s}".format(
            "I (W/cm2)", "|theta| deg", "fringe", "contrast", "DoLP", "local slope"))
        for i, r in enumerate(rows):
            if 0 < i < len(rows) - 1:
                sl = (np.log(rows[i + 1]["theta"] / rows[i - 1]["theta"]) /
                      np.log(rows[i + 1]["I"] / rows[i - 1]["I"]))
                sls = "{:12.2f}".format(sl)
            else:
                sls = "{:>12s}".format("-")
            flag = "" if r["dolp"] > 0.95 else ("  <- DoLP low" if r["dolp"] > 0.90
                                                else "  <- DEPOLARIZED")
            print("  {:10.2e} {:11.5f} {:10.5f} {:8.2f} {:9.4f} {}{}".format(
                r["I"], r["theta"], r["fringe"], r["contrast"], r["dolp"], sls, flag))

    json.dump({"intensities": INTENSITIES, "samples": report},
              open(OUT / "s0_result.json", "w"), indent=2, default=float)
    print("\n-> {}".format(OUT / "s0_result.json"))


if __name__ == "__main__":
    main()
