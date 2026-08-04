#!/usr/bin/env python
"""Stage 5 -- how much more can the EXISTING fabricated sample give, with no new fabrication?

The geometry is FROZEN at SiN_optimizations/best_absolute (the sample already made and measured).
The only knobs the lab actually has are:

  * probe wavelength   -- tune the probe laser to a different cavity mode of the same stack
  * pump center        -- where the sigma+/sigma- pair sits in the mid-IR
  * pump splitting Delta -- the pair separation (the M5 lever)
  * pump intensity     -- the I^2 law (measured separately in s3_validate --what intensity)
  * measurement procedure -- carrier-phase dithering (delay_physics / Stage 4)

Stage 2 already showed a 2.03x gain from retuning the pumps alone, but it only saw 18 operating
points on this geometry and restricted the probe to the 790-810 nm window. This stage maps the
(probe x center x Delta) landscape properly, so the answer is "here is the optimum and here is
how precisely you must hit it", not "here is the best of the few points we happened to try".

Two phases, both carrier-averaged (4 sub-samples -- Stage 0 proved no cheaper estimator ranks
correctly):
  A  coarse: every cavity mode in 750-1000 nm x 6 pump centers x 3 Delta
  B  fine:   the best probe from A x 12 centers x 6 Delta  -> the map and the sensitivity

  python chi5_dbr_design/s5_existing.py --phase A --workers 26 --slice 0/6
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

OUT = HERE / "runs" / "s5_existing"

PROBE_BAND = (0.750, 1.000)          # um -- every mode the STACK supports, accessible or not
# What the probe laser can actually reach (user, 2026-08-02/03). The scan deliberately covers a
# wider band so the cost of the restriction is quantified, but only these windows are quotable
# as achievable today: 763.6 nm is the global optimum and is NOT reachable.
LAB_PROBE_WINDOWS = [(0.790, 0.810), (0.850, 0.950)]


def lab_accessible(probe_nm: float) -> bool:
    return any(lo * 1000 <= probe_nm <= hi * 1000 for lo, hi in LAB_PROBE_WINDOWS)


CENTERS_A = np.linspace(0.56, 0.72, 6)      # 1/um  (1.79 -> 1.39 um)
DELTAS_A = [0.010, 0.016, 0.023]
CENTERS_B = np.linspace(0.60, 0.72, 12)
DELTAS_B = [0.008, 0.011, 0.014, 0.017, 0.020, 0.023]


def geometry():
    return C.load_base_geometry()


def probe_candidates(geom):
    """Every TMM cavity mode in the probe band, high-Q first (the lab tunes the laser to one)."""
    return C.probe_modes(geom, windows=[PROBE_BAND])


def nearest_mode(modes, f_target):
    return min(modes, key=lambda m: abs(m["freq"] - f_target))


def ops_for(phase, geom, probe_freqs):
    """Operating points. Pump centers are snapped to the nearest cavity mode where one is close,
    otherwise used as-is -- the lab can set any wavelength, and Stage 2 showed the pump-band
    modes are all far broader than the 100 fs pulse resolves, so exact mode-centring is not
    required."""
    centers = CENTERS_A if phase == "A" else CENTERS_B
    deltas = DELTAS_A if phase == "A" else DELTAS_B
    ops = []
    for fp in probe_freqs:
        for c in centers:
            for d in deltas:
                ops.append({"probe": float(fp), "center": float(c),
                            "pump1": float(c) + 0.5 * d, "pump2": float(c) - 0.5 * d,
                            "delta": float(d), "probe_nm": 1000.0 / float(fp)})
    return ops


def tag(op):
    return "p{:.4f}_c{:.4f}_d{:.3f}".format(op["probe"], op["center"], op["delta"])


def case_dir(op, sub):
    return OUT / tag(op) / "s{}".format(sub)


def run_job(geom, op, sub, tau, res, decay):
    C.run_case(case_dir(op, sub), geom,
               {"probe": op["probe"], "pump1": op["pump1"], "pump2": op["pump2"]},
               tau_fs=tau, res=res, decay=decay)
    return tag(op), sub


def collect(op):
    recs = [C.read_case(case_dir(op, s)) for s in range(C.SUBSAMPLES)]
    if any(r is None for r in recs):
        return None
    return C.carrier_average(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "both"], default="both")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--res", type=int, default=C.RES_1D)
    ap.add_argument("--decay", default=C.DECAY_1D)
    ap.add_argument("--slice", default=None)
    ap.add_argument("--probe-nm", default=None,
                    help="comma-separated probe wavelengths (nm) to use for the fine phase B, "
                         "instead of the best from phase A. Use this to refine the LAB-ACCESSIBLE "
                         "modes, since the global optimum (763.6 nm) is out of reach.")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    geom = geometry()
    modes = probe_candidates(geom)
    print("=== Stage 5 | existing sample (best_absolute), no refabrication ===")
    print("  probe modes available in {:.0f}-{:.0f} nm:".format(PROBE_BAND[0] * 1000,
                                                                PROBE_BAND[1] * 1000))
    for m in modes:
        print("    {:7.1f} nm  Q {:6.1f}  T_peak {:.3f}".format(
            m["lambda_um"] * 1000, m["Q"], m["T_peak"]))

    phases = ["A", "B"] if args.phase == "both" else [args.phase]
    all_ops = []
    for ph in phases:
        if ph == "A":
            probes = [m["freq"] for m in modes]
        else:
            # best probe from phase A, falling back to the highest-Q mode if A is not in yet
            if args.probe_nm:
                want = [float(x) for x in args.probe_nm.split(",")]
                probes = [nearest_mode(modes, 1000.0 / w)["freq"] for w in want]
                print("  phase B probes (requested): "
                      + ", ".join("{:.1f} nm".format(1000.0 / f) for f in probes))
            else:
                scored = []
                for op in ops_for("A", geom, [m["freq"] for m in modes]):
                    r = collect(op)
                    if r:
                        scored.append((abs(r["theta_chi5_deg"]), op["probe"]))
                if scored:
                    scored.sort(reverse=True)
                    probes = [scored[0][1]]
                    print("  phase B probe (best from A): {:.1f} nm".format(1000.0 / probes[0]))
                else:
                    probes = [modes[0]["freq"]]
        all_ops += ops_for(ph, geom, probes)

    # dedup (phase A and B grids can overlap)
    seen, ops = set(), []
    for op in all_ops:
        if tag(op) not in seen:
            seen.add(tag(op))
            ops.append(op)

    jobs = []
    for op in ops:
        for s, tau in enumerate(C.subsample_taus(op["pump1"])):
            jobs.append((geom, op, s, tau))
    print("  {} operating points -> {} sims (res {}, decay {}, I {:.0e})".format(
        len(ops), len(jobs), args.res, args.decay, C.PUMP_INTENSITY))

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
                if done % 25 == 0 or done == len(mine):
                    print("    {}/{} ({:.0f}s)".format(done, len(mine), time.time() - t0),
                          flush=True)

    rows = []
    for op in ops:
        r = collect(op)
        if r:
            rows.append({"op": op, **{k: v for k, v in r.items()
                                      if k not in ("theta_sub_deg", "vmh_sub", "legacy_sub_deg")}})
    rows.sort(key=lambda r: -abs(r["theta_chi5_deg"]))

    design = 0.003519   # as-fabricated operating point, true 100 fs pulse (Stage 0 Part A)
    print("\n=== best operating points for the EXISTING sample ===")
    print("  {:>9s} {:>10s} {:>10s} {:>8s} {:>11s} {:>8s} {:>7s} {:>8s}".format(
        "probe_nm", "pump1_nm", "pump2_nm", "Delta", "|theta_chi5|", "vs design", "DoLP",
        "contrast"))
    for r in rows[:12]:
        op = r["op"]
        t = abs(r["theta_chi5_deg"])
        print("  {:>9.1f} {:>10.1f} {:>10.1f} {:>8.4f} {:>11.5f} {:>7.2f}x {:>7.4f} {:>8.2f}"
              .format(op["probe_nm"], 1000.0 / op["pump1"], 1000.0 / op["pump2"], op["delta"],
                      t, t / design, r["dolp"], t / max(r["theta_fringe_amp_deg"], 1e-12)))
    json.dump({"design_reference_deg": design, "n_ops": len(rows),
               "probe_modes": [{"lambda_nm": m["lambda_um"] * 1000, "Q": m["Q"],
                                "T_peak": m["T_peak"]} for m in modes],
               "config": {"res": args.res, "decay": args.decay,
                          "pump_intensity": C.PUMP_INTENSITY,
                          "pulse_label_fs": C.PULSE_LABEL_FS},
               "rows": rows}, open(OUT / "s5_result.json", "w"), indent=2)
    print("\n-> {}".format(OUT / "s5_result.json"))


if __name__ == "__main__":
    main()
