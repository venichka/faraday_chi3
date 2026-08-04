#!/usr/bin/env python
"""Stage 2 -- 1D FDTD ranking on the physical objective. This is the campaign's ground truth.

The analytic proxy proposes; FDTD disposes. Two separate lessons from 2026-06 force this:
  * the analytic FoM mis-ranks GEOMETRY (it was anti-correlated, Spearman -0.92, before the
    v2/v3 correction, and v3 is only a coarse pool selector);
  * it cannot pick the OPERATING POINT at all -- the symmetry-break Re(Sigma) is a delicate
    difference of two nearly-cancelling arms, and the FDTD operating-point diagnostic put the
    true optimum somewhere the selector never looked.
So every geometry is swept over its own operating-point grid in FDTD and scored by the best
result it can reach: max |theta_chi5| = the geometry's actual capability.

TWO-PHASE, to keep the 4x carrier-averaging cost off the wide sweep:
  Phase A (screen)   -- `--screen-subsamples` runs per operating point over the full grid.
  Phase B (confirm)  -- the top `--confirm-top` operating points per geometry get the full
                        4-sub-sample carrier average, which is the reported objective.
Whether Phase A may use 1 sub-sample is decided by Stage 0's correlation study, NOT assumed:
if a single carrier phase does not rank operating points like the carrier average, the screen
is dominated by the fringe and must itself be averaged (`--screen-subsamples 4`).

DoLP is recorded everywhere and gated on, not silently ignored. Rotation is Re[Delta chi] and
ellipticity is Im[Delta chi]; a design that scores a big angle at DoLP 0.8 has converted the
probe to an ellipse rather than rotating it, which is not the effect the experiment wants.

  python chi5_dbr_design/s2_fdtd.py --workers 26 --slice 0/8
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

OUT = HERE / "runs" / "s2_fdtd"
S1 = HERE / "runs" / "s1_screen" / "s1_result.json"

# Operating-point grid per geometry = MAX_PROBES x (MAX_CENTERS + SPAN_CENTERS) x DELTA_GRID = 40.
#
# The center count was raised from 2 to 5 mid-campaign after the first partial ranking exposed a
# real flaw: centers were being chosen as the top-Q pump modes, but EVERY pump-band mode here has
# Q ~ 40-130 against a 100 fs Q_cap of ~12, so all of them are unresolved by the pulse, buildup is
# saturated, and Q says nothing about which center is better. Worse, the fabricated baseline has
# two anomalously high-Q modes at 1.75/1.87 um that crowded out its own 1.547 um design point
# (5th by Q) -- so the baseline scored 0.00115 deg against its known 0.00352 deg, and every
# "N x vs baseline" number was inflated. See common.pump_centers.
MAX_CENTERS = 2          # highest-Q modes (kept: cheap, occasionally right)
SPAN_CENTERS = 3         # plus modes spread across the pump band -- treats all geometries alike
MAX_PAIRS = 4            # plus resonant pump PAIRS (Delta derived) -- the fabricated design's config
MAX_PROBES = 1
# Probe restricted to the ~800 nm window on measured evidence: across 8 geometries with both
# windows evaluated, the 790-810 nm probe beat the 850-950 nm one EVERY time, median 4.0x
# (near-octave matching -- 2 f_pump ~ 1.30 vs f_probe 1.25, 4% off, against 15% off at 900 nm).
# Note probe_modes sorts by Q and often ranks the 860 nm mode first, so the window filter --
# not max_probes -- is what selects correctly.
PROBE_WINDOWS_S2 = [(0.790, 0.810)]
DOLP_MIN = 0.95          # below this the "rotation" is really ellipticity conversion


def load_pool(s1_path: Path, include_baseline: bool = True):
    """[(label, geom, pool_tag, ops)] for every geometry Stage 2 must evaluate."""
    res = json.load(open(s1_path))
    base = C.load_base_geometry()
    entries = []
    if include_baseline:
        entries.append(("baseline", base, "control"))
    for i, r in enumerate(res["top"]):
        p = r["params"]
        g = C.build_geometry(base, int(p["n_left"]), int(p["n_right"]),
                             float(p["t_hi"]), float(p["t_lo"]), float(p["L_cav"]))
        entries.append(("cand{:02d}".format(i), g, r.get("pool", "proxy")))
    out = []
    for label, g, tag in entries:
        ops = C.operating_points(g, max_centers=MAX_CENTERS, max_probes=MAX_PROBES,
                                 span_centers=SPAN_CENTERS, max_pairs=MAX_PAIRS,
                                 probe_windows=PROBE_WINDOWS_S2)
        if not ops:
            print("  skip {}: no TMM probe/pump modes".format(label))
            continue
        out.append((label, g, tag, ops))
    return out


def op_tag(op):
    # probe frequency is part of the identity: the same center/Delta at a different probe mode
    # is a different operating point and must not collide on disk
    return "p{:.4f}_c{:.4f}_d{:.3f}".format(op["probe"], op["center"], op["delta"])


def case_dir(label, op, sub):
    return OUT / label / op_tag(op) / "s{}".format(sub)


def jobs_for(pool, n_sub):
    jobs = []
    for label, geom, _tag, ops in pool:
        for op in ops:
            freqs = {"probe": op["probe"], "pump1": op["pump1"], "pump2": op["pump2"]}
            for s, tau in enumerate(C.subsample_taus(op["pump1"], n_sub)):
                jobs.append((label, geom, op, freqs, s, tau))
    return jobs


def run_job(label, geom, op, freqs, sub, tau, res, decay):
    C.run_case(case_dir(label, op, sub), geom, freqs, tau_fs=tau, res=res, decay=decay)
    return label, op_tag(op), sub


def collect_op(label, op, n_sub):
    recs = [C.read_case(case_dir(label, op, s)) for s in range(n_sub)]
    if any(r is None for r in recs):
        return None
    return C.carrier_average(recs)


def execute(jobs, workers, res, decay, tag=""):
    if not jobs:
        return
    print("  {}running {} sims on {} workers...".format(tag, len(jobs), workers), flush=True)
    t0, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_job, *j, res, decay) for j in jobs]
        for _ in as_completed(futs):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print("    {}/{} ({:.0f}s)".format(done, len(jobs), time.time() - t0), flush=True)
    print("  {}done in {:.0f}s".format(tag, time.time() - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--res", type=int, default=C.RES_1D)
    ap.add_argument("--decay", default=C.DECAY_1D)
    ap.add_argument("--screen-subsamples", type=int, default=C.SUBSAMPLES,
                    help="carrier sub-samples during the wide operating-point screen. "
                         "Set to 1 ONLY if Stage 0 showed single-phase ranks like the average.")
    ap.add_argument("--confirm-top", type=int, default=2,
                    help="operating points per geometry promoted to the full carrier average")
    ap.add_argument("--phase", choices=["screen", "confirm", "both"], default="both")
    ap.add_argument("--slice", default=None, help="i/n -- run only slice i of n")
    ap.add_argument("--s1", default=str(S1))
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_pool(Path(args.s1))
    n_screen = max(1, args.screen_subsamples)
    n_ops = sum(len(ops) for _, _, _, ops in pool)
    print("=== Stage 2 | 1D FDTD carrier-averaged ranking ===")
    print("  {} geometries, {} operating points, screen at {} sub-sample(s)".format(
        len(pool), n_ops, n_screen))
    print("  res {}, decay {}, pad {:.0f} fs, I_pump {:.0e} W/cm2, pulse {:.0f} fs int-FWHM".format(
        args.res, args.decay, C.PAD_FS, C.PUMP_INTENSITY, C.PULSE_INTENSITY_FWHM_FS))

    # ---------------------------------------------------------------- Phase A: screen --- #
    if args.phase in ("screen", "both") and not args.aggregate_only:
        jobs = jobs_for(pool, n_screen)
        if args.slice:
            i, n = (int(x) for x in args.slice.split("/"))
            jobs = [j for k, j in enumerate(jobs) if k % n == i]
        execute(jobs, args.workers, args.res, args.decay, tag="[screen] ")

    # rank operating points per geometry from whatever the screen produced
    promoted = {}
    for label, geom, tag, ops in pool:
        scored = []
        for op in ops:
            r = collect_op(label, op, n_screen)
            if r is not None:
                scored.append((abs(r["theta_chi5_deg"]), op))
        scored.sort(key=lambda t: t[0], reverse=True)
        promoted[label] = [op for _, op in scored[:args.confirm_top]]

    # ---------------------------------------------------------------- Phase B: confirm --- #
    if args.phase in ("confirm", "both") and not args.aggregate_only and n_screen < C.SUBSAMPLES:
        jobs = []
        for label, geom, tag, ops in pool:
            for op in promoted.get(label, []):
                freqs = {"probe": op["probe"], "pump1": op["pump1"], "pump2": op["pump2"]}
                for s, tau in enumerate(C.subsample_taus(op["pump1"], C.SUBSAMPLES)):
                    if s < n_screen:
                        continue          # already have it from the screen
                    jobs.append((label, geom, op, freqs, s, tau))
        if args.slice:
            i, n = (int(x) for x in args.slice.split("/"))
            jobs = [j for k, j in enumerate(jobs) if k % n == i]
        execute(jobs, args.workers, args.res, args.decay, tag="[confirm] ")

    # ------------------------------------------------------------------------ report --- #
    ranking = []
    for label, geom, tag, ops in pool:
        best = None
        per_op = []
        for op in ops:
            # prefer the full carrier average; fall back to the screen depth where Phase B
            # did not promote this operating point
            full = collect_op(label, op, C.SUBSAMPLES)
            r = full if full is not None else collect_op(label, op, n_screen)
            if r is None:
                continue
            n_have = C.SUBSAMPLES if full is not None else n_screen
            rec = {"op": op, "n_sub": n_have, **r}
            per_op.append(rec)
            # only a fully carrier-averaged point may set the reported score
            if n_have == C.SUBSAMPLES and (best is None or
                                           abs(r["theta_chi5_deg"]) > abs(best["theta_chi5_deg"])):
                best = rec
        if best is None and per_op:
            best = max(per_op, key=lambda r: abs(r["theta_chi5_deg"]))
        if best is None:
            continue
        ranking.append({"label": label, "pool": tag, "params": C.geometry_params(geom),
                        "best": best, "per_op": per_op})
    ranking.sort(key=lambda r: abs(r["best"]["theta_chi5_deg"]), reverse=True)

    base = next((r for r in ranking if r["label"] == "baseline"), None)
    bt = abs(base["best"]["theta_chi5_deg"]) if base else None
    print("\n=== ranking by |theta_chi5| (carrier-averaged, pulse-integrated) ===")
    print("{:>9s} {:>8s} {:>3s} {:>3s} {:>7s} {:>7s} {:>7s} {:>11s} {:>7s} {:>6s} {:>6s} {:>8s}"
          .format("label", "pool", "nL", "nR", "t_hi", "t_lo", "L_cav", "|th_chi5|", "vs base",
                  "Delta", "DoLP", "fringe"))
    for r in ranking:
        p, b = r["params"], r["best"]
        th = abs(b["theta_chi5_deg"])
        rel = "{:6.2f}x".format(th / bt) if bt else "   --  "
        flag = "" if b["dolp"] >= DOLP_MIN else "  <-- DoLP low: ellipticity, not rotation"
        print("{:>9s} {:>8s} {:>3d} {:>3d} {:>7.4f} {:>7.4f} {:>7.3f} {:>11.5f} {:>7s} "
              "{:>6.3f} {:>6.3f} {:>8.4f}{}".format(
                  r["label"], r["pool"], p["n_left"], p["n_right"], p["t_hi"], p["t_lo"],
                  p["L_cav"], th, rel, b["op"]["delta"], b["dolp"],
                  b["theta_fringe_amp_deg"], flag))

    # did the analytic pre-rank have any skill? (proxy half vs diversity half)
    for tag in ("proxy", "diverse"):
        vals = [abs(r["best"]["theta_chi5_deg"]) for r in ranking if r["pool"] == tag]
        if vals:
            print("  {:>8s} half: n={}, median |theta| = {:.5f}, best = {:.5f}".format(
                tag, len(vals), float(np.median(vals)), float(np.max(vals))))
    proxy_fom = {}
    try:
        s1 = json.load(open(args.s1))
        for i, r in enumerate(s1["top"]):
            proxy_fom["cand{:02d}".format(i)] = r["fom"]
        pairs = [(proxy_fom[r["label"]], abs(r["best"]["theta_chi5_deg"]))
                 for r in ranking if r["label"] in proxy_fom]
        if len(pairs) >= 3:
            rho = C.spearman([a for a, _ in pairs], [b for _, b in pairs])
            print("  Spearman(analytic v3 FoM, FDTD |theta_chi5|) = {:+.3f}  over {} candidates"
                  .format(rho, len(pairs)))
    except Exception as e:
        print("  (proxy-skill check skipped: {})".format(e))

    path = OUT / "s2_result.json"
    json.dump({"config": {"res": args.res, "decay": args.decay,
                          "screen_subsamples": n_screen, "confirm_top": args.confirm_top,
                          "pulse_label_fs": C.PULSE_LABEL_FS, "dolp_min": DOLP_MIN},
               "ranking": ranking}, open(path, "w"), indent=2)
    print("\n-> {}".format(path))

    # stage the winners for Stage 3
    for r in ranking[:3]:
        if r["label"] == "baseline":
            continue
        wd = OUT / "winners" / r["label"]
        wd.mkdir(parents=True, exist_ok=True)
        p = r["params"]
        g = C.build_geometry(C.load_base_geometry(), int(p["n_left"]), int(p["n_right"]),
                             p["t_hi"], p["t_lo"], p["L_cav"])
        op = r["best"]["op"]
        json.dump(g, open(wd / "geometry.json", "w"), indent=2)
        json.dump(C.modes_json(op["probe"], op["pump1"], op["pump2"]),
                  open(wd / "cavity_modes.json", "w"), indent=2)


if __name__ == "__main__":
    main()
