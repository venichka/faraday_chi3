#!/usr/bin/env python
"""Narrow-Delta operating-point refinement of the SiN hybrid winners.

The hybrid Stage-B grid (Delta in {0.015, 0.022, 0.030}) railed at its LOW edge: every SiN winner
peaked at Delta=0.015 with DoLP ~0.99, so the small-Delta M5 lever was never bracketed and the
reported max|theta| are LOWER BOUNDS. Here we refine the top-3 SiN winners (cand07/cand01/cand02)
on a finer grid that brackets Delta BELOW 0.015 and scans the pump center finely around each
winner's anchor (to land on the true FDTD resonance, not the discrete TMM mode).

Probe stays fixed at the geometry's TMM probe mode (same as the hybrid). Same FDTD config as the
hybrid (res 80, decay 1e-4, I=1e12, --materials fit) so the numbers are directly comparable.

  python chi5_optimization/refine_sin.py --workers 75
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hybrid as H  # noqa: E402  (reuse run_fdtd / fdtd_theta / probe_mode / MAT)

WINNERS = ["cand07", "cand01", "cand02"]
# Finer grid: brackets Delta BELOW the railed 0.015, and a +-0.006 center neighbourhood (well within
# the ~0.047 100fs pump bandwidth, so buildup stays flat -- this scans the FWM/sideband alignment).
DELTAS = [0.008, 0.010, 0.012, 0.014, 0.016]
CENTER_OFFSETS = [-0.006, -0.003, 0.0, 0.003, 0.006]


def _flist(s):
    return [float(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=75)
    ap.add_argument("--res", type=int, default=80)
    ap.add_argument("--decay", default="1e-4")
    ap.add_argument("--deltas", type=_flist, default=DELTAS,
                    help="comma list of Delta values")
    ap.add_argument("--center-offsets", type=_flist, default=CENTER_OFFSETS,
                    help="comma list of pump-center offsets around the anchor")
    ap.add_argument("--anchor", choices=["hybrid", "refine"], default="hybrid",
                    help="center anchor: hybrid result (default) or a prior refine_result.json best center")
    ap.add_argument("--anchor-file", default=None,
                    help="refine_result.json to read best centers from when --anchor refine")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    deltas, center_offsets = args.deltas, args.center_offsets

    mat = "sin"
    win_root = HERE / "hybrid" / "sin" / "winners"
    if args.anchor == "refine":
        af = Path(args.anchor_file) if args.anchor_file else HERE / "refine_sin" / "refine_result.json"
        prev = json.load(open(af))["winners"]
        anchor = {lab: {"center": v["best"]["center"], "max_theta": v["best"]["theta"],
                        "delta": v["best"]["delta"]} for lab, v in prev.items()}
    else:
        result = json.load(open(HERE / "hybrid" / "sin" / "hybrid_result.json"))
        anchor = {r["label"]: r for r in result["ranking"]}
    outroot = Path(args.out) if args.out else HERE / "refine_sin"
    outroot.mkdir(parents=True, exist_ok=True)

    # Build the job list.
    plan, jobs = {}, []
    print("=== SiN narrow-Delta refinement | winners={} ===".format(WINNERS), flush=True)
    for label in WINNERS:
        geom = json.load(open(win_root / label / "geometry.json"))
        fp = H.probe_mode(geom, "SiO2")
        c0 = anchor[label]["center"]
        centers = [round(c0 + off, 5) for off in center_offsets]
        plan[label] = {"geom": geom, "fprobe": fp, "centers": centers,
                       "hybrid_theta": anchor[label]["max_theta"],
                       "hybrid_center": c0, "hybrid_delta": anchor[label]["delta"]}
        print("  {}: probe={:.1f}nm  anchor center={:.4f} (hybrid {:.4f} deg @Delta {:.3f}) "
              "centers={} Deltas={}".format(label, 1000.0 / fp, c0, anchor[label]["max_theta"],
                                            anchor[label]["delta"], centers, deltas), flush=True)
        for c in centers:
            for d in deltas:
                jobs.append((mat, label, geom, fp, float(c), float(d), args.res, args.decay, outroot))

    print("\n{} 1D-FDTD sims ({} winners x {} centers x {} Deltas, res {}, decay {}, {} workers)...".format(
        len(jobs), len(WINNERS), len(center_offsets), len(deltas), args.res, args.decay, args.workers),
        flush=True)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(H.run_fdtd, *j) for j in jobs]):
            done += 1
            if done % 10 == 0:
                print("  {}/{} ({:.0f}s)".format(done, len(jobs), time.time() - t0), flush=True)
    print("sims in {:.0f}s".format(time.time() - t0), flush=True)

    # Collect the full grid + best per winner.
    out = {"material": mat, "deltas": deltas, "center_offsets": center_offsets, "winners": {}}
    print("\n=== Refinement result (max|theta| per winner; hybrid Stage-B in parens) ===")
    print("{:>8s} {:>9s} {:>9s} {:>8s} {:>6s}   {:>22s}".format(
        "winner", "max|th|", "@center", "@Delta", "DoLP", "vs hybrid Stage-B"))
    for label, info in plan.items():
        grid, best = [], None
        for c in info["centers"]:
            for d in deltas:
                r = H.fdtd_theta(outroot, label, c, d)
                if not r:
                    continue
                rec = {"center": c, "delta": d, "theta": r[0], "dolp": r[1]}
                grid.append(rec)
                if best is None or r[0] > best["theta"]:
                    best = rec
        out["winners"][label] = {"hybrid_theta": info["hybrid_theta"], "hybrid_center": info["hybrid_center"],
                                 "hybrid_delta": info["hybrid_delta"], "probe_nm": 1000.0 / info["fprobe"],
                                 "best": best, "grid": grid}
        if best:
            gain = best["theta"] / info["hybrid_theta"] if info["hybrid_theta"] else float("nan")
            railed = "(Delta railed)" if abs(best["delta"] - deltas[0]) < 1e-9 or abs(best["delta"] - deltas[-1]) < 1e-9 else ""
            print("{:>8s} {:9.4f} {:9.4f} {:8.3f} {:6.3f}   {:.4f} deg -> {:.2f}x {}".format(
                label, best["theta"], best["center"], best["delta"], best["dolp"],
                info["hybrid_theta"], gain, railed))
    json.dump(out, open(outroot / "refine_result.json", "w"), indent=2)
    print("-> {}".format(outroot / "refine_result.json"))


if __name__ == "__main__":
    main()
