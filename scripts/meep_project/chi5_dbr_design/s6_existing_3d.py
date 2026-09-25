#!/usr/bin/env python
"""Stage 6 -- 3D check of the EXISTING sample's two lab-accessible optima from Stage 5.

Stage 5 (1D) found that the fabricated best_absolute sample has two accessible optima that share
the SAME pumps (1563.2/1598.2 nm, Delta = 0.014) and differ only in the probe mode:

  p867_contrast  probe 867.3 nm -> contrast 4.10, 0.00572 deg   (effect beats the fringe)
  p800_signal    probe 800.1 nm -> contrast 0.11, 0.00822 deg   (most signal)

Both numbers are 1D.  The SiC campaign showed a 1D contrast can be a plane-wave artifact
(L=3.2: 1.08 in 1D -> 0.07 in 3D, its fringe grew 43x), so the 867.3 nm contrast must be
re-checked in 3D before the lab relies on it.  Same 3D settings as s3_validate (res 30,
decay 1e-3, 4 carrier sub-samples, I = 1e12).

  python chi5_dbr_design/s6_existing_3d.py --print-args --index 0     # used by the sbatch
  python chi5_dbr_design/s6_existing_3d.py --aggregate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT = HERE / "runs" / "s6_existing_3d"
S5 = HERE / "runs" / "s5_existing" / "s5_result_all.json"
FROZEN = OUT / "ops_result.json"
RES_3D, DECAY_3D = 30, "1e-3"


def contrast(r):
    return abs(r["theta_chi5_deg"]) / max(r["theta_fringe_amp_deg"], 1e-12)


def freeze():
    """Pick both operating points from the Stage 5 map once, then never re-pick."""
    if FROZEN.exists():
        return json.load(open(FROZEN))
    rows = json.load(open(S5))["rows"]
    at = lambda nm: [r for r in rows if abs(r["op"]["probe_nm"] - nm) < 1.0]  # noqa: E731
    best_c = max(at(867.3), key=contrast)
    best_s = max(at(800.1), key=lambda r: abs(r["theta_chi5_deg"]))
    ops = []
    for label, r in (("p867_contrast", best_c), ("p800_signal", best_s)):
        ops.append({"label": label, "op": r["op"],
                    "theta_1d_deg": r["theta_chi5_deg"],
                    "fringe_1d_deg": r["theta_fringe_amp_deg"],
                    "contrast_1d": contrast(r), "dolp_1d": r["dolp"]})
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(ops, open(FROZEN, "w"), indent=2)
    return ops


def jobs():
    return [(o, s) for o in freeze() for s in range(C.SUBSAMPLES)]


def print_args(index):
    """Serial, rank-free: emit one 3D job's argument list, or nothing if it is already done."""
    js = jobs()
    if not (0 <= index < len(js)):
        return
    o, sub = js[index]
    op = o["op"]
    out = OUT / o["label"] / "s{}".format(sub)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "faraday_summary.json").exists():
        return
    json.dump(C.load_base_geometry(), open(out / "geometry.json", "w"))
    json.dump(C.modes_json(op["probe"], op["pump1"], op["pump2"]),
              open(out / "cavity_modes.json", "w"))
    tau = C.subsample_taus(op["pump1"])[sub]
    cmd = C.fdtd_cmd(out, RES_3D, DECAY_3D, tau, C.PAD_FS, dim=3)
    print(" ".join(cmd[2:]))          # drop [python, script]; the sbatch supplies those


def aggregate():
    rows = {}
    print("  {:15s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>7s}".format(
        "label", "theta_1D", "theta_3D", "fringe_3D", "contr_1D", "contr_3D", "DoLP"))
    for o in freeze():
        recs = [C.read_case(OUT / o["label"] / "s{}".format(s)) for s in range(C.SUBSAMPLES)]
        if any(r is None for r in recs):
            print("  {:15s} (incomplete)".format(o["label"]))
            continue
        a = C.carrier_average(recs)
        t3, f3 = abs(a["theta_chi5_deg"]), a["theta_fringe_amp_deg"]
        rows[o["label"]] = {"theta_3d_deg": a["theta_chi5_deg"], "fringe_3d_deg": f3,
                            "contrast_3d": t3 / max(f3, 1e-12), "dolp_3d": a["dolp"],
                            "theta_1d_deg": o["theta_1d_deg"], "fringe_1d_deg": o["fringe_1d_deg"],
                            "contrast_1d": o["contrast_1d"],
                            "ratio_3d_1d": t3 / abs(o["theta_1d_deg"]), "op": o["op"]}
        r = rows[o["label"]]
        print("  {:15s} {:10.5f} {:10.5f} {:10.5f} {:10.2f} {:10.2f} {:7.4f}".format(
            o["label"], abs(o["theta_1d_deg"]), t3, f3, o["contrast_1d"], r["contrast_3d"],
            r["dolp_3d"]))
    json.dump(rows, open(OUT / "s6_result.json", "w"), indent=2)
    print("-> {}".format(OUT / "s6_result.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-args", action="store_true")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.print_args:
        print_args(args.index)
    elif args.aggregate:
        aggregate()
    else:
        for o in freeze():
            print(o["label"], json.dumps(o["op"]))
        print("{} 3D jobs".format(len(jobs())))


if __name__ == "__main__":
    main()
