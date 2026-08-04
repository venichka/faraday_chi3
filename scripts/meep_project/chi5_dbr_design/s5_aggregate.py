#!/usr/bin/env python
"""Aggregate EVERY Stage 5 case on disk, independent of the current op list.

`s5_existing.py` writes `s5_result.json` from the operating points of the invocation that is
running, so a phase-B run at one probe overwrites the phase-A map. This walks the run directory
instead, so the file always reflects everything that has actually been simulated.

  python chi5_dbr_design/s5_aggregate.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

OUT = HERE / "runs" / "s5_existing"
TAG = re.compile(r"^p(?P<probe>[\d.]+)_c(?P<center>[\d.]+)_d(?P<delta>[\d.]+)$")
DESIGN_REF = 0.003519      # as-fabricated operating point, true 100 fs (Stage 0 Part A)
LAB_WINDOWS = [(0.790, 0.810), (0.850, 0.950)]


def accessible(probe_nm):
    return any(lo * 1000 <= probe_nm <= hi * 1000 for lo, hi in LAB_WINDOWS)


def main():
    rows = []
    for d in sorted(OUT.iterdir()):
        m = TAG.match(d.name) if d.is_dir() else None
        if not m:
            continue
        recs = [C.read_case(d / "s{}".format(s)) for s in range(C.SUBSAMPLES)]
        if any(r is None for r in recs):
            continue
        a = C.carrier_average(recs)
        probe = float(m.group("probe"))
        center = float(m.group("center"))
        delta = float(m.group("delta"))
        rows.append({
            "op": {"probe": probe, "probe_nm": 1000.0 / probe,
                   "center": center, "center_nm": 1000.0 / center,
                   "pump1": center + 0.5 * delta, "pump2": center - 0.5 * delta,
                   "pump1_nm": 1000.0 / (center + 0.5 * delta),
                   "pump2_nm": 1000.0 / (center - 0.5 * delta),
                   "delta": delta,
                   "fwm_mismatch_pct": 100.0 * abs(2.0 * center - probe) / probe,
                   "lab_accessible": accessible(1000.0 / probe)},
            **{k: v for k, v in a.items()
               if k not in ("theta_sub_deg", "vmh_sub", "legacy_sub_deg")},
        })
    rows.sort(key=lambda r: -abs(r["theta_chi5_deg"]))
    json.dump({"design_reference_deg": DESIGN_REF, "n_ops": len(rows),
               "lab_probe_windows_um": LAB_WINDOWS, "rows": rows},
              open(OUT / "s5_result_all.json", "w"), indent=2)

    probes = sorted({round(r["op"]["probe_nm"], 1) for r in rows})
    print("Stage 5 -- complete map: {} operating points over {} probe modes".format(
        len(rows), len(probes)))
    print("\n{:>9s} {:>5s} {:>4s} {:>11s} {:>9s} {:>9s} {:>9s} {:>7s}".format(
        "probe_nm", "reach", "nops", "best|theta|", "vs design", "fringe", "contrast", "DoLP"))
    for p in probes:
        sel = [r for r in rows if abs(r["op"]["probe_nm"] - p) < 0.05]
        b = max(sel, key=lambda r: abs(r["theta_chi5_deg"]))
        bc = max(sel, key=lambda r: abs(r["theta_chi5_deg"]) /
                 max(r["theta_fringe_amp_deg"], 1e-12))
        t = abs(b["theta_chi5_deg"])
        print("{:>9.1f} {:>5s} {:>4d} {:>11.5f} {:>8.2f}x {:>9.5f} {:>9.2f} {:>7.4f}".format(
            p, "YES" if b["op"]["lab_accessible"] else "no", len(sel), t, t / DESIGN_REF,
            b["theta_fringe_amp_deg"],
            abs(bc["theta_chi5_deg"]) / max(bc["theta_fringe_amp_deg"], 1e-12), b["dolp"]))

    acc = [r for r in rows if r["op"]["lab_accessible"]]
    for label, sel in (("ALL probes", rows), ("LAB-ACCESSIBLE only", acc)):
        if not sel:
            continue
        bt = max(sel, key=lambda r: abs(r["theta_chi5_deg"]))
        bc = max(sel, key=lambda r: abs(r["theta_chi5_deg"]) /
                 max(r["theta_fringe_amp_deg"], 1e-12))
        print("\n{}:".format(label))
        for tag, r in (("max signal ", bt), ("max contrast", bc)):
            op = r["op"]
            t = abs(r["theta_chi5_deg"])
            print("  {}: probe {:.1f} nm, pumps {:.1f}/{:.1f} nm, Delta {:.4f} -> {:.5f} deg "
                  "({:.2f}x), contrast {:.2f}, DoLP {:.4f}, FWM mismatch {:.2f}%".format(
                      tag, op["probe_nm"], op["pump1_nm"], op["pump2_nm"], op["delta"], t,
                      t / DESIGN_REF, t / max(r["theta_fringe_amp_deg"], 1e-12), r["dolp"],
                      op["fwm_mismatch_pct"]))
    print("\n-> {}".format(OUT / "s5_result_all.json"))


if __name__ == "__main__":
    main()
