#!/usr/bin/env python
"""Stage 2 -- read the SiC operating-point map and answer the lab's three questions.

For each sample (SiC cavity L=3.2, L=4.8) and each probe scenario:
  * which probe / pump1 / pump2 / Delta to use,
  * what rotation to expect (and how it compares to the fabricated SiN sample),
  * whether the effect beats the carrier fringe (contrast > 1) or needs delay dithering.

The scan covered 600-1000 nm; the scenarios are applied here as filters:
  now     {~800} u [850, 950] nm   -- achievable today
  future  [600, 900] nm            -- if the probe source is extended
  all     whatever was scanned     -- to expose optima outside both windows

  python chi5_sic_samples/s2_analyze.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_sic as S  # noqa: E402

RUNS = HERE / "runs" / "s1_scan"
DOCS = HERE / "docs"

# The fabricated SiN sample, same estimator, same pulse, at I = 1e12 W/cm^2
# (chi5_dbr_design Stage 0 Part A / Stage 5).
SIN_REF_I = 1e12
SIN_AS_FAB_DEG = 0.003519      # as built
SIN_RETUNED_DEG = 0.00822      # best retune inside today's probe range (Stage 5 fine scan)
SIN_BEST_ANY_DEG = 0.02185     # best over ALL probe modes (763.6 nm, not lab-reachable)
SIN_SLOPE = 1.99               # measured local log-log slope of the SiN baseline

# ⚠️ The SiC samples must be pumped ~10x weaker than the SiN one (above ~2e11 W/cm^2 their
# probe depolarizes and the I^2 law fails -- see s0_intensity). So "SiC vs SiN" has TWO honest
# readings and both are reported:
#   equal-intensity  -- the MATERIAL gain, SiN extrapolated down to the SiC intensity by I^1.99
#   as-operated      -- what each sample actually delivers at ITS OWN usable intensity, which
#                       is the number that decides which sample to put on the bench


def sin_at(intensity):
    """SiN baseline scaled to a given intensity by its measured I^1.99 law."""
    return SIN_AS_FAB_DEG * (intensity / SIN_REF_I) ** SIN_SLOPE


def contrast(r):
    return abs(r["theta_chi5_deg"]) / max(r["theta_fringe_amp_deg"], 1e-12)


def load(sample):
    p = RUNS / sample / "result.json"
    if not p.exists():
        return None
    return json.load(open(p))


def scenario_rows(res, scenario):
    if scenario == "all":
        return list(res["rows"])
    wins = S.probe_scenario_windows(scenario)
    return [r for r in res["rows"]
            if S.in_windows(r["op"]["probe_nm"] / 1000.0, wins)]


def best(rows, key):
    if not rows:
        return None
    if key == "theta":
        return max(rows, key=lambda r: abs(r["theta_chi5_deg"]))
    return max(rows, key=contrast)


def fmt_op(r):
    op = r["op"]
    t = abs(r["theta_chi5_deg"])
    return ("probe {:.1f} nm | pumps {:.1f} / {:.1f} nm (centre {:.1f}, Delta {:.4f}) | "
            "theta {:.5f} deg | fringe {:.5f} | contrast {:.2f} | DoLP {:.3f} | "
            "FWM mismatch {:.2f}%".format(
                op["probe_nm"], op["pump1_nm"], op["pump2_nm"], op["center_nm"], op["delta"],
                t, r["theta_fringe_amp_deg"], contrast(r), r["dolp"], op["fwm_mismatch_pct"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="L3p2,L4p8")
    args = ap.parse_args()
    samples = args.samples.split(",")

    print("=" * 100)
    print("FABRICATED SiC-CAVITY SAMPLES -- operating-point analysis")
    print("  SiN best_absolute reference, all at I = {:.0e} W/cm^2 (its own usable intensity):"
          .format(SIN_REF_I))
    print("    as fabricated {:.5f} deg | best retune in today's probe range {:.5f} deg | "
          "best over all probes {:.5f} deg".format(
              SIN_AS_FAB_DEG, SIN_RETUNED_DEG, SIN_BEST_ANY_DEG))
    print("=" * 100)

    summary = {}
    for s in samples:
        res = load(s)
        if res is None:
            print("\n### {}: no result.json yet".format(s))
            continue
        I_sic = float(res["config"]["pump_intensity"])
        sin_eq = sin_at(I_sic)
        print("\n### {}  (SiC cavity L = {} um)   {} operating points   I = {:.0e} W/cm^2"
              .format(s, res["cavity_L_um"], res["n_ops"], I_sic))
        print("    SiN at the SAME intensity would give {:.6f} deg "
              "(scaled by its measured I^{:.2f})".format(sin_eq, SIN_SLOPE))
        summary[s] = {}
        for sc in ("now", "future", "all"):
            rows = scenario_rows(res, sc)
            if not rows:
                print("  [{}] no operating points".format(sc))
                continue
            bt, bc = best(rows, "theta"), best(rows, "contrast")
            print("  [{}] {} ops".format(sc, len(rows)))
            print("      max signal   : " + fmt_op(bt))
            print("                     -> equal-intensity vs SiN: {:.0f}x  |  as-operated "
                  "(SiC@{:.0e} vs SiN@{:.0e}): {:.2f}x as-fab, {:.2f}x SiN's best retune"
                  .format(abs(bt["theta_chi5_deg"]) / sin_eq, I_sic, SIN_REF_I,
                          abs(bt["theta_chi5_deg"]) / SIN_AS_FAB_DEG,
                          abs(bt["theta_chi5_deg"]) / SIN_RETUNED_DEG))
            print("      max contrast : " + fmt_op(bc))
            summary[s][sc] = {"n_ops": len(rows),
                              "best_theta": {"op": bt["op"], "theta_deg": bt["theta_chi5_deg"],
                                             "fringe_deg": bt["theta_fringe_amp_deg"],
                                             "contrast": contrast(bt), "dolp": bt["dolp"]},
                              "best_contrast": {"op": bc["op"], "theta_deg": bc["theta_chi5_deg"],
                                                "fringe_deg": bc["theta_fringe_amp_deg"],
                                                "contrast": contrast(bc), "dolp": bc["dolp"]}}

        # per-probe-mode best, the dominant knob in the SiN campaign
        print("\n  best achievable at each probe mode:")
        print("    {:>9s} {:>7s} {:>6s} {:>11s} {:>9s} {:>8s} {:>9s}  {}".format(
            "probe_nm", "Q", "T", "|theta|", "fringe", "contrast", "vs SiN", "reachable"))
        by_probe = {}
        for r in res["rows"]:
            k = round(r["op"]["probe_nm"], 1)
            if k not in by_probe or abs(r["theta_chi5_deg"]) > abs(by_probe[k]["theta_chi5_deg"]):
                by_probe[k] = r
        for k in sorted(by_probe):
            r = by_probe[k]
            t = abs(r["theta_chi5_deg"])
            lam = k / 1000.0
            tags = []
            if S.in_windows(lam, S.PROBE_WINDOWS_NOW):
                tags.append("NOW")
            if S.in_windows(lam, S.PROBE_WINDOWS_FUTURE):
                tags.append("future")
            print("    {:>9.1f} {:>7.0f} {:>6.2f} {:>11.5f} {:>9.5f} {:>8.2f} {:>8.2f}x  {}"
                  .format(k, r["op"]["probe_Q"], r["op"]["probe_T"], t,
                          r["theta_fringe_amp_deg"], contrast(r), t / SIN_AS_FAB_DEG,
                          ",".join(tags) if tags else "-"))

    if summary:
        DOCS.mkdir(parents=True, exist_ok=True)
        json.dump({"reference": {"sin_reference_intensity": SIN_REF_I,
                                 "sin_as_fabricated_deg": SIN_AS_FAB_DEG,
                                 "sin_best_retune_now_deg": SIN_RETUNED_DEG,
                                 "sin_best_any_probe_deg": SIN_BEST_ANY_DEG,
                                 "sin_local_slope": SIN_SLOPE},
                   "samples": summary}, open(DOCS / "s2_summary.json", "w"), indent=2)
        print("\n-> {}".format(DOCS / "s2_summary.json"))


if __name__ == "__main__":
    main()
