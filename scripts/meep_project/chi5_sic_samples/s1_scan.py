#!/usr/bin/env python
"""Stage 1 -- the operating-point map of both FABRICATED SiC-cavity samples.

Scans (probe cavity mode) x (pump centre) x (Delta), carrier-averaged over 4 pump-1 phases, for
the two fabricated geometries (SiC cavity L = 3.2 and 4.8 um inside the unchanged SiN/SiO2
best_absolute mirrors).

The scan runs over the FULL 600-1000 nm probe band -- the union of both lab scenarios plus some
headroom -- and the two scenarios are applied as filters at analysis time. That way a single set
of simulations answers "what can you do today" and "what would an extended probe source buy",
and it also quantifies anything good that sits outside both windows (as the 763.6 nm result did
for the SiN sample).

  python chi5_sic_samples/s1_scan.py --sample both --workers 26 --slice 0/8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TAG_RE = re.compile(r"^p(?P<probe>[\d.]+)_c(?P<center>[\d.]+)_d(?P<delta>[\d.]+)$")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_sic as S  # noqa: E402
import common as C      # noqa: E402  (chi5_dbr_design harness, via common_sic's sys.path)

OUT = HERE / "runs" / "s1_scan"


# Two phases, the structure Stage 5 of the SiN campaign validated: the probe mode is the
# dominant knob and the pump centre/Delta are forgiving, so scan every probe mode coarsely
# first and refine only around the winners. This costs ~45% of a uniformly fine grid.
OFFSETS_A = (-0.03, 0.0, 0.03)
DELTAS_A = (0.010, 0.016, 0.023)
OFFSETS_B = (-0.03, -0.015, 0.0, 0.015, 0.03)
DELTAS_B = (0.008, 0.011, 0.014, 0.017, 0.020, 0.023)


def phase_a_ops(geom):
    return S.operating_points(geom, [S.PROBE_BAND_SCAN],
                              center_offsets=OFFSETS_A, deltas=DELTAS_A)


def phase_b_ops(geom, probe_nms):
    ops = []
    for nm in probe_nms:
        lam = nm / 1000.0
        win = (lam - 0.002, lam + 0.002)
        ops += S.operating_points(geom, [win], center_offsets=OFFSETS_B, deltas=DELTAS_B)
    return ops


def best_probes_from_a(sample, geom):
    """Probe modes worth refining -- selected on BOTH axes and in BOTH lab scenarios.

    ⚠️ Refining only the top modes by |theta| would be a trap. On the L=3.2 sample the two
    largest-|theta| modes (695, 714 nm) have contrast 0.06-0.07 -- buried under the fringe --
    while the one operating point that actually answers the lab's question (850.2 nm,
    contrast 1.44, reachable today) is 7th by |theta| and would never be refined.
    So take, per scenario: the best by |theta| AND the best by contrast.
    """
    best = {}          # probe_nm -> (theta, contrast)
    for op in phase_a_ops(geom):
        r = S.collect(OUT / sample, op)
        if not r:
            continue
        k = round(op["probe_nm"], 1)
        t = abs(r["theta_chi5_deg"])
        c = t / max(r["theta_fringe_amp_deg"], 1e-12)
        prev = best.get(k, (0.0, 0.0))
        best[k] = (max(prev[0], t), max(prev[1], c))
    if not best:
        modes = S.probe_modes(geom, [S.PROBE_BAND_SCAN])
        modes.sort(key=lambda m: abs(m["lambda_um"] - 0.800))
        return [round(m["lambda_um"] * 1000, 1) for m in modes[:2]]

    picks = []
    for windows in (S.PROBE_WINDOWS_NOW, S.PROBE_WINDOWS_FUTURE):
        sel = {k: v for k, v in best.items() if S.in_windows(k / 1000.0, windows)}
        if not sel:
            continue
        picks.append(max(sel, key=lambda k: sel[k][0]))      # best signal
        picks.append(max(sel, key=lambda k: sel[k][1]))      # best contrast
    out = []
    for p in picks:
        if p not in out:
            out.append(p)
    return out


def jobs_for(sample: str, phase: str):
    geom = S.sic_geometry(S.CAVITY_LENGTHS_UM[sample])
    if phase == "A":
        ops = phase_a_ops(geom)
    else:
        probes = best_probes_from_a(sample, geom)
        print("  [{}] phase B refines probe modes: {}".format(sample, probes))
        ops = phase_b_ops(geom, probes)
    seen, uniq = set(), []
    for op in ops:
        if S.tag(op) not in seen:
            seen.add(S.tag(op))
            uniq.append(op)
    out = []
    for op in uniq:
        for sub, tau in enumerate(C.subsample_taus(op["pump1"])):
            out.append((sample, geom, op, sub, tau))
    return geom, uniq, out


def run_one(sample, geom, op, sub, tau, res, decay, intensity):
    S.run_case(OUT / sample / S.tag(op) / "s{}".format(sub), geom, op, sub, tau,
               res=res, decay=decay, pump_intensity=intensity)
    return sample, S.tag(op), sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", choices=["L3p2", "L4p8", "both"], default="both")
    ap.add_argument("--phase", choices=["A", "B"], default="A")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--res", type=int, default=C.RES_1D)
    ap.add_argument("--decay", default=C.DECAY_1D)
    ap.add_argument("--intensity", type=float, default=None,
                    help="Pump intensity (W/cm^2). REQUIRED for SiC -- stage 0 picks it. "
                         "The SiN default of 1e12 drives these cavities into the large-signal, "
                         "depolarizing regime (DoLP 0.72), where theta is not a meaningful "
                         "azimuth and the I^2 law does not hold.")
    ap.add_argument("--slice", default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    samples = ["L3p2", "L4p8"] if args.sample == "both" else [args.sample]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.intensity is None and not args.aggregate_only:
        ap.error("--intensity is required (run s0_intensity.py first; the SiN default of "
                 "1e12 W/cm^2 is large-signal for these SiC cavities)")
    intensity = args.intensity if args.intensity is not None else C.PUMP_INTENSITY
    print("pump intensity = {:.3e} W/cm^2".format(intensity))

    all_jobs, meta = [], {}
    for s in samples:
        geom, ops, jobs = jobs_for(s, args.phase)
        meta[s] = {"geom": geom, "ops": ops}
        all_jobs += jobs
        sb = S.stopband(geom)
        print("=== {}  L={} um  stack {:.3f} um  stopband {:.0f}-{:.0f} nm ===".format(
            s, geom["cavity"]["L_um"], S.stack_um(geom), sb[0] * 1000, sb[1] * 1000))
        print("    {} operating points -> {} sims".format(len(ops), len(ops) * C.SUBSAMPLES))

    if not args.aggregate_only:
        mine = all_jobs
        if args.slice:
            i, n = (int(x) for x in args.slice.split("/"))
            mine = [j for k, j in enumerate(all_jobs) if k % n == i]
        print("  running {} of {} on {} workers".format(len(mine), len(all_jobs), args.workers))
        t0, done = time.time(), 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, *j, args.res, args.decay, intensity) for j in mine]
            for _ in as_completed(futs):
                done += 1
                if done % 25 == 0 or done == len(mine):
                    el = time.time() - t0
                    print("    {}/{}  {:.0f}s  ({:.1f}s/sim)".format(
                        done, len(mine), el, el / max(done, 1)), flush=True)

    # ---- aggregate ------------------------------------------------------------------ #
    # Walk the run directory rather than the current phase's op list: a phase-B invocation must
    # not overwrite the phase-A map with its own 2-probe subset. (Stage 5 of the SiN campaign
    # did exactly that and the coarse map had to be rebuilt.)
    for s in samples:
        geom = meta[s]["geom"]
        modes = {round(m["freq"], 4): m for m in S.probe_modes(geom, [S.PROBE_BAND_SCAN])}
        rows = []
        run_intensity = [None]
        for d in sorted((OUT / s).iterdir()) if (OUT / s).is_dir() else []:
            m = TAG_RE.match(d.name) if d.is_dir() else None
            if not m:
                continue
            probe = float(m.group("probe"))
            center = float(m.group("center"))
            delta = float(m.group("delta"))
            recs = [C.read_case(d / "s{}".format(k)) for k in range(C.SUBSAMPLES)]
            if any(x is None for x in recs):
                continue
            r = C.carrier_average(recs)
            mode = modes.get(round(probe, 4), {})
            op = {"probe": probe, "probe_nm": 1000.0 / probe,
                  "probe_Q": float(mode.get("Q", float("nan"))),
                  "probe_T": float(mode.get("T_peak", float("nan"))),
                  "center": center, "center_nm": 1000.0 / center,
                  "center_offset": center / (0.5 * probe) - 1.0,
                  "pump1": center + 0.5 * delta, "pump2": center - 0.5 * delta,
                  "pump1_nm": 1000.0 / (center + 0.5 * delta),
                  "pump2_nm": 1000.0 / (center - 0.5 * delta),
                  "delta": delta,
                  "fwm_mismatch": abs(2.0 * center - probe),
                  "fwm_mismatch_pct": 100.0 * abs(2.0 * center - probe) / probe}
            rows.append({"op": op, **{k: v for k, v in r.items()
                                      if k not in ("theta_sub_deg", "vmh_sub",
                                                   "legacy_sub_deg")}})
            if run_intensity[0] is None:
                # Ground truth: read the intensity the sims ACTUALLY ran at, rather than
                # trusting --intensity, which is absent in --aggregate-only mode and would
                # otherwise record the default and mislabel the whole result file.
                try:
                    sm = json.load(open(d / "s0" / "faraday_summary.json"))
                    run_intensity[0] = float(sm["run_params"]["pump_intensity_w_cm2"])
                except Exception:
                    pass
        rows.sort(key=lambda r: -abs(r["theta_chi5_deg"]))
        payload = {
            "sample": s,
            "cavity_L_um": S.CAVITY_LENGTHS_UM[s],
            "geometry": meta[s]["geom"],
            "config": {"res": args.res, "decay": args.decay,
                       "pump_intensity": run_intensity[0] or intensity,
                       "pump_intensity_source": ("read from run" if run_intensity[0]
                                                 else "cli default"),
                       "probe_intensity": C.PROBE_INTENSITY,
                       "pulse_label_fs": C.PULSE_LABEL_FS,
                       "subsamples": C.SUBSAMPLES,
                       "sic_n2_m2_per_w": S.SIC_N2_M2_PER_W,
                       "sin_n2_m2_per_w": S.SIN_N2_M2_PER_W,
                       "probe_band_scanned": list(S.PROBE_BAND_SCAN)},
            "n_ops": len(rows),
            "rows": rows,
        }
        (OUT / s).mkdir(parents=True, exist_ok=True)
        json.dump(payload, open(OUT / s / "result.json", "w"), indent=2)
        print("\n-> {}  ({} operating points complete)".format(OUT / s / "result.json", len(rows)))
        if rows:
            print("  {:>9s} {:>9s} {:>9s} {:>7s} {:>11s} {:>9s} {:>8s} {:>6s}".format(
                "probe_nm", "pump1_nm", "pump2_nm", "Delta", "|theta_chi5|", "fringe",
                "contrast", "DoLP"))
            for r in rows[:8]:
                op = r["op"]
                t = abs(r["theta_chi5_deg"])
                print("  {:>9.1f} {:>9.1f} {:>9.1f} {:>7.4f} {:>11.5f} {:>9.5f} {:>8.2f} "
                      "{:>6.3f}".format(op["probe_nm"], op["pump1_nm"], op["pump2_nm"],
                                        op["delta"], t, r["theta_fringe_amp_deg"],
                                        t / max(r["theta_fringe_amp_deg"], 1e-12), r["dolp"]))


if __name__ == "__main__":
    main()
