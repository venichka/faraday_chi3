#!/usr/bin/env python
"""Pump-probe DELAY scan on the SiN best_absolute design, matching the lab measurement.

Experiment being reproduced: the probe transmitted through the sample is split into V and H
by a polarizing beamsplitter and (signal_V - signal_H) is recorded as a function of the delay
tau of pump1 relative to (pump2, probe), which stay locked together.

Observable. For a probe launched along +45 deg, balanced detection reads

    V - H  =  -S1  =  S0 . cos(2 chi) . sin(2 theta)  ~  2 S0 theta,

so the balanced signal IS the rotation. We read the PULSE-ENERGY-INTEGRATED Stokes vector
(`probe_pulse_integrated` in faraday_summary.json), which by Parseval is the time integral of
the transmitted probe over the whole pulse -- what a detector that integrates the pulse
reports. The legacy tail/final-window rotation is a settled-state measure and is NOT
comparable across delays; it is recorded alongside for continuity only.

Delay-step choice. The pump1 optical carrier period is T1 = lambda1/c = 5.075 fs. A delay scan
sampled at an arbitrary step ALIASES any carrier-phase-sensitive component into a slow fake
oscillation. Stage 1 therefore steps by exactly 3*T1, which holds omega1*tau fixed modulo 2pi:
a carrier component then contributes a constant offset rather than a spurious period. The
separate --fine scan steps by T1/6 to measure that carrier component head-on.

Physics predictions being tested (design values, TMM mode comb of this geometry):
  * pump1/pump2 (1525.1 / 1577.5 nm) are ADJACENT longitudinal modes, local FSR 0.0218 /um,
    so pump beat period = cavity round-trip time = 152 fs;
  * finesse ~2.3 => intracavity energy falls ~15x per round trip => ONE weakly damped ringing,
    not 2-3. Seeing 2-3 in the lab implies pump-band Q ~ 190-470 rather than the designed 70;
  * at |tau| >> 300 fs pump1 has left but pump2 still overlaps the probe, leaving a residual
    SINGLE-PUMP chi3 pedestal that does not decay to zero. Pedestal-to-peak measures the
    chi3/chi5 split directly, with no power-law fitting.

Stages:
  --stage1   tau scan, clean balanced sigma+/sigma- config, linear 45 deg probe   (61 sims)
  --fine     carrier-resolved mini-scan near tau=0                                (24 sims)
  --stage2   systematics: probe ellipticity, azimuth misalignment, pump imbalance (182 sims)

  python chi5_optimization/delay_scan.py --stage1 --fine --stage2 --workers 90
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent

C0_UM_FS = 0.299792458  # um per fs

# The design under test. --materials fit with these CSVs is the configuration that reproduced
# the published |theta| = 0.137 deg for this geometry (diagnose_oppoint / clean_lsweep).
BASE_DIR = MEEP / "SiN_optimizations" / "best_absolute"
FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
         "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
PUMP_INTENSITY = 1e12    # W/cm^2, as in the experiment
PROBE_INTENSITY = 5e7    # W/cm^2

LAM_PUMP1_UM = 1.5214626391096977        # design pump1 (cavity_modes.json)
T_CARRIER_FS = LAM_PUMP1_UM / C0_UM_FS   # 5.0751 fs

# Stage 1: +-30 steps of 3*T1 => +-456.8 fs, carrier-phase-locked.
STAGE1_STEP_FS = 3.0 * T_CARRIER_FS
STAGE1_HALF_STEPS = 30
# Fine scan: 4 carrier periods at T1/6, to detect a carrier-phase component directly.
FINE_STEP_FS = T_CARRIER_FS / 6.0
FINE_N = 24
# Stage 2: same span, coarser (every 4th stage-1 step), one family per systematic.
STAGE2_STRIDE = 4
STAGE2_FAMILIES = [
    # (tag,            azimuth_deg, ellipticity_deg, pump_imbalance)
    ("ellip05",        45.0,  5.0, 1.00),
    ("ellip10",        45.0, 10.0, 1.00),
    ("ellip20",        45.0, 20.0, 1.00),
    ("azim1",          46.0,  0.0, 1.00),
    ("azim3",          48.0,  0.0, 1.00),
    ("imbal095",       45.0,  0.0, 0.95),
    ("imbal090",       45.0,  0.0, 0.90),
]


def job_tag(tau_fs: float, azim: float, ellip: float, imbal: float) -> str:
    return "tau{:+08.2f}_az{:.1f}_el{:.1f}_ib{:.3f}".format(tau_fs, azim, ellip, imbal)


def run_one(tau_fs, azim, ellip, imbal, res, decay, outroot):
    """One 1D-FDTD run at a given delay / probe state / pump balance. Skips completed runs."""
    out = Path(outroot) / job_tag(tau_fs, azim, ellip, imbal)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "faraday_summary.json").exists():
        cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *FLAGS,
               "--geometry-file", str(BASE_DIR / "geometry.json"),
               "--cavity-modes-file", str(BASE_DIR / "cavity_modes.json"),
               "--resolution", str(res), "--decay-threshold", str(decay),
               "--pump-intensity", str(PUMP_INTENSITY), "--probe-intensity", str(PROBE_INTENSITY),
               "--pump1-delay-fs", "{:.6f}".format(tau_fs),
               "--probe-azimuth-deg", "{:.6f}".format(azim),
               "--probe-ellipticity-deg", "{:.6f}".format(ellip),
               "--pump-imbalance", "{:.6f}".format(imbal),
               "--output-dir", str(out)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
        with open(out / "run.log", "w") as lf:
            subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return tau_fs, azim, ellip, imbal


def read_one(tau_fs, azim, ellip, imbal, outroot):
    p = Path(outroot) / job_tag(tau_fs, azim, ellip, imbal) / "faraday_summary.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    pi = d.get("probe_pulse_integrated", {})
    if not pi:
        return None
    rec = {"tau_fs": float(tau_fs), "azimuth_deg": float(azim),
           "ellipticity_deg": float(ellip), "imbalance": float(imbal),
           "vmh_norm": pi.get("balanced_V_minus_H_norm"),
           "vmh": pi.get("balanced_V_minus_H"),
           "rotation_deg": pi.get("rotation_deg"),
           "chi_deg": pi.get("chi_deg"),
           "dchi_deg": pi.get("ellipticity_change_deg"),
           "dolp": pi.get("dolp"),
           "S0": pi.get("S0")}
    # legacy settled-state readout, for continuity with the published 0.137 deg
    rec["legacy_final_rel_deg"] = d.get("probe_rotation_deg", {}).get("final_relative_deg")
    return rec


def launch(jobs, res, decay, outroot, workers, label):
    print("\n{}: {} 1D-FDTD sims (res {}, decay {}, {} workers)...".format(
        label, len(jobs), res, decay, workers), flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_one, *j, res, decay, outroot) for j in jobs]
        for _ in as_completed(futs):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print("  {}/{} ({:.0f}s)".format(done, len(jobs), time.time() - t0), flush=True)
    print("  {} done in {:.0f}s".format(label, time.time() - t0), flush=True)


def summarize(recs, tag):
    """Print the delay trace and the numbers that discriminate the mechanisms."""
    recs = sorted([r for r in recs if r and r["vmh_norm"] is not None], key=lambda r: r["tau_fs"])
    if not recs:
        print("  (no results for {})".format(tag))
        return {}
    tau = np.array([r["tau_fs"] for r in recs])
    sig = np.array([r["vmh_norm"] for r in recs])
    rot = np.array([r["rotation_deg"] for r in recs])
    # Pedestal = mean over the outer 20% of |tau| (pump1 well away from the probe);
    # peak = value at the tau closest to 0.
    edge = np.abs(tau) >= 0.8 * np.max(np.abs(tau))
    pedestal = float(np.mean(sig[edge])) if edge.any() else float("nan")
    # Same decomposition on the rotation ANGLE. theta and (V-H) carry the same information
    # -- (V-H)/S0 = cos(2chi) sin(2theta), so theta ~ (V-H)/2 in rad for small angles -- but
    # theta is the physical quantity to quote, so both are reported everywhere.
    ped_rot = float(np.mean(rot[edge])) if edge.any() else float("nan")
    i0 = int(np.argmin(np.abs(tau)))
    scatter = float(np.std(sig[edge])) if edge.any() else float("nan")
    scatter_rot = float(np.std(rot[edge])) if edge.any() else float("nan")
    out = {"tau_fs": tau.tolist(), "vmh_norm": sig.tolist(), "rotation_deg": rot.tolist(),
           "dolp": [r["dolp"] for r in recs], "chi_deg": [r["chi_deg"] for r in recs],
           "legacy_final_rel_deg": [r["legacy_final_rel_deg"] for r in recs],
           "pedestal_vmh_norm": pedestal, "peak_vmh_norm": float(sig[i0]),
           "peak_rotation_deg": float(rot[i0]), "pedestal_rotation_deg": ped_rot,
           "edge_scatter_vmh_norm": scatter, "edge_scatter_rotation_deg": scatter_rot}
    contrast = sig[i0] - pedestal
    out["peak_minus_pedestal"] = float(contrast)
    out["peak_minus_pedestal_rotation_deg"] = float(rot[i0] - ped_rot)
    print("  {:<10s} n={:3d}  V-H:   peak(tau=0)={:+.5e}  pedestal={:+.5e}  "
          "peak-ped={:+.5e}".format(tag, len(recs), sig[i0], pedestal, contrast))
    print("  {:<10s}          theta: peak(tau=0)={:+.5f} deg  pedestal={:+.5f} deg  "
          "peak-ped={:+.5f} deg".format("", rot[i0], ped_rot, rot[i0] - ped_rot))
    # The pedestal is only meaningful if it stands above the point-to-point scatter of the
    # very points it averages; otherwise the peak/pedestal ratio below is not a measurement.
    if np.isfinite(scatter) and int(edge.sum()) > 1:
        err = scatter / np.sqrt(int(edge.sum()))
        print("  {:<10s}          pedestal = {:+.3e} +/- {:.3e} ({:.1f} sigma, n={:d} edge pts;"
              " per-point scatter {:.3e})".format(
                  "", pedestal, err, abs(pedestal) / err if err > 0 else float("nan"),
                  int(edge.sum()), scatter))
        out["pedestal_vmh_norm_stderr"] = float(err)
    if abs(pedestal) > 1e-30:
        ratio = abs(contrast / pedestal)
        out["chi5_over_chi3_ratio"] = float(ratio)
        print("  {:<10s} chi5(peak-pedestal) / chi3(pedestal) = {:.2f}"
              "  [meaningful only if the pedestal is many sigma above 0]".format("", ratio))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=90)
    ap.add_argument("--res", type=int, default=80)
    ap.add_argument("--decay", default="1e-4")
    ap.add_argument("--stage1", action="store_true", help="carrier-locked delay scan (61 sims)")
    ap.add_argument("--fine", action="store_true", help="carrier-resolved mini-scan (24 sims)")
    ap.add_argument("--stage2", action="store_true", help="ellipticity/azimuth/imbalance families")
    ap.add_argument("--families", default=None,
                    help="comma list of stage-2 family tags to run (default: all). Families write "
                         "to disjoint output dirs, so they can be farmed out to separate nodes and "
                         "aggregated later by re-running the driver (completed sims are skipped).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not (args.stage1 or args.fine or args.stage2):
        args.stage1 = True

    outroot = Path(args.out) if args.out else HERE / "delay_scan"
    outroot.mkdir(parents=True, exist_ok=True)

    modes = json.load(open(BASE_DIR / "cavity_modes.json"))
    delta = modes["sidebands"]["delta_frequency"]
    beat_fs = 1.0 / (delta * 1e6 * 299792458.0) * 1e15
    print("=== Delay scan | SiN best_absolute | I_pump={:.0e} I_probe={:.0e} W/cm^2 ===".format(
        PUMP_INTENSITY, PROBE_INTENSITY))
    print("  probe {:.1f} nm | pumps {:.1f} / {:.1f} nm | Delta={:.5f}/um".format(
        modes["probe"]["lambda_um"] * 1000, modes["pump1"]["lambda_um"] * 1000,
        modes["pump2"]["lambda_um"] * 1000, delta))
    print("  predicted beat / round-trip period = {:.1f} fs;  carrier period T1 = {:.3f} fs".format(
        beat_fs, T_CARRIER_FS))
    print("  stage-1 step = 3*T1 = {:.3f} fs (carrier-phase-locked), span +-{:.1f} fs".format(
        STAGE1_STEP_FS, STAGE1_HALF_STEPS * STAGE1_STEP_FS))

    taus1 = [round(k * STAGE1_STEP_FS, 4)
             for k in range(-STAGE1_HALF_STEPS, STAGE1_HALF_STEPS + 1)]
    result = {"design": "SiN_optimizations/best_absolute",
              "pump_intensity_w_cm2": PUMP_INTENSITY, "probe_intensity_w_cm2": PROBE_INTENSITY,
              "resolution": args.res, "decay_threshold": args.decay,
              "delta_inv_um": delta, "predicted_beat_fs": beat_fs,
              "carrier_period_fs": T_CARRIER_FS, "stage1_step_fs": STAGE1_STEP_FS}

    if args.stage1:
        jobs = [(t, 45.0, 0.0, 1.0) for t in taus1]
        launch(jobs, args.res, args.decay, outroot, args.workers, "stage1")
        print("\n=== Stage 1: clean balanced config ===")
        result["stage1"] = summarize([read_one(*j, outroot) for j in jobs], "stage1")

    if args.fine:
        taus = [round(k * FINE_STEP_FS, 4) for k in range(FINE_N)]
        jobs = [(t, 45.0, 0.0, 1.0) for t in taus]
        launch(jobs, args.res, args.decay, outroot, args.workers, "fine")
        print("\n=== Fine scan: is there a carrier-period component? ===")
        fine = summarize([read_one(*j, outroot) for j in jobs], "fine")
        if fine:
            s = np.array(fine["vmh_norm"])
            # peak-to-peak modulation across 4 carrier periods vs the mean level
            ptp = float(np.ptp(s))
            fine["carrier_ptp"] = ptp
            fine["carrier_ptp_over_mean"] = float(ptp / max(abs(np.mean(s)), 1e-30))
            print("  carrier-period modulation: ptp={:.3e} ({:.2%} of mean level)".format(
                ptp, fine["carrier_ptp_over_mean"]))
            print("  -> large => the lab oscillation may be an ALIASED carrier fringe;"
                  " small => the delay trace is genuinely envelope-level")
        result["fine"] = fine

    if args.stage2:
        families = STAGE2_FAMILIES
        if args.families:
            want = {s.strip() for s in args.families.split(",") if s.strip()}
            families = [f for f in STAGE2_FAMILIES if f[0] in want]
            unknown = want - {f[0] for f in STAGE2_FAMILIES}
            if unknown:
                raise SystemExit("unknown stage-2 families: {}".format(sorted(unknown)))
        taus2 = taus1[::STAGE2_STRIDE]
        jobs, fam_jobs = [], {}
        for tag, az, el, ib in families:
            fam_jobs[tag] = [(t, az, el, ib) for t in taus2]
            jobs += fam_jobs[tag]
        launch(jobs, args.res, args.decay, outroot, args.workers, "stage2")
        print("\n=== Stage 2: systematics (span {} tau points each) ===".format(len(taus2)))
        # chi0=0 reference on the same coarse grid, taken from the stage-1 runs
        ref = summarize([read_one(t, 45.0, 0.0, 1.0, outroot) for t in taus2], "reference")
        result["stage2"] = {"reference": ref}
        for tag, az, el, ib in families:
            fam = summarize([read_one(*j, outroot) for j in fam_jobs[tag]], tag)
            if fam and ref:
                # predicted contrast law for a pure ellipticity effect
                fam["cos2chi_prediction"] = float(np.cos(np.radians(2.0 * el)))
                if abs(ref["peak_minus_pedestal"]) > 1e-30:
                    fam["contrast_vs_reference"] = float(
                        fam["peak_minus_pedestal"] / ref["peak_minus_pedestal"])
                    print("  {:<10s} contrast vs reference = {:.4f}   (cos(2chi0) = {:.4f})".format(
                        "", fam["contrast_vs_reference"], fam["cos2chi_prediction"]))
            result["stage2"][tag] = fam

    path = outroot / "delay_scan_result.json"
    json.dump(result, open(path, "w"), indent=2)
    print("\n-> {}".format(path))


if __name__ == "__main__":
    main()
