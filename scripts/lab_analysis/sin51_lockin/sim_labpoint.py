#!/usr/bin/env python
"""Simulated signal of the AS-BUILT SIN51 sample at the lab's operating point.

Geometry: the fabricated sample is not the best_absolute design. `asbuilt_geometry.py` fits
per-material thickness scales to where the lab finds its resonances (SiN +3.0%, SiO2 +0.95%;
1.5 nm rms), and the fit is degenerate along a valley, so the uniform-scale member (+2.5%
everywhere) is simulated too, as the other end of the range.
Operating point (user, 2026-09-23/24): probe 800 nm, pumps 1560 and 1620 nm with the 1620 nm one
DELAYED, peak intensity ~1e12 W/cm^2, pulses ~100-150 fs.

The simulator only delays its "pump1", so pump1 is set to 1620 nm and pump2 to 1560 nm. That
puts the right carrier (the delayed pump's) in the fringe. Swapping which frequency carries
which helicity mirrors the configuration: the SIGN of the rotation may flip, not its magnitude.

  trace    as-built (best), tau = -240 ... +240 fs in 20 fs steps, 125 fs pulses, 4 carrier
           sub-samples per delay; pad fixed at 350 fs so only pump1 moves
  bracket  tau = 0 for {best, uniform} x {100, 125, 150 fs} pulses
  3D       as-built (best), tau = 0, 125 fs, res 30, decay 1e-3: 4 MPI jobs (--print-args-3d)

All 1D runs: res 80, decay 1e-4, I = 1e12 W/cm^2, idempotent. From the sub-samples at each delay
C0 (the carrier-averaged effect) and C1 (the complex fringe) are stored, so s6_compare.py can
re-sample the trace at the lab's own delays.

  python sim_labpoint.py --workers 70          # all 1D runs, then aggregate
  python sim_labpoint.py --print-args-3d --index k
  python sim_labpoint.py --aggregate-only
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import common as K

C = K.dbr_harness()
HERE = Path(__file__).resolve().parent
OUT = HERE / "sim"
GEOMS = {"best": HERE / "asbuilt" / "geometry.json",
         "uniform": HERE / "asbuilt" / "geometry_uniform.json"}
F_PROBE = 1.0 / 0.800
F_DELAYED = 1.0 / 1.620              # sim "pump1" = the lab's delayed pump
F_FIXED = 1.0 / 1.560                # sim "pump2"
I_PEAK = K.I_PEAK_W_CM2
PAD_FS = 350.0
TRACE_TAUS = np.round(np.arange(-240.0, 240.0 + 1e-9, 20.0), 3)
TRACE_FWHM = 125.0
BRACKET_FWHM = (100.0, 125.0, 150.0)
RES_3D, DECAY_3D = 30, "1e-3"


def label_of(fwhm_fs):
    """df_from_pulse_duration's label for a true intensity FWHM (see chi5_dbr_design.common)."""
    return fwhm_fs * np.sqrt(np.log(2.0))


def geom(name):
    return json.load(open(GEOMS[name]))


def freqs():
    return {"probe": F_PROBE, "pump1": F_DELAYED, "pump2": F_FIXED}


def case_dir(kind, gname, fwhm, tau, sub, dim=1):
    return (OUT / "runs{}".format("_3d" if dim == 3 else "") / kind / gname /
            "fwhm{:.0f}".format(fwhm) / "t{:+08.2f}".format(tau) / "s{}".format(sub))


def jobs():
    js = []
    for tau in TRACE_TAUS:
        for sub, t in enumerate(C.subsample_taus(F_DELAYED, C.SUBSAMPLES, float(tau))):
            js.append(("trace", "best", TRACE_FWHM, float(tau), sub, t))
    for gname in GEOMS:
        for fw in BRACKET_FWHM:
            for sub, t in enumerate(C.subsample_taus(F_DELAYED, C.SUBSAMPLES, 0.0)):
                js.append(("bracket", gname, fw, 0.0, sub, t))
    return js


def run(job):
    kind, gname, fwhm, tau, sub, t = job
    return C.run_case(case_dir(kind, gname, fwhm, tau, sub), geom(gname), freqs(),
                      tau_fs=t, pad_fs=PAD_FS, pump_intensity=I_PEAK,
                      pulse_label_fs=label_of(fwhm))


def print_args_3d(index):
    """Serial, rank-free: one 3D job's argument list, or nothing if it is already done."""
    taus = C.subsample_taus(F_DELAYED, C.SUBSAMPLES, 0.0)
    if not 0 <= index < len(taus):
        return
    out = case_dir("tau0", "best", TRACE_FWHM, 0.0, index, dim=3)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "faraday_summary.json").exists():
        return
    json.dump(geom("best"), open(out / "geometry.json", "w"))
    f = freqs()
    json.dump(C.modes_json(f["probe"], f["pump1"], f["pump2"]), open(out / "cavity_modes.json", "w"))
    cmd = C.fdtd_cmd(out, RES_3D, DECAY_3D, taus[index], PAD_FS, dim=3, pump_intensity=I_PEAK,
                     pulse_label_fs=label_of(TRACE_FWHM))
    print(" ".join(cmd[2:]))


def collect(kind, gname, fwhm, tau, dim=1):
    recs = [C.read_case(case_dir(kind, gname, fwhm, tau, s, dim)) for s in range(C.SUBSAMPLES)]
    if any(r is None for r in recs):
        return None
    a = C.carrier_average(recs)
    y = np.array(a["theta_sub_deg"])
    phi = 2 * np.pi * np.arange(len(y)) / len(y)
    c1 = 2.0 * np.sum(y * np.exp(-1j * phi)) / len(y)    # fringe phasor at this tau
    return {"tau_fs": tau, "effect_deg": a["theta_chi5_deg"], "fringe_amp_deg": abs(c1),
            "fringe_re": float(c1.real), "fringe_im": float(c1.imag),
            "contrast": abs(a["theta_chi5_deg"]) / max(abs(c1), 1e-12),
            "theta_sub_deg": a["theta_sub_deg"], "dolp": a["dolp"]}


def aggregate():
    trace = [r for r in (collect("trace", "best", TRACE_FWHM, float(t)) for t in TRACE_TAUS) if r]
    bracket = {g: {str(int(fw)): collect("bracket", g, fw, 0.0) for fw in BRACKET_FWHM} for g in GEOMS}
    d3 = collect("tau0", "best", TRACE_FWHM, 0.0, dim=3)
    res = {"operating_point": {"probe_nm": 1000 / F_PROBE, "delayed_pump_nm": 1000 / F_DELAYED,
                               "fixed_pump_nm": 1000 / F_FIXED, "intensity_W_cm2": I_PEAK,
                               "trace_fwhm_fs": TRACE_FWHM},
           "geometries": {k: str(v.relative_to(HERE)) for k, v in GEOMS.items()},
           "trace_1d": trace, "bracket_1d": bracket, "tau0_3d": d3}
    OUT.mkdir(exist_ok=True)
    json.dump(res, open(OUT / "sim_result.json", "w"), indent=2)
    print("1D trace: {} / {} delays".format(len(trace), len(TRACE_TAUS)))
    for r in trace:
        print("  tau {:+7.1f}  effect {:+.5f}  fringe {:.5f}  DoLP {:.4f}".format(
            r["tau_fs"], r["effect_deg"], r["fringe_amp_deg"], r["dolp"]))
    for g, b in bracket.items():
        for fw, r in b.items():
            if r:
                print("  1D tau=0 {:7s} {:>3s} fs: effect {:+.5f}, fringe {:.5f}, contrast {:.2f}".format(
                    g, fw, r["effect_deg"], r["fringe_amp_deg"], r["contrast"]))
    if d3:
        print("  3D tau=0 best 125 fs: effect {:+.5f}, fringe {:.5f}, contrast {:.2f}, DoLP {:.3f}".format(
            d3["effect_deg"], d3["fringe_amp_deg"], d3["contrast"], d3["dolp"]))
    print("-> {}".format(OUT / "sim_result.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=70)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--print-args-3d", action="store_true")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()
    if args.print_args_3d:
        print_args_3d(args.index)
        return
    if not args.aggregate_only:
        js = jobs()
        print("{} sims".format(len(js)), flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for k, _ in enumerate(as_completed([ex.submit(run, j) for j in js]), 1):
                if k % 10 == 0 or k == len(js):
                    print("  {}/{} ({:.0f}s)".format(k, len(js), time.time() - t0), flush=True)
    aggregate()


if __name__ == "__main__":
    main()
