#!/usr/bin/env python
"""EMPIRICAL DECOMPOSITION of theta(L)~L^+1.2 into its physical factors, to settle the
buildup-vs-sideband-resonance question the hand-derivation can't (FDTD says theta rises with L,
but the 100fs broadband buildup ~1/Q says it should fall -> something else carries the growth).

For each L (best_absolute mirror, N=3) at its FDTD-OPTIMAL operating point (from the clean L-sweep
lsweep_summary.json), run faraday_meep_fp_circ with --enable-nonlinear-diagnostics and read off the
INTRACAVITY tail-weighted fields at the cavity center:
  - pump1 |e-|, pump2 |e+|       -> pump buildup product P = |A1||A2|  (cascade sigma ~ (P)^2)
  - probe field                  -> probe buildup B_s
  - sb_minus, sb_plus field      -> intracavity SIDEBAND amplitude (the cascade output / resonance)
Then decompose theta(L) = buildup(L) x residual(L); the residual = the dispersion/sideband-resonance
factor the analytic must capture. Same FDTD settings as the clean L-sweep (res 80, decay 1e-5, I=1e12).
"""
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
sys.path.insert(0, str(HERE))
import tmm          # noqa: E402
import optimize as O  # noqa: E402

BASE = json.load(open(MEEP / "SiN_optimizations/best_absolute/geometry.json"))
HI = BASE["cavity"]["mat"]
T_HI = float(np.mean([l["thk_um"] for l in BASE["mirrors"]["left"] if l["mat"] == HI]))
T_LO = float(np.mean([l["thk_um"] for l in BASE["mirrors"]["left"] if l["mat"] != HI]))

FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
         "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
INTENSITY = 1e12
RES = 80
DECAY = "1e-4"   # faster; residual theta/buildup is self-consistent at fixed decay (verify theta(L) still rises)
N_PAIRS = 3
WORKERS = 10
OUT = HERE / "decompose"
SUMM = json.load(open(HERE / "clean_lsweep/lsweep_summary.json"))


def probe_freq(geom):
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    ms = tmm.find_modes_in_band(layers, idx, 1.0 / 0.810, 1.0 / 0.790)
    return max(ms, key=lambda m: m["Q"])["freq"] if ms else None


def run_one(s):
    L = s["L"]; center = s["center"]; delta = s["delta"]
    f1, f2 = center + 0.5 * delta, center - 0.5 * delta
    geom = O.build_geometry(BASE, N_PAIRS, T_HI, T_LO, L)
    fp = probe_freq(geom)
    out = OUT / "L{:.2f}".format(L)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "faraday_summary.json").exists():
        json.dump(geom, open(out / "geometry.json", "w"))
        modes = {"probe": {"frequency": fp, "lambda_um": 1.0 / fp},
                 "pump1": {"frequency": f1, "lambda_um": 1.0 / f1},
                 "pump2": {"frequency": f2, "lambda_um": 1.0 / f2},
                 "sidebands": {"frequency_plus": fp + delta, "frequency_minus": fp - delta,
                               "delta_frequency": delta, "pump_separation_um": abs(1.0 / f2 - 1.0 / f1)}}
        json.dump(modes, open(out / "cavity_modes.json", "w"))
        cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *FLAGS,
               "--enable-nonlinear-diagnostics",
               "--geometry-file", str(out / "geometry.json"), "--cavity-modes-file", str(out / "cavity_modes.json"),
               "--resolution", str(RES), "--decay-threshold", DECAY, "--pump-intensity", str(INTENSITY),
               "--pump1-frequency", "{:.9f}".format(f1), "--pump2-frequency", "{:.9f}".format(f2),
               "--output-dir", str(out)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
        with open(out / "run.log", "w") as lf:
            subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return L, fp


def extract(L, fp, delta):
    p = OUT / "L{:.2f}/faraday_summary.json".format(L)
    if not p.exists():
        return None
    d = json.load(open(p))
    th = abs(d["probe_rotation_deg"]["final_relative_deg"])
    nd = d.get("nonlinear_diagnostics", {})
    cav = nd.get("intracavity_fixed_freqs", {}) or {}

    def g(label, key):
        return float(cav.get(label, {}).get(key, float("nan")))
    P1 = g("pump1", "eminus_rms_tail")   # pump1 dominant = |e-| at f_p1
    P2 = g("pump2", "eplus_rms_tail")    # pump2 dominant = |e+| at f_p2
    Bs = g("probe", "field_rms_tail")
    if not np.isfinite(Bs):
        Bs = float(np.hypot(g("probe", "eplus_rms_tail"), g("probe", "eminus_rms_tail")))
    sbm = g("sb_minus", "field_rms_tail")
    sbp = g("sb_plus", "field_rms_tail")
    return {"L": L, "theta": th, "P1": P1, "P2": P2, "pump_amp_prod": P1 * P2,
            "pump_energy_prod": (P1 * P2) ** 2, "Bs": Bs, "sb_minus": sbm, "sb_plus": sbp,
            "probe_nm": 1000.0 / fp, "delta": delta}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("running {} FDTD sims (--enable-nonlinear-diagnostics, decay {})...".format(len(SUMM), DECAY), flush=True)
    t0 = time.time()
    fps = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fu in as_completed([ex.submit(run_one, s) for s in SUMM]):
            L, fp = fu.result(); fps[L] = fp
            print("  done L={:.2f} ({:.0f}s)".format(L, time.time() - t0), flush=True)
    print("all sims in {:.0f}s\n".format(time.time() - t0))

    rows = []
    for s in SUMM:
        r = extract(s["L"], fps.get(s["L"]), s["delta"])
        if r:
            rows.append(r)
    rows.sort(key=lambda r: r["L"])
    json.dump(rows, open(OUT / "decompose_summary.json", "w"), indent=2)

    print("=== empirical decomposition of theta(L) (intracavity tail-weighted fields) ===")
    print("{:>5s} {:>9s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "L", "theta", "Ppump_amp", "Penergy", "B_probe", "sb_minus", "sb_plus", "th/Penergy"))
    for r in rows:
        resid = r["theta"] / r["pump_energy_prod"] if r["pump_energy_prod"] else float("nan")
        print("{:5.2f} {:9.4f} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e} {:10.3e}".format(
            r["L"], r["theta"], r["pump_amp_prod"], r["pump_energy_prod"], r["Bs"],
            r["sb_minus"], r["sb_plus"], resid))

    Ls = np.array([r["L"] for r in rows])

    def sl(key, transform=lambda x: x):
        y = np.array([transform(r[key]) for r in rows], float)
        m = np.isfinite(y) & (y > 0)
        return np.polyfit(np.log(Ls[m]), np.log(y[m]), 1)[0] if m.sum() >= 3 else float("nan")
    resid = [r["theta"] / r["pump_energy_prod"] for r in rows]
    print("\nscalings ~L^p:")
    print("  theta          ~ L^{:+.2f}".format(sl("theta")))
    print("  pump_energy    ~ L^{:+.2f}   (|A1|^2|A2|^2 intracavity; hand-deriv predicts ~1/L if broadband)".format(sl("pump_energy_prod")))
    print("  probe buildup  ~ L^{:+.2f}".format(sl("Bs")))
    print("  sb_minus(cav)  ~ L^{:+.2f}".format(sl("sb_minus")))
    print("  sb_plus(cav)   ~ L^{:+.2f}".format(sl("sb_plus")))
    yr = np.array(resid, float); mr = np.isfinite(yr) & (yr > 0)
    sr = np.polyfit(np.log(Ls[mr]), np.log(yr[mr]), 1)[0] if mr.sum() >= 3 else float("nan")
    print("  theta/pumpE    ~ L^{:+.2f}   (residual = dispersion/sideband-resonance + probe response)".format(sr))
    print("-> {}".format(OUT / "decompose_summary.json"))


if __name__ == "__main__":
    main()
