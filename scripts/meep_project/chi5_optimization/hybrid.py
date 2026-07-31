#!/usr/bin/env python
"""Hybrid chi5 geometry optimizer: analytic v2 GEOMETRY pre-rank  ->  1D-FDTD operating-point rank.

Division of labor (settled 2026-06-11 by the FDTD-anchored analytics revision):
  * Stage A (cheap, no FDTD): Sobol over [n_pairs, t_hi, t_lo, L_cav] near the base design; score every
    candidate with objective.chi5_score_v3 -- the clean-normalization FoM (100fs-aware, true-cavity-response,
    buildup once per carrier; validated Spearman ~+0.84 vs FDTD over the L-family; the legacy energy-normalized
    score was -0.92, sign-flipped). Keep the top-K geometries.
  * Stage B (FDTD truth): the analytic score does NOT reliably pick the OPERATING POINT (the symmetry-break
    Re(Sigma) is a delicate difference). So for each top-K geometry + the baseline, sweep the pump
    operating point (pumps placed on each resonant pump mode x a few Delta; probe fixed at its TMM mode)
    in 1D FDTD and take MAX |theta| = the geometry's true capability. Rank by that.
  * Stage C (optional): hand the FDTD winners to 3D / the TCMT(FaradayJL) cross-check (not run here).

Material-agnostic (SiN over SiO2, SiC over SiO2). Run under meep-mpi on a free 96-core node:
  python chi5_optimization/hybrid.py --material sin --n-samples 256 --topk 10 --workers 90
Stage A only (fast, anywhere):
  python chi5_optimization/hybrid.py --material sin --n-samples 256 --topk 10 --skip-fdtd
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
sys.path.insert(0, str(HERE))
import tmm          # noqa: E402
import objective    # noqa: E402
import optimize as O  # noqa: E402

# Per-material FDTD config + base design (same as validate_phase1 / clean_lsweep).
MAT = {
    "sin": {"flags": ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
                      "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"],
            "intensity": 1e12, "base_dir": "SiN_optimizations/best_absolute"},
    "sic": {"flags": ["--materials", "fit", "--sin-fit", "sic.csv", "--sio2-fit", "sio2.csv",
                      "--fit-poles", "3", "--fit-window", "600", "2000", "--high-index-material", "sic"],
            "intensity": 2e11, "base_dir": "SiC_optimizations/sic_L3p2um"},
}

PROBE_WINDOWS = O.PROBE_WINDOWS          # {~800} U [850,950] nm
PUMP_BAND = O.PUMP_BAND                  # (1.40, 1.95) um
# Pre-rank scores v2 over a small-Delta window (the validated-good M5 regime); avoids the
# spurious large-Delta peak in v2's noisy operating-point landscape.
PRERANK_DELTA = (0.012, 0.025)
# Stage-B FDTD operating-point grid: pumps on each resonant pump mode (high buildup) x these Delta.
FDTD_DELTAS = [0.015, 0.022, 0.030]
FDTD_MAX_CENTERS = 5                      # cap resonant-pump centers per geometry


# ------------------------------- Stage A: analytic ------------------------------- #
def prerank(geom, chi_iso, sub_label="SiO2"):
    """Best chi5_score_v2 over probe x resonant-pump-pair (small Delta) -> (score, freqs) or (None, None)."""
    ctx = objective.make_ctx(geom, sub_label)
    layers, idx = ctx["layers"], ctx["idx"]
    probes = []
    for lo, hi in PROBE_WINDOWS:
        probes += tmm.find_modes_in_band(layers, idx, 1.0 / hi, 1.0 / lo, sub_label)
    pumps = tmm.find_modes_in_band(layers, idx, 1.0 / PUMP_BAND[1], 1.0 / PUMP_BAND[0], sub_label)
    if not probes or len(pumps) < 1:
        return None, None
    pm = max(probes, key=lambda m: m["Q"])          # highest-Q probe mode
    fs = pm["freq"]
    best, best_freqs = None, None
    for i in range(len(pumps)):
        for j in range(i, len(pumps)):
            f1 = max(pumps[i]["freq"], pumps[j]["freq"])
            f2 = min(pumps[i]["freq"], pumps[j]["freq"])
            d = f1 - f2
            if not (PRERANK_DELTA[0] <= d <= PRERANK_DELTA[1]):
                continue
            freqs = {"probe": fs, "pump1": f1, "pump2": f2, "sb_plus": fs + d, "sb_minus": fs - d}
            try:
                s = objective.chi5_score_v3(geom, freqs, chi_iso, sub_label, ctx=ctx)
            except Exception:
                continue
            if best is None or s["fom_rotation"] > best["fom_rotation"]:
                best, best_freqs = s, freqs
    return best, best_freqs


def stage_a(base, chi_iso, n_samples, scope, seed, sub_label, topk):
    bounds, base_params = O.phase1_bounds(base, scope)
    pts = O.sobol(bounds, n_samples, seed)
    out = []
    for p in pts:
        n_pairs = int(round(p[0]))
        geom = O.build_geometry(base, n_pairs, p[1], p[2], p[3])
        s, freqs = prerank(geom, chi_iso, sub_label)
        if s is None:
            continue
        out.append({"params": {"n_pairs": n_pairs, "t_hi": float(p[1]), "t_lo": float(p[2]),
                                "L_cav": float(p[3])},
                    "prerank_freqs": {k: float(v) for k, v in freqs.items()},
                    "v2_fom_rotation": s["fom_rotation"], "v2_buildup": s["buildup"],
                    "v2_L_interaction": s["L_interaction"], "Qprobe": None, "geometry": geom})
    out.sort(key=lambda r: r["v2_fom_rotation"], reverse=True)
    return out[:topk], base_params


# ------------------------------- Stage B: 1D FDTD ------------------------------- #
def probe_mode(geom, sub_label="SiO2"):
    idx = tmm.index_map(); layers = tmm.build_layers(geom)
    ms = []
    for lo, hi in PROBE_WINDOWS:
        ms += tmm.find_modes_in_band(layers, idx, 1.0 / hi, 1.0 / lo, sub_label)
    return max(ms, key=lambda m: m["Q"])["freq"] if ms else None


def fdtd_centers(geom, sub_label="SiO2"):
    """Resonant pump-mode frequencies (each a high-buildup operating center)."""
    idx = tmm.index_map(); layers = tmm.build_layers(geom)
    pumps = tmm.find_modes_in_band(layers, idx, 1.0 / PUMP_BAND[1], 1.0 / PUMP_BAND[0], sub_label)
    pumps.sort(key=lambda m: m["Q"], reverse=True)
    return [m["freq"] for m in pumps[:FDTD_MAX_CENTERS]]


def run_fdtd(mat, label, geom, fprobe, center, delta, res, decay, outroot):
    f1, f2 = center + 0.5 * delta, center - 0.5 * delta
    tag = "{}/c{:.4f}_d{:.3f}".format(label, center, delta)
    out = outroot / tag
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "faraday_summary.json").exists():
        json.dump(geom, open(out / "geometry.json", "w"))
        json.dump({"probe": {"frequency": fprobe, "lambda_um": 1.0 / fprobe},
                   "pump1": {"frequency": f1, "lambda_um": 1.0 / f1},
                   "pump2": {"frequency": f2, "lambda_um": 1.0 / f2},
                   "sidebands": {"frequency_plus": fprobe + delta, "frequency_minus": fprobe - delta,
                                 "delta_frequency": delta, "pump_separation_um": abs(1.0 / f2 - 1.0 / f1)}},
                  open(out / "cavity_modes.json", "w"))
        cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", "1", "--mode", "full", *MAT[mat]["flags"],
               "--geometry-file", str(out / "geometry.json"), "--cavity-modes-file", str(out / "cavity_modes.json"),
               "--resolution", str(res), "--decay-threshold", decay, "--pump-intensity", str(MAT[mat]["intensity"]),
               "--pump1-frequency", "{:.9f}".format(f1), "--pump2-frequency", "{:.9f}".format(f2),
               "--output-dir", str(out)]
        env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
        with open(out / "run.log", "w") as lf:
            subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return label, center, delta


def fdtd_theta(outroot, label, center, delta):
    p = outroot / "{}/c{:.4f}_d{:.3f}/faraday_summary.json".format(label, center, delta)
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
        return abs(d["probe_rotation_deg"]["final_relative_deg"]), \
            d["probe_rotation_deg"].get("coherent_window_estimate", {}).get("dolp", float("nan"))
    except Exception:
        return None


def stage_b(mat, candidates, base_geom, res, decay, workers, outroot, sub_label="SiO2"):
    """1D-FDTD operating-point sweep over top-K candidates + baseline; rank by max|theta|."""
    entries = [("baseline", base_geom)] + [("cand{:02d}".format(k), c["geometry"])
                                           for k, c in enumerate(candidates)]
    jobs, plan = [], {}
    for label, geom in entries:
        fp = probe_mode(geom, sub_label)
        if fp is None:
            continue
        centers = fdtd_centers(geom, sub_label)
        plan[label] = {"fprobe": fp, "centers": centers, "geom": geom}
        for c in centers:
            for d in FDTD_DELTAS:
                jobs.append((mat, label, geom, fp, float(c), d, res, decay, outroot))
    print("Stage B: {} 1D-FDTD sims ({} geoms x centers x {} Delta, res {}, decay {}, {} workers)...".format(
        len(jobs), len(plan), len(FDTD_DELTAS), res, decay, workers), flush=True)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fu in as_completed([ex.submit(run_fdtd, *j) for j in jobs]):
            done += 1
            if done % 20 == 0:
                print("  {}/{} ({:.0f}s)".format(done, len(jobs), time.time() - t0), flush=True)
    print("Stage B sims in {:.0f}s".format(time.time() - t0))

    ranked = []
    for label, info in plan.items():
        best = None
        for c in info["centers"]:
            for d in FDTD_DELTAS:
                r = fdtd_theta(outroot, label, c, d)
                if r and (best is None or r[0] > best[0]):
                    best = (r[0], float(c), d, r[1])
        if best:
            ranked.append({"label": label, "max_theta": best[0], "center": best[1], "delta": best[2],
                           "dolp": best[3], "probe_nm": 1000.0 / info["fprobe"]})
    ranked.sort(key=lambda r: r["max_theta"], reverse=True)
    return ranked, plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", choices=["sin", "sic"], default="sin")
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--scope", choices=["local", "regime"], default="regime")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=90)
    ap.add_argument("--res", type=int, default=80)
    ap.add_argument("--decay", default="1e-4")
    ap.add_argument("--skip-fdtd", action="store_true", help="Stage A (analytic pre-rank) only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mat = args.material
    base = json.load(open(MEEP / MAT[mat]["base_dir"] / "geometry.json"))
    outroot = Path(args.out) if args.out else HERE / "hybrid" / mat
    outroot.mkdir(parents=True, exist_ok=True)
    chi_iso = 1.0   # ranking-invariant within a material

    print("=== Hybrid chi5 optimizer | material={} | Stage A: {} Sobol candidates ===".format(mat, args.n_samples),
          flush=True)
    t0 = time.time()
    cands, base_params = stage_a(base, chi_iso, args.n_samples, args.scope, args.seed, "SiO2", args.topk)
    print("Stage A done in {:.0f}s -> top {} by v2 fom_rotation:".format(time.time() - t0, len(cands)))
    for k, c in enumerate(cands):
        p = c["params"]
        print("  cand{:02d}: n={} t_hi={:.4f} t_lo={:.4f} L={:.3f}  v2_fom={:.3e}".format(
            k, p["n_pairs"], p["t_hi"], p["t_lo"], p["L_cav"], c["v2_fom_rotation"]))
    json.dump({"material": mat, "base_params": base_params, "candidates":
               [{kk: vv for kk, vv in c.items() if kk != "geometry"} for c in cands]},
              open(outroot / "stage_a.json", "w"), indent=2)

    if args.skip_fdtd:
        print("--skip-fdtd: stopping after Stage A.  -> {}".format(outroot / "stage_a.json"))
        return

    ranked, plan = stage_b(mat, cands, base, args.res, args.decay, args.workers, outroot)
    print("\n=== Stage B: 1D-FDTD operating-point ranking (max|theta| per geometry) ===")
    print("{:>10s} {:>9s} {:>9s} {:>7s} {:>6s} {:>9s}".format("label", "max|th|", "@center", "@Delta", "DoLP", "probe_nm"))
    bt = next((r["max_theta"] for r in ranked if r["label"] == "baseline"), None)
    for r in ranked:
        x = "  ({:.2f}x base)".format(r["max_theta"] / bt) if bt else ""
        print("{:>10s} {:9.4f} {:9.4f} {:7.3f} {:6.3f} {:9.1f}{}".format(
            r["label"], r["max_theta"], r["center"], r["delta"], r["dolp"], r["probe_nm"], x))

    # write the winning geometry + operating point for the next stage (3D / TCMT)
    for r in ranked[:3]:
        info = plan[r["label"]]; f1, f2 = r["center"] + 0.5 * r["delta"], r["center"] - 0.5 * r["delta"]
        wd = outroot / "winners" / r["label"]; wd.mkdir(parents=True, exist_ok=True)
        json.dump(info["geom"], open(wd / "geometry.json", "w"), indent=2)
        json.dump({"probe": {"frequency": info["fprobe"], "lambda_um": 1.0 / info["fprobe"]},
                   "pump1": {"frequency": f1, "lambda_um": 1.0 / f1},
                   "pump2": {"frequency": f2, "lambda_um": 1.0 / f2},
                   "sidebands": {"frequency_plus": info["fprobe"] + r["delta"],
                                 "frequency_minus": info["fprobe"] - r["delta"], "delta_frequency": r["delta"],
                                 "pump_separation_um": abs(1.0 / f2 - 1.0 / f1)}},
                  open(wd / "cavity_modes.json", "w"), indent=2)
    json.dump({"material": mat, "ranking": ranked}, open(outroot / "hybrid_result.json", "w"), indent=2)
    print("-> {}".format(outroot / "hybrid_result.json"))


if __name__ == "__main__":
    main()
