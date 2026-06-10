#!/usr/bin/env python
"""FDTD-light chi5 geometry optimizer (Phase 1: refine an existing DBR; Phase 2: open bounds).

Because the objective is analytic TMM+TCMT (~ms/candidate, chi5_optimization.objective.chi5_score),
we use a dense Sobol scan over the bounded geometry space rather than a sample-efficient BO.
For each candidate geometry we auto-select the operating point (hybrid, user 2026-06-10):
  - probe: best cavity mode in {~800} U [850,950] nm,
  - pumps: the resonant mode PAIR (f1,f2) with f1+f2 ~ f_probe (FWM) and small Delta,
  - Delta = f1 - f2 is DERIVED from the chosen pumps (not a free variable).
Top candidates are written out for the TCMT(FaradayJL)+1D+3D-FDTD validation gate.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
sys.path.insert(0, str(HERE))
import tmm          # noqa: E402
import objective    # noqa: E402

# probe windows (um) and pump-band / Delta tolerances
PROBE_WINDOWS = [(0.790, 0.810), (0.850, 0.950)]
PUMP_BAND = (1.40, 1.95)
# NOTE: the f1+f2~=f_probe (octave) constraint was REMOVED (2026-06-10) -- the 1D-FDTD
# operating-point diagnostic showed it is wrong: max |theta| is at the resonant pump modes
# (NOT octave-matched) with small Delta. So pumps are chosen by buildup+small-Delta only.
DELTA_RANGE = (0.005, 0.030)  # Delta = f1-f2 small (sidebands inside the probe mode, M5)


def sobol(bounds, count, seed=0):
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    try:
        from scipy.stats.qmc import Sobol
        m = max(1, int(np.ceil(np.log2(max(count, 1)))))
        raw = Sobol(d=len(bounds), scramble=True, seed=seed).random_base2(m)[:count]
    except Exception:
        raw = np.random.default_rng(seed).uniform(size=(count, len(bounds)))
    return raw * (hi - lo) + lo


def build_geometry(base, n_pairs, t_hi, t_lo, L_cav):
    """Detuned single-defect DBR like the base, with given pair count, layer thicknesses, cavity L.
    Keeps the base's material labels, pads, and left/right starting-layer arrangement."""
    g = copy.deepcopy(base)
    hi = g["cavity"]["mat"]
    lo = next(m for m in g["materials"] if m != hi)
    g["cavity"]["L_um"] = float(L_cav)
    base_left = base["mirrors"]["left"]
    left_hi_first = base_left[0]["mat"] == hi
    left, right = [], []
    for p in range(int(n_pairs)):
        a = ({"mat": hi, "thk_um": float(t_hi)}, {"mat": lo, "thk_um": float(t_lo)})
        left += list(a if left_hi_first else a[::-1])
        right += list(a[::-1] if left_hi_first else a)
    g["mirrors"]["left"] = left
    g["mirrors"]["right"] = right
    return g


def select_operating_point(geom, chi_iso, sub_label="SiO2"):
    """Hybrid operating-point selector -> (best score dict, freqs) or (None, None).
    Builds one per-geometry context (field cache) and passes the already-found mode Q's
    so the inner pump-pair loop only recomputes the sidebands."""
    ctx = objective.make_ctx(geom, sub_label)
    layers, idx = ctx["layers"], ctx["idx"]
    probe_modes = []
    for lo, hi in PROBE_WINDOWS:
        probe_modes += tmm.find_modes_in_band(layers, idx, 1.0 / hi, 1.0 / lo, sub_label)
    pumps = tmm.find_modes_in_band(layers, idx, 1.0 / PUMP_BAND[1], 1.0 / PUMP_BAND[0], sub_label)
    if not probe_modes or len(pumps) < 1:
        return None, None
    best, best_freqs = None, None
    for pm in probe_modes:
        fs = pm["freq"]
        for i in range(len(pumps)):
            for j in range(i, len(pumps)):
                hi_p = pumps[i] if pumps[i]["freq"] >= pumps[j]["freq"] else pumps[j]
                lo_p = pumps[j] if pumps[i]["freq"] >= pumps[j]["freq"] else pumps[i]
                f1, f2 = hi_p["freq"], lo_p["freq"]
                d = f1 - f2
                if not (DELTA_RANGE[0] <= d <= DELTA_RANGE[1]):   # small Delta only (no octave constraint)
                    continue
                freqs = {"probe": fs, "pump1": f1, "pump2": f2,
                         "sb_plus": fs + d, "sb_minus": fs - d}
                q_known = {"probe": pm["Q"], "pump1": hi_p["Q"], "pump2": lo_p["Q"]}
                try:
                    s = objective.chi5_score(geom, freqs, chi_iso, ctx=ctx, q_known=q_known)
                except Exception:
                    continue
                if best is None or s["fom_rotation"] > best["fom_rotation"]:
                    best, best_freqs = s, freqs
    return best, best_freqs


def phase1_bounds(base, scope="regime"):
    """Param bounds [n_pairs, t_hi, t_lo, L_cav] near the base design."""
    hi = base["cavity"]["mat"]
    bl = base["mirrors"]["left"]
    t_hi0 = float(np.mean([l["thk_um"] for l in bl if l["mat"] == hi]))
    t_lo0 = float(np.mean([l["thk_um"] for l in bl if l["mat"] != hi]))
    L0 = float(base["cavity"]["L_um"])
    n0 = len(bl) // 2
    f = 0.12 if scope == "local" else 0.30      # +- fractional window on thicknesses/L
    return [(max(2, n0 - 1) - 0.49, n0 + 1 + 0.49),
            (t_hi0 * (1 - f), t_hi0 * (1 + f)),
            (t_lo0 * (1 - f), t_lo0 * (1 + f)),
            (L0 * (1 - f), L0 * (1 + f))], (n0, t_hi0, t_lo0, L0)


def run(base_geom_path, chi_iso, n_samples=512, scope="regime", seed=0, sub_label="SiO2", topk=8):
    base = json.load(open(base_geom_path))
    bounds, base_params = phase1_bounds(base, scope)
    pts = sobol(bounds, n_samples, seed)
    results = []
    for p in pts:
        n_pairs = int(round(p[0]))
        geom = build_geometry(base, n_pairs, p[1], p[2], p[3])
        s, freqs = select_operating_point(geom, chi_iso, sub_label)
        if s is None:
            continue
        results.append({"params": {"n_pairs": n_pairs, "t_hi": float(p[1]),
                                    "t_lo": float(p[2]), "L_cav": float(p[3])},
                        "freqs": {k: float(v) for k, v in freqs.items()},
                        "fom_rotation": s["fom_rotation"], "fom_ellipticity": s["fom_ellipticity"],
                        "B1": s["B1"], "B2": s["B2"], "Bs": s["Bs"],
                        "Qprobe": s["Qprobe"], "geometry": geom})
    results.sort(key=lambda r: r["fom_rotation"], reverse=True)
    # baseline (the existing design, at its own operating point)
    base_s, base_freqs = select_operating_point(base, chi_iso, sub_label)
    return results[:topk], (base_s, base_freqs, base_params)
