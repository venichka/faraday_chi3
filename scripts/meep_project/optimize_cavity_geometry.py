#!/usr/bin/env python3
"""
optimize_cavity_geometry.py

Optimize a 1D DBR cavity (no spacers) so that reflectance dips align near
λ = 800, 1550, 1650 nm by tuning: [L_cav, t_SiN, t_SiO2, cell_margin].

INPUT (either):
  1) --base <python module>  (e.g. fp_cavity_modes_spectrum)
     The module should define: mat_SiN, mat_SiO2, resolution, dpml, pad_air, pad_sub
     Optionally: N_per, t_SiN, t_SiO2, t_cav
  2) --in-json <geometry JSON> compatible with geometry_io.py

OUTPUTS:
  - optimized_geometry.json  (fully compatible with geometry_io.py)
  - optimize_report.json     (best params + score + R at targets)
  - reflectance_plot.png     (spectrum with markers)

Meep API calls follow the official Python UI (add_flux / get_flux_data / load_minus_flux_data). 
"""

import argparse
import importlib
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import meep as mp

# --- our helpers ---
from mode_targeting import (
    # mirrors + cavity optimizer (no spacers)
    optimize_geometry_mirrors,
    # build geometry from (N_per, t_SiN, t_SiO2, L_cav)
    build_stack_from_params,
    # 2-run normalized reflectance (R >= 0)
    reflectance_spectrum,
    TARGET_WL, fcen, df, nfreq
)

# optional I/O helpers (only used for JSON import/export)
try:
    from geometry_io import read_json as geo_read_json, write_json as geo_write_json
except Exception:
    geo_read_json = geo_write_json = None


# --------------------- utilities ---------------------
def _try_import_base(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"[WARN] Could not import base module '{module_name}': {e}")
        return None


def _discover_base_module(preferred: str):
    for name in [preferred, "fp_cavity_modes_spectrum", "meep_fp_cavity_modes_and_reflectance", "cavity_base"]:
        if not name:
            continue
        mod = _try_import_base(name)
        if mod is not None:
            print(f"[INFO] Using base module: {name}")
            return mod
    raise RuntimeError("Failed to import any base module.")


def _get_attr(mod, name, default=None):
    return getattr(mod, name, default)


def _infer_from_json(spec):
    """Return tuple:
       (n_SiN, n_SiO2, dpml, pad_air, pad_sub, N_per, t_SiN, t_SiO2, L_cav)"""
    mats = spec["materials"]
    n_SiN = float(mats["SiN"]["params"].get("index", 2.0))
    n_SiO2 = float(mats["SiO2"]["params"].get("index", 1.45))

    dpml = float(spec["pads"]["pml_um"])
    pad_air = float(spec["pads"]["air_um"])
    pad_sub = float(spec["pads"]["substrate_um"])

    L_cav = float(spec["cavity"]["L_um"])

    # Deduce N_per and average t_SiN/t_SiO2 from left mirror
    left = spec["mirrors"]["left"]
    tH = [layer["thk_um"] for layer in left if layer["mat"] == "SiN"]
    tL = [layer["thk_um"] for layer in left if layer["mat"] == "SiO2"]
    if not tH or not tL:
        # fallback: try right mirror
        right = spec["mirrors"]["right"]
        tH = [layer["thk_um"]
              for layer in right if layer["mat"] == "SiN"] or [0.26]
        tL = [layer["thk_um"]
              for layer in right if layer["mat"] == "SiO2"] or [0.16]
    t_SiN = float(np.mean(tH))
    t_SiO2 = float(np.mean(tL))
    N_per = int(min(len(tH), len(tL))) if (tH and tL) else 3

    return n_SiN, n_SiO2, dpml, pad_air, pad_sub, N_per, t_SiN, t_SiO2, L_cav


def _infer_from_base(base):
    """Try to extract everything from the Python module; fall back sanely."""
    # materials (index only needed for soft quarter-wave regularizer)
    mat_SiN = _get_attr(base, "mat_SiN", mp.Medium(index=2.00))
    mat_SiO2 = _get_attr(base, "mat_SiO2", mp.Medium(index=1.45))
    try:
        n_SiN = float(getattr(mat_SiN, "index", 2.0))
        n_SiO2 = float(getattr(mat_SiO2, "index", 1.45))
    except Exception:
        n_SiN, n_SiO2 = 2.0, 1.45

    resolution = int(_get_attr(base, "resolution", 100))
    dpml = float(_get_attr(base, "dpml", 1.0))
    pad_air = float(_get_attr(base, "pad_air", 0.8))
    pad_sub = float(_get_attr(base, "pad_sub", 0.8))

    # geometric hints (optional)
    N_per = int(_get_attr(base, "N_per", 3))
    t_SiN = float(_get_attr(base, "t_SiN", 0.26))
    t_SiO2 = float(_get_attr(base, "t_SiO2", 0.16))
    L_cav = float(_get_attr(base, "t_cav", 6.4))

    return mat_SiN, mat_SiO2, n_SiN, n_SiO2, resolution, dpml, pad_air, pad_sub, N_per, t_SiN, t_SiO2, L_cav


def _stack_len(N_per, t_SiN, t_SiO2, L_cav, pad_air, pad_sub):
    return pad_air + N_per*(t_SiN + t_SiO2) + L_cav + N_per*(t_SiO2 + t_SiN) + pad_sub


def _make_geometry_spec(n_SiN, n_SiO2, dpml, pad_air, pad_sub,
                        N_per, t_SiN, t_SiO2, L_cav):
    """Build JSON compatible with geometry_io: mirrors list explicitly, no spacers."""
    left = []
    for _ in range(N_per):
        left += [{"mat": "SiN",
                  "thk_um": float(t_SiN)}, {"mat": "SiO2", "thk_um": float(t_SiO2)}]
    right = []
    for _ in range(N_per):
        right += [{"mat": "SiO2",
                   "thk_um": float(t_SiO2)}, {"mat": "SiN", "thk_um": float(t_SiN)}]

    spec = {
        "materials": {
            "SiN":  {"type": "Medium", "params": {"index": float(n_SiN)}},
            "SiO2": {"type": "Medium", "params": {"index": float(n_SiO2)}},
        },
        "pads": {"pml_um": float(dpml), "air_um": float(pad_air), "substrate_um": float(pad_sub)},
        "spacers": {"left_um": 0.0, "right_um": 0.0},
        "cavity": {"mat": "SiN", "L_um": float(L_cav)},
        "mirrors": {"left": left, "right": right},
        "meta": {"generated_on": datetime.utcnow().isoformat()+"Z"}
    }
    return spec


# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(
        description="Optimize DBR cavity (no spacers) for dips at 800/1550/1650 nm.")
    ap.add_argument("--base", type=str, default=None,
                    help="Python module with current Meep setup")
    ap.add_argument("--in-json", type=str, default=None,
                    help="Geometry JSON compatible with geometry_io.py")
    ap.add_argument("--L0", type=float, default=6.4,
                    help="Initial cavity length (μm)")
    ap.add_argument("--tH0", type=float, default=0.20,
                    help="Initial SiN layer thickness (μm)")
    ap.add_argument("--tL0", type=float, default=0.28,
                    help="Initial SiO2 layer thickness (μm)")
    ap.add_argument("--Nper", type=int, default=None,
                    help="Number of DBR periods (override)")
    ap.add_argument("--maxiter", type=int, default=100,
                    help="Nelder–Mead iterations")
    ap.add_argument("--bounds", type=float, nargs=8,
                    metavar=("Lmin", "Lmax", "tHmin", "tHmax",
                             "tLmin", "tLmax", "margMin", "margMax"),
                    default=(0.5, 8.0, 0.05, 0.50, 0.05, 0.60, 0.0, 1.0),
                    help="Bounds for [L_cav, t_SiN, t_SiO2, cell_margin]")
    ap.add_argument("--outfile", type=str,
                    default="optimize_report.json", help="Report JSON filename")
    ap.add_argument("--out-geom", type=str,
                    default="optimized_geometry.json", help="Geometry JSON filename")
    ap.add_argument("--plot", type=str, default="reflectance_plot.png",
                    help="Reflectance plot filename")
    args = ap.parse_args()

    # ---------- input: either JSON or base module ----------
    if args.in_json:
        if geo_read_json is None:
            raise RuntimeError(
                "geometry_io.read_json not available. Please ensure geometry_io.py is importable.")
        spec_in = geo_read_json(args.in_json)
        n_SiN, n_SiO2, dpml, pad_air, pad_sub, N_per_j, t_SiN_j, t_SiO2_j, L_cav_j = _infer_from_json(
            spec_in)
        # materials for meep
        mat_SiN = mp.Medium(index=n_SiN)
        mat_SiO2 = mp.Medium(index=n_SiO2)
        resolution = 200  # default; JSON doesn't store resolution
        # starting guesses
        N_per = args.Nper if args.Nper is not None else N_per_j
        L0 = args.L0 if args.L0 != 6.4 else L_cav_j
        tH0 = args.tH0 if args.tH0 != 0.20 else t_SiN_j
        tL0 = args.tL0 if args.tL0 != 0.28 else t_SiO2_j
    else:
        if not args.base:
            raise SystemExit("Provide --base <module> or --in-json <file>")
        base = _discover_base_module(args.base)
        mat_SiN, mat_SiO2, n_SiN, n_SiO2, resolution, dpml, pad_air, pad_sub, \
            N_per_b, t_SiN_b, t_SiO2_b, L_cav_b = _infer_from_base(base)
        N_per = args.Nper if args.Nper is not None else N_per_b
        n_SiN = getattr(mat_SiN, "index", 2.0)
        n_SiO2 = getattr(mat_SiO2, "index", 1.45)
        L0 = args.L0 if args.L0 != 6.4 else L_cav_b
        tH0 = args.tH0 if args.tH0 != 0.20 else t_SiN_b
        tL0 = args.tL0 if args.tL0 != 0.28 else t_SiO2_b

    # ---------- optimize mirrors + cavity ----------
    bounds = ((args.bounds[0], args.bounds[1]),
              (args.bounds[2], args.bounds[3]),
              (args.bounds[4], args.bounds[5]),
              (args.bounds[6], args.bounds[7]))

    best_theta, bestJ = optimize_geometry_mirrors(
        N_per=N_per,
        dpml=dpml, pad_air=pad_air, pad_sub=pad_sub, resolution=resolution,
        mat_SiN=mat_SiN, mat_SiO2=mat_SiO2, n_SiN=n_SiN, n_SiO2=n_SiO2,
        L0=L0, tH0=tH0, tL0=tL0, margin0=bounds[3][0],
        bounds=bounds, maxiter=args.maxiter
    )
    L_cav, t_SiN_opt, t_SiO2_opt, cell_margin = best_theta

    # ---------- build final geometry & spectrum for the optimum ----------
    stack_len = _stack_len(N_per, t_SiN_opt, t_SiO2_opt,
                           L_cav, pad_air, pad_sub)
    cell_z = stack_len + 2*dpml + cell_margin
    cell = mp.Vector3(0, 0, cell_z)
    geometry = build_stack_from_params(N_per, t_SiN_opt, t_SiO2_opt, L_cav,
                                       dpml, pad_air, pad_sub, mat_SiN, mat_SiO2, cell_z)
    freqs, R = reflectance_spectrum(
        cell, geometry, resolution, dpml, fcen, df, nfreq)
    wl = 1.0/freqs
    R_targets = np.interp(TARGET_WL, wl[::-1], R[::-1])

    # ---------- save spectrum plot ----------
    import matplotlib
    matplotlib.use("Agg")
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(1e3*wl, R, lw=1.5)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Optimized reflectance")
    ax.invert_xaxis()
    for lam_um, Rt in zip(TARGET_WL, R_targets):
        ax.scatter([1e3*lam_um], [Rt])
        ax.annotate(f"{int(1e3*lam_um)} nm", xy=(1e3*lam_um, Rt),
                    xytext=(5, 5), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=200)

    # ---------- write geometry JSON (compatible with geometry_io) ----------
    spec_out = _make_geometry_spec(n_SiN, n_SiO2, dpml, pad_air, pad_sub,
                                   N_per, t_SiN_opt, t_SiO2_opt, L_cav)
    if geo_write_json is None:
        with open(args.out_geom, "w") as f:
            json.dump(spec_out, f, indent=2)
    else:
        geo_write_json(spec_out, args.out_geom)

    # ---------- write optimization report ----------
    report = {
        "best_theta": {
            "L_cav_um": float(L_cav),
            "t_SiN_um": float(t_SiN_opt),
            "t_SiO2_um": float(t_SiO2_opt),
            "cell_margin_um": float(cell_margin),
            "N_per": int(N_per),
        },
        "score": float(bestJ),
        "R_targets": {
            "800nm": float(R_targets[0]),
            "1550nm": float(R_targets[1]),
            "1650nm": float(R_targets[2]),
        },
        "sim": {
            "resolution_px_per_um": int(resolution),
            "dpml_um": float(dpml),
            "pad_air_um": float(pad_air),
            "pad_sub_um": float(pad_sub),
            "cell_z_um": float(cell_z),
        },
        "files": {
            "geometry_json": args.out_geom,
            "plot_png": args.plot
        },
        "notes": "Mirror+cavity optimization. Reflectance via two-run normalization (get_flux_data/load_minus_flux_data)."
    }
    with open(args.outfile, "w") as f:
        json.dump(report, f, indent=2)

    print("[DONE] Best parameters:", report["best_theta"])
    print("[DONE] R@targets:", report["R_targets"])
    print(
        f"[DONE] Saved geometry to {args.out_geom}, plot to {args.plot}, report to {args.outfile}")


if __name__ == "__main__":
    main()
