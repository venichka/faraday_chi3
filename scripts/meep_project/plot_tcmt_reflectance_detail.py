#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from extract_tcmt_params import _build_materials
from optimize_cavity_geometry import build_1d_geometry_from_spec, debug_reflectance


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rt_spectra(
    *,
    spec: Dict,
    mats: Dict[str, mp.Medium],
    resolution: int,
    nfreq: int,
    wl_min: float,
    wl_max: float,
    decay_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geom, cell_z, _ = build_1d_geometry_from_spec(spec, mats)
    dpml = float(spec["pads"]["pml_um"])

    wl_lo = float(min(wl_min, wl_max))
    wl_hi = float(max(wl_min, wl_max))
    fmin = float(1.0 / wl_hi)
    fmax = float(1.0 / wl_lo)
    fcen = 0.5 * (fmin + fmax)
    df = fmax - fmin

    src_z = -0.5 * cell_z + dpml + 0.2
    refl_z = src_z + 0.1
    tran_z = 0.5 * cell_z - dpml - 0.2
    src = [
        mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=df),
            component=mp.Ex,
            center=mp.Vector3(0, 0, src_z),
        )
    ]

    def _make_sim(geometry: List[mp.Block]) -> mp.Simulation:
        return mp.Simulation(
            cell_size=mp.Vector3(0, 0, cell_z),
            geometry=geometry,
            sources=src,
            boundary_layers=[mp.PML(dpml)],
            default_material=mp.air,
            resolution=int(resolution),
            dimensions=1,
            force_complex_fields=True,
        )

    sim_ref = _make_sim([])
    refl_ref = sim_ref.add_flux(fcen, df, int(nfreq), mp.FluxRegion(center=mp.Vector3(0, 0, refl_z)))
    tran_ref = sim_ref.add_flux(fcen, df, int(nfreq), mp.FluxRegion(center=mp.Vector3(0, 0, tran_z)))
    sim_ref.run(
        until_after_sources=mp.stop_when_fields_decayed(
            60, mp.Ex, mp.Vector3(0, 0, refl_z), float(decay_threshold)
        )
    )
    inc_refl = np.asarray(mp.get_fluxes(refl_ref), dtype=float)
    inc_tran = np.asarray(mp.get_fluxes(tran_ref), dtype=float)
    freqs = np.asarray(mp.get_flux_freqs(refl_ref), dtype=float)
    refl_ref_data = sim_ref.get_flux_data(refl_ref)

    sim = _make_sim(geom)
    refl = sim.add_flux(fcen, df, int(nfreq), mp.FluxRegion(center=mp.Vector3(0, 0, refl_z)))
    tran = sim.add_flux(fcen, df, int(nfreq), mp.FluxRegion(center=mp.Vector3(0, 0, tran_z)))
    sim.load_minus_flux_data(refl, refl_ref_data)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            60, mp.Ex, mp.Vector3(0, 0, refl_z), float(decay_threshold)
        )
    )

    refl_flux = np.asarray(mp.get_fluxes(refl), dtype=float)
    tran_flux = np.asarray(mp.get_fluxes(tran), dtype=float)
    den_refl = np.where(np.abs(inc_refl) > 1e-30, inc_refl, np.nan)
    den_tran = np.where(np.abs(inc_tran) > 1e-30, inc_tran, np.nan)
    r_raw = -refl_flux / den_refl
    t_raw = tran_flux / den_tran
    r = np.maximum(0.0, np.nan_to_num(r_raw, nan=0.0, posinf=0.0, neginf=0.0))
    t = np.maximum(0.0, np.nan_to_num(t_raw, nan=0.0, posinf=0.0, neginf=0.0))
    wl = 1.0 / freqs
    order = np.argsort(wl)
    wl = wl[order]
    r = r[order]
    t = t[order]
    return wl, r, t, r + t


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot detailed reflectance diagnostics for TCMT extraction."
    )
    root = Path(__file__).resolve().parent / "pipeline_tio2_20260302_162215" / "optimizers" / "mf"
    ap.add_argument("--geometry-file", type=Path, default=root / "optimized_geometry.json")
    ap.add_argument("--modes-file", type=Path, default=root / "cavity_modes.json")
    ap.add_argument("--tcmt-json", type=Path, default=root / "tcmt_extracted_params.json")
    ap.add_argument("--output", type=Path, default=root / "tcmt_reflectance_detailed.png")
    ap.add_argument("--wl-min", type=float, default=0.6)
    ap.add_argument("--wl-max", type=float, default=2.0)
    ap.add_argument("--nfreq", type=int, default=801)
    ap.add_argument("--resolution", type=int, default=60)
    ap.add_argument("--decay-threshold", type=float, default=1e-8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    spec = _load_json(args.geometry_file)
    mode_json = _load_json(args.modes_file)
    tcmt = _load_json(args.tcmt_json)

    mat = tcmt.get("material_constants", {})
    nH = float(mat.get("nH", 2.31))
    kH = float(mat.get("kH", 8e-6))
    nL = float(mat.get("nL", 1.45))
    kref = float(mat.get("kappa_ref_lambda_um", 1.55))
    mats = _build_materials(nH=nH, kH=kH, nL=nL, kref_um=kref)

    wl, r, t, rt = _rt_spectra(
        spec=spec,
        mats=mats,
        resolution=int(args.resolution),
        nfreq=int(args.nfreq),
        wl_min=float(args.wl_min),
        wl_max=float(args.wl_max),
        decay_threshold=float(args.decay_threshold),
    )
    wl_old, r_old = debug_reflectance(
        spec=spec,
        mats=mats,
        resolution=int(args.resolution),
        nfreq=int(args.nfreq),
        decay_threshold=float(args.decay_threshold),
        wl_min=float(args.wl_min),
        wl_max=float(args.wl_max),
    )
    order_old = np.argsort(wl_old)
    wl_old = wl_old[order_old]
    r_old = r_old[order_old]
    r_old_interp = np.interp(wl, wl_old, r_old)
    d_r = r - r_old_interp

    target_wl = {
        "pump1": 1.0 / float(mode_json["pump1"]["frequency"]),
        "pump2": 1.0 / float(mode_json["pump2"]["frequency"]),
        "probe": 1.0 / float(mode_json["probe"]["frequency"]),
        "sb+": 1.0 / float(mode_json["sidebands"]["frequency_plus"]),
        "sb-": 1.0 / float(mode_json["sidebands"]["frequency_minus"]),
    }
    harminv = tcmt.get("resonance_fit_loaded_lossy_harminv", {})

    fig = plt.figure(figsize=(10.5, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.3, 1.7])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    ax0.plot(wl, r, lw=1.6, label="R (independent)")
    ax0.plot(wl, t, lw=1.4, label="T (independent)")
    ax0.plot(wl, rt, lw=1.2, ls="--", label="R+T")
    ax0.plot(wl, r_old_interp, lw=1.0, alpha=0.75, label="R (debug_reflectance)")

    colors = {
        "pump1": "#d62728",
        "pump2": "#ff7f0e",
        "probe": "#1f77b4",
        "sb+": "#2ca02c",
        "sb-": "#9467bd",
    }
    for key, x in target_wl.items():
        c = colors[key]
        ax0.axvline(x, color=c, lw=0.8, alpha=0.5)
        y = float(np.interp(x, wl, r))
        ax0.scatter([x], [y], s=22, color=c, zorder=4)
        ax0.text(x, min(1.11, y + 0.04), f"{key}\n{1e3*x:.1f} nm", color=c, fontsize=8, ha="center")

    harminv_map = {
        "pump1": "pump1",
        "pump2": "pump2",
        "probe": "probe",
        "sb+": "sb_plus",
        "sb-": "sb_minus",
    }
    for short, full in harminv_map.items():
        h = harminv.get(full, {})
        lam = float(h.get("lam", np.nan))
        q = float(h.get("Q", np.nan))
        if np.isfinite(lam) and np.isfinite(q):
            yy = float(np.interp(lam, wl, r))
            ax0.scatter([lam], [yy], marker="x", s=36, color=colors[short], zorder=5)
            ax0.text(lam, max(0.03, yy - 0.07), f"Q={q:.1f}", color=colors[short], fontsize=7, ha="center")

    rv = tcmt.get("reflectance_validation", {})
    text = (
        f"range={rv.get('wl_range_um', [args.wl_min, args.wl_max])} um\n"
        f"nfreq={rv.get('nfreq', args.nfreq)}, res={args.resolution}\n"
        f"R+T mean={rv.get('RT_sum_mean', float('nan')):.6f}, p99={rv.get('RT_sum_p99', float('nan')):.6f}\n"
        f"R diff RMS={rv.get('R_old_vs_new_abs_diff_rms', float('nan')):.6f}"
    )
    ax0.text(
        0.013,
        0.98,
        text,
        transform=ax0.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.9),
    )

    ax0.set_ylabel("Spectral value")
    ax0.set_ylim(-0.02, 1.14)
    ax0.grid(alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8, ncol=2)
    ax0.set_title("Detailed Reflectance Diagnostics (TiO2 mf geometry)")

    ax1.plot(wl, d_r, lw=1.2, color="#333333", label="R(independent) - R(debug)")
    ax1.axhline(0.0, lw=0.8, color="k", alpha=0.5)
    ax1.set_xlabel("Wavelength (um)")
    ax1.set_ylabel("Delta R")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
