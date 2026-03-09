#!/usr/bin/env python3
"""
Extract TCMT-ready parameters from direct Meep diagnostics.

This helper targets the optimized cavity artifacts (geometry + modes) and
computes:
1) Loaded/lossless resonance fits and Q for pump/probe/sideband frequencies.
2) Decay-rate decomposition (kappa_total, kappa_ext, kappa_int) plus a
   left/right port split estimate for kappa_ext.
3) Overlap-based proxies for FaradayJL Norms terms.

All quantities are written to a JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import meep as mp
import numpy as np

# Keep matplotlib caches out of non-writable home paths.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from mode_targeting import get_cavity_materials, material_index_at_wavelength
from nonlinear_materials import n2_to_chi3_si
from optimize_cavity_geometry import (
    build_1d_geometry_from_spec,
    cavity_overlaps,
    debug_reflectance,
    debug_mode_profiles,
    refine_local_resonances,
)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_materials(*, nH: float, kH: float, nL: float, kref_um: float) -> Dict[str, mp.Medium]:
    mat_high, mat_low = get_cavity_materials(
        model="constant",
        index_high=float(nH),
        kappa_high=float(kH),
        index_low=float(nL),
        high_index_material="tio2",
        kappa_ref_wavelength_um=float(kref_um),
    )
    return {"SiN": mat_high, "SiO2": mat_low}


def _mode_freqs(mode_json: Dict) -> Dict[str, float]:
    return {
        "pump1": float(mode_json["pump1"]["frequency"]),
        "pump2": float(mode_json["pump2"]["frequency"]),
        "probe": float(mode_json["probe"]["frequency"]),
        "sb_plus": float(mode_json["sidebands"]["frequency_plus"]),
        "sb_minus": float(mode_json["sidebands"]["frequency_minus"]),
    }


def _mode_lams(mode_json: Dict) -> Dict[str, float]:
    freqs = _mode_freqs(mode_json)
    out = {}
    for k, f in freqs.items():
        out[k] = float(1.0 / f) if f > 0 else float("nan")
    return out


def _harminv_global_resonances(
    *,
    spec: Dict,
    mats: Dict[str, mp.Medium],
    target_freqs: Dict[str, float],
    resolution: int,
    run_time: float,
    band_pad: float,
    max_rel_detune: float,
    error_max: float,
) -> Dict[str, Dict[str, float]]:
    geom, cell_z, cavity_bounds = build_1d_geometry_from_spec(spec, mats)
    dpml = float(spec["pads"]["pml_um"])

    fvals = np.array([float(v) for v in target_freqs.values() if np.isfinite(v)], dtype=float)
    fmin = float(np.min(fvals))
    fmax = float(np.max(fvals))
    f_center = 0.5 * (fmin + fmax)
    f_span = max(fmax - fmin, 1e-3)
    fwidth = float(f_span * (1.0 + 2.0 * float(max(band_pad, 0.0))))

    src_z = -0.5 * cell_z + dpml + 0.2
    mon_z = 0.5 * (float(cavity_bounds[0]) + float(cavity_bounds[1]))
    src = [
        mp.Source(
            mp.GaussianSource(frequency=f_center, fwidth=fwidth),
            component=mp.Ex,
            center=mp.Vector3(0, 0, src_z),
            amplitude=1.0,
        )
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(0, 0, cell_z),
        geometry=geom,
        sources=src,
        boundary_layers=[mp.PML(dpml)],
        default_material=mp.air,
        resolution=int(resolution),
        dimensions=1,
        force_complex_fields=True,
    )
    mon = mp.Harminv(mp.Ex, mp.Vector3(0, 0, mon_z), f_center, fwidth)
    sim.run(mp.after_sources(mon), until=float(run_time))

    candidates: List[Dict[str, float]] = []
    for mode in list(mon.modes):
        freq = float(getattr(mode, "freq", float("nan")))
        q = float(getattr(mode, "Q", float("nan")))
        err = float(getattr(mode, "error", float("nan")))
        decay = float(getattr(mode, "decay", float("nan")))
        if (not np.isfinite(freq)) or freq <= 0.0:
            continue
        if (not np.isfinite(q)) or q <= 0.0:
            continue
        if np.isfinite(err) and err > float(error_max):
            continue
        candidates.append(
            {
                "freq": freq,
                "lam": float(1.0 / freq),
                "Q": q,
                "error": err,
                "decay": decay,
            }
        )

    out: Dict[str, Dict[str, float]] = {}
    for key, f_target in target_freqs.items():
        f0 = float(f_target)
        max_df = max(float(max_rel_detune) * abs(f0), 0.005)
        viable = [m for m in candidates if abs(float(m["freq"]) - f0) <= max_df]
        if not viable:
            out[key] = {
                "freq": float("nan"),
                "lam": float("nan"),
                "Q": float("nan"),
                "error": float("nan"),
                "decay": float("nan"),
                "detune_inv_um": float("nan"),
                "selection_note": "no_harminv_mode_within_detune_window",
            }
            continue
        best = min(
            viable,
            key=lambda m: (
                abs(float(m["freq"]) - f0),
                float(m["error"]) if np.isfinite(float(m["error"])) else 1e9,
            ),
        )
        out[key] = {
            "freq": float(best["freq"]),
            "lam": float(best["lam"]),
            "Q": float(best["Q"]),
            "error": float(best["error"]),
            "decay": float(best["decay"]),
            "detune_inv_um": float(best["freq"] - f0),
            "selection_note": "nearest_mode_in_detune_window",
        }
    return out


def _reflectance_rt_validation(
    *,
    spec: Dict,
    mats: Dict[str, mp.Medium],
    resolution: int,
    nfreq: int,
    wl_min: float,
    wl_max: float,
    decay_threshold: float,
) -> Dict[str, object]:
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
    r_clip = np.maximum(0.0, np.nan_to_num(r_raw, nan=0.0, posinf=0.0, neginf=0.0))
    t_clip = np.maximum(0.0, np.nan_to_num(t_raw, nan=0.0, posinf=0.0, neginf=0.0))
    rt_sum = r_clip + t_clip

    wl_old, r_old = debug_reflectance(
        spec=spec,
        mats=mats,
        resolution=int(resolution),
        nfreq=int(nfreq),
        decay_threshold=float(decay_threshold),
        wl_min=wl_lo,
        wl_max=wl_hi,
    )
    wl_new = 1.0 / freqs
    if wl_old[0] > wl_old[-1]:
        wl_old_i = wl_old[::-1]
        r_old_i = r_old[::-1]
    else:
        wl_old_i = wl_old
        r_old_i = r_old
    if wl_new[0] > wl_new[-1]:
        wl_new_i = wl_new[::-1]
        r_new_i = r_clip[::-1]
    else:
        wl_new_i = wl_new
        r_new_i = r_clip
    r_old_interp = np.interp(wl_new_i, wl_old_i, r_old_i)
    diff = r_new_i - r_old_interp

    return {
        "wl_range_um": [wl_lo, wl_hi],
        "nfreq": int(nfreq),
        "R_min": float(np.nanmin(r_clip)),
        "R_max": float(np.nanmax(r_clip)),
        "T_min": float(np.nanmin(t_clip)),
        "T_max": float(np.nanmax(t_clip)),
        "RT_sum_min": float(np.nanmin(rt_sum)),
        "RT_sum_max": float(np.nanmax(rt_sum)),
        "RT_sum_mean": float(np.nanmean(rt_sum)),
        "RT_sum_p99": float(np.nanpercentile(rt_sum, 99)),
        "count_rt_gt_1p01": int(np.sum(rt_sum > 1.01)),
        "count_rt_gt_1p05": int(np.sum(rt_sum > 1.05)),
        "R_old_vs_new_abs_diff_rms": float(np.sqrt(np.nanmean(diff * diff))),
        "R_old_vs_new_abs_diff_max": float(np.nanmax(np.abs(diff))),
    }


def _resonance_fit(
    *,
    spec: Dict,
    mats: Dict[str, mp.Medium],
    targets_um: Dict[str, float],
    linewidth_level: float,
    resolution: int,
    nfreq: int,
    decay_threshold: float,
    window_um: float,
) -> Dict[str, Dict[str, float]]:
    return refine_local_resonances(
        spec=spec,
        mats=mats,
        targets_um=targets_um,
        linewidth_level=float(linewidth_level),
        resolution=int(resolution),
        nfreq=int(nfreq),
        decay_threshold=float(decay_threshold),
        window_um=float(window_um),
    )


def _estimate_port_split(
    *,
    spec: Dict,
    mats: Dict[str, mp.Medium],
    freq: float,
    resolution: int,
    decay_threshold: float,
) -> Dict[str, float]:
    geom, cell_z, _ = build_1d_geometry_from_spec(spec, mats)
    dpml = float(spec["pads"]["pml_um"])
    left_z = -0.5 * cell_z + dpml + 0.12
    right_z = 0.5 * cell_z - dpml - 0.12
    f0 = float(freq)
    fwidth = max(0.03 * f0, 0.006)

    src = [
        mp.Source(
            mp.GaussianSource(frequency=f0, fwidth=fwidth),
            component=mp.Ex,
            center=mp.Vector3(0, 0, 0.0),
            amplitude=1.0,
        )
    ]
    sim = mp.Simulation(
        cell_size=mp.Vector3(0, 0, cell_z),
        geometry=geom,
        sources=src,
        boundary_layers=[mp.PML(dpml)],
        default_material=mp.air,
        resolution=int(resolution),
        dimensions=1,
        force_complex_fields=True,
    )
    fl = sim.add_flux(f0, 0.0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, left_z)))
    fr = sim.add_flux(f0, 0.0, 1, mp.FluxRegion(center=mp.Vector3(0, 0, right_z)))
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            100,
            mp.Ex,
            mp.Vector3(0, 0, 0.0),
            float(decay_threshold),
        )
    )

    left_flux = float(np.asarray(mp.get_fluxes(fl), dtype=float)[0])
    right_flux = float(np.asarray(mp.get_fluxes(fr), dtype=float)[0])
    left_mag = abs(left_flux)
    right_mag = abs(right_flux)
    denom = left_mag + right_mag
    if denom <= 1e-30:
        frac_left = float("nan")
        frac_right = float("nan")
    else:
        frac_left = float(left_mag / denom)
        frac_right = float(right_mag / denom)
    return {
        "left_flux_raw": float(left_flux),
        "right_flux_raw": float(right_flux),
        "left_flux_mag": float(left_mag),
        "right_flux_mag": float(right_mag),
        "frac_left": float(frac_left),
        "frac_right": float(frac_right),
    }


def _mode_rate_block(
    *,
    key: str,
    freq_target: float,
    fit_loaded: Dict[str, float],
    fit_lossless: Dict[str, float],
    port_split: Dict[str, float],
) -> Dict[str, float]:
    lam_loaded = float(fit_loaded.get("lam", float("nan")))
    lam_lossless = float(fit_lossless.get("lam", float("nan")))
    q_loaded = float(fit_loaded.get("Q", float("nan")))
    q_ext = float(fit_lossless.get("Q", float("nan")))

    f_loaded = float(1.0 / lam_loaded) if np.isfinite(lam_loaded) and lam_loaded > 0 else float("nan")
    f_lossless = float(1.0 / lam_lossless) if np.isfinite(lam_lossless) and lam_lossless > 0 else float("nan")

    omega_loaded = float(2.0 * math.pi * f_loaded) if np.isfinite(f_loaded) else float("nan")
    omega_lossless = float(2.0 * math.pi * f_lossless) if np.isfinite(f_lossless) else float("nan")

    k_loaded = (
        float(omega_loaded / q_loaded)
        if np.isfinite(omega_loaded) and np.isfinite(q_loaded) and q_loaded > 0
        else float("nan")
    )
    k_ext = (
        float(omega_lossless / q_ext)
        if np.isfinite(omega_lossless) and np.isfinite(q_ext) and q_ext > 0
        else float("nan")
    )

    if np.isfinite(k_loaded) and np.isfinite(k_ext):
        k_diff = float(k_loaded - k_ext)
        # Loaded vs. lossless runs can differ slightly due independent spectral fits.
        # Keep any physically valid positive k_int, but clamp small negative
        # differences to 0 as numerical fitting jitter.
        tol_neg = max(5e-3 * max(abs(k_loaded), abs(k_ext)), 1e-8)
        if k_diff >= 0.0:
            k_int = float(k_diff)
        elif abs(k_diff) <= tol_neg:
            k_int = 0.0
        else:
            k_int = float("nan")
    else:
        k_int = float("nan")

    if np.isfinite(omega_loaded) and np.isfinite(k_int):
        if k_int > 0.0:
            q_int = float(omega_loaded / k_int)
        elif k_int == 0.0:
            q_int = float("inf")
        else:
            q_int = float("nan")
    else:
        q_int = float("nan")

    frac_left = float(port_split.get("frac_left", float("nan")))
    frac_right = float(port_split.get("frac_right", float("nan")))
    k_ext_left = (
        float(frac_left * k_ext)
        if np.isfinite(frac_left) and np.isfinite(k_ext)
        else float("nan")
    )
    k_ext_right = (
        float(frac_right * k_ext)
        if np.isfinite(frac_right) and np.isfinite(k_ext)
        else float("nan")
    )

    return {
        "mode": key,
        "freq_target_inv_um": float(freq_target),
        "freq_loaded_inv_um": float(f_loaded),
        "freq_lossless_inv_um": float(f_lossless),
        "detune_loaded_inv_um": float(f_loaded - freq_target) if np.isfinite(f_loaded) else float("nan"),
        "detune_loaded_omega": float(2.0 * math.pi * (f_loaded - freq_target))
        if np.isfinite(f_loaded)
        else float("nan"),
        "Q_loaded": float(q_loaded),
        "Q_ext_lossless": float(q_ext),
        "Q_int_absorption": float(q_int),
        "kappa_loaded": float(k_loaded),
        "kappa_ext": float(k_ext),
        "kappa_int": float(k_int),
        "kappa_ext_left": float(k_ext_left),
        "kappa_ext_right": float(k_ext_right),
        "port_split": port_split,
    }


def _eta_proxies(
    *,
    z: np.ndarray,
    profiles: Dict[str, np.ndarray],
    cavity_bounds: Tuple[float, float],
) -> Dict[str, float]:
    z0, z1 = cavity_bounds
    mask = (z >= z0) & (z <= z1)
    zc = z[mask]

    if zc.size < 4:
        return {}

    def _norm(v: np.ndarray) -> np.ndarray:
        vv = np.asarray(v[mask], dtype=float)
        nrm = float(np.sqrt(np.trapezoid(vv * vv, zc)))
        return vv / max(nrm, 1e-30)

    u_probe = _norm(profiles["probe"])
    u_p1 = _norm(profiles["pump1"])
    u_p2 = _norm(profiles["pump2"])
    u_sb_p = _norm(profiles["sb_plus"])
    u_sb_m = _norm(profiles["sb_minus"])

    eta_s_u1_us = float(np.trapezoid((u_probe * u_probe) * (u_p1 * u_p1), zc))
    eta_s_u2_us = float(np.trapezoid((u_probe * u_probe) * (u_p2 * u_p2), zc))
    eta_Omega_p = float(np.trapezoid((u_probe * u_probe) * u_p1 * u_p2, zc))
    eta_Omega_m = float(np.trapezoid((u_probe * u_probe) * u_p2 * u_p1, zc))

    # Optional extra proxies involving sideband profiles.
    eta_sb_probe_p = float(np.trapezoid((u_probe * u_probe) * (u_sb_p * u_sb_p), zc))
    eta_sb_probe_m = float(np.trapezoid((u_probe * u_probe) * (u_sb_m * u_sb_m), zc))

    return {
        "eta_s_u1_us_proxy": eta_s_u1_us,
        "eta_s_u2_us_proxy": eta_s_u2_us,
        "eta_Omega_p_proxy": eta_Omega_p,
        "eta_Omega_m_proxy": eta_Omega_m,
        "eta_sb_probe_p_proxy": eta_sb_probe_p,
        "eta_sb_probe_m_proxy": eta_sb_probe_m,
    }


def parse_args() -> argparse.Namespace:
    default_root = Path("meep_project/pipeline_tio2_20260302_162215/optimizers/mf")
    ap = argparse.ArgumentParser(description="Extract TCMT mapping parameters from Meep.")
    ap.add_argument("--geometry-file", type=Path, default=default_root / "optimized_geometry.json")
    ap.add_argument("--modes-file", type=Path, default=default_root / "cavity_modes.json")
    ap.add_argument("--output-json", type=Path, default=default_root / "tcmt_extracted_params.json")
    ap.add_argument("--nH", type=float, default=2.31)
    ap.add_argument("--kH", type=float, default=8e-6)
    ap.add_argument("--nL", type=float, default=1.45)
    ap.add_argument("--n2", type=float, default=2.3e-18)
    ap.add_argument("--kappa-ref-lambda", type=float, default=1.55)
    ap.add_argument("--resolution", type=int, default=60)
    ap.add_argument("--nfreq", type=int, default=1601)
    ap.add_argument("--window-um", type=float, default=0.25)
    ap.add_argument("--linewidth-level", type=float, default=0.5)
    ap.add_argument("--decay-threshold", type=float, default=1e-6)
    ap.add_argument("--harminv-runtime", type=float, default=700.0)
    ap.add_argument("--harminv-band-pad", type=float, default=0.25)
    ap.add_argument("--harminv-max-rel-detune", type=float, default=0.08)
    ap.add_argument("--harminv-max-error", type=float, default=1e-3)
    ap.add_argument("--reflectance-check-wl-min", type=float, default=0.6)
    ap.add_argument("--reflectance-check-wl-max", type=float, default=2.0)
    ap.add_argument("--reflectance-check-nfreq", type=int, default=801)
    ap.add_argument("--reflectance-check-decay-threshold", type=float, default=1e-8)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    spec = _load_json(args.geometry_file)
    modes = _load_json(args.modes_file)

    mats_lossy = _build_materials(nH=args.nH, kH=args.kH, nL=args.nL, kref_um=args.kappa_ref_lambda)
    mats_lossless = _build_materials(nH=args.nH, kH=0.0, nL=args.nL, kref_um=args.kappa_ref_lambda)
    targets_um = _mode_lams(modes)
    mode_freqs = _mode_freqs(modes)

    # Secondary estimate: linewidth from reflectance dips.
    resonances_loaded_linewidth = _resonance_fit(
        spec=spec,
        mats=mats_lossy,
        targets_um=targets_um,
        linewidth_level=args.linewidth_level,
        resolution=args.resolution,
        nfreq=args.nfreq,
        decay_threshold=args.decay_threshold,
        window_um=args.window_um,
    )
    resonances_lossless_linewidth = _resonance_fit(
        spec=spec,
        mats=mats_lossless,
        targets_um=targets_um,
        linewidth_level=args.linewidth_level,
        resolution=args.resolution,
        nfreq=args.nfreq,
        decay_threshold=args.decay_threshold,
        window_um=args.window_um,
    )

    # Primary robust estimate: Harminv modal extraction.
    resonances_loaded_harminv = _harminv_global_resonances(
        spec=spec,
        mats=mats_lossy,
        target_freqs=mode_freqs,
        resolution=int(args.resolution),
        run_time=float(args.harminv_runtime),
        band_pad=float(args.harminv_band_pad),
        max_rel_detune=float(args.harminv_max_rel_detune),
        error_max=float(args.harminv_max_error),
    )
    resonances_lossless_harminv = _harminv_global_resonances(
        spec=spec,
        mats=mats_lossless,
        target_freqs=mode_freqs,
        resolution=int(args.resolution),
        run_time=float(args.harminv_runtime),
        band_pad=float(args.harminv_band_pad),
        max_rel_detune=float(args.harminv_max_rel_detune),
        error_max=float(args.harminv_max_error),
    )

    # Port split is estimated from lossless runs so it represents radiative leakage.
    port_splits: Dict[str, Dict[str, float]] = {}
    for key in mode_freqs:
        f_use = float(mode_freqs[key])
        fit_ll = resonances_lossless_harminv.get(key, {})
        if fit_ll and np.isfinite(float(fit_ll.get("lam", float("nan")))):
            f_use = 1.0 / float(fit_ll["lam"])
        port_splits[key] = _estimate_port_split(
            spec=spec,
            mats=mats_lossless,
            freq=float(f_use),
            resolution=args.resolution,
            decay_threshold=args.decay_threshold,
        )

    rates: Dict[str, Dict[str, float]] = {}
    for key, f_tgt in mode_freqs.items():
        rates[key] = _mode_rate_block(
            key=key,
            freq_target=float(f_tgt),
            fit_loaded=resonances_loaded_harminv.get(key, {}),
            fit_lossless=resonances_lossless_harminv.get(key, {}),
            port_split=port_splits.get(key, {}),
        )

    rates_linewidth: Dict[str, Dict[str, float]] = {}
    for key, f_tgt in mode_freqs.items():
        rates_linewidth[key] = _mode_rate_block(
            key=key,
            freq_target=float(f_tgt),
            fit_loaded=resonances_loaded_linewidth.get(key, {}),
            fit_lossless=resonances_lossless_linewidth.get(key, {}),
            port_split=port_splits.get(key, {}),
        )

    kappa_s = float(rates["probe"]["kappa_loaded"])
    rates_norm = {}
    for key, entry in rates.items():
        out = {}
        for rk in ("kappa_loaded", "kappa_ext", "kappa_int", "kappa_ext_left", "kappa_ext_right", "detune_loaded_omega"):
            val = float(entry.get(rk, float("nan")))
            out[rk + "_norm_to_probe_kappa"] = float(val / kappa_s) if np.isfinite(val) and kappa_s > 0 else float("nan")
        rates_norm[key] = out

    freq_map = {
        "pump1": float(mode_freqs["pump1"]),
        "pump2": float(mode_freqs["pump2"]),
        "probe": float(mode_freqs["probe"]),
        "sb_minus": float(mode_freqs["sb_minus"]),
        "sb_plus": float(mode_freqs["sb_plus"]),
    }
    z_mode, profiles, cavity_bounds = debug_mode_profiles(
        spec=spec,
        mats=mats_lossy,
        freqs=freq_map,
        resolution=int(args.resolution),
    )
    overlaps = cavity_overlaps(z_mode, profiles, cavity_bounds)
    eta = _eta_proxies(z=z_mode, profiles=profiles, cavity_bounds=cavity_bounds)

    reflectance_validation = _reflectance_rt_validation(
        spec=spec,
        mats=mats_lossy,
        resolution=int(args.resolution),
        nfreq=int(args.reflectance_check_nfreq),
        wl_min=float(args.reflectance_check_wl_min),
        wl_max=float(args.reflectance_check_wl_max),
        decay_threshold=float(args.reflectance_check_decay_threshold),
    )

    n_probe = float(material_index_at_wavelength(mats_lossy["SiN"], 1.0 / mode_freqs["probe"]))
    chi3_si = float(n2_to_chi3_si(float(args.n2), n_probe))
    abc = float(chi3_si / 3.0)

    out = {
        "input_files": {
            "geometry_file": str(args.geometry_file.resolve()),
            "modes_file": str(args.modes_file.resolve()),
        },
        "material_constants": {
            "nH": float(args.nH),
            "kH": float(args.kH),
            "nL": float(args.nL),
            "kappa_ref_lambda_um": float(args.kappa_ref_lambda),
            "n2_m2_per_w": float(args.n2),
            "n_linear_probe": float(n_probe),
            "chi3_si": float(chi3_si),
            "A_equals_B_equals_C": float(abc),
        },
        "targets": {
            "frequencies_inv_um": mode_freqs,
            "wavelengths_um": targets_um,
        },
        "resonance_fit_loaded_lossy_linewidth": resonances_loaded_linewidth,
        "resonance_fit_lossless_ext_linewidth": resonances_lossless_linewidth,
        "resonance_fit_loaded_lossy_harminv": resonances_loaded_harminv,
        "resonance_fit_lossless_ext_harminv": resonances_lossless_harminv,
        "rates_harminv_primary": rates,
        "rates_linewidth_secondary": rates_linewidth,
        "rates_normalized_to_probe_kappa": rates_norm,
        "overlap_matrix_debug_mode_profiles": overlaps,
        "eta_proxies_from_mode_profiles": eta,
        "reflectance_validation": reflectance_validation,
        "notes": [
            "Primary Q/rates are from Harminv; linewidth-derived rates are retained as secondary diagnostics.",
            "kappa decomposition uses kappa_total(lossy) and kappa_ext(lossless): kappa_int = kappa_total - kappa_ext.",
            "port split uses a centered pulse and left/right emitted flux magnitudes in lossless geometry.",
            "reflectance_validation compares independent R/T energy balance and R-consistency with debug_reflectance.",
            "reflectance_validation uses --reflectance-check-decay-threshold (default 1e-8) for tighter convergence.",
            "eta values are scalar overlap proxies from |E| DFT profiles inside the cavity region.",
            "A=B=C is set from chi3_si/3 per requested isotropic simplification.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
