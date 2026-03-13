#!/usr/bin/env python3
"""
Extract derivation-consistent TCMT coefficients from Meep diagnostics.

This script keeps the legacy proxy extraction untouched and adds a separate
path that:
1) refines modal frequencies / rates for a given geometry + mode target file,
2) extracts complex field profiles for the loaded modes,
3) computes overlap coefficients on the nonlinear material support, and
4) writes a Julia include file that can drive both the legacy and the new
   derivation-consistent FaradayJL implementations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from extract_tcmt_params import (
    _estimate_port_split,
    _harminv_global_resonances,
    _load_json,
    _mode_freqs,
    _mode_lams,
    _mode_rate_block,
    _reflectance_rt_validation,
    _resonance_fit,
)
from mode_targeting import get_cavity_materials, material_index_at_wavelength
from nonlinear_materials import (
    canonical_high_index_material,
    chi3_si_to_meep_e_chi3,
    n2_to_chi3_si,
    resolve_high_index_index,
    resolve_high_index_kappa,
    resolve_high_index_n2,
)
from optimize_cavity_geometry import build_1d_geometry_from_spec, debug_reflectance


EPS0 = 8.854187817e-12
C0 = 299792458.0
SCALE_E = 1.0 / (1e-6 * EPS0 * C0)


def _build_materials(args: argparse.Namespace) -> Dict[str, mp.Medium]:
    mat_high, mat_low = get_cavity_materials(
        model=str(args.materials),
        index_high=float(args.nH),
        kappa_high=float(args.kH),
        index_low=float(args.nL),
        high_index_material=str(args.high_index_material),
        kappa_ref_wavelength_um=float(args.kappa_ref_lambda),
        sin_csv=str(args.sin_fit),
        sio2_csv=str(args.sio2_fit),
        lam_min=int(args.fit_window[0]),
        lam_max=int(args.fit_window[1]),
        fit_poles=int(args.fit_poles),
    )
    return {"SiN": mat_high, "SiO2": mat_low}


def _jl_float(value: float) -> str:
    x = float(value)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Inf" if x > 0 else "-Inf"
    return repr(x)


def _to_complex(value: Any) -> complex:
    if isinstance(value, dict) and "re" in value and "im" in value:
        return complex(float(value["re"]), float(value["im"]))
    return complex(value)


def _jl_complex(value: Any) -> str:
    z = _to_complex(value)
    return f"ComplexF64({_jl_float(z.real)}, {_jl_float(z.imag)})"


def _jl_string(value: str) -> str:
    return json.dumps(str(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _choose_analysis_freqs(
    *,
    targets: Dict[str, float],
    loaded_harminv: Dict[str, Dict[str, float]],
    loaded_linewidth: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, f_target in targets.items():
        fit_h = loaded_harminv.get(key, {})
        fit_lw = loaded_linewidth.get(key, {})
        chosen, source = _select_preferred_fit(
            primary=fit_h,
            fallback=fit_lw,
            target_freq=float(f_target),
            primary_label="loaded_harminv",
            fallback_label="loaded_linewidth",
            detune_improvement_factor=2.0,
        )
        f_sel = _fit_freq_inv_um(chosen)
        if np.isfinite(f_sel) and f_sel > 0.0:
            out[key] = {
                "freq_inv_um": float(f_sel),
                "source": source,
                "target_freq_inv_um": float(f_target),
            }
            continue

        out[key] = {
            "freq_inv_um": float(f_target),
            "source": "target_frequency",
            "target_freq_inv_um": float(f_target),
        }
    return out


def _fit_freq_inv_um(fit: Dict[str, Any]) -> float:
    freq = float(fit.get("freq", float("nan")))
    if np.isfinite(freq) and freq > 0.0:
        return float(freq)
    lam = float(fit.get("lam", float("nan")))
    if np.isfinite(lam) and lam > 0.0:
        return float(1.0 / lam)
    return float("nan")


def _select_preferred_fit(
    *,
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    target_freq: float,
    primary_label: str,
    fallback_label: str,
    detune_improvement_factor: float,
) -> Tuple[Dict[str, Any], str]:
    f_primary = _fit_freq_inv_um(primary)
    f_fallback = _fit_freq_inv_um(fallback)
    primary_ok = np.isfinite(f_primary) and f_primary > 0.0
    fallback_ok = np.isfinite(f_fallback) and f_fallback > 0.0

    if primary_ok and fallback_ok:
        d_primary = abs(f_primary - target_freq)
        d_fallback = abs(f_fallback - target_freq)
        if d_fallback * detune_improvement_factor < d_primary:
            return dict(fallback), fallback_label
        return dict(primary), primary_label
    if primary_ok:
        return dict(primary), primary_label
    if fallback_ok:
        return dict(fallback), fallback_label
    return {}, "none"


def _fit_with_linewidth_fallback(
    primary: Dict[str, Any],
    fallback: Dict[str, Any],
    *,
    tag: str,
    target_freq: float,
) -> Dict[str, Any]:
    fit, source = _select_preferred_fit(
        primary=primary,
        fallback=fallback,
        target_freq=float(target_freq),
        primary_label=tag + "_harminv",
        fallback_label=tag + "_linewidth",
        detune_improvement_factor=2.0,
    )
    fit = dict(fit)
    fit["source_for_rates"] = source
    if "freq" not in fit:
        lam_fb = float(fit.get("lam", float("nan")))
        fit["freq"] = float(1.0 / lam_fb) if np.isfinite(lam_fb) and lam_fb > 0.0 else float("nan")
    return fit


def _material_eps_real(material: mp.Medium, freq: float) -> float:
    if freq <= 0.0:
        return 1.0
    if hasattr(material, "epsilon"):
        try:
            eps_val = material.epsilon(freq)[0][0]
            eps_real = float(np.real(eps_val))
            if np.isfinite(eps_real) and eps_real > 0.0:
                return eps_real
        except Exception:
            pass
    lam = 1.0 / max(freq, 1e-12)
    n = float(material_index_at_wavelength(material, lam))
    return float(max(n * n, 1e-9))


def _grid_material_metadata(
    geometry: List[mp.Block],
    *,
    z: np.ndarray,
    mats: Dict[str, mp.Medium],
    freq: float,
) -> Tuple[np.ndarray, np.ndarray]:
    eps = np.ones_like(z, dtype=float)
    nonlinear_mask = np.zeros_like(z, dtype=bool)
    sin_mat = mats["SiN"]
    for block in geometry:
        half_z = 0.5 * float(block.size.z)
        z0 = float(block.center.z - half_z)
        z1 = float(block.center.z + half_z)
        mask = (z >= z0) & (z <= z1)
        if not np.any(mask):
            continue
        eps_val = _material_eps_real(block.material, freq)
        eps[mask] = eps_val
        if block.material is sin_mat:
            nonlinear_mask[mask] = True
    return eps, nonlinear_mask


def _phase_fix(reference: np.ndarray) -> complex:
    n = int(reference.size)
    if n == 0:
        return 1.0 + 0.0j
    window = max(8, n // 32)
    left_mean = np.sum(reference[:window])
    if abs(left_mean) > 1e-20:
        return np.exp(-1j * np.angle(left_mean))
    peak = int(np.argmax(np.abs(reference)))
    return np.exp(-1j * np.angle(reference[peak]))


def _extract_mode_profile(
    *,
    spec: Dict[str, Any],
    mats: Dict[str, mp.Medium],
    key: str,
    freq: float,
    resolution: int,
    decay_threshold: float,
    source_band_fraction: float,
) -> Dict[str, Any]:
    geom, cell_z, cavity_bounds = build_1d_geometry_from_spec(spec, mats)
    dpml = float(spec["pads"]["pml_um"])
    monitor_len = cell_z - 2.0 * dpml - 0.02
    src_z = -0.5 * cell_z + dpml + 0.2
    f0 = float(freq)
    fwidth = max(float(source_band_fraction) * f0, 0.003)

    sim = mp.Simulation(
        cell_size=mp.Vector3(0, 0, cell_z),
        geometry=geom,
        sources=[
            mp.Source(
                mp.GaussianSource(frequency=f0, fwidth=fwidth),
                component=mp.Ex,
                center=mp.Vector3(0, 0, src_z),
                amplitude=1.0,
            )
        ],
        boundary_layers=[mp.PML(dpml)],
        default_material=mp.air,
        resolution=int(resolution),
        dimensions=1,
        force_complex_fields=True,
    )

    vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 0, monitor_len))
    dft = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Hx, mp.Hy], [f0], where=vol)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            100, mp.Ex, mp.Vector3(0, 0, 0.0), float(decay_threshold)
        )
    )

    ex = np.asarray(sim.get_dft_array(dft, mp.Ex, 0), dtype=complex)
    ey = np.asarray(sim.get_dft_array(dft, mp.Ey, 0), dtype=complex)
    hx = np.asarray(sim.get_dft_array(dft, mp.Hx, 0), dtype=complex)
    hy = np.asarray(sim.get_dft_array(dft, mp.Hy, 0), dtype=complex)
    z = np.linspace(-0.5 * monitor_len, 0.5 * monitor_len, ex.size, dtype=float)

    phase = _phase_fix(ex if np.max(np.abs(ex)) >= np.max(np.abs(ey)) else ey)
    ex *= phase
    ey *= phase
    hx *= phase
    hy *= phase

    eps_z, nonlinear_mask = _grid_material_metadata(geom, z=z, mats=mats, freq=f0)
    e_sq = np.abs(ex) ** 2 + np.abs(ey) ** 2
    h_sq = np.abs(hx) ** 2 + np.abs(hy) ** 2
    energy_density = 0.25 * (eps_z * e_sq + h_sq)
    energy = float(np.trapezoid(energy_density, z))
    if not np.isfinite(energy) or energy <= 1e-30:
        raise RuntimeError(f"Failed to normalize mode '{key}': energy={energy!r}")
    scale = math.sqrt(energy)

    scalar_mode = ex / scale
    return {
        "key": key,
        "freq_inv_um": float(f0),
        "lambda_um": float(1.0 / f0),
        "z_um": z,
        "Ex": ex / scale,
        "Ey": ey / scale,
        "Hx": hx / scale,
        "Hy": hy / scale,
        "scalar_mode": scalar_mode,
        "eps_real": eps_z,
        "energy_density": energy_density / energy,
        "nonlinear_mask": nonlinear_mask,
        "cavity_bounds_um": (float(cavity_bounds[0]), float(cavity_bounds[1])),
        "energy_raw": energy,
    }


def _complex_trapz(z: np.ndarray, values: np.ndarray, mask: np.ndarray) -> complex:
    if not np.any(mask):
        return 0.0 + 0.0j
    return complex(np.trapezoid(values[mask], z[mask]))


def _direct_overlap(z: np.ndarray, us: np.ndarray, up: np.ndarray, mask: np.ndarray) -> complex:
    return _complex_trapz(z, (np.abs(us) ** 2) * (np.abs(up) ** 2), mask)


def _gen_overlap(
    z: np.ndarray,
    ub: np.ndarray,
    uin: np.ndarray,
    uout: np.ndarray,
    us: np.ndarray,
    mask: np.ndarray,
) -> complex:
    return _complex_trapz(z, np.conjugate(ub) * uin * np.conjugate(uout) * us, mask)


def _mix_overlap(
    z: np.ndarray,
    us: np.ndarray,
    uout: np.ndarray,
    uin: np.ndarray,
    ub: np.ndarray,
    mask: np.ndarray,
) -> complex:
    return _complex_trapz(z, np.conjugate(us) * uout * np.conjugate(uin) * ub, mask)


def _legacy_proxy_norms(
    profiles: Dict[str, Dict[str, Any]],
    cavity_bounds: Tuple[float, float],
) -> Dict[str, float]:
    z = np.asarray(profiles["probe"]["z_um"], dtype=float)
    z0, z1 = cavity_bounds
    mask = (z >= z0) & (z <= z1)
    zc = z[mask]
    if zc.size < 4:
        return {}

    def _norm(key: str) -> np.ndarray:
        vv = np.asarray(np.abs(profiles[key]["scalar_mode"][mask]), dtype=float)
        nrm = float(np.sqrt(np.trapezoid(vv * vv, zc)))
        return vv / max(nrm, 1e-30)

    u_probe = _norm("probe")
    u_p1 = _norm("pump1")
    u_p2 = _norm("pump2")
    u_sb_p = _norm("sb_plus")
    u_sb_m = _norm("sb_minus")
    return {
        "eta_s_u1_us_proxy": float(np.trapezoid((u_probe * u_probe) * (u_p1 * u_p1), zc)),
        "eta_s_u2_us_proxy": float(np.trapezoid((u_probe * u_probe) * (u_p2 * u_p2), zc)),
        "eta_Omega_p_proxy": float(np.trapezoid((u_probe * u_probe) * u_p1 * u_p2, zc)),
        "eta_Omega_m_proxy": float(np.trapezoid((u_probe * u_probe) * u_p2 * u_p1, zc)),
        "eta_sb_probe_p_proxy": float(np.trapezoid((u_probe * u_probe) * (u_sb_p * u_sb_p), zc)),
        "eta_sb_probe_m_proxy": float(np.trapezoid((u_probe * u_probe) * (u_sb_m * u_sb_m), zc)),
    }


def _derived_coefficients(
    *,
    profiles: Dict[str, Dict[str, Any]],
    chi_iso_meep: float,
    kappa_s: float,
) -> Dict[str, Any]:
    z = np.asarray(profiles["probe"]["z_um"], dtype=float)
    nonlinear_mask = np.asarray(profiles["probe"]["nonlinear_mask"], dtype=bool)
    u_probe = np.asarray(profiles["probe"]["scalar_mode"], dtype=complex)
    u_p1 = np.asarray(profiles["pump1"]["scalar_mode"], dtype=complex)
    u_p2 = np.asarray(profiles["pump2"]["scalar_mode"], dtype=complex)
    u_sb_p = np.asarray(profiles["sb_plus"]["scalar_mode"], dtype=complex)
    u_sb_m = np.asarray(profiles["sb_minus"]["scalar_mode"], dtype=complex)

    ωs = float(profiles["probe"]["freq_inv_um"])
    ωΩp = float(profiles["sb_plus"]["freq_inv_um"])
    ωΩm = float(profiles["sb_minus"]["freq_inv_um"])
    χiso = complex(float(chi_iso_meep), 0.0)

    A1 = B1 = C1 = χiso
    A2 = B2 = C2 = χiso
    A_sb_p = B_sb_p = C_sb_p = χiso
    A_sb_m = B_sb_m = C_sb_m = χiso
    A_mx_p = B_mx_p = C_mx_p = χiso
    A_mx_m = B_mx_m = C_mx_m = χiso

    pref_s = complex(3.0 * ωs / 8.0, 0.0)
    pref_Ωp = complex(3.0 * ωΩp / 8.0, 0.0)
    pref_Ωm = complex(3.0 * ωΩm / 8.0, 0.0)

    counter = {
        "alpha1_plus_raw": pref_s * (A1 + B1) * _direct_overlap(z, u_probe, u_p1, nonlinear_mask),
        "alpha2_plus_raw": pref_s * (A2 + C2) * _direct_overlap(z, u_probe, u_p2, nonlinear_mask),
        "alpha1_minus_raw": pref_s * (A1 + C1) * _direct_overlap(z, u_probe, u_p1, nonlinear_mask),
        "alpha2_minus_raw": pref_s * (A2 + B2) * _direct_overlap(z, u_probe, u_p2, nonlinear_mask),
        "zeta_plus_raw": pref_Ωp
        * (B_sb_p + C_sb_p)
        * _gen_overlap(z, u_sb_p, u_p1, u_p2, u_probe, nonlinear_mask),
        "zeta_minus_raw": pref_Ωm
        * (B_sb_m + C_sb_m)
        * _gen_overlap(z, u_sb_m, u_p2, u_p1, u_probe, nonlinear_mask),
        "eta_plus_raw": pref_s
        * (B_mx_p + C_mx_p)
        * _mix_overlap(z, u_probe, u_p2, u_p1, u_sb_p, nonlinear_mask),
        "eta_minus_raw": pref_s
        * (B_mx_m + C_mx_m)
        * _mix_overlap(z, u_probe, u_p1, u_p2, u_sb_m, nonlinear_mask),
    }

    coro = {
        "alpha1_plus_raw": pref_s * (A1 + B1) * _direct_overlap(z, u_probe, u_p1, nonlinear_mask),
        "alpha2_plus_raw": pref_s * (A2 + B2) * _direct_overlap(z, u_probe, u_p2, nonlinear_mask),
        "alpha1_minus_raw": pref_s * (A1 + C1) * _direct_overlap(z, u_probe, u_p1, nonlinear_mask),
        "alpha2_minus_raw": pref_s * (A2 + C2) * _direct_overlap(z, u_probe, u_p2, nonlinear_mask),
        "zeta_pp_raw": pref_Ωp
        * (A_sb_p + B_sb_p)
        * _gen_overlap(z, u_sb_p, u_p1, u_p2, u_probe, nonlinear_mask),
        "zeta_pm_raw": pref_Ωp
        * (A_sb_p + C_sb_p)
        * _gen_overlap(z, u_sb_p, u_p1, u_p2, u_probe, nonlinear_mask),
        "zeta_mp_raw": pref_Ωm
        * (A_sb_m + B_sb_m)
        * _gen_overlap(z, u_sb_m, u_p2, u_p1, u_probe, nonlinear_mask),
        "zeta_mm_raw": pref_Ωm
        * (A_sb_m + C_sb_m)
        * _gen_overlap(z, u_sb_m, u_p2, u_p1, u_probe, nonlinear_mask),
        "eta_pp_raw": pref_s
        * (A_mx_p + B_mx_p)
        * _mix_overlap(z, u_probe, u_p2, u_p1, u_sb_p, nonlinear_mask),
        "eta_pm_raw": pref_s
        * (A_mx_p + C_mx_p)
        * _mix_overlap(z, u_probe, u_p2, u_p1, u_sb_p, nonlinear_mask),
        "eta_mp_raw": pref_s
        * (A_mx_m + B_mx_m)
        * _mix_overlap(z, u_probe, u_p1, u_p2, u_sb_m, nonlinear_mask),
        "eta_mm_raw": pref_s
        * (A_mx_m + C_mx_m)
        * _mix_overlap(z, u_probe, u_p1, u_p2, u_sb_m, nonlinear_mask),
        "Lambda_Omegap_raw": 0.0 + 0.0j,
        "Lambda_Omegam_raw": 0.0 + 0.0j,
    }

    if np.isfinite(kappa_s) and kappa_s > 0.0:
        for section in (counter, coro):
            keys = list(section.keys())
            for key in keys:
                section[key.replace("_raw", "_norm")] = complex(section[key] / kappa_s)

    return {
        "counter": counter,
        "coro": coro,
        "normalization": {
            "kappa_probe_loaded_meep": float(kappa_s),
            "chi_iso_meep": float(chi_iso_meep),
        },
    }


def _plot_reflectance(
    *,
    spec: Dict[str, Any],
    mats: Dict[str, mp.Medium],
    targets: Dict[str, float],
    analysis_freqs: Dict[str, Dict[str, Any]],
    out_path: Path,
    resolution: int,
    nfreq: int,
    decay_threshold: float,
    title: str,
) -> None:
    wl, rr = debug_reflectance(
        spec=spec,
        mats=mats,
        resolution=int(resolution),
        nfreq=int(nfreq),
        decay_threshold=float(decay_threshold),
        wl_min=0.6,
        wl_max=2.0,
    )
    order = np.argsort(wl)
    wl = wl[order]
    rr = rr[order]

    colors = {
        "pump1": "#d62728",
        "pump2": "#ff7f0e",
        "probe": "#1f77b4",
        "sb_plus": "#2ca02c",
        "sb_minus": "#9467bd",
    }
    labels = {
        "pump1": "pump1",
        "pump2": "pump2",
        "probe": "probe",
        "sb_plus": "sb+",
        "sb_minus": "sb-",
    }

    fig = plt.figure(figsize=(9.4, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(1e3 * wl, rr, lw=1.5, color="k", label="Reflectance")

    for key, f_tgt in targets.items():
        lam_tgt = 1e3 / float(f_tgt)
        lam_use = 1e3 / float(analysis_freqs[key]["freq_inv_um"])
        color = colors[key]
        y_tgt = float(np.interp(1e-3 * lam_tgt, wl, rr))
        y_use = float(np.interp(1e-3 * lam_use, wl, rr))
        ax.axvline(lam_tgt, color=color, ls="--", lw=0.9, alpha=0.45)
        ax.scatter([lam_tgt], [y_tgt], s=20, color=color, alpha=0.6)
        ax.scatter([lam_use], [y_use], s=24, color=color, marker="x")
        ax.text(
            lam_use + 2.0,
            min(1.05, y_use + 0.02),
            f"{labels[key]} ({analysis_freqs[key]['source']})",
            fontsize=8,
            color=color,
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_mode_profiles(
    *,
    profiles: Dict[str, Dict[str, Any]],
    out_amp: Path,
    out_phase: Path,
) -> None:
    order = ["pump1", "pump2", "probe", "sb_plus", "sb_minus"]
    fig_amp = plt.figure(figsize=(10.0, 5.4))
    ax_amp = fig_amp.add_subplot(1, 1, 1)
    fig_phase = plt.figure(figsize=(10.0, 5.4))
    ax_phase = fig_phase.add_subplot(1, 1, 1)

    for key in order:
        data = profiles[key]
        z = np.asarray(data["z_um"], dtype=float)
        u = np.asarray(data["scalar_mode"], dtype=complex)
        ax_amp.plot(z, np.abs(u), lw=1.3, label=key)
        phase = np.unwrap(np.angle(u))
        phase -= phase[np.argmax(np.abs(u))]
        ax_phase.plot(z, phase, lw=1.1, label=key)

    cav0, cav1 = profiles["probe"]["cavity_bounds_um"]
    for ax in (ax_amp, ax_phase):
        ax.axvspan(float(cav0), float(cav1), color="gray", alpha=0.12)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=3, fontsize=8)
        ax.set_xlabel("z (um)")

    ax_amp.set_ylabel("|u(z)|")
    ax_amp.set_title("Normalized mode amplitudes")
    ax_phase.set_ylabel("phase(u) [rad]")
    ax_phase.set_title("Normalized mode phases")

    fig_amp.tight_layout()
    fig_phase.tight_layout()
    out_amp.parent.mkdir(parents=True, exist_ok=True)
    fig_amp.savefig(out_amp, dpi=180, bbox_inches="tight")
    fig_phase.savefig(out_phase, dpi=180, bbox_inches="tight")
    plt.close(fig_amp)
    plt.close(fig_phase)


def _write_julia_case_file(
    *,
    out_path: Path,
    case_name: str,
    data: Dict[str, Any],
) -> None:
    rates_norm = data["rates_normalized_to_probe_kappa"]
    derived = data["derived_coefficients"]
    legacy = data["legacy_proxy_norms"]
    material = data["material_constants"]
    pulse = data["pulse_settings"]
    plots = data["plots"]

    def rate_expr(entry: Dict[str, float]) -> str:
        return (
            "("
            f"kappa_loaded = {_jl_float(entry.get('kappa_loaded_norm_to_probe_kappa', float('nan')))}, "
            f"detune = {_jl_float(entry.get('detune_loaded_omega_norm_to_probe_kappa', float('nan')))}, "
            f"kappa_ext = {_jl_float(entry.get('kappa_ext_norm_to_probe_kappa', float('nan')))}, "
            f"kappa_ext_left = {_jl_float(entry.get('kappa_ext_left_norm_to_probe_kappa', float('nan')))}, "
            f"kappa_ext_right = {_jl_float(entry.get('kappa_ext_right_norm_to_probe_kappa', float('nan')))}"
            ")"
        )

    content = f"""# This file is autogenerated by extract_tcmt_params_derivation.py.
const TCMT_CASE = (
    name = {_jl_string(case_name)},
    source = (
        geometry_file = {_jl_string(str(data['input_files']['geometry_file']))},
        modes_file = {_jl_string(str(data['input_files']['modes_file']))},
        extracted_json = {_jl_string(str(data['output_json']))},
    ),
    material = (
        materials_model = {_jl_string(str(material['materials_model']))},
        high_index_material = {_jl_string(str(material['high_index_material']))},
        n2_m2_per_w = {_jl_float(material['n2_m2_per_w'])},
        n_linear_probe = {_jl_float(material['n_linear_probe'])},
        chi3_si = {_jl_float(material['chi3_si'])},
        chi3_meep = {_jl_float(material['chi3_meep'])},
        chi_iso_meep = {_jl_float(material['chi_iso_meep'])},
    ),
    pulse = (
        pump_intensity_w_cm2 = {_jl_float(pulse['pump_intensity_w_cm2'])},
        probe_intensity_w_cm2 = {_jl_float(pulse['probe_intensity_w_cm2'])},
        pulse_fwhm_intensity_fs = {_jl_float(pulse['pulse_fwhm_intensity_fs'])},
    ),
    kappa_probe_meep = {_jl_float(data['rates_harminv_primary']['probe']['kappa_loaded'])},
    legacy = (
        norms = (
            eta_s_u1_us = {_jl_complex(complex(legacy.get('eta_s_u1_us_proxy', float('nan')), 0.0))},
            eta_s_u2_us = {_jl_complex(complex(legacy.get('eta_s_u2_us_proxy', float('nan')), 0.0))},
            eta_Omega_p = {_jl_complex(complex(legacy.get('eta_Omega_p_proxy', float('nan')), 0.0))},
            eta_Omega_m = {_jl_complex(complex(legacy.get('eta_Omega_m_proxy', float('nan')), 0.0))},
        ),
    ),
    rates = (
        pump1 = {rate_expr(rates_norm['pump1'])},
        pump2 = {rate_expr(rates_norm['pump2'])},
        probe = {rate_expr(rates_norm['probe'])},
        sb_plus = {rate_expr(rates_norm['sb_plus'])},
        sb_minus = {rate_expr(rates_norm['sb_minus'])},
    ),
    derived = (
        counter = (
            alpha1_plus = {_jl_complex(derived['counter']['alpha1_plus_norm'])},
            alpha2_plus = {_jl_complex(derived['counter']['alpha2_plus_norm'])},
            alpha1_minus = {_jl_complex(derived['counter']['alpha1_minus_norm'])},
            alpha2_minus = {_jl_complex(derived['counter']['alpha2_minus_norm'])},
            zeta_plus = {_jl_complex(derived['counter']['zeta_plus_norm'])},
            zeta_minus = {_jl_complex(derived['counter']['zeta_minus_norm'])},
            eta_plus = {_jl_complex(derived['counter']['eta_plus_norm'])},
            eta_minus = {_jl_complex(derived['counter']['eta_minus_norm'])},
        ),
        coro = (
            alpha1_plus = {_jl_complex(derived['coro']['alpha1_plus_norm'])},
            alpha2_plus = {_jl_complex(derived['coro']['alpha2_plus_norm'])},
            alpha1_minus = {_jl_complex(derived['coro']['alpha1_minus_norm'])},
            alpha2_minus = {_jl_complex(derived['coro']['alpha2_minus_norm'])},
            zeta_pp = {_jl_complex(derived['coro']['zeta_pp_norm'])},
            zeta_pm = {_jl_complex(derived['coro']['zeta_pm_norm'])},
            zeta_mp = {_jl_complex(derived['coro']['zeta_mp_norm'])},
            zeta_mm = {_jl_complex(derived['coro']['zeta_mm_norm'])},
            eta_pp = {_jl_complex(derived['coro']['eta_pp_norm'])},
            eta_pm = {_jl_complex(derived['coro']['eta_pm_norm'])},
            eta_mp = {_jl_complex(derived['coro']['eta_mp_norm'])},
            eta_mm = {_jl_complex(derived['coro']['eta_mm_norm'])},
            Lambda_Omegap = {_jl_complex(derived['coro']['Lambda_Omegap_norm'])},
            Lambda_Omegam = {_jl_complex(derived['coro']['Lambda_Omegam_norm'])},
        ),
        output = (
            kappa_out_plus = 1.0,
            kappa_out_minus = 1.0,
            c_plus = {_jl_complex(0.0 + 0.0j)},
            c_minus = {_jl_complex(0.0 + 0.0j)},
        ),
    ),
    plots = (
        reflectance = {_jl_string(str(plots['reflectance']))},
        mode_amplitude = {_jl_string(str(plots['mode_amplitude']))},
        mode_phase = {_jl_string(str(plots['mode_phase']))},
    ),
)
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path("meep_project/SiN_optimizations")
    ap = argparse.ArgumentParser(
        description="Extract derivation-consistent TCMT coefficients from Meep."
    )
    ap.add_argument("--geometry-file", type=Path, default=root / "optimized_geometry_sin_090326_new.json")
    ap.add_argument("--modes-file", type=Path, default=root / "cavity_modes_sin_090326_new.json")
    ap.add_argument(
        "--output-json",
        type=Path,
        default=root / "tcmt_derivation_analysis" / "sin_090326_new" / "tcmt_extracted_params_derivation.json",
    )
    ap.add_argument(
        "--julia-case-file",
        type=Path,
        default=Path("FaradayJL/examples/generated/tcmt_case_sin_090326_new.jl"),
    )
    ap.add_argument("--case-name", type=str, default="sin_090326_new")

    ap.add_argument("--materials", choices=("library", "constant", "fit"), default="fit")
    ap.add_argument("--high-index-material", type=str, default="sin")
    ap.add_argument("--sin-fit", type=str, default="meep_project/si3n4.csv")
    ap.add_argument("--sio2-fit", type=str, default="meep_project/sio2.csv")
    ap.add_argument("--fit-window", type=int, nargs=2, default=(600, 2000))
    ap.add_argument("--fit-poles", type=int, default=2)
    ap.add_argument("--nH", type=float, default=None)
    ap.add_argument("--kH", type=float, default=None)
    ap.add_argument("--nL", type=float, default=1.45)
    ap.add_argument("--kappa-ref-lambda", type=float, default=1.55)
    ap.add_argument("--high-index-n2", type=float, default=None)

    ap.add_argument("--resolution", type=int, default=80)
    ap.add_argument("--nfreq", type=int, default=1601)
    ap.add_argument("--window-um", type=float, default=0.20)
    ap.add_argument("--linewidth-level", type=float, default=0.5)
    ap.add_argument("--decay-threshold", type=float, default=1e-7)
    ap.add_argument("--harminv-runtime", type=float, default=900.0)
    ap.add_argument("--harminv-band-pad", type=float, default=0.25)
    ap.add_argument("--harminv-max-rel-detune", type=float, default=0.08)
    ap.add_argument("--harminv-max-error", type=float, default=1e-3)
    ap.add_argument("--field-source-band-fraction", type=float, default=0.03)
    ap.add_argument("--reflectance-check-nfreq", type=int, default=1001)
    ap.add_argument("--reflectance-check-decay-threshold", type=float, default=1e-8)

    ap.add_argument("--pump-intensity", type=float, default=1.0e12)
    ap.add_argument("--probe-intensity", type=float, default=1.0e8)
    ap.add_argument("--pulse-duration-fs", type=float, default=100.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.high_index_material = canonical_high_index_material(args.high_index_material)
    args.nH = resolve_high_index_index(args.nH, args.high_index_material)
    args.kH = resolve_high_index_kappa(args.kH, args.high_index_material)
    args.high_index_n2 = resolve_high_index_n2(args.high_index_n2, args.high_index_material)

    spec = _load_json(args.geometry_file)
    modes = _load_json(args.modes_file)
    mats_lossy = _build_materials(args)

    args_lossless = argparse.Namespace(**vars(args))
    args_lossless.kH = 0.0
    mats_lossless = _build_materials(args_lossless)

    targets_um = _mode_lams(modes)
    mode_freqs = _mode_freqs(modes)

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

    port_splits: Dict[str, Dict[str, float]] = {}
    for key in mode_freqs:
        fit_port, _ = _select_preferred_fit(
            primary=resonances_lossless_harminv.get(key, {}),
            fallback=resonances_lossless_linewidth.get(key, {}),
            target_freq=float(mode_freqs[key]),
            primary_label="lossless_harminv",
            fallback_label="lossless_linewidth",
            detune_improvement_factor=2.0,
        )
        f_use = _fit_freq_inv_um(fit_port)
        if not (np.isfinite(f_use) and f_use > 0.0):
            f_use = float(mode_freqs[key])
        port_splits[key] = _estimate_port_split(
            spec=spec,
            mats=mats_lossless,
            freq=float(f_use),
            resolution=args.resolution,
            decay_threshold=args.decay_threshold,
        )

    rates: Dict[str, Dict[str, float]] = {}
    for key, f_tgt in mode_freqs.items():
        fit_loaded = _fit_with_linewidth_fallback(
            resonances_loaded_harminv.get(key, {}),
            resonances_loaded_linewidth.get(key, {}),
            tag="loaded",
            target_freq=float(f_tgt),
        )
        fit_lossless = _fit_with_linewidth_fallback(
            resonances_lossless_harminv.get(key, {}),
            resonances_lossless_linewidth.get(key, {}),
            tag="lossless",
            target_freq=float(f_tgt),
        )
        rates[key] = _mode_rate_block(
            key=key,
            freq_target=float(f_tgt),
            fit_loaded=fit_loaded,
            fit_lossless=fit_lossless,
            port_split=port_splits.get(key, {}),
        )

    rates_norm: Dict[str, Dict[str, float]] = {}
    kappa_s = float(rates["probe"]["kappa_loaded"])
    for key, entry in rates.items():
        block: Dict[str, float] = {}
        for name in (
            "kappa_loaded",
            "kappa_ext",
            "kappa_int",
            "kappa_ext_left",
            "kappa_ext_right",
            "detune_loaded_omega",
        ):
            val = float(entry.get(name, float("nan")))
            block[name + "_norm_to_probe_kappa"] = (
                float(val / kappa_s) if np.isfinite(val) and kappa_s > 0.0 else float("nan")
            )
        rates_norm[key] = block

    analysis_freqs = _choose_analysis_freqs(
        targets=mode_freqs,
        loaded_harminv=resonances_loaded_harminv,
        loaded_linewidth=resonances_loaded_linewidth,
    )

    profiles: Dict[str, Dict[str, Any]] = {}
    for key in ("pump1", "pump2", "probe", "sb_plus", "sb_minus"):
        profiles[key] = _extract_mode_profile(
            spec=spec,
            mats=mats_lossy,
            key=key,
            freq=float(analysis_freqs[key]["freq_inv_um"]),
            resolution=int(args.resolution),
            decay_threshold=float(args.decay_threshold),
            source_band_fraction=float(args.field_source_band_fraction),
        )

    cavity_bounds = tuple(profiles["probe"]["cavity_bounds_um"])
    legacy_proxy = _legacy_proxy_norms(profiles, cavity_bounds)

    n_probe = float(material_index_at_wavelength(mats_lossy["SiN"], 1.0 / mode_freqs["probe"]))
    chi3_si = float(n2_to_chi3_si(float(args.high_index_n2), n_probe))
    chi3_meep = float(chi3_si_to_meep_e_chi3(chi3_si, scale_e=SCALE_E, nonlinear_scale=1.0))
    chi_iso_meep = float(chi3_meep / 3.0)
    derived = _derived_coefficients(profiles=profiles, chi_iso_meep=chi_iso_meep, kappa_s=kappa_s)

    reflectance_validation = _reflectance_rt_validation(
        spec=spec,
        mats=mats_lossy,
        resolution=int(args.resolution),
        nfreq=int(args.reflectance_check_nfreq),
        wl_min=0.6,
        wl_max=2.0,
        decay_threshold=float(args.reflectance_check_decay_threshold),
    )

    analysis_dir = args.output_json.parent.resolve()
    reflectance_plot = analysis_dir / "reflectance_marked.png"
    amplitude_plot = analysis_dir / "mode_amplitudes.png"
    phase_plot = analysis_dir / "mode_phases.png"
    _plot_reflectance(
        spec=spec,
        mats=mats_lossy,
        targets=mode_freqs,
        analysis_freqs=analysis_freqs,
        out_path=reflectance_plot,
        resolution=int(args.resolution),
        nfreq=int(args.reflectance_check_nfreq),
        decay_threshold=float(args.reflectance_check_decay_threshold),
        title=f"Reflectance + analyzed modes ({args.case_name})",
    )
    _plot_mode_profiles(profiles=profiles, out_amp=amplitude_plot, out_phase=phase_plot)

    mode_profile_summary = {}
    for key, data in profiles.items():
        nonlinear_mask = np.asarray(data["nonlinear_mask"], dtype=bool)
        z = np.asarray(data["z_um"], dtype=float)
        u = np.asarray(data["scalar_mode"], dtype=complex)
        mode_profile_summary[key] = {
            "analysis_freq_inv_um": float(data["freq_inv_um"]),
            "analysis_lambda_um": float(data["lambda_um"]),
            "energy_raw": float(data["energy_raw"]),
            "nonlinear_region_weight": float(
                np.trapezoid(np.abs(u[nonlinear_mask]) ** 2, z[nonlinear_mask])
            )
            if np.any(nonlinear_mask)
            else float("nan"),
            "peak_abs_field": float(np.max(np.abs(u))),
        }

    out = {
        "input_files": {
            "geometry_file": str(args.geometry_file.resolve()),
            "modes_file": str(args.modes_file.resolve()),
        },
        "output_json": str(args.output_json.resolve()),
        "material_constants": {
            "materials_model": str(args.materials),
            "high_index_material": str(args.high_index_material),
            "nH": float(args.nH),
            "kH": float(args.kH),
            "nL": float(args.nL),
            "kappa_ref_lambda_um": float(args.kappa_ref_lambda),
            "n2_m2_per_w": float(args.high_index_n2),
            "n_linear_probe": float(n_probe),
            "chi3_si": float(chi3_si),
            "chi3_meep": float(chi3_meep),
            "chi_iso_meep": float(chi_iso_meep),
            "sin_fit": str(Path(args.sin_fit).resolve()),
            "sio2_fit": str(Path(args.sio2_fit).resolve()),
            "fit_window_nm": [int(args.fit_window[0]), int(args.fit_window[1])],
            "fit_poles": int(args.fit_poles),
        },
        "pulse_settings": {
            "pump_intensity_w_cm2": float(args.pump_intensity),
            "probe_intensity_w_cm2": float(args.probe_intensity),
            "pulse_fwhm_intensity_fs": float(args.pulse_duration_fs),
        },
        "targets": {
            "frequencies_inv_um": mode_freqs,
            "wavelengths_um": targets_um,
        },
        "analysis_frequencies": analysis_freqs,
        "resonance_fit_loaded_lossy_linewidth": resonances_loaded_linewidth,
        "resonance_fit_lossless_linewidth": resonances_lossless_linewidth,
        "resonance_fit_loaded_lossy_harminv": resonances_loaded_harminv,
        "resonance_fit_lossless_harminv": resonances_lossless_harminv,
        "rates_harminv_primary": rates,
        "rates_normalized_to_probe_kappa": rates_norm,
        "port_splits": port_splits,
        "legacy_proxy_norms": legacy_proxy,
        "derived_coefficients": derived,
        "mode_profile_summary": mode_profile_summary,
        "reflectance_validation": reflectance_validation,
        "plots": {
            "reflectance": str(reflectance_plot),
            "mode_amplitude": str(amplitude_plot),
            "mode_phase": str(phase_plot),
        },
        "notes": [
            "Loaded-mode analysis frequencies prefer Harminv frequencies, then linewidth-refined resonances, then the target frequencies.",
            "Mode profiles are extracted as complex DFT fields from separate narrowband simulations.",
            "Mode energies are normalized over the physical cell excluding PMLs with U = 1/4 integral(eps|E|^2 + |H|^2) dz.",
            "Nonlinear overlap integrals are restricted to SiN regions, where chi3 is non-zero.",
            "Legacy proxy norms are retained for side-by-side comparison with the original FaradayJL implementation.",
            "Derived coefficients are normalized to the loaded probe kappa, matching the Julia time normalization tau = kappa_s * t.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_json_safe(out), indent=2), encoding="utf-8")
    _write_julia_case_file(
        out_path=args.julia_case_file.resolve(),
        case_name=str(args.case_name),
        data=out,
    )
    print(f"Wrote {args.output_json.resolve()}")
    print(f"Wrote {args.julia_case_file.resolve()}")


if __name__ == "__main__":
    main()
