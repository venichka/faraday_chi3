#!/usr/bin/env python
"""Analytic chi5 objective: 4-mode overlaps + derived TCMT coefficients from the TMM
field profiles (FDTD-light). Replicates extract_tcmt_params_derivation._derived_coefficients
exactly, but with TMM E(z) instead of Meep DFT fields, so the rotation can be scored without
FDTD. Validated by comparing the "_raw" counter coefficients to the Meep extraction JSON.

Energy normalization (matches the extractor): U = int 1/4 (eps|E|^2 + |H|^2) dz = 1, so the
scalar mode is u = E / sqrt(U). Overlaps are over the high-index (chi3) region only.
Prefactor 3*omega/8; (A+B)=(A+C)=(B+C)=2*chi_iso (isotropic, single chi).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
sys.path.insert(0, str(HERE))   # so `import tmm` works however this module is loaded
import tmm  # noqa: E402


def normalized_mode(layers, idx, f0, zc, sub_label="SiO2"):
    """Energy-normalized complex scalar mode u(z)=E/sqrt(U) resampled onto common grid zc."""
    z, E, H, eps = tmm.field_profile(layers, idx, f0, sub_label=sub_label)
    U = np.trapezoid(0.25 * (eps * np.abs(E) ** 2 + np.abs(H) ** 2), z)
    u = E / np.sqrt(U)
    return np.interp(zc, z, u.real) + 1j * np.interp(zc, z, u.imag)


def chi3_mask(geom, zc):
    """Boolean mask of the high-index (cavity-material) region = where chi3 != 0."""
    hi = geom["cavity"]["mat"]
    mask = np.zeros_like(zc, dtype=bool)
    z = 0.0
    sp = geom.get("spacers", {}) or {}
    z += float(sp.get("left_um", 0.0))
    for l in geom["mirrors"]["left"]:
        d = float(l["thk_um"])
        if l["mat"] == hi:
            mask |= (zc >= z) & (zc <= z + d)
        z += d
    mask |= (zc >= z) & (zc <= z + float(geom["cavity"]["L_um"]))   # cavity is high-index
    z += float(geom["cavity"]["L_um"])
    for l in geom["mirrors"]["right"]:
        d = float(l["thk_um"])
        if l["mat"] == hi:
            mask |= (zc >= z) & (zc <= z + d)
        z += d
    return mask


def counter_coefficients(geom, freqs, chi_iso, kappa_s, sub_label="SiO2", npts=8000):
    """Derived counter-rotating coefficients (raw and /kappa_s) from TMM fields.
    freqs: dict with probe/pump1/pump2/sb_plus/sb_minus -> frequency (1/um)."""
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    Ltot = sum(d for d, _ in layers)
    zc = np.linspace(0.0, Ltot, npts)
    mask = chi3_mask(geom, zc)
    u = {k: normalized_mode(layers, idx, freqs[k], zc, sub_label)
         for k in ("probe", "pump1", "pump2", "sb_plus", "sb_minus")}
    us, u1, u2, ubp, ubm = u["probe"], u["pump1"], u["pump2"], u["sb_plus"], u["sb_minus"]
    chi = complex(chi_iso)

    def I(v):
        return complex(np.trapezoid(v[mask], zc[mask]))

    ps, pp, pm = 3 * freqs["probe"] / 8, 3 * freqs["sb_plus"] / 8, 3 * freqs["sb_minus"] / 8
    raw = {
        # direct Kerr (alpha): pref_s * 2chi * int |us|^2 |up|^2
        "alpha1_plus": ps * 2 * chi * I(np.abs(us) ** 2 * np.abs(u1) ** 2),
        "alpha2_plus": ps * 2 * chi * I(np.abs(us) ** 2 * np.abs(u2) ** 2),
        # generation (zeta): pref_Omega * 2chi * conj(ub) uin conj(uout) us
        "zeta_plus": pp * 2 * chi * I(np.conj(ubp) * u1 * np.conj(u2) * us),
        "zeta_minus": pm * 2 * chi * I(np.conj(ubm) * u2 * np.conj(u1) * us),
        # back-mixing (eta): pref_s * 2chi * conj(us) uout conj(uin) ub
        "eta_plus": ps * 2 * chi * I(np.conj(us) * u2 * np.conj(u1) * ubp),
        "eta_minus": ps * 2 * chi * I(np.conj(us) * u1 * np.conj(u2) * ubm),
    }
    norm = {k: v / kappa_s for k, v in raw.items()}
    return raw, norm


# ------------------------------- the chi5 score ------------------------------ #
# Source bandwidth cap (fixed 100 fs pulses): fwidth = 1/width_meep (matches
# faraday_meep_fp_circ.df_from_pulse_duration). Q_cap(f) = f / fwidth.
FWIDTH_100FS = 1.0 / ((100.0 / (2.0 * np.log(2.0))) * (299792458.0 / 1e-6 * 1e-15))


def q_cap(f, pulse_fs=100.0):
    fwidth = 1.0 / ((pulse_fs / (2.0 * np.log(2.0))) * (299792458.0 / 1e-6 * 1e-15))
    return f / fwidth


def chi5_score(geom, freqs, chi_iso, kappa_s, sub_label="SiO2", pulse_fs=100.0):
    """FDTD-light chi5 rotation figure of merit from TMM + the derived coefficients.

    Steady state (balanced sigma+sigma-): theta ~ (B_s/omega_s) * |Re(Sigma+ - Sigma-)|,
    Sigma+- = -eta*zeta*|p1|^2|p2|^2 / (kappa_Omega/2 - i*Delta_Omega), with the pump buildup
    |p_i|^2 ~ B_i |S_i|^2 / omega_i and B_i = min(Q_i, Q_cap_i) (the fixed-pulse bandwidth cap).
    Re(.) is the rotation channel and is the symmetry-break factor (a +-Delta-symmetric cavity
    gives Re=0 = pure ellipticity); Im(.) is the ellipticity/DoLP-loss channel.
    Drops the fixed |S1|^2|S2|^2; relative score for ranking geometries.
    """
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    raw, _ = counter_coefficients(geom, freqs, chi_iso, kappa_s, sub_label)
    Q = {k: tmm.find_mode(layers, idx, freqs[k])["Q"] for k in ("probe", "pump1", "pump2")}
    # sideband modes nearest the generated sideband frequencies omega_s +- Delta
    msbp = tmm.find_mode(layers, idx, freqs["sb_plus"])
    msbm = tmm.find_mode(layers, idx, freqs["sb_minus"])
    Dp = msbp["freq"] / msbp["Q"] / 2 - 1j * (freqs["sb_plus"] - msbp["freq"])
    Dm = msbm["freq"] / msbm["Q"] / 2 - 1j * (freqs["sb_minus"] - msbm["freq"])
    # cascade differential (a+ via sb_minus, a- via sb_plus); symmetry-break in Re
    S = raw["eta_minus"] * raw["zeta_minus"] / Dm - raw["eta_plus"] * raw["zeta_plus"] / Dp
    B1 = min(Q["pump1"], q_cap(freqs["pump1"], pulse_fs))
    B2 = min(Q["pump2"], q_cap(freqs["pump2"], pulse_fs))
    Bs = min(Q["probe"], q_cap(freqs["probe"], pulse_fs))
    buildup = (B1 * B2 / (freqs["pump1"] * freqs["pump2"])) * (Bs / freqs["probe"])
    return {
        "fom_rotation": float(buildup * abs(S.real)),
        "fom_ellipticity": float(buildup * abs(S.imag)),
        "B1": B1, "B2": B2, "Bs": Bs,
        "Q1": Q["pump1"], "Q2": Q["pump2"], "Qprobe": Q["probe"],
        "Qsb_plus": msbp["Q"], "Qsb_minus": msbm["Q"],
        "ReS": float(S.real), "ImS": float(S.imag), "buildup": float(buildup),
    }


# --------------------------------- validation -------------------------------- #
def _validate(name, geom_path, json_path):
    geom = json.load(open(geom_path))
    ex = json.load(open(json_path))
    af = ex["analysis_frequencies"]
    freqs = {k: float(af[k]["freq_inv_um"]) for k in
             ("probe", "pump1", "pump2", "sb_plus", "sb_minus")}
    chi_iso = float(ex["material_constants"]["chi_iso_meep"])
    kappa_s = float(ex["rates_harminv_primary"]["probe"]["kappa_loaded"])
    raw, _ = counter_coefficients(geom, freqs, chi_iso, kappa_s)

    def cx(v):
        return complex(v["re"], v["im"]) if isinstance(v, dict) else complex(v)

    meep = {k: cx(ex["derived_coefficients"]["counter"][k + "_raw"]) for k in raw}
    print(f"\n=== {name}: TMM-analytic vs Meep-extracted RAW counter coefficients ===")
    print(f"{'coeff':12s} {'|TMM|':>10s} {'|Meep|':>10s} {'ratio':>7s} {'dphase°':>8s}")
    for k in ("zeta_plus", "zeta_minus", "eta_plus", "eta_minus", "alpha1_plus", "alpha2_plus"):
        t, m = raw[k], meep[k]
        dph = np.degrees(np.angle(t / m)) if abs(m) > 0 else float("nan")
        print(f"{k:12s} {abs(t):10.3e} {abs(m):10.3e} {abs(t)/abs(m):7.2f} {dph:8.1f}")
    # structural check: eta ~ conj(zeta)?
    for arm in ("plus", "minus"):
        z, e = raw["zeta_" + arm], raw["eta_" + arm]
        print(f"  eta_{arm} vs conj(zeta_{arm}): |diff|/|zeta| = {abs(e - z.conjugate())/abs(z):.2e}")


if __name__ == "__main__":
    _validate("sin_best_absolute",
              MEEP / "SiN_optimizations/best_absolute/geometry.json",
              MEEP / "SiN_optimizations/best_absolute/tcmt_derivation_analysis/tcmt_extracted_params_derivation.json")
    _validate("sic_L3p2um",
              MEEP / "SiC_optimizations/sic_L3p2um/geometry.json",
              MEEP / "SiC_optimizations/sic_L3p2um/tcmt_derivation_analysis/tcmt_extracted_params_derivation.json")
