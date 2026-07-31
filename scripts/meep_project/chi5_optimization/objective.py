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


def make_ctx(geom, sub_label="SiO2", npts=8000):
    """Per-geometry context (grid, chi3 mask, mode-field cache) reused across the
    operating-point loop so probe/pump fields are computed once, not per pump pair."""
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    Ltot = sum(d for d, _ in layers)
    zc = np.linspace(0.0, Ltot, npts)
    return {"idx": idx, "layers": layers, "zc": zc,
            "mask": chi3_mask(geom, zc), "cache": {}, "sub": sub_label}


def _u(ctx, f):
    k = round(float(f), 9)
    c = ctx["cache"]
    if k not in c:
        c[k] = normalized_mode(ctx["layers"], ctx["idx"], f, ctx["zc"], ctx["sub"])
    return c[k]


def counter_coefficients(geom, freqs, chi_iso, kappa_s=1.0, sub_label="SiO2", npts=8000, ctx=None):
    """Derived counter-rotating coefficients (raw and /kappa_s) from TMM fields.
    freqs: dict with probe/pump1/pump2/sb_plus/sb_minus -> frequency (1/um)."""
    if ctx is None:
        ctx = make_ctx(geom, sub_label, npts)
    zc, mask = ctx["zc"], ctx["mask"]
    us, u1, u2, ubp, ubm = (_u(ctx, freqs[k]) for k in
                            ("probe", "pump1", "pump2", "sb_plus", "sb_minus"))
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


def chi5_score(geom, freqs, chi_iso, kappa_s=1.0, sub_label="SiO2", pulse_fs=100.0,
               ctx=None, q_known=None):
    """FDTD-light chi5 rotation figure of merit from TMM + the derived coefficients.

    Steady state (balanced sigma+sigma-): theta ~ (B_s/omega_s) * |Re(Sigma+ - Sigma-)|,
    Sigma+- = -eta*zeta*|p1|^2|p2|^2 / (kappa_Omega/2 - i*Delta_Omega), with the pump buildup
    |p_i|^2 ~ B_i |S_i|^2 / omega_i and B_i = min(Q_i, Q_cap_i) (the fixed-pulse bandwidth cap).
    Re(.) is the rotation channel and is the symmetry-break factor (a +-Delta-symmetric cavity
    gives Re=0 = pure ellipticity); Im(.) is the ellipticity/DoLP-loss channel.
    Drops the fixed |S1|^2|S2|^2; relative score for ranking geometries.
    """
    if ctx is None:
        ctx = make_ctx(geom, sub_label)
    layers, idx = ctx["layers"], ctx["idx"]
    raw, _ = counter_coefficients(geom, freqs, chi_iso, kappa_s, sub_label, ctx=ctx)
    if q_known is not None:
        Q = q_known
    else:
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


# ============================================================================= #
#  CORRECTED chi5 FoM (v2, 2026-06-11) — empirically anchored to the FDTD L-sweep
# ============================================================================= #
# WHY v2 replaces chi5_score for geometry ranking: the legacy score mis-ranked geometry
# (proxy fom ~ L^-3, FDTD theta ~ L^+1.2 — sign-flipped). Two root causes, both fixed here:
#   (1) Energy normalization u=E/sqrt(U) imposed fixed-energy single-mode TCMT scaling
#       (overlaps ~1/V). But the FDTD decomposition (chi5_optimization/decompose/) shows the
#       rotation is a PROPAGATION effect: theta ~ (interaction length L) x (intracavity pump
#       intensity, FLAT in L) x (probe enhancement, FLAT) x (mild symmetry-break). We use the
#       physical per-incident intracavity field tmm.cav_field (NOT energy-normalized) and the
#       explicit interaction length, so the length dependence is right.
#   (2) The symmetry-break Re(S) was a near-cancellation of two snapped-Lorentzian arms
#       (D+ ~= D-^*), dominated by mode-finding noise -> spurious L^-3. We evaluate the cascade
#       on the TRUE complex cavity response cav_field(omega_s +- Delta); a symmetric cavity
#       gives cav_field(-d)=cav_field(+d)^* -> Re=0 (the correct symmetric-cavity null), and real
#       cavity dispersion gives a robust nonzero rotation.
# 100 fs EVERYWHERE: every buildup factor is the pulse-spectrum-weighted intracavity intensity
# (buildup_100fs); this reproduces the FDTD-measured FLAT buildup (Q>Q_cap -> saturated, not 1/Q).
# Validated on the SiN best_absolute L-family: theta_fom rises with L (Spearman ~+0.88 vs FDTD),
# the right SIGN (legacy was anti-correlated). See chi5_optimization/decompose_buildup.py.

def _cav(ctx, f):
    """Cached per-incident intracavity field cav_field(f) on the ctx grid."""
    k = ("cav", round(float(f), 9))
    c = ctx["cache"]
    if k not in c:
        c[k] = tmm.cav_field(ctx["layers"], ctx["idx"], f, ctx["zc"], ctx["sub"])
    return c[k]


def buildup_100fs(ctx, f0, fwidth=FWIDTH_100FS, n_pts=9):
    """100fs-bandwidth-limited intracavity intensity (cavity-region average) at carrier f0.
    Pulse-spectrum (Gaussian, source fwidth) weighted |cav_field|^2 -> reproduces the FDTD-flat
    buildup (the cavity line is narrower than the 100fs source, so the buildup saturates)."""
    sig = fwidth / np.sqrt(2.0 * np.log(2.0))
    fs = f0 + sig * np.linspace(-3.0, 3.0, n_pts)
    w = np.exp(-(fs - f0) ** 2 / (2.0 * sig ** 2)); w /= w.sum()
    mask = ctx["mask"]
    return float(sum(wi * np.mean(np.abs(_cav(ctx, f)[mask]) ** 2) for f, wi in zip(fs, w)))


def corrected_cascade(ctx, freqs, chi_iso):
    """Cascade self-energy difference Sigma = sigma+_arm - sigma-_arm on the TRUE cavity response.
    Same overlap structure as counter_coefficients (eta, zeta) but with physical per-incident
    cav_field and the actual cavity response at omega_s +- Delta in place of the snapped
    sideband Lorentzian. Re(Sigma) = rotation channel (symmetry-break), Im(Sigma) = ellipticity."""
    zc, mask = ctx["zc"], ctx["mask"]
    us, u1, u2 = (_cav(ctx, freqs[k]) for k in ("probe", "pump1", "pump2"))
    ubp, ubm = _cav(ctx, freqs["sb_plus"]), _cav(ctx, freqs["sb_minus"])
    chi = complex(chi_iso)

    def I(v):
        return complex(np.trapezoid(v[mask], zc[mask]))

    ps, pp, pm = 3 * freqs["probe"] / 8, 3 * freqs["sb_plus"] / 8, 3 * freqs["sb_minus"] / 8
    # lower-sideband arm (drives a+); upper-sideband arm (drives a-)
    z_m = pm * 2 * chi * I(np.conj(ubm) * u2 * np.conj(u1) * us)
    e_m = ps * 2 * chi * I(np.conj(us) * u1 * np.conj(u2) * ubm)
    z_p = pp * 2 * chi * I(np.conj(ubp) * u1 * np.conj(u2) * us)
    e_p = ps * 2 * chi * I(np.conj(us) * u2 * np.conj(u1) * ubp)
    return e_m * z_m - e_p * z_p


def chi5_score_v2(geom, freqs, chi_iso, sub_label="SiO2", pulse_fs=100.0, ctx=None):
    """Corrected, 100fs-aware, FDTD-anchored chi5 rotation FoM (see header above).

      theta_fom ~ [B1 B2 Bs : 100fs intracavity buildup, flat in L]
                  x |Re Sigma|  (cascade symmetry-break on the true cavity response)
                  x L_interaction  (chi3 region length; the probe accumulates rotation over it)

    Re(Sigma)=rotation, Im(Sigma)=ellipticity. freqs: probe/pump1/pump2/sb_plus/sb_minus (1/um)."""
    if ctx is None:
        ctx = make_ctx(geom, sub_label)
    fwidth = 1.0 / ((pulse_fs / (2.0 * np.log(2.0))) * (299792458.0 / 1e-6 * 1e-15))
    B1 = buildup_100fs(ctx, freqs["pump1"], fwidth)
    B2 = buildup_100fs(ctx, freqs["pump2"], fwidth)
    Bs = buildup_100fs(ctx, freqs["probe"], fwidth)
    Sigma = corrected_cascade(ctx, freqs, chi_iso)
    L_int = float(np.trapezoid(ctx["mask"].astype(float), ctx["zc"]))   # interaction length ~ L
    buildup = B1 * B2 * Bs
    return {
        "fom_rotation": float(buildup * abs(Sigma.real) * L_int),
        "fom_ellipticity": float(buildup * abs(Sigma.imag) * L_int),
        "B1": B1, "B2": B2, "Bs": Bs, "buildup": float(buildup),
        "L_interaction": L_int,
        "ReSigma": float(Sigma.real), "ImSigma": float(Sigma.imag),
    }


# ============================================================================= #
#  v3 (2026-06-11): clean normalization — fixes the v2 buildup over-counting
# ============================================================================= #
# The v2 cascade used un-normalized cav_field for ALL carriers, so |Sigma_v2| ~ Bs.B1.B2.B_sb;
# multiplying by B1.B2.Bs again gave fom ~ B1^2 B2^2 Bs^2 B_sb.L -- pump buildup SQUARED (I^4,
# chi9-like) and a spurious probe-intensity^2 (the rotation ANGLE is probe-intensity independent).
# Masked on the flat-buildup L-family but biases cross-geometry ranking (see analytic_model.md S5).
# v3: the cascade Sigma_hat uses NORMALIZED fields so it carries ONLY overlap-shape + sideband
# dispersion (the symmetry-break), NOT buildup. Pumps & probe -> peak-normalized (shape, |peak|=1).
# Sidebands -> normalized by the PROBE peak, so their resonance RELATIVE to the probe (the M5 lever)
# and the +-Delta dispersion asymmetry (the rotation) survive while the absolute buildup is removed.
# Then buildup appears EXACTLY ONCE per carrier: B1.B2 (pumps, = I_pump^2) x Bs (probe dwell, LINEAR).

def cascade_normalized(ctx, freqs, chi_iso):
    """Symmetry-break cascade on SHAPE-normalized fields (no buildup). Re=rotation, Im=ellipticity.
    Pumps/probe peak-normalized; sidebands normalized by the probe peak (keeps resonance-vs-probe +
    dispersion = the M5 / symmetry-break, drops absolute sideband buildup)."""
    zc, mask = ctx["zc"], ctx["mask"]
    us, u1, u2 = (_cav(ctx, freqs[k]) for k in ("probe", "pump1", "pump2"))
    ubp, ubm = _cav(ctx, freqs["sb_plus"]), _cav(ctx, freqs["sb_minus"])
    ps_s = np.max(np.abs(us))
    hs = us / ps_s                                   # probe shape (peak 1)
    h1 = u1 / np.max(np.abs(u1)); h2 = u2 / np.max(np.abs(u2))   # pump shapes (peak 1)
    hbm = ubm / ps_s; hbp = ubp / ps_s               # sidebands relative to the probe peak
    chi = complex(chi_iso)

    def I(v):
        return complex(np.trapezoid(v[mask], zc[mask]))

    ps, pp, pm = 3 * freqs["probe"] / 8, 3 * freqs["sb_plus"] / 8, 3 * freqs["sb_minus"] / 8
    z_m = pm * 2 * chi * I(np.conj(hbm) * h2 * np.conj(h1) * hs)
    e_m = ps * 2 * chi * I(np.conj(hs) * h1 * np.conj(h2) * hbm)
    z_p = pp * 2 * chi * I(np.conj(hbp) * h1 * np.conj(h2) * hs)
    e_p = ps * 2 * chi * I(np.conj(hs) * h2 * np.conj(h1) * hbp)
    return e_m * z_m - e_p * z_p


def chi5_score_v3(geom, freqs, chi_iso, sub_label="SiO2", pulse_fs=100.0, ctx=None):
    """Clean-normalization chi5 rotation FoM (see header). Correct buildup powers:

      theta_fom = B1 . B2 . Bs . |Re Sigma_hat| . L_interaction
        B1,B2 = pump 100fs buildup (I_pump^2, once each)   Bs = probe buildup (LINEAR dwell)
        Sigma_hat = symmetry-break cascade on SHAPE-normalized fields (no buildup; M5 + dispersion)
        L_interaction = chi3 region length (the probe accumulates rotation over it)

    Re=rotation, Im=ellipticity. freqs: probe/pump1/pump2/sb_plus/sb_minus (1/um)."""
    if ctx is None:
        ctx = make_ctx(geom, sub_label)
    fwidth = 1.0 / ((pulse_fs / (2.0 * np.log(2.0))) * (299792458.0 / 1e-6 * 1e-15))
    B1 = buildup_100fs(ctx, freqs["pump1"], fwidth)
    B2 = buildup_100fs(ctx, freqs["pump2"], fwidth)
    Bs = buildup_100fs(ctx, freqs["probe"], fwidth)
    Sigma = cascade_normalized(ctx, freqs, chi_iso)
    L_int = float(np.trapezoid(ctx["mask"].astype(float), ctx["zc"]))
    buildup = B1 * B2 * Bs
    return {
        "fom_rotation": float(buildup * abs(Sigma.real) * L_int),
        "fom_ellipticity": float(buildup * abs(Sigma.imag) * L_int),
        "B1": B1, "B2": B2, "Bs": Bs, "buildup": float(buildup), "L_interaction": L_int,
        "ReSigma": float(Sigma.real), "ImSigma": float(Sigma.imag),
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
