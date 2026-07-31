#!/usr/bin/env python
"""Analytic 1-D transfer-matrix (TMM) linear engine for the FDTD-light chi5 optimizer.

Solves the linear optics of a DBR + Fabry-Perot-defect stack EXACTLY (normal incidence,
characteristic-matrix TMM) from the geometry JSON + dispersive n(lambda) interpolated from
the ellipsometry CSVs. No Meep / no FDTD. Provides:
  - reflectance/transmittance R/T(lambda)
  - cavity-mode finder: resonance f0 + loaded Q (transmission-peak FWHM)
  - (next: complex field profile E(z), mode volume, 4-mode overlaps)

This is the keystone that replaces per-candidate FDTD in the search loop (design_optimization_plan.md
section 5). Validate against the committed cavity_modes.json before building on it.

Units: a = 1 um, frequency f = 1/lambda_um (Meep convention).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MEEP_DIR = Path(__file__).resolve().parent.parent  # scripts/meep_project

# material label -> ellipsometry CSV (columns: wavelength_nm, n, k)
CSV_FOR = {"SiN": "si3n4.csv", "SiO2": "sio2.csv", "SiC": "sic.csv"}


def make_index(csv_name: str):
    """Return a callable n_complex(lambda_um) interpolating (n + i k) from a CSV."""
    data = np.loadtxt(MEEP_DIR / csv_name, delimiter=",", skiprows=1)
    lam_um = data[:, 0] / 1000.0
    order = np.argsort(lam_um)
    lam_um, n, k = lam_um[order], data[order, 1], data[order, 2]

    def idx(L):
        # Meep time convention e^{-i w t}: a lossy medium is n - i k (so the TMM
        # phase e^{i delta} decays). Using +ik would give unphysical gain (R+T>1).
        return np.interp(L, lam_um, n) - 1j * np.interp(L, lam_um, k)

    return idx


def index_map(extra_high: str | None = None):
    """Build {label: n(lambda)} for the materials a geometry may reference."""
    m = {lab: make_index(csv) for lab, csv in CSV_FOR.items()}
    m["air"] = lambda L: np.ones_like(np.asarray(L, dtype=float)) + 0j
    return m


def build_layers(geom: dict):
    """Physical stack air(incident) | left mirror | cavity | right mirror | substrate(exit).
    Returns list of (thickness_um, material_label), ordered incident->exit. PML/air pads are
    simulation buffers and excluded; spacers (if any) are treated as air gaps."""
    layers = []
    sp = geom.get("spacers", {}) or {}
    if float(sp.get("left_um", 0.0)) > 0:
        layers.append((float(sp["left_um"]), "air"))
    for l in geom["mirrors"]["left"]:
        layers.append((float(l["thk_um"]), l["mat"]))
    layers.append((float(geom["cavity"]["L_um"]), geom["cavity"]["mat"]))
    for l in geom["mirrors"]["right"]:
        layers.append((float(l["thk_um"]), l["mat"]))
    if float(sp.get("right_um", 0.0)) > 0:
        layers.append((float(sp["right_um"]), "air"))
    return layers


def rt_at(layers, idx, lam, n_inc=1.0 + 0j, sub_label="SiO2"):
    """Reflectance & transmittance at one wavelength lam (um). Characteristic-matrix TMM."""
    M = np.eye(2, dtype=complex)
    for d, mat in layers:
        N = complex(idx[mat](lam))
        delta = 2.0 * np.pi * N * d / lam
        c, s = np.cos(delta), np.sin(delta)
        M = M @ np.array([[c, 1j * s / N], [1j * N * s, c]], dtype=complex)
    ns = complex(idx[sub_label](lam))
    B = M[0, 0] + M[0, 1] * ns
    C = M[1, 0] + M[1, 1] * ns
    r = (n_inc * B - C) / (n_inc * B + C)
    R = abs(r) ** 2
    T = 4.0 * n_inc.real * ns.real / abs(n_inc * B + C) ** 2
    return R, float(np.real(T))


def spectrum(layers, idx, freqs, sub_label="SiO2", n_inc=1.0):
    """R, T over an array of frequencies f=1/lambda (1/um). Vectorized over frequency
    (only the 13-ish layers are a Python loop) -> ~50x faster than scalar rt_at."""
    f = np.asarray(freqs, dtype=float)
    lam = 1.0 / f
    m00 = np.ones_like(f, dtype=complex); m01 = np.zeros_like(f, dtype=complex)
    m10 = np.zeros_like(f, dtype=complex); m11 = np.ones_like(f, dtype=complex)
    for d, mat in layers:
        N = np.asarray(idx[mat](lam), dtype=complex)
        delta = 2.0 * np.pi * N * d * f
        c, s = np.cos(delta), np.sin(delta)
        a00, a01, a10, a11 = c, 1j * s / N, 1j * N * s, c
        m00, m01, m10, m11 = (m00 * a00 + m01 * a10, m00 * a01 + m01 * a11,
                              m10 * a00 + m11 * a10, m10 * a01 + m11 * a11)
    ns = np.asarray(idx[sub_label](lam), dtype=complex)
    B, C = m00 + m01 * ns, m10 + m11 * ns
    den = n_inc * B + C
    R = np.abs((n_inc * B - C) / den) ** 2
    T = 4.0 * n_inc * np.real(ns) / np.abs(den) ** 2
    return R, np.real(T)


def _layer_matrix(N, d, f):
    delta = 2.0 * np.pi * N * d * f          # f = 1/lambda, so delta = 2*pi*N*d/lambda
    c, s = np.cos(delta), np.sin(delta)
    return np.array([[c, 1j * s / N], [1j * N * s, c]], dtype=complex)


def t_amp(layers, idx, lam, n_inc=1.0, sub_label="SiO2"):
    """Complex transmission amplitude t = 2 n0 / (n0 B + C)."""
    M = np.eye(2, dtype=complex)
    for d, mat in layers:
        N = complex(idx[mat](lam))
        M = M @ _layer_matrix(N, d, 1.0 / lam)
    ns = complex(idx[sub_label](lam))
    D = n_inc * (M[0, 0] + M[0, 1] * ns) + (M[1, 0] + M[1, 1] * ns)
    return 2.0 * n_inc / D


def mode_Q(layers, idx, f0, sub_label="SiO2", n_inc=1.0, rel_df=1e-5):
    """Loaded Q from the transmission group delay at resonance: Q = (f0/2)|dphi_t/df|.
    Robust for leaky/low-finesse modes (no half-max bracketing needed)."""
    df = f0 * rel_df
    ph = np.unwrap([np.angle(t_amp(layers, idx, 1.0 / f, n_inc, sub_label))
                    for f in (f0 - df, f0, f0 + df)])
    dphi = (ph[2] - ph[0]) / (2.0 * df)
    return float(abs(0.5 * f0 * dphi))


def find_mode(layers, idx, f_center, f_halfwin=0.02, sub_label="SiO2", n=4000):
    """Locate the transmission-peak cavity mode NEAREST f_center (dense FP comb -> pick
    the closest peak, not the tallest) and its loaded Q from the complex pole.
    Returns dict(freq, lambda_um, Q, T_peak, f_pole) or None."""
    fs = np.linspace(f_center - f_halfwin, f_center + f_halfwin, n)
    _, T = spectrum(layers, idx, fs, sub_label=sub_label)
    peaks = [i for i in range(1, len(fs) - 1) if T[i] >= T[i - 1] and T[i] > T[i + 1]]
    if not peaks:
        return None
    j = min(peaks, key=lambda i: abs(fs[i] - f_center))
    f0, Tpk = float(fs[j]), float(T[j])
    Q = mode_Q(layers, idx, f0, sub_label=sub_label)
    return {"freq": f0, "lambda_um": 1.0 / f0, "Q": Q, "T_peak": Tpk}


def find_modes_in_band(layers, idx, fmin, fmax, sub_label="SiO2", n=4000, t_min=0.05):
    """All transmission-peak cavity modes in [fmin, fmax] with loaded Q (group delay).
    Returns list of dict(freq, lambda_um, Q, T_peak), low to high frequency."""
    fs = np.linspace(fmin, fmax, n)
    _, T = spectrum(layers, idx, fs, sub_label=sub_label)
    out = []
    for i in range(1, len(fs) - 1):
        if T[i] >= T[i - 1] and T[i] > T[i + 1] and T[i] > t_min:
            f0 = float(fs[i])
            out.append({"freq": f0, "lambda_um": 1.0 / f0,
                        "Q": mode_Q(layers, idx, f0, sub_label=sub_label),
                        "T_peak": float(T[i])})
    return out


def field_profile(layers, idx, f0, sub_label="SiO2", ppw=80):
    """Complex E(z) of the resonant field at f0 (drive normalized to a unit transmitted
    wave). Returns (z_um, E_complex, eps_real). |E|^2 peaks in the defect."""
    lam0 = 1.0 / f0
    ns = complex(idx[sub_label](lam0))
    slices = []
    for d, mat in layers:
        N = complex(idx[mat](lam0))
        ns_sub = max(2, int(np.ceil(ppw * abs(N) * d / lam0)))
        for _ in range(ns_sub):
            slices.append((d / ns_sub, N))
    Ltot = sum(s[0] for s in slices)
    EH = np.array([1.0, ns], dtype=complex)     # outgoing wave at the substrate face
    z = Ltot
    zg, Eg, Hg, epsg = [z], [EH[0]], [EH[1]], [ns.real ** 2]
    for dz, N in reversed(slices):              # propagate substrate -> air
        EH = _layer_matrix(N, dz, f0) @ EH
        z -= dz
        zg.append(z); Eg.append(EH[0]); Hg.append(EH[1]); epsg.append(N.real ** 2)
    return (np.array(zg[::-1]), np.array(Eg[::-1]), np.array(Hg[::-1]), np.array(epsg[::-1]))


def mode_volume(z, E, eps):
    """1-D effective mode length V = int eps|E|^2 dz / max(eps|E|^2) (um)."""
    w = eps * np.abs(E) ** 2
    return float(np.trapezoid(w, z) / np.max(w))


def cav_field(layers, idx, f, zc, sub_label="SiO2", n_inc=1.0):
    """Complex intracavity E(z) per unit INCIDENT amplitude, resampled to grid zc.

    Unlike field_profile (normalized to unit *transmitted* wave), this divides out the
    incident amplitude E_inc = (E_air + H_air/n_inc)/2, so |cav_field|^2 is the physical
    intracavity field-intensity ENHANCEMENT and the COMPLEX value carries the true cavity
    dispersion/phase at f. This is the keystone for both the 100fs buildup and the
    symmetry-break: it is the actual (generally asymmetric) cavity response, not a snapped
    single-mode Lorentzian. For a symmetric Lorentzian cav_field(w0-d)=cav_field(w0+d)^*,
    so Re[cav_field(w0-d)-cav_field(w0+d)]=0 -> no rotation (the symmetric-cavity null)."""
    z, E, H, eps = field_profile(layers, idx, f, sub_label=sub_label)
    e_inc = 0.5 * (E[0] + H[0] / n_inc)        # air face: E_air = E_inc+E_refl, H_air = n(E_inc-E_refl)
    u = E / e_inc
    return np.interp(zc, z, u.real) + 1j * np.interp(zc, z, u.imag)


# --------------------------------- validation -------------------------------- #
def _validate(geom_path, modes_path, label, sub_label="SiO2"):
    geom = json.load(open(geom_path))
    modes = json.load(open(modes_path))
    cav_mat = geom["cavity"]["mat"]
    idx = index_map()
    layers = build_layers(geom)
    print(f"\n=== {label} ({cav_mat}, {len(layers)} layers, cavity L={geom['cavity']['L_um']:.4f} um, sub={sub_label}) ===")
    print(f"{'mode':8s} {'f_fdtd':>9s} {'f_tmm':>9s} {'dlam_nm':>8s} {'Q_fdtd':>8s} {'Q_tmm':>8s} {'Tpk':>6s}")
    targets = {"probe": modes["probe"]["frequency"],
               "pump1": modes["pump1"]["frequency"],
               "pump2": modes["pump2"]["frequency"]}
    for name, f_t in targets.items():
        Q_fdtd = modes[name].get("Q", float("nan"))
        m = find_mode(layers, idx, f_t, sub_label=sub_label)
        if m is None:
            print(f"{name:8s} {f_t:9.4f}   (no clean peak found near target)")
            continue
        dlam_nm = (m["lambda_um"] - 1.0 / f_t) * 1000.0
        print(f"{name:8s} {f_t:9.4f} {m['freq']:9.4f} {dlam_nm:8.2f} "
              f"{Q_fdtd:8.1f} {m['Q']:8.1f} {m['T_peak']:6.3f}")


if __name__ == "__main__":
    base = MEEP_DIR
    _validate(base / "SiN_optimizations/best_absolute/geometry.json",
              base / "SiN_optimizations/best_absolute/cavity_modes.json",
              "SiN best_absolute")
    _validate(base / "SiC_optimizations/sic_L3p2um/geometry.json",
              base / "SiC_optimizations/sic_L3p2um/cavity_modes.json",
              "SiC L3.2um")
