#!/usr/bin/env python
"""Bulk c-cut sapphire: plane-wave estimate of the two-pump all-optical Faraday rotation.

sigma+ at f1 and sigma- at f2 write a Kerr birefringence whose axis turns at (f1 - f2)/2. A probe
time-slice sees the axis at phi(z) = phi0 + q z, with q = pi * df * (1/vg_probe - 1/vg_pump) from
group-velocity walk-off, and a retardance rate kappa = (2 pi / lambda_probe) * (8/3) * n2 * I
(I = peak intensity per pump; isotropic XPM with B/A = 1). The probe Stokes vector precesses about
(cos 2phi, sin 2phi, 0) at rate kappa. Averaging over phi0 (the pump relative phase, random from
scan to scan) leaves the k = 0 rotation; the phi0 swing is the carrier fringe.

Thin-sample limit of the averaged rotation: theta ~ delta^2 * qL / 12, delta = kappa L.

Assumptions (2026-11-13, written for the lab's preliminary sapphire data):
  - plane wave, flat-top pulses in time, no pump depletion or pump walk-off between the two pumps
  - ordinary ray only (c-cut, normal incidence), Malitson Sellmeier, n2 = 3e-20 m^2/W
  - instantaneous electronic chi3 with the isotropic transverse structure (no Raman, no dispersion
    of the tensor), so single-pump rotation and inverse Faraday terms are zero by construction
  - thickness and wavelengths are guesses: the lab has not reported them
Real Gaussian pulses lower the numbers by roughly 2-3x. No FDTD cross-check.

  python sapphire_bulk.py
"""
import numpy as np

B = [1.4313493, 0.65054713, 5.3414021]      # Malitson, ordinary ray, lambda in um
C = [0.0726631, 0.1193242, 18.028251]
C0 = 299792458.0
N2 = 3e-20                                  # m^2/W


def n(lam):
    return np.sqrt(1 + sum(b * lam * lam / (lam * lam - c * c) for b, c in zip(B, C)))


def n_group(lam, h=1e-4):
    return n(lam) - lam * (n(lam + h) - n(lam - h)) / (2 * h)


def run(lp, l1, l2, L, I, nz=4000, nphi=64):
    """Probe lp, pumps l1/l2 (um), thickness L (m), intensity per pump I (W/cm^2)."""
    df = C0 / (l1 * 1e-6) - C0 / (l2 * 1e-6)
    dgv = (n_group(lp) - n_group(0.5 * (l1 + l2))) / C0              # s/m
    q = np.pi * abs(df) * dgv
    kap = 2 * np.pi / (lp * 1e-6) * (8 / 3) * N2 * I * 1e4
    dz = L / nz
    out = []
    for p0 in np.linspace(0, np.pi, nphi, endpoint=False):
        S = np.array([0, 1.0, 0])                                     # 45 deg linear probe
        for k in range(nz):
            ph = p0 + q * (k + 0.5) * dz
            a = np.array([np.cos(2 * ph), np.sin(2 * ph), 0])
            t = kap * dz
            S = S * np.cos(t) + np.cross(a, S) * np.sin(t) + a * np.dot(a, S) * (1 - np.cos(t))
        out.append(S)
    out = np.array(out)
    m = out.mean(0)
    theta = 0.5 * np.degrees(np.arctan2(-m[0], m[1]))                 # azimuth of the phase-averaged Stokes vector
    ths = 0.5 * np.degrees(np.arctan2(-out[:, 0], out[:, 1]))
    return {"df_THz": df / 1e12, "walkoff_fs_per_mm": dgv * 1e12, "qL": q * L, "deltaL": kap * L,
            "theta_deg": theta, "swing_deg": (ths.max() - ths.min()) / 2,
            "max_S3": np.abs(out[:, 2]).max(), "dolp": np.hypot(m[0], m[1])}


if __name__ == "__main__":
    for name, (lp, l1, l2) in {"800 / 1560+1620": (0.8, 1.56, 1.62),
                               "850 / 1656+1695": (0.8502, 1.6559, 1.6952)}.items():
        print(f"{name}: n_g probe {n_group(lp):.4f}, pump {n_group(0.5 * (l1 + l2)):.4f}")
        print("  L mm   I W/cm2   df THz  walkoff fs/mm   qL   delta   <theta> deg  swing deg  max|S3|  DoLP(avg)")
        for L in (0.5e-3, 1e-3, 3e-3):
            for I in (1e11, 3e11, 1e12):
                r = run(lp, l1, l2, L, I)
                print(f"  {L * 1e3:4.1f}  {I:8.0e}  {r['df_THz']:6.2f}  {r['walkoff_fs_per_mm']:8.1f}  "
                      f"{r['qL']:6.2f}  {r['deltaL']:6.2f}  {r['theta_deg']:+9.2f}  {r['swing_deg']:9.2f}  "
                      f"{r['max_S3']:6.2f}  {r['dolp']:6.2f}")
