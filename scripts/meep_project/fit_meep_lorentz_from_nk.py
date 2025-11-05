#!/usr/bin/env python3
"""
Fit (n,k) data to a Meep-style sum of Lorentzian susceptibilities and plot ε(λ).

- Uses SciPy's least_squares for the fit.
- Input files are CSV with columns: wavelength_nm, n, k
- No pandas; uses numpy.genfromtxt.
- Material construction matches your attached example:
    * If f0 == 0  -> DrudeSusceptibility with sigma = A (and frequency=1.0)
    * If f0 > 0   -> LorentzianSusceptibility with sigma = A / f0**2

Model (Meep frequency f = 1/λ_μm):
    ε(f) = ε_inf + sum_j A_j / ( f0_j**2 - f**2 - i*γ_j*f )

NOTE: Parameter layout:
    params = [eps_inf, f0_1..f0_N, gamma_1..gamma_N, A_1..A_N]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- user inputs ----
FILE_SIO2 = "sio2_from_slowsio2_2min42s_fastsn_3min2s_no6_tape_backside.csv"
FILE_SINX = "sn_from_slowsio2_2min42s_fastsn_3min2s_no6_tape_backside.csv"
FIT_FILE = FILE_SINX     # choose which file to fit
N_POLES = 2             # number of Lorentz/Drude poles
OUT_PREFIX = "fit_sinx"   # prefix for outputs (png)
# Optional wavelength window (nm). Use None to disable.
LAMBDA_MIN = 600         # e.g., 450.0
LAMBDA_MAX = 2000         # e.g., 900.0
# ---------------------


# ------------------------ Data loading (CSV, no pandas) ------------------------
def load_nk_csv(path: Path):
    """
    Load CSV with columns: wavelength_nm, n, k (with or without a header).
    Returns (wl_nm, wl_um, f_meep, eps_real, eps_imag).
    """
    arr = np.genfromtxt(
        path, delimiter=",", comments="#", dtype=float, invalid_raise=False
    )
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Could not parse 3 numeric columns from {path}")
    mask = np.all(np.isfinite(arr[:, :3]), axis=1)
    arr = arr[mask]
    wl_nm = arr[:, 0]
    n = arr[:, 1] + 1j * arr[:, 2]
    wl_um = wl_nm / 1000.0
    eps = n**2
    eps_real = np.real(eps)
    eps_imag = np.imag(eps)
    f_meep = 1.0 / wl_um  # Meep frequency units: 1/μm
    # sort by increasing frequency to help optimizers
    order = np.argsort(f_meep)
    return wl_nm[order], wl_um[order], f_meep[order], eps_real[order], eps_imag[order]


# ----------------------------- Fitter (unchanged) ------------------------------
def eps_model(f, params):
    """
    Complex ε for Lorentz poles:
    params = [eps_inf, f0_1..f0_N, gamma_1..gamma_N, A_1..A_N]
    """
    n = (len(params) - 1) // 3
    eps_inf = float(params[0])
    f0 = np.abs(np.array(params[1:1+n], dtype=float))
    gam = np.abs(np.array(params[1+n:1+2*n], dtype=float))
    A = np.abs(np.array(params[1+2*n:1+3*n], dtype=float))
    f = np.atleast_1d(f).astype(float)
    denom = (f0[None, :]**2 - f[:, None]**2) - 1j * (gam[None, :] * f[:, None])
    chi = (A[None, :] / denom).sum(axis=1)
    return eps_inf + chi


def pack_params(eps_inf, f0, gamma, A):
    return np.concatenate([[eps_inf], f0, gamma, A])


def initial_guess(f, eps_real, n_poles):
    eps_inf = max(float(np.median(eps_real)), 1.0)
    f0 = np.quantile(f, np.linspace(0.2, 0.8, n_poles))
    gamma = 0.1 * (f.max() - f.min()) * np.ones(n_poles)
    dyn = max(1e-3, float(eps_real.max() - eps_real.min()))
    A = 0.5 * dyn * np.ones(n_poles)
    return pack_params(eps_inf, f0, gamma, A)


def fit_lorentz(f, eps_real, eps_imag, n_poles=3):
    try:
        from scipy.optimize import least_squares
    except Exception:
        print(
            "[error] SciPy is required for fitting. Try: pip install scipy", file=sys.stderr)
        sys.exit(2)

    target = eps_real + 1j * eps_imag

    def resid(p):
        est = eps_model(f, p)
        r = np.concatenate([est.real - target.real, est.imag - target.imag])
        return r

    p0 = initial_guess(f, eps_real, n_poles)
    n = n_poles
    lb = np.concatenate(([-np.inf], 1e-6*np.ones(3*n))
                        )   # eps_inf free; others >= 0
    ub = np.concatenate(([np.inf],  np.inf*np.ones(3*n)))
    sol = least_squares(resid, p0, bounds=(lb, ub), max_nfev=10000, verbose=1)
    return sol.x, sol.cost, sol.success


# ---------------- Material creation (matches your attached example) -------------
def build_meep_medium_from_params(params, tiny=1e-12):
    """
    Build a meep.Medium using the same construction logic as your reference file:

      - For each pole:
          if f0 ~ 0: DrudeSusceptibility(frequency=1.0, gamma=γ, sigma=A)
          else:      LorentzianSusceptibility(frequency=f0, gamma=γ, sigma=A/f0**2)
    """
    try:
        import meep as mp
    except Exception as e:
        raise RuntimeError(
            "meep is required to construct the Medium. Try: pip install meep") from e

    n = (len(params) - 1) // 3
    eps_inf = float(params[0])
    f0 = np.abs(np.array(params[1:1+n], dtype=float))
    gam = np.abs(np.array(params[1+n:1+2*n], dtype=float))
    A = np.abs(np.array(params[1+2*n:1+3*n], dtype=float))

    E_susceptibilities = []
    for j in range(n):
        f0j = float(f0[j])
        gamj = float(gam[j])
        Aj = float(A[j])
        if abs(f0j) < tiny:
            E_susceptibilities.append(
                mp.DrudeSusceptibility(frequency=1.0, gamma=gamj, sigma=Aj)
            )
        else:
            E_susceptibilities.append(
                mp.LorentzianSusceptibility(
                    frequency=f0j, gamma=gamj, sigma=Aj / (f0j**2)
                )
            )
    return mp.Medium(epsilon=eps_inf, E_susceptibilities=E_susceptibilities)


# ---------------------------------- Main ---------------------------------------
def main():
    wl_nm, wl_um, f_meep, eps_real, eps_imag = load_nk_csv(Path(FIT_FILE))

    # Apply wavelength window (nm) if provided
    mask = np.ones_like(wl_nm, dtype=bool)
    if LAMBDA_MIN is not None:
        mask &= (wl_nm >= float(LAMBDA_MIN))
    if LAMBDA_MAX is not None:
        mask &= (wl_nm <= float(LAMBDA_MAX))
    if not np.any(mask):
        raise ValueError(
            "No data points remain after applying wavelength window.")

    wl_nm = wl_nm[mask]
    wl_um = wl_um[mask]
    f_meep_win = f_meep[mask]
    eps_r_win = eps_real[mask]
    eps_i_win = eps_imag[mask]

    # sanity: at least a few points vs parameters
    if f_meep_win.size < max(6, 3*N_POLES):
        raise ValueError(f"Not enough data ({f_meep_win.size}) for {N_POLES} poles; "
                         f"need at least ~{max(6, 3*N_POLES)} points.")

    params, cost, ok = fit_lorentz(f_meep_win, eps_r_win, eps_i_win, N_POLES)
    print("\n=== Fit status ===")
    print("success:", ok, "cost:", cost)
    n = (len(params)-1)//3
    eps_inf = params[0]
    f0 = np.abs(params[1:1+n])
    gam = np.abs(params[1+n:1+2*n])
    A = np.abs(params[1+2*n:1+3*n])
    print(f"eps_inf = {eps_inf:.6g}")
    for j in range(n):
        print(f"pole {j+1}: f0={f0[j]:.6g}, gamma={gam[j]:.6g}, A={A[j]:.6g}")

    # Create Medium using the (fixed) construction method
    medium = build_meep_medium_from_params(params)

    # Plot comparison of ε from data vs. fitted medium
    # medium.epsilon(f) may return scalar or 3x3; handle both robustly.
    eps_fit = []
    for f in f_meep_win:
        val = medium.epsilon(float(f))
        arr = np.asarray(val)
        eps_fit.append(arr[0, 0] if arr.ndim >= 2 else arr)
    eps_fit = np.array(eps_fit, dtype=complex)

    plt.figure(figsize=(9, 4.5), dpi=140)
    plt.subplot(1, 2, 1)
    plt.plot(wl_um, eps_r_win, "k.", label="data Re{ε}")
    plt.plot(wl_um, eps_fit.real, "-", label="fit Re{ε}")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Re{ε}")
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(wl_um, eps_i_win, "k.", label="data Im{ε}")
    plt.plot(wl_um, eps_fit.imag, "-", label="fit Im{ε}")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Im{ε}")
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_PREFIX + "_eps_fit.png")
    print("Saved", OUT_PREFIX + "_eps_fit.png")


if __name__ == "__main__":
    main()
