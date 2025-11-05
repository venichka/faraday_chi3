#!/usr/bin/env python3
"""
Fit (n,k) data to a Meep-style sum of Lorentzian susceptibilities and plot ε(λ).

- Uses SciPy's least_squares for the fit.
- Input files are CSV with columns: wavelength_nm, n, k
- No pandas; uses numpy.genfromtxt.
- Material construction matches your attached example:
    * If f0 == 0  -> DrudeSusceptibility with sigma = A (and frequency=1.0)
    * If f0 > 0   -> LorentzianSusceptibility with sigma = A / f0**2
  (See Meep docs for susceptibility parameters.)  # refs: Materials / Python API

Model (Meep frequency f = 1/λ_μm):
    ε(f) = ε_inf + sum_j A_j / ( f0_j**2 - f**2 - i*γ_j*f )

NOTE: The fitter keeps the same parameter layout as before:
    params = [eps_inf, f0_1..f0_N, gamma_1..gamma_N, A_1..A_N]
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- user inputs ----
FILE_SIO2 = "sio2_from_slowsio2_2min42s_fastsn_3min2s_no6_tape_backside.csv"
FILE_SINX = "sn_from_slowsio2_2min42s_fastsn_3min2s_no6_tape_backside.csv"
FIT_FILE = FILE_SINX     # choose which file to fit
N_POLES = 10            # number of Lorentz poles
OUT_PREFIX = "fit_sinx"   # prefix for outputs (png, params.py)
# ---------------------


# ------------------------ Data loading (CSV, no pandas) ------------------------
def load_nk_csv(path: Path):
    """
    Load CSV with columns: wavelength_nm, n, k (with or without a header).
    Returns (wl_um, f_meep, eps_real, eps_imag).
    """
    arr = np.genfromtxt(
        path, delimiter=",", comments="#", dtype=float, invalid_raise=False
    )
    # If the file has a header, genfromtxt may return NaNs in the first row; drop those.
    if arr.ndim == 1:
        raise RuntimeError(f"Could not parse numeric columns from {path}")
    # Keep only rows with 3 finite columns
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
    return wl_um[order], f_meep[order], eps_real[order], eps_imag[order]


# ----------------------------- Fitter (unchanged) ------------------------------
def eps_model(f, params):
    """
    Complex ε for Lorentz poles, with the same parameterization as before:
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
    # crude guesses, same spirit as before
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
    lb = np.concatenate(([-np.inf], 1e-6*np.ones(3*n)))
    ub = np.concatenate(([np.inf],  np.inf*np.ones(3*n)))
    sol = least_squares(resid, p0, bounds=(lb, ub), max_nfev=10000, verbose=1)
    return sol.x, sol.cost, sol.success


# ---------------- Material creation (matches your attached example) -------------
def build_meep_medium_from_params(params):
    """
    Build a meep.Medium using the same construction logic as your reference file:

      - For each pole:
          if f0 == 0: use DrudeSusceptibility(frequency=1.0, gamma=γ, sigma=A)
          else:       use LorentzianSusceptibility(frequency=f0, gamma=γ, sigma=A/f0**2)

    See Meep's susceptibility docs for parameter meanings.  # refs: Materials / Python API
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
        if f0j == 0.0:
            # Drude term: use sigma = A (as in your example)
            E_susceptibilities.append(
                mp.DrudeSusceptibility(frequency=1.0, gamma=gamj, sigma=Aj)
            )
        else:
            # Lorentz term: Meep expects sigma consistent with its internal form;
            # your example converts A -> sigma via division by f0^2
            E_susceptibilities.append(
                mp.LorentzianSusceptibility(
                    frequency=f0j,
                    gamma=gamj,
                    sigma=Aj / (f0j**2),
                )
            )

    return mp.Medium(epsilon=eps_inf, E_susceptibilities=E_susceptibilities)


# ---------------------------------- Main ---------------------------------------
def main():
    wl_um, f_meep, eps_real, eps_imag = load_nk_csv(Path(FIT_FILE))

    params, cost, ok = fit_lorentz(f_meep, eps_real, eps_imag, N_POLES)
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
    eps_fit = np.array([complex(medium.epsilon(float(f))[0, 0])
                       for f in f_meep])

    plt.figure(figsize=(9, 4.5), dpi=140)
    plt.subplot(1, 2, 1)
    plt.plot(wl_um, eps_real, "k.", label="data Re{ε}")
    plt.plot(wl_um, eps_fit.real, "-", label="fit Re{ε}")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Re{ε}")
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(wl_um, eps_imag, "k.", label="data Im{ε}")
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
