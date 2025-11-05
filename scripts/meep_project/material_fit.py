# fit_meep_material_from_csv.py
# Minimal importable function: returns a meep.Medium from an n,k CSV.

import sys
import numpy as np
from pathlib import Path

# ---------- data loading (CSV, no pandas) ----------


def _load_nk_csv(path: Path):
    """CSV columns: wavelength_nm, n, k (with/without header)."""
    arr = np.genfromtxt(path, delimiter=",", comments="#",
                        dtype=float, invalid_raise=False)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Could not parse 3 numeric columns from {path}")
    # keep only finite rows
    mask = np.all(np.isfinite(arr[:, :3]), axis=1)
    arr = arr[mask]
    wl_nm = arr[:, 0]
    n_complex = arr[:, 1] + 1j*arr[:, 2]
    wl_um = wl_nm / 1000.0
    # Meep frequency units: 1/μm  (Materials doc)
    f_meep = 1.0 / wl_um
    eps = n_complex**2
    eps_r = np.real(eps)
    eps_i = np.imag(eps)
    # sort by frequency for optimizer stability
    order = np.argsort(f_meep)
    return f_meep[order], eps_r[order], eps_i[order]

# ---------- fitter (unchanged structure) ----------


def _eps_model(f, params):
    """
    ε(f) = ε_inf + Σ_j A_j / (f0_j**2 - f**2 - i γ_j f)
    params = [eps_inf, f0_1..f0_N, gamma_1..gamma_N, A_1..A_N]
    """
    n = (len(params)-1)//3
    eps_inf = float(params[0])
    f0 = np.abs(np.array(params[1:1+n], dtype=float))
    gam = np.abs(np.array(params[1+n:1+2*n], dtype=float))
    A = np.abs(np.array(params[1+2*n:1+3*n], dtype=float))
    f = np.atleast_1d(f).astype(float)
    denom = (f0[None, :]**2 - f[:, None]**2) - 1j*(gam[None, :]*f[:, None])
    chi = (A[None, :] / denom).sum(axis=1)
    return eps_inf + chi


def _pack_params(eps_inf, f0, gamma, A):
    return np.concatenate([[eps_inf], f0, gamma, A])


def _initial_guess(f, eps_r, n_poles):
    eps_inf = max(float(np.median(eps_r)), 1.0)
    f0 = np.quantile(f, np.linspace(0.2, 0.8, n_poles))
    gamma = 0.1*(f.max()-f.min())*np.ones(n_poles)
    dyn = max(1e-3, float(eps_r.max() - eps_r.min()))
    A = 0.5*dyn*np.ones(n_poles)
    return _pack_params(eps_inf, f0, gamma, A)


def _fit_lorentz(f, eps_r, eps_i, n_poles):
    try:
        from scipy.optimize import least_squares
    except Exception:
        raise RuntimeError(
            "SciPy is required for fitting. Try: pip install scipy")
    target = eps_r + 1j*eps_i

    def resid(p):
        est = _eps_model(f, p)
        return np.concatenate([est.real - target.real, est.imag - target.imag])
    p0 = _initial_guess(f, eps_r, n_poles)
    # eps_inf unbounded; others >= 0
    lb = np.concatenate(([-np.inf], 1e-6*np.ones(3*n_poles)))
    ub = np.concatenate(([np.inf],  np.inf*np.ones(3*n_poles)))
    sol = least_squares(resid, p0, bounds=(lb, ub), max_nfev=10000, verbose=0)
    return sol.x

# ---------- build meep material (matches your reference) ----------


def _build_meep_medium_from_params(params, tiny=1e-12):
    """
    For each pole:
      - if f0 ~ 0:    DrudeSusceptibility(frequency=1.0, gamma=γ, sigma=A)
      - else:         LorentzianSusceptibility(frequency=f0, gamma=γ, sigma=A/f0**2)
    """
    try:
        import meep as mp
    except Exception as e:
        raise RuntimeError(
            "meep is required to construct the Medium. Try: pip install meep") from e

    n = (len(params)-1)//3
    eps_inf = float(params[0])
    f0 = np.abs(np.array(params[1:1+n], dtype=float))
    gam = np.abs(np.array(params[1+n:1+2*n], dtype=float))
    A = np.abs(np.array(params[1+2*n:1+3*n], dtype=float))

    E_sus = []
    for j in range(n):
        f0j, gamj, Aj = float(f0[j]), float(gam[j]), float(A[j])
        if abs(f0j) < tiny:
            E_sus.append(mp.DrudeSusceptibility(
                frequency=1.0, gamma=gamj, sigma=Aj))
        else:
            E_sus.append(mp.LorentzianSusceptibility(
                frequency=f0j, gamma=gamj, sigma=Aj/(f0j**2)))
    return mp.Medium(epsilon=eps_inf, E_susceptibilities=E_sus)

# ---------- public function ----------


def fit_meep_material_from_csv(csv_path, *, material_name="fitted_mat", n_poles=3):
    """
    Return a meep.Medium whose dispersive ε(f) fits the (n,k) table in `csv_path`.

    Parameters
    ----------
    csv_path : str or Path
        CSV with columns: wavelength_nm, n, k
    material_name : str (unused at runtime; for your own variable naming)
    n_poles : int
        Number of Lorentz/Drude poles in the fit.

    Returns
    -------
    mp.Medium
        A Meep material usable directly in geometries/simulations.
    """
    f, eps_r, eps_i = _load_nk_csv(Path(csv_path))
    params = _fit_lorentz(f, eps_r, eps_i, n_poles)
    return _build_meep_medium_from_params(params)
