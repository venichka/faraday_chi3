# fit_meep_material_from_csv.py
# Returns a meep.Medium fitted to an n,k CSV; optional wavelength filter.
import sys
import numpy as np
from pathlib import Path

# ---------- data loading (CSV, no pandas) ----------


def _load_nk_csv(path: Path):
    """CSV columns: wavelength_nm, n, k (with or without a header)."""
    arr = np.genfromtxt(path, delimiter=",", comments="#",
                        dtype=float, invalid_raise=False)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise RuntimeError(f"Could not parse 3 numeric columns from {path}")
    mask = np.all(np.isfinite(arr[:, :3]), axis=1)
    arr = arr[mask]
    wl_nm = arr[:, 0]
    n_complex = arr[:, 1] + 1j*arr[:, 2]
    wl_um = wl_nm / 1000.0
    f_meep = 1.0 / wl_um                 # Meep frequencies are in 1/μm
    eps = n_complex**2
    eps_r = np.real(eps)
    eps_i = np.imag(eps)
    # sort by frequency for optimizer stability
    order = np.argsort(f_meep)
    return wl_nm[order], f_meep[order], eps_r[order], eps_i[order]

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

# ---------- build meep material (same mapping as your reference) ----------


def _build_meep_medium_from_params(params, *, tiny=1e-12):
    """
    For each pole:
      if f0 ~ 0 -> DrudeSusceptibility(frequency=1.0, gamma=γ, sigma=A)
      else      -> LorentzianSusceptibility(frequency=f0, gamma=γ, sigma=A/f0**2)
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

# ---------- public function with wavelength window ----------


def fit_meep_material_from_csv(csv_path, *, material_name="fitted_mat", n_poles=3,
                               lambda_min=None, lambda_max=None):
    """
    Return a meep.Medium whose dispersive ε(f) fits the (n,k) table in `csv_path`.

    Parameters
    ----------
    csv_path : str or Path
        CSV with columns: wavelength_nm, n, k
    material_name : str
        For your own variable naming (not used internally).
    n_poles : int
        Number of Lorentz/Drude poles in the fit.
    lambda_min, lambda_max : float | None
        If provided, only rows with lambda in [lambda_min, lambda_max] **in nm** are fitted.

    Returns
    -------
    mp.Medium
    """
    wl_nm, f, eps_r, eps_i = _load_nk_csv(Path(csv_path))

    # wavelength-windowing (nm)
    mask = np.ones_like(wl_nm, dtype=bool)
    if lambda_min is not None:
        mask &= (wl_nm >= float(lambda_min))
    if lambda_max is not None:
        mask &= (wl_nm <= float(lambda_max))

    if not np.any(mask):
        raise ValueError(
            "No data points remain after applying wavelength window.")
    f_win, er_win, ei_win = f[mask], eps_r[mask], eps_i[mask]

    # sanity: need at least a few points to fit
    if f_win.size < max(6, 3*n_poles):
        # heuristic: want more points than parameters
        raise ValueError(f"Not enough data ({f_win.size}) for {n_poles} poles; "
                         f"need at least ~{max(6, 3*n_poles)} points.")

    params = _fit_lorentz(f_win, er_win, ei_win, n_poles)
    return _build_meep_medium_from_params(params)
