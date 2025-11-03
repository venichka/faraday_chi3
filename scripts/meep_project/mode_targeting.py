# ========= mode_targeting.py (mirror + cavity optimizer; no spacers) =========
import numpy as np
import meep as mp
from math import isfinite

# ---- Targets (μm) ----
TARGET_WL = np.array([0.800, 1.550, 1.650])      # 800, 1550, 1650 nm
FSR_TARGET = 0.100                                # 100 nm near 1.6 μm
ALPHA_FSR = 2.0                                   # weight for FSR penalty
# gentle DBR quarter-wave regularizer
ALPHA_QW = 0.05

# ---- Spectrum setup (covers 0.6–2.0 μm) ----
wl_min, wl_max = 0.6, 2.0
fmin, fmax = 1/wl_max, 1/wl_min
fcen, df = 0.5*(fmin+fmax), (fmax-fmin)
nfreq = 1200

# --- NEW: mirror-baseline constraints near targets ---
R_MIN = 0.80                 # desired baseline reflectivity floor
BASELINE_WINDOW_NM = 25.0    # +/- window around each target (nm)
BASELINE_PERCENTILE = 0.85   # how we define "baseline" in the window
ALPHA_RBASE = 10.0           # penalty weight if baseline < R_MIN
HARD_REJECT_BELOW = 0.55     # optional early reject if really bad mirrors

LAMBDA0_COMB   = 0.800        # um, put the grid center near 800 nm
ALPHA_QW       = 5.0          # weight for quarter-wave centering
ALPHA_MID      = 8.0          # weight for comb midpoint alignment
MID_WINDOW_NM  = 150.0        # search ± window for dips around lambda0 (nm)


# helper: get material indices (so J_QW can be computed)
def _n_of_medium(m):
    try:
        return float(m.index)
    except Exception:
        return 1.0

# ---- Geometry builder: mirrors only + cavity (no spacers) ----


def build_stack_from_params(N_per, t_SiN, t_SiO2, L_cav,
                            dpml, pad_air, pad_sub, mat_SiN, mat_SiO2, cell_z):
    """Construct 1D stack: air pad, (SiN,SiO2)^N_per, cavity(SiN), (SiO2,SiN)^N_per, SiO2 substrate pad."""
    geom = []
    z = -0.5*cell_z + dpml

    def add(thk, mat):
        nonlocal z
        c = z + 0.5*thk
        geom.append(mp.Block(center=mp.Vector3(0, 0, c),
                             size=mp.Vector3(mp.inf, mp.inf, thk),
                             material=mat))
        z += thk

    # left air spacer
    z += pad_air
    # LEFT DBR → ends with SiO2 against cavity
    for _ in range(N_per):
        add(t_SiN,  mat_SiN)
        add(t_SiO2, mat_SiO2)
    # CAVITY (SiN)
    add(L_cav, mat_SiN)
    # RIGHT DBR → starts with SiO2 against cavity
    for _ in range(N_per):
        add(t_SiO2, mat_SiO2)
        add(t_SiN,  mat_SiN)
    # SiO2 substrate spacer
    add(pad_sub, mat_SiO2)
    return geom

# ---- Reflectance spectrum with proper 2-run normalization (non-negative) ----


def reflectance_spectrum(cell, geometry, resolution, dpml, fcen, df, nfreq):
    src_z = -0.5*cell.z + dpml + 0.2
    refl_z = src_z + 0.1
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ex, center=mp.Vector3(0, 0, src_z))]

    # reference (no structure)
    sim_ref = mp.Simulation(cell_size=cell, boundary_layers=[mp.PML(dpml)],
                            default_material=mp.air, resolution=resolution,
                            dimensions=1, sources=src)
    refl_ref = sim_ref.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, refl_z)))
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(0, 0, refl_z), 1e-8))
    ref_data = sim_ref.get_flux_data(refl_ref)
    Inc = np.array(mp.get_fluxes(refl_ref))
    freqs = np.array(mp.get_flux_freqs(refl_ref))

    # structure
    sim = mp.Simulation(cell_size=cell, geometry=geometry,
                        boundary_layers=[mp.PML(dpml)], default_material=mp.air,
                        resolution=resolution, dimensions=1, sources=src)
    refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
        center=mp.Vector3(0, 0, refl_z)))
    sim.load_minus_flux_data(refl, ref_data)  # subtract incident fields
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(0, 0, refl_z), 1e-8))
    R_flux = np.array(mp.get_fluxes(refl))
    R = np.maximum(0.0, -R_flux / Inc)        # reflectance ≥ 0
    return freqs, R

# ---- Score: dips at targets + FSR penalty; small DBR quarter-wave regularizer at 1.6 μm ----


def score_for_params(theta, N_per,
                     dpml, pad_air, pad_sub, resolution,
                     mat_SiN, mat_SiO2, n_SiN, n_SiO2):
    L_cav, t_SiN, t_SiO2, cell_margin = theta
    if L_cav <= 0 or t_SiN <= 0 or t_SiO2 <= 0 or cell_margin < 0:
        return 1e3

    # cell size from geometry span + margins
    stack_len = pad_air + N_per*(t_SiN + t_SiO2) + \
        L_cav + N_per*(t_SiO2 + t_SiN) + pad_sub
    cell_z = stack_len + 2*dpml + cell_margin
    cell = mp.Vector3(0, 0, cell_z)

    geom = build_stack_from_params(N_per, t_SiN, t_SiO2, L_cav,
                                   dpml, pad_air, pad_sub, mat_SiN, mat_SiO2, cell_z)

    freqs, R = reflectance_spectrum(
        cell, geom, resolution, dpml, fcen, df, nfreq)
    wl = 1.0/freqs

    # interpolate R at targets
    R_targets = np.interp(TARGET_WL, wl[::-1], R[::-1], left=1.0, right=1.0)

    # --- NEW: baseline reflectivity constraints near each target ---
    baseline_pen = 0.0
    win = BASELINE_WINDOW_NM * 1e-3  # nm -> um
    for lam0 in TARGET_WL:
        mask = (wl >= lam0 - win) & (wl <= lam0 + win)
        if np.count_nonzero(mask) >= 6:
            R_local = R[mask]
            # define a "baseline" as a high percentile within the window
            R_base = float(np.quantile(R_local, BASELINE_PERCENTILE))
            if R_base < HARD_REJECT_BELOW:
                return 1e3  # early kill: mirrors clearly too weak here
            if R_base < R_MIN:
                baseline_pen += (R_MIN - R_base)**2
        else:
            # if sampling too sparse, add gentle penalty to encourage refinement
            baseline_pen += 0.05

    # FSR near 1.6 μm: pick two deepest minima in [1.45,1.75] μm
    mask = (wl > 1.45) & (wl < 1.75)
    w_local, R_local = wl[mask], R[mask]
    if w_local.size >= 20:
        try:
            from scipy.ndimage import minimum_filter1d
            Rmin = minimum_filter1d(R_local, size=9, mode='nearest')
        except Exception:
            Rmin = R_local
        idx = np.argsort(Rmin)[:6]
        ws = np.sort(w_local[idx])
        FSR_pen = 0.0
        if ws.size >= 2:
            dws = np.diff(ws)
            FSR_pen = (np.min(np.abs(dws - FSR_TARGET))**2)
    else:
        FSR_pen = 0.1

    # DBR quarter-wave proximity (centered at 1.6 μm helps mirror R there)
    lam0 = 1.6
    qw_err = ((n_SiN*t_SiN - lam0/4.0)**2 +
              (n_SiO2*t_SiO2 - lam0/4.0)**2) / (lam0**2)

    J = float(np.sum(R_targets) + ALPHA_FSR*FSR_pen# +
              # ALPHA_RBASE*baseline_pen +
              # ALPHA_QW*qw_err
              )
    return J if isfinite(J) else 1e3

# ---- Optimizer: Nelder–Mead with bounds ----


def optimize_geometry_mirrors(N_per,
                              dpml, pad_air, pad_sub, resolution,
                              mat_SiN, mat_SiO2, n_SiN, n_SiO2,
                              L0=6.4, tH0=0.20, tL0=0.28, margin0=0.4,
                              bounds=((1.0, 4.0),   # L_cav
                                      (0.05, 0.50),  # t_SiN  (μm)
                                      (0.05, 0.60),  # t_SiO2 (μm)
                                      (0.2,  1.0)),  # cell_margin
                              maxiter=50):
    """
    Returns best (L_cav, t_SiN, t_SiO2, cell_margin), best_score
    """
    try:
        from scipy.optimize import minimize
        x0 = np.array([L0, tH0, tL0, margin0])

        def proj(x):
            out = np.array(x, dtype=float)
            for i, (lo, hi) in enumerate(bounds):
                out[i] = min(max(out[i], lo), hi)
            return out

        def fun(x):
            x = proj(x)
            return score_for_params(x, N_per,
                                    dpml, pad_air, pad_sub, resolution,
                                    mat_SiN, mat_SiO2, n_SiN, n_SiO2)

        res = minimize(fun, x0, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=5e-5, fatol=5e-6, disp=True))
        xbest = proj(res.x)
        fbest = fun(xbest)
        return xbest, fbest
    except Exception as e:
        print("Optimizer failed or SciPy missing:", e)
        # fallback grid
        L_grid = np.linspace(bounds[0][0], bounds[0][1], 8)
        tH_grid = np.linspace(bounds[1][0], bounds[1][1], 6)
        tL_grid = np.linspace(bounds[2][0], bounds[2][1], 6)
        m_grid = np.linspace(bounds[3][0], bounds[3][1], 3)
        bestJ, bestx = 1e9, None
        for L in L_grid:
            for tH in tH_grid:
                for tL in tL_grid:
                    for m in m_grid:
                        x = np.array([L, tH, tL, m])
                        J = score_for_params(x, N_per,
                                             dpml, pad_air, pad_sub, resolution,
                                             mat_SiN, mat_SiO2, n_SiN, n_SiO2)
                        if J < bestJ:
                            bestJ, bestx = J, x
        return bestx, bestJ

# ========= end module =========
