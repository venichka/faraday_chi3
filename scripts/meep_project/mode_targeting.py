# ========= mode_targeting.py (mirror + cavity optimizer; no spacers) =========
import numpy as np
import meep as mp
from math import isfinite

# ------------------ Targets & weights (all wavelengths in μm) ------------------
LAM_PROBE = 0.800
LAM_PUMP1 = 1.550
LAM_PUMP2 = 1.650
TARGET_WL = np.array([LAM_PROBE, LAM_PUMP1, LAM_PUMP2])

# Harminv search window (fraction of f0) and Q-requirements
BAND_FRAC = dict(probe=0.06, pump1=0.15, pump2=0.15)   # ±6% around target
Q_MIN = dict(probe=50.0, pump1=30.0, pump2=30.0)

# Modal objective weights: strong lock at probe, softer at pumps
ALPHA_DETUNE = dict(probe=80.0, pump1=25.0, pump2=25.0)   # multiplies (Δλ)^2
ALPHA_QPEN = dict(probe=10.0, pump1=6.0,  pump2=6.0)    # penalty if Q < Qmin
# gentle reward for high Q
ALPHA_QREWARD = dict(probe=1.0,  pump1=0.6,  pump2=0.6)

# Reflectance baseline floor near each target (stopband must be high-R)
R_MIN = 0.80       # desired floor
BASELINE_WINDOW_NM = 25.0       # ± window (nm) around each target
BASELINE_PERCENTILE = 0.85       # high-percentile used as local baseline
ALPHA_RBASE = 10.0       # weight for baseline deficit
HARD_REJECT_BELOW = 0.55       # early reject if baseline catastrophically low

# DBR regularizers around 800 nm: quarter-wave + local comb midpoint
ALPHA_QW = 5.0                  # quarter-wave centering weight
ALPHA_MID = 8.0                  # comb-midpoint weight
MID_WINDOW_NM = 150.0             # search window for two dips to bracket 800 nm

# FSR shaping near ~1.6 μm
FSR_TARGET = 0.100                # ~100 nm around 1.6 μm
ALPHA_FSR = 2.0

# Spectrum for R(λ) (covers 0.6–2.0 μm)
wl_min, wl_max = 0.6, 2.0
fmin, fmax = 1/wl_max, 1/wl_min
fcen, df = 0.5*(fmin+fmax), (fmax-fmin)
nfreq = 1200

# Harminv controls (time after source-off to ring down)
HARMINV_RUN_TIME = 600
SRC_AMPLITUDE = 1.0


# ------------------------------ Utilities -------------------------------------
def _n_of_medium(m):
    try:
        return float(m.index)
    except Exception:
        return 1.0


def _harminv_modes_for_window(cell, geometry, resolution, dpml,
                              f_center, f_width, mon_pt=mp.Vector3(),
                              component=mp.Ex, src_z=None):
    """Run a single-window Harminv: return list of (freq, Q, err)."""
    if src_z is None:
        src_z = -0.5*cell.z + dpml + 0.2
    src = [mp.Source(mp.GaussianSource(f_center, fwidth=f_width),
                     component=component,
                     center=mp.Vector3(0, 0, src_z),
                     amplitude=SRC_AMPLITUDE)]
    sim = mp.Simulation(cell_size=cell, geometry=geometry,
                        boundary_layers=[mp.PML(dpml)],
                        default_material=mp.air,
                        resolution=resolution, dimensions=1, sources=src)
    h = mp.Harminv(component, mon_pt, f_center, f_width)
    sim.run(mp.after_sources(h), until=HARMINV_RUN_TIME)
    out = []
    for m in h.modes:
        out.append((m.freq, m.Q, getattr(m, 'error', 0.0)))
    return out


def _score_one_target(modes, lam_target_um, half_window_um,
                      w_detune, w_qpen, w_qreward):
    # Accept only sane modes
    filtered = []
    f0 = 1.0/lam_target_um
    df = f0 - 1.0/(lam_target_um + half_window_um), f0 + \
        1.0/(lam_target_um - half_window_um)
    f_lo, f_hi = min(df), max(df)
    for (f, Q, err) in modes:
        if not np.isfinite(f) or not np.isfinite(Q) or f <= 0:
            continue
        if not (f_lo <= f <= f_hi):
            continue
        if err is not None and err > 1e-3:
            continue
        filtered.append((f, Q, err))

    if not filtered:
        # bounded fallback: 1 window off in detune, no Q reward
        return float(w_detune + w_qpen*1.0), dict(found=False, lam=None, Q=None)

    # pick closest in wavelength
    lamodes = np.array([1.0/f for (f, _, _) in filtered])
    idx = int(np.argmin(np.abs(lamodes - lam_target_um)))
    f, Q, err = filtered[idx]
    lam = 1.0/f

    # window-normalized & clipped detune
    dnorm = (lam - lam_target_um)/half_window_um
    detune = w_detune * min(1.0, dnorm*dnorm)

    # log-Q penalty/reward (bounded)
    if Q <= 0 or not np.isfinite(Q):
        qpen = w_qpen
        qrew = 0.0
    else:
        from math import log
        Qmin = 1500.0 if np.isclose(lam_target_um, 0.800) else 800.0
        qdef = max(0.0, (log(Qmin) - log(Q))/log(Qmin))
        qpen = w_qpen * (qdef*qdef)
        qrew = - w_qreward * min(1.0, Q/Qmin)

    return float(detune + qpen + qrew), dict(found=True, lam=lam, Q=Q, err=err)


def modal_objective(cell, geometry, resolution, dpml):
    """Sum modal penalties for probe(800), pump1(1550), pump2(1650)."""
    J = 0.0
    diag = {}
    for key, lam0 in (("probe", LAM_PROBE), ("pump1", LAM_PUMP1), ("pump2", LAM_PUMP2)):
        f0 = 1.0/lam0
        df = BAND_FRAC[key] * f0
        modes = _harminv_modes_for_window(
            cell, geometry, resolution, dpml, f0, df)
        term, d = _score_one_target(modes, lam0,
                                    ALPHA_DETUNE[key], Q_MIN[key],
                                    ALPHA_QPEN[key], ALPHA_QREWARD[key])
        J += term
        diag[key] = d
    return J, diag


# -------------------------- Geometry & spectra --------------------------------
def build_stack_from_params(N_per, t_SiN, t_SiO2, L_cav,
                            dpml, pad_air, pad_sub, mat_SiN, mat_SiO2, cell_z):
    """Air pad | (SiN,SiO2)^N | SiN cavity | (SiO2,SiN)^N | SiO2 pad."""
    geom = []
    z = -0.5*cell_z + dpml

    def add(thk, mat):
        nonlocal z
        c = z + 0.5*thk
        geom.append(mp.Block(center=mp.Vector3(0, 0, c),
                             size=mp.Vector3(mp.inf, mp.inf, thk),
                             material=mat))
        z += thk
    z += pad_air
    for _ in range(N_per):
        add(t_SiN, mat_SiN)
        add(t_SiO2, mat_SiO2)
    add(L_cav, mat_SiN)
    for _ in range(N_per):
        add(t_SiO2, mat_SiO2)
        add(t_SiN, mat_SiN)
    add(pad_sub, mat_SiO2)
    return geom


def reflectance_spectrum(cell, geometry, resolution, dpml, fcen, df, nfreq):
    """Two-run normalization: save incident DFT fields then subtract to get R(λ)."""
    src_z = -0.5*cell.z + dpml + 0.2
    refl_z = src_z + 0.1
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ex, center=mp.Vector3(0, 0, src_z))]
    # reference
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
    sim.load_minus_flux_data(refl, ref_data)   # per Meep docs
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(0, 0, refl_z), 1e-8))
    R_flux = np.array(mp.get_fluxes(refl))
    R = np.maximum(0.0, -R_flux / Inc)         # reflectance ≥ 0
    return freqs, R


# --------------------------- Penalty components -------------------------------
def _baseline_penalty(wl, R):
    """High-R stopband near each target: baseline (percentile) ≥ R_MIN."""
    win_um = BASELINE_WINDOW_NM * 1e-3
    pen = 0.0
    for lam0 in TARGET_WL:
        m = (wl >= lam0 - win_um) & (wl <= lam0 + win_um)
        if np.count_nonzero(m) >= 6:
            Rb = float(np.quantile(R[m], BASELINE_PERCENTILE))
            if Rb < HARD_REJECT_BELOW:   # early kill
                return 1e3
            if Rb < R_MIN:
                pen += (R_MIN - Rb)**2
        else:
            pen += 0.05
    return pen


def _fsr_penalty(wl, R):
    """Encourage ~100 nm spacing near 1.6 μm using deepest local minima."""
    mask = (wl > 1.45) & (wl < 1.75)
    w, y = wl[mask], R[mask]
    if w.size < 20:
        return 0.1
    try:
        from scipy.ndimage import minimum_filter1d
        ysm = minimum_filter1d(y, size=9, mode='nearest')
    except Exception:
        ysm = y
    idx = np.argsort(ysm)[:6]
    ws = np.sort(w[idx])
    if ws.size >= 2:
        dws = np.diff(ws)
        return (np.min(np.abs(dws - FSR_TARGET))**2)
    return 0.1


def _comb_midpoint_penalty(wl, R, lam0=LAM_PROBE):
    """Find dips around lam0; pull their midpoint to lam0 (uses find_peaks(-R))."""
    win_um = MID_WINDOW_NM * 1e-3
    m = (wl >= lam0 - win_um) & (wl <= lam0 + win_um)
    if np.count_nonzero(m) < 8:
        return 0.2
    try:
        from scipy.signal import find_peaks
        wloc, Rloc = wl[m], R[m]
        peaks, _ = find_peaks(-Rloc)  # dips of R
        if peaks.size >= 2:
            below = wloc[peaks][wloc[peaks] <= lam0]
            above = wloc[peaks][wloc[peaks] >= lam0]
            if below.size and above.size:
                lam_low, lam_high = below[-1], above[0]
                lam_mid = 0.5*(lam_low + lam_high)
                return ALPHA_MID * (lam_mid - lam0)**2
    except Exception:
        pass
    return 0.2


# ----------------------------- Master objective --------------------------------
def score_for_params(theta, N_per,
                     dpml, pad_air, pad_sub, resolution,
                     mat_SiN, mat_SiO2):
    """
    theta = (t_SiN, t_SiO2, L_cav, cell_margin)
    Returns scalar J (lower is better).
    """
    t_SiN, t_SiO2, L_cav, cell_margin = theta
    if min(t_SiN, t_SiO2, L_cav) <= 0 or cell_margin < 0:
        return 1e3

    # total length & cell
    stack_len = pad_air + N_per*(t_SiN + t_SiO2) + \
        L_cav + N_per*(t_SiO2 + t_SiN) + pad_sub
    cell_z = stack_len + 2*dpml + cell_margin
    cell = mp.Vector3(0, 0, cell_z)

    # geometry
    geom = build_stack_from_params(N_per, t_SiN, t_SiO2, L_cav,
                                   dpml, pad_air, pad_sub, mat_SiN, mat_SiO2, cell_z)

    # 1) Modal locking via Harminv (probe locked, pumps near targets with Q constraints)
    # J_modes, _diag_modes = modal_objective(cell, geom, resolution, dpml)

    # 2) Reflectance spectrum & penalties (baseline + FSR + comb midpoint)
    freqs, R = reflectance_spectrum(
        cell, geom, resolution, dpml, fcen, df, nfreq)
    wl = 1.0/freqs
    # interpolate R at targets
    R_targets = np.interp(TARGET_WL, wl[::-1], R[::-1], left=1.0, right=1.0)
    baseline_pen = _baseline_penalty(wl, R)
    FSR_pen = _fsr_penalty(wl, R)
    J_mid = _comb_midpoint_penalty(wl, R, lam0=LAM_PROBE)

    # 3) Quarter-wave regularizer at 800 nm (centers DBR stopband)
    nH, nL = _n_of_medium(mat_SiN), _n_of_medium(mat_SiO2)
    J_QW = ALPHA_QW * (((nH*t_SiN - LAM_PROBE/4.0)**2 +
                        (nL*t_SiO2 - LAM_PROBE/4.0)**2) / LAM_PROBE**2)

    # combine
    J = float(np.sum(R_targets) + ALPHA_RBASE*baseline_pen +
              ALPHA_FSR*FSR_pen + J_mid + J_QW)
    return J if isfinite(J) else 1e3


# ------------------------------- Optimizer ------------------------------------
def optimize_geometry_mirrors(N_per,
                              dpml, pad_air, pad_sub, resolution,
                              mat_SiN, mat_SiO2,
                              tH0=0.260, tL0=0.160, L0=1.543, margin0=0.4,
                              bounds=((0.05, 0.60),  # t_SiN  (μm)
                                      (0.05, 0.60),  # t_SiO2 (μm)
                                      (0.50, 8.00),  # L_cav  (μm)
                                      (0.20, 1.00)),  # cell_margin
                              maxiter=60):
    """
    Optimize (t_SiN, t_SiO2, L_cav, cell_margin) for:
      • fixed probe mode at 800 nm (high Q),
      • two pump modes near 1550/1650 nm (high Q),
      • high reflectance baseline near targets,
      • sensible FSR near 1.6 μm,
      • quarter-wave centering near 800 nm.
    """
    try:
        from scipy.optimize import minimize
        x0 = np.array([tH0, tL0, L0, margin0])

        def proj(x):
            y = np.array(x, dtype=float)
            for i, (lo, hi) in enumerate(bounds):
                y[i] = min(max(y[i], lo), hi)
            return y

        def fun(x):
            x = proj(x)
            return score_for_params(x, N_per, dpml, pad_air, pad_sub, resolution, mat_SiN, mat_SiO2)

        res = minimize(fun, x0, method="Nelder-Mead",
                       options=dict(maxiter=maxiter, xatol=5e-3, fatol=5e-4, disp=True))
        xbest = proj(res.x)
        fbest = fun(xbest)
        return xbest, fbest
    except Exception as e:
        print("Optimizer failed or SciPy missing:", e)
        # coarse fallback
        tH_grid = np.linspace(bounds[0][0], bounds[0][1], 5)
        tL_grid = np.linspace(bounds[1][0], bounds[1][1], 5)
        L_grid = np.linspace(bounds[2][0], bounds[2][1], 6)
        m_grid = np.linspace(bounds[3][0], bounds[3][1], 3)
        bestJ, bestx = 1e9, None
        for tH in tH_grid:
            for tL in tL_grid:
                for L in L_grid:
                    for m in m_grid:
                        x = np.array([tH, tL, L, m])
                        J = score_for_params(
                            x, N_per, dpml, pad_air, pad_sub, resolution, mat_SiN, mat_SiO2)
                        if J < bestJ:
                            bestJ, bestx = J, x
        return bestx, bestJ

# ========= end module =========
