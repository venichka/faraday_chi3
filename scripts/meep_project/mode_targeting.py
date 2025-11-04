# ========= mode_targeting.py (mirror + cavity optimizer; no spacers) =========
import numpy as np
import meep as mp
from math import isfinite, log
from copy import deepcopy
from typing import Dict, List, Any, Optional, Sequence, Tuple

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

# Reflectance dip shaping (local minima near targets)
DIP_WINDOW_NM = dict(probe=25.0, pump1=35.0, pump2=35.0)
ALPHA_DIP_DETUNE = dict(probe=40.0, pump1=24.0, pump2=24.0)
ALPHA_DIP_DEPTH = dict(probe=12.0, pump1=8.0, pump2=8.0)
DIP_TARGET_R = dict(probe=0.10, pump1=0.12, pump2=0.12)

# Baseline smoothing kernel length (number of samples)
BASELINE_SMOOTH_LEN = 9

# Spectrum for R(λ) (covers 0.6–2.0 μm)
wl_min, wl_max = 0.6, 2.0
fmin, fmax = 1/wl_max, 1/wl_min
fcen, df = 0.5*(fmin+fmax), (fmax-fmin)
nfreq = 1200

# Harminv controls (time after source-off to ring down)
HARMINV_RUN_TIME = 600
SRC_AMPLITUDE = 1.0


# ------------------------------ Utilities -------------------------------------
def _instantiate_medium(candidate) -> Optional[mp.Medium]:
    if candidate is None:
        return None
    if isinstance(candidate, mp.Medium):
        return deepcopy(candidate)
    if callable(candidate):
        maybe = candidate()
        if isinstance(maybe, mp.Medium):
            return maybe
    return None


def _material_from_library(names: Sequence[str]) -> Optional[mp.Medium]:
    try:
        from meep import materials  # type: ignore
    except Exception:
        return None
    for name in names:
        candidate = getattr(materials, name, None)
        mat = _instantiate_medium(candidate)
        if mat is not None:
            return mat
    return None


def _linearize_medium(mat: mp.Medium) -> mp.Medium:
    """Ensure no nonlinear terms are present (keep optimization purely linear)."""
    if hasattr(mat, "chi2"):
        mat.chi2 = 0.0
    if hasattr(mat, "chi3"):
        mat.chi3 = 0.0
    return mat


def get_cavity_materials(model: str = "library",
                         index_high: float = 2.0,
                         index_low: float = 1.45) -> Tuple[mp.Medium, mp.Medium]:
    """
    Return (SiN, SiO2) media according to the requested model:
      • "library": try mp.materials.Si3N4_NIR and mp.materials.SiO2 (fallback to constants)
      • "constant": simple lossless media with provided indices.
    Nonlinear susceptibilities are cleared to keep the response linear.
    """
    model_lc = (model or "").lower()
    if model_lc == "library":
        mat_sin = _material_from_library(["Si3N4_NIR", "SiN"])
        mat_sio2 = _material_from_library(["SiO2", "FusedSilica", "Silica"])
        if mat_sin is None:
            mat_sin = mp.Medium(index=index_high)
        if mat_sio2 is None:
            mat_sio2 = mp.Medium(index=index_low)
    elif model_lc == "constant":
        mat_sin = mp.Medium(index=index_high)
        mat_sio2 = mp.Medium(index=index_low)
    else:
        raise ValueError(f"Unknown material model '{model}'. Expected 'library' or 'constant'.")
    return _linearize_medium(mat_sin), _linearize_medium(mat_sio2)


def material_index_at_wavelength(material: mp.Medium,
                                 wavelength_um: float) -> float:
    """Approximate the refractive index at wavelength λ (μm) for quarter-wave penalties."""
    if wavelength_um <= 0:
        return 1.0
    freq = 1.0 / wavelength_um
    if hasattr(material, "epsilon"):
        try:
            eps_val = material.epsilon(freq)
            eps_real = float(np.real(eps_val))
            if np.isfinite(eps_real) and eps_real > 1e-9:
                return float(np.sqrt(eps_real))
        except Exception:
            pass
    idx = getattr(material, "index", None)
    if idx is not None:
        try:
            n = float(idx)
            if np.isfinite(n) and n > 0:
                return n
        except Exception:
            pass
    return 1.0


def _n_of_medium(material, wavelength_um: float = LAM_PROBE) -> float:
    return material_index_at_wavelength(material, wavelength_um)


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
                      w_detune, q_min, w_qpen, w_qreward):
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
        return float(w_detune + w_qpen*1.0), dict(
            found=False, lam=None, detune_um=None, detune_nm=None, Q=None, penalty=float(w_detune + w_qpen*1.0))

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
        Qmin = max(q_min, 1.0)
        qdef = max(0.0, (log(Qmin) - log(Q))/log(Qmin))
        qpen = w_qpen * (qdef*qdef)
        qrew = - w_qreward * min(1.0, max(0.0, (Q - Qmin)/Qmin))

    penalty = float(detune + qpen + qrew)
    return penalty, dict(found=True,
                         lam=float(lam),
                         detune_um=float(lam - lam_target_um),
                         detune_nm=float(lam - lam_target_um)*1e3,
                         Q=float(Q),
                         err=float(err) if err is not None else None,
                         penalty=penalty,
                         detune_term=float(detune),
                         q_penalty=float(qpen),
                         q_reward=float(qrew))


def modal_objective(cell, geometry, resolution, dpml):
    """Sum modal penalties for probe(800), pump1(1550), pump2(1650)."""
    J = 0.0
    diag = {}
    for key, lam0 in (("probe", LAM_PROBE), ("pump1", LAM_PUMP1), ("pump2", LAM_PUMP2)):
        f0 = 1.0/lam0
        df = BAND_FRAC[key] * f0
        modes = _harminv_modes_for_window(
            cell, geometry, resolution, dpml, f0, df)
        half_window = BAND_FRAC[key] * lam0
        term, d = _score_one_target(
            modes, lam0, half_window,
            ALPHA_DETUNE[key], Q_MIN[key],
            ALPHA_QPEN[key], ALPHA_QREWARD[key])
        d["half_window_um"] = float(half_window)
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
    diag: Dict[float, Dict[str, Any]] = {}
    for lam0 in TARGET_WL:
        entry: Dict[str, Any] = dict(target=float(lam0))
        m = (wl >= lam0 - win_um) & (wl <= lam0 + win_um)
        entry["samples"] = int(np.count_nonzero(m))
        if entry["samples"] >= 6:
            Rloc = np.asarray(R[m], dtype=float)
            baseline_series = None
            try:
                from scipy.signal import savgol_filter  # type: ignore

                win_len = len(Rloc) if len(Rloc) % 2 == 1 else len(Rloc) - 1
                if win_len < 5:
                    raise ValueError("window too small")
                baseline_series = savgol_filter(
                    Rloc, win_len, 2, mode="interp")
            except Exception:
                k = min(BASELINE_SMOOTH_LEN, len(Rloc))
                if k % 2 == 0:
                    k = max(1, k - 1)
                if k >= 3:
                    kernel = np.ones(k, dtype=float) / float(k)
                    baseline_series = np.convolve(
                        Rloc, kernel, mode="same")
                else:
                    baseline_series = Rloc
            baseline_series = np.asarray(baseline_series, dtype=float)
            Rb = float(np.quantile(
                baseline_series, BASELINE_PERCENTILE))
            entry["baseline"] = Rb
            if Rb < HARD_REJECT_BELOW:   # early kill
                entry["hard_reject"] = True
                diag[lam0] = entry
                return 1e3, diag
            if Rb < R_MIN:
                deficit = (R_MIN - Rb)**2
                pen += deficit
                entry["penalty"] = deficit
            else:
                entry["penalty"] = 0.0
        else:
            pen += 0.05
            entry["penalty"] = 0.05
            entry["insufficient"] = True
        diag[lam0] = entry
    return pen, diag


def _fsr_penalty(wl, R):
    """Encourage ~100 nm spacing near 1.6 μm using deepest local minima."""
    mask = (wl > 1.45) & (wl < 1.75)
    w, y = wl[mask], R[mask]
    diag = dict(samples=int(w.size))
    if w.size < 20:
        diag["reason"] = "insufficient_samples"
        return 0.1, diag
    try:
        from scipy.ndimage import minimum_filter1d  # type: ignore

        ysm = minimum_filter1d(y, size=9, mode='nearest')
    except Exception:
        ysm = y
    idx = np.argsort(ysm)[:6]
    ws = np.sort(w[idx])
    if ws.size >= 2:
        dws = np.diff(ws)
        spacing = float(np.min(np.abs(dws - FSR_TARGET)))
        penalty = spacing**2
        diag.update(dict(
            min_spacing_um=float(np.min(dws)),
            penalty=penalty,
            target_spacing_um=float(FSR_TARGET)
        ))
        return penalty, diag
    diag["reason"] = "insufficient_minima"
    return 0.1, diag


def _comb_midpoint_penalty(wl, R, lam0=LAM_PROBE):
    """Find dips around lam0; pull their midpoint to lam0 (uses find_peaks(-R))."""
    win_um = MID_WINDOW_NM * 1e-3
    diag = dict(target=float(lam0))
    m = (wl >= lam0 - win_um) & (wl <= lam0 + win_um)
    if np.count_nonzero(m) < 8:
        diag["reason"] = "insufficient_samples"
        return 0.2, diag
    try:
        from scipy.signal import find_peaks  # type: ignore

        wloc, Rloc = wl[m], R[m]
        peaks, _ = find_peaks(-Rloc)  # dips of R
        if peaks.size >= 2:
            below = wloc[peaks][wloc[peaks] <= lam0]
            above = wloc[peaks][wloc[peaks] >= lam0]
            if below.size and above.size:
                lam_low, lam_high = below[-1], above[0]
                lam_mid = 0.5*(lam_low + lam_high)
                penalty = ALPHA_MID * (lam_mid - lam0)**2
                diag.update(dict(
                    found=True,
                    lam_low=float(lam_low),
                    lam_high=float(lam_high),
                    midpoint=float(lam_mid),
                    penalty=penalty
                ))
                return penalty, diag
    except Exception as exc:
        diag["exception"] = repr(exc)
    diag["reason"] = "peaks_not_found"
    return 0.2, diag


def _dip_penalty(wl, R):
    """Locate nearest reflectance dip around each target and penalize detune/depth."""
    total = 0.0
    diag: Dict[str, Dict[str, Any]] = {}
    for key, lam0 in (("probe", LAM_PROBE), ("pump1", LAM_PUMP1), ("pump2", LAM_PUMP2)):
        win_um = DIP_WINDOW_NM[key] * 1e-3
        m = (wl >= lam0 - win_um) & (wl <= lam0 + win_um)
        entry: Dict[str, Any] = dict(target=float(lam0),
                                     window_um=float(win_um))
        entry["samples"] = int(np.count_nonzero(m))
        if entry["samples"] < 6:
            entry["found"] = False
            entry["reason"] = "insufficient_samples"
            penalty = 0.5
            total += penalty
            entry["penalty"] = penalty
            diag[key] = entry
            continue
        wloc = np.asarray(wl[m], dtype=float)
        Rloc = np.asarray(R[m], dtype=float)
        try:
            from scipy.signal import find_peaks  # type: ignore

            peaks, _ = find_peaks(-Rloc)
        except Exception:
            peaks = np.empty(0, dtype=int)
        if peaks.size == 0:
            idx = int(np.argmin(Rloc))
            entry["reason"] = "fallback_min"
        else:
            idx = int(peaks[np.argmin(Rloc[peaks])])
        lam_dip = float(wloc[idx])
        R_dip = float(Rloc[idx])
        detune_um = lam_dip - lam0
        det_norm = detune_um / win_um if win_um > 0 else 0.0
        depth_def = max(0.0, R_dip - DIP_TARGET_R[key])
        detune_pen = ALPHA_DIP_DETUNE[key] * det_norm * det_norm
        depth_pen = ALPHA_DIP_DEPTH[key] * (depth_def ** 2)
        penalty = detune_pen + depth_pen
        total += penalty
        entry.update(dict(
            found=True,
            lam=float(lam_dip),
            R=float(R_dip),
            detune_um=float(detune_um),
            detune_nm=float(detune_um*1e3),
            detune_pen=float(detune_pen),
            depth_pen=float(depth_pen),
            penalty=float(penalty)
        ))
        diag[key] = entry
    return total, diag


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
    J_modes, diag_modes = modal_objective(cell, geom, resolution, dpml)

    # 2) Reflectance spectrum & penalties (baseline + FSR + comb midpoint)
    freqs, R = reflectance_spectrum(
        cell, geom, resolution, dpml, fcen, df, nfreq)
    wl = 1.0/freqs
    R_targets = np.interp(TARGET_WL, wl[::-1], R[::-1], left=1.0, right=1.0)
    dip_pen, dip_diag = _dip_penalty(wl, R)
    baseline_pen, baseline_diag = _baseline_penalty(wl, R)
    FSR_pen, fsr_diag = _fsr_penalty(wl, R)
    J_mid, mid_diag = _comb_midpoint_penalty(wl, R, lam0=LAM_PROBE)

    # 3) Quarter-wave regularizer at 800 nm (centers DBR stopband)
    nH, nL = _n_of_medium(mat_SiN), _n_of_medium(mat_SiO2)
    J_QW = ALPHA_QW * (((nH*t_SiN - LAM_PROBE/4.0)**2 +
                        (nL*t_SiO2 - LAM_PROBE/4.0)**2) / LAM_PROBE**2)

    # combine
    if baseline_pen >= 1e3:
        total = float(baseline_pen)
    else:
        total = float(
            J_modes +
            dip_pen +
            ALPHA_RBASE*baseline_pen +
            ALPHA_FSR*FSR_pen +
            J_mid + J_QW
        )

    diagnostics = dict(
        theta=dict(t_SiN=float(t_SiN), t_SiO2=float(t_SiO2),
                   L_cav=float(L_cav), cell_margin=float(cell_margin)),
        total=total,
        modes=diag_modes,
        dips=dip_diag,
        baseline=baseline_diag,
        fsr=fsr_diag,
        comb_mid=mid_diag,
        quarter_wave=dict(
            penalty=float(J_QW),
            n_high=float(nH),
            n_low=float(nL)
        ),
        reflectance=dict(
            targets=TARGET_WL.tolist(),
            values=R_targets.tolist()
        )
    )

    # Pareto bookkeeping (min detune, maximize min Q)
    target_map = dict(probe=LAM_PROBE, pump1=LAM_PUMP1, pump2=LAM_PUMP2)
    detune_sum = 0.0
    min_Q = float('inf')
    for key, entry in diag_modes.items():
        if entry.get("found"):
            detune_sum += abs(entry.get("detune_um", 0.0))
            q_val = entry.get("Q", 0.0) if entry.get("Q") is not None else 0.0
            if q_val > 0:
                min_Q = min(min_Q, q_val) if min_Q != float('inf') else q_val
        else:
            detune_sum += BAND_FRAC[key] * target_map[key]
            min_Q = min(min_Q, 0.0)
    if min_Q == float('inf'):
        min_Q = 0.0

    record = dict(theta=np.array(theta, dtype=float),
                  J=float(total),
                  detune=float(detune_sum),
                  min_Q=float(min_Q),
                  diagnostics=diagnostics)

    def _dominates(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Return True if record a Pareto-dominates record b."""
        better_detune = a["detune"] <= b["detune"]
        better_q = a["min_Q"] >= b["min_Q"]
        strictly = (a["detune"] < b["detune"]) or (a["min_Q"] > b["min_Q"])
        return better_detune and better_q and strictly

    pareto: List[Dict[str, Any]] = getattr(score_for_params, "_pareto", [])
    if not any(_dominates(r, record) for r in pareto):
        pareto = [r for r in pareto if not _dominates(record, r)]
        pareto.append(record)
    score_for_params._pareto = pareto
    history: List[Dict[str, Any]] = getattr(score_for_params, "history", [])
    history.append(record)
    if len(history) > 200:
        history = history[-200:]
    score_for_params.history = history
    score_for_params.last_eval = diagnostics

    return total if isfinite(total) else 1e3


# ------------------------------- Optimizer ------------------------------------
def optimize_geometry_mirrors(N_per,
                              dpml, pad_air, pad_sub, resolution,
                              mat_SiN, mat_SiO2,
                              tH0=0.260, tL0=0.160, L0=1.543, margin0=0.4,
                              bounds=((0.05, 0.60),  # t_SiN  (μm)
                                      (0.05, 0.60),  # t_SiO2 (μm)
                                      (0.50, 8.00),  # L_cav  (μm)
                                      (0.20, 1.00)),  # cell_margin
                              maxiter=60,
                              n_restarts=3,
                              random_seed=None):
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
        bounds = tuple(bounds)
        rng = np.random.default_rng(random_seed)
        starts = [np.array([tH0, tL0, L0, margin0], dtype=float)]
        for _ in range(1, max(1, n_restarts)):
            perturb = np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)
            starts.append(perturb)

        def proj(x):
            y = np.array(x, dtype=float)
            for i, (lo, hi) in enumerate(bounds):
                y[i] = min(max(y[i], lo), hi)
            return y

        def fun(x):
            x_proj = proj(x)
            return score_for_params(x_proj, N_per, dpml, pad_air, pad_sub, resolution, mat_SiN, mat_SiO2)

        best_record: Dict[str, Any] = {}
        restart_summaries: List[Dict[str, Any]] = []

        for idx, x0 in enumerate(starts):
            res = minimize(fun, x0, method="Nelder-Mead",
                           options=dict(maxiter=maxiter, xatol=5e-3, fatol=5e-4, disp=False))
            xbest = proj(res.x)
            cost = float(res.fun)
            diagnostics = getattr(score_for_params, "last_eval", None)
            pareto_snapshot = getattr(score_for_params, "_pareto", [])
            restart_summary = dict(
                restart=idx,
                start=x0.tolist(),
                xbest=xbest.tolist(),
                cost=cost,
                diagnostics=diagnostics,
                pareto=[dict(theta=r["theta"].tolist(), J=r["J"],
                             detune=r["detune"], min_Q=r["min_Q"])
                        for r in pareto_snapshot]
            )
            restart_summaries.append(restart_summary)
            if not best_record or cost < best_record["cost"]:
                best_record = dict(
                    theta=xbest,
                    cost=cost,
                    diagnostics=diagnostics,
                    restart=idx,
                    optimizer_result=res)

            # restart Nelder-Mead around best found so far
            if idx + 1 < len(starts) and best_record:
                starts[idx + 1] = proj(
                    best_record["theta"] + 0.15 * (starts[idx + 1] - best_record["theta"]))

            if diagnostics and isinstance(diagnostics, dict):
                dip_summary = diagnostics.get("dips", {})
                detune_nm = sum(abs(v.get("detune_nm", 0.0))
                                for v in dip_summary.values()
                                if isinstance(v, dict) and v.get("detune_nm") is not None)
                modes_diag = diagnostics.get("modes", {}) if isinstance(
                    diagnostics.get("modes"), dict) else {}
                probe_entry = modes_diag.get("probe", {}) if isinstance(
                    modes_diag, dict) else {}
                probe_Q = probe_entry.get("Q") if isinstance(
                    probe_entry, dict) else None
                print(f"[restart {idx}] cost={cost:.5f}, Σ|detune|≈{detune_nm:.2f} nm, "
                      f"min_Q_probe={probe_Q if probe_Q is not None else 'n/a'}")

        optimize_geometry_mirrors.last_result = dict(
            best=best_record,
            restarts=restart_summaries,
            pareto=getattr(score_for_params, "_pareto", []),
            history=getattr(score_for_params, "history", [])
        )
        if not best_record:
            return None, None
        return best_record.get("theta"), best_record.get("cost")

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
        optimize_geometry_mirrors.last_result = dict(
            best=dict(theta=bestx, cost=bestJ,
                      diagnostics=getattr(score_for_params, "last_eval", None),
                      restart="grid"),
            restarts=[],
            pareto=getattr(score_for_params, "_pareto", []),
            history=getattr(score_for_params, "history", [])
        )
        return bestx, bestJ

# ========= end module =========
