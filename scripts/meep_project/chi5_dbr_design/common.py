#!/usr/bin/env python
"""Shared configuration and helpers for the chi5 DBR design campaign.

The campaign optimizes a SiN/SiO2 DBR Fabry-Perot cavity for the *physical* all-optical
chi5 Faraday rotation. Two things separate it from every earlier optimizer run in this repo,
and both come from the 2026-08 experimental audit (see chi5_optimization/delay_physics.md and
the measurement-estimators notes):

  1. ESTIMATOR.  Earlier optimizers maximized `probe_rotation_deg.final_relative_deg` -- the
     polarization azimuth averaged over the last few time samples, at ONE pump carrier phase.
     That quantity is (a) not energy-weighted, so it keeps accumulating phase through ring-down
     as the field decays to nothing, and (b) evaluated at the maximum of a coherent carrier
     FRINGE which is ~8x (1D) / ~35x (3D) larger than the rotation and averages to zero.  The
     published 0.137 deg / 1.991 deg headline numbers are that fringe maximum, not the effect.
     Here the objective is `probe_pulse_integrated` (= int|E_V|^2 - int|E_H|^2 by Parseval, the
     balanced-detector observable) averaged over the pump1 carrier phase.

  2. CARRIER AVERAGING.  Delaying pump1 by tau multiplies its field by exp(i w1 tau), so a term
     with n powers of E1 and m of E1* carries exp(i(n-m) w1 tau).  Averaging the STOKES VECTOR
     uniformly over one pump1 optical period T1 annihilates every n != m term and leaves exactly
     the rectified chi3/chi5 response.  N = 4 sub-samples cancels harmonics 1, 2 and 3.
     N = 4 is chosen for robustness, not because k=2 is large: on the pulse-integrated channel the
     fringe is a pure fundamental to within measurement error (ptp/A1 = 1.807..2.007 over
     |tau| <= 100 fs, against the pure-sinusoid bound [sqrt2, 2]; directly measured c2/A1 = 1.0% at
     the fabricated operating point).  But N = 2 cancels only ODD harmonics, so it would rest the
     result on that purity holding for every new geometry, and N = 3 leaks at k = 3 for the same
     cost as N = 4.  (The legacy tail-window channel is NOT pure -- ptp/A1 reaches 2.832.)
     The fringe amplitude is retained as a diagnostic, via the exact discrete Fourier
     projection rather than peak-to-peak (ptp of 4 samples is biased sqrt(2)A..2A).

Fabrication constraints (user, 2026-08-02): SiN/SiO2 only, <= 6 mirror pairs per side,
total stack <= 12 um, every layer >= 80 nm.  These are hard bounds on the search space, and
the stack cap is binding: the validated max|theta| ~ L^+1.2 trend wants a long cavity while
more mirror pairs want thickness, so the two compete inside the 12 um budget.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
sys.path.insert(0, str(MEEP / "chi5_optimization"))

C0_UM_FS = 0.299792458          # um per fs

# ------------------------------------------------------------------ materials / sources --- #
# SiN over SiO2, dispersive 2-pole fits of the measured ellipsometry CSVs (same flags as every
# prior campaign, so numbers stay comparable to best_absolute).
FDTD_FLAGS = ["--materials", "fit", "--sin-fit", "si3n4.csv", "--sio2-fit", "sio2.csv",
              "--fit-poles", "2", "--fit-window", "600", "2000", "--high-index-material", "sin"]
PUMP_INTENSITY = 1e12           # W/cm^2, the reference operating intensity
PROBE_INTENSITY = 5e7           # W/cm^2, weak probe (rotation is probe-intensity independent)

# --- pulse duration -------------------------------------------------------------------- #
# The lab pulses are 100 fs FWHM in INTENSITY (user, 2026-08-02).  faraday_meep_fp_circ's
# `pulse_duration_fs` is NOT that: df_from_pulse_duration sets the Gaussian amplitude width to
# T/(2 ln 2), so the amplitude sigma is T/(2 ln 2) and the INTENSITY FWHM is
#     2 sqrt(ln 2) . T/(2 ln 2) = T / sqrt(ln 2) = 1.2011 T.
# The historical label 100.0 is therefore a 120.1 fs pulse -- 20% too long.  For a true 100 fs
# intensity FWHM the label must be T = 100 sqrt(ln 2) = 83.2555 fs, passed via the additive
# --pulse-duration-fs flag (default 100.0, so every earlier result still reproduces).
PULSE_INTENSITY_FWHM_FS = 100.0
PULSE_LABEL_FS = PULSE_INTENSITY_FWHM_FS * np.sqrt(np.log(2.0))      # 83.2555


def fwidth_of(label_fs: float) -> float:
    """Meep Gaussian `fwidth` (1/um) for a pulse-duration label, matching the simulator
    exactly (faraday_meep_fp_circ.df_from_pulse_duration): width_meep = T/(2 ln 2) * c,
    with c = 0.299792458 um/fs, and fwidth = 1/width_meep."""
    return 1.0 / ((label_fs / (2.0 * np.log(2.0))) * C0_UM_FS)


FWIDTH = fwidth_of(PULSE_LABEL_FS)          # 0.055542 /um  (was 0.046249 at the 100.0 label)
FWIDTH_LEGACY = fwidth_of(100.0)            # for continuity comparisons only

# ------------------------------------------------------------------ operating-point space --- #
# Both probe and pumps are tunable in the lab (user, 2026-08-02).
PROBE_WINDOWS = [(0.790, 0.810), (0.850, 0.950)]     # um
PUMP_BAND = (1.40, 1.95)                             # um
# Delta = f1 - f2, bounded from BOTH sides:
#  * upper -- the pulse-integrated readout sums the probe DFT band freq_probe +- df_probe/2, so
#    the FWM sidebands at +-Delta must sit inside it or the objective silently loses the very
#    signal it measures.  Half-width = FWIDTH/2 = 0.02777; we cap at ~85% of that.  (The true
#    100 fs pulse is 20% broader than the old label, so this is roomier than the 94.9% the
#    delay study ran at -- that setup had no margin.)
#  * lower -- below ~1/3 of the pump bandwidth the two sigma+ sigma- pulses overlap spectrally
#    almost completely and "two pumps split by Delta" stops being an experimentally
#    distinguishable configuration.  Earlier scans railed at Delta -> 0 for exactly this reason.
# We sweep across the interesting range and REPORT the dependence rather than railing.
DELTA_MAX_INBAND = 0.85 * 0.5 * FWIDTH               # 0.02361
DELTA_GRID = [0.010, 0.014, 0.018, 0.023]
DELTA_RANGE = (0.008, 0.0236)

# ------------------------------------------------------------------ fabrication limits --- #
N_PAIRS_MAX = 6                 # mirror pairs per side
N_PAIRS_MIN = 2
T_LAYER_MIN = 0.080             # um, PECVD thickness-control floor
T_LAYER_MAX = 0.600             # um, keeps layers near the first stopband order
STACK_MAX_UM = 12.0             # um, total deposited thickness (mirrors + cavity)
L_CAV_MIN = 1.0
L_CAV_MAX = 9.0

# ------------------------------------------------------------------ carrier averaging --- #
SUBSAMPLES = 4                  # uniform phases over one T1; cancels harmonics 1,2,3
PAD_FS = 25.0                   # fixed common source start-time pad.  Must be > 0: pad = 0 is a
                                # +2.1% outlier (sources turning on exactly at t=0), while every
                                # pad > 0 reproduces to 0.23%.  All historical numbers used pad=0.

RES_1D = 80
DECAY_1D = "1e-4"


# =========================================================================== geometry === #
def build_geometry(base: dict, n_left: int, n_right: int, t_hi: float, t_lo: float,
                   L_cav: float) -> dict:
    """Single-defect DBR with independent left/right pair counts.

    Generalizes chi5_optimization.optimize.build_geometry, which forced n_left == n_right.
    Asymmetric mirrors are a deliberate new degree of freedom: the derivations
    (isotropic_derivation, very_general_derivation) show a cavity whose response is symmetric
    about w_s gives Re[Delta chi] = 0, i.e. PURE ELLIPTICITY AND ZERO NET ROTATION -- net
    rotation requires breaking the w_s +- Delta symmetry.  Unequal mirrors are the most direct
    structural way to do that, and no previous search in this repo had the freedom.

    Keeps the base's material labels, pads and layer ordering (right mirror is the mirror image
    of the left, so the stack reads air | left | cavity | right | substrate).
    """
    g = copy.deepcopy(base)
    hi = g["cavity"]["mat"]
    lo = next(m for m in g["materials"] if m != hi)
    g["cavity"]["L_um"] = float(L_cav)
    left_hi_first = base["mirrors"]["left"][0]["mat"] == hi
    pair = ({"mat": hi, "thk_um": float(t_hi)}, {"mat": lo, "thk_um": float(t_lo)})
    left: List[dict] = []
    right: List[dict] = []
    for _ in range(int(n_left)):
        left += [dict(x) for x in (pair if left_hi_first else pair[::-1])]
    for _ in range(int(n_right)):
        right += [dict(x) for x in (pair[::-1] if left_hi_first else pair)]
    g["mirrors"]["left"] = left
    g["mirrors"]["right"] = right
    return g


def stack_thickness_um(geom: dict) -> float:
    """Total deposited thickness = both mirrors + cavity (the fab budget)."""
    t = float(geom["cavity"]["L_um"])
    for side in ("left", "right"):
        t += sum(float(l["thk_um"]) for l in geom["mirrors"][side])
    return t


def fab_violations(geom: dict) -> List[str]:
    """Empty list iff the geometry is fabricable under the agreed constraints."""
    bad = []
    n_l = len(geom["mirrors"]["left"]) // 2
    n_r = len(geom["mirrors"]["right"]) // 2
    if not (N_PAIRS_MIN <= n_l <= N_PAIRS_MAX) or not (N_PAIRS_MIN <= n_r <= N_PAIRS_MAX):
        bad.append("pairs {}/{} outside [{}, {}]".format(n_l, n_r, N_PAIRS_MIN, N_PAIRS_MAX))
    thin = [float(l["thk_um"]) for side in ("left", "right") for l in geom["mirrors"][side]
            if float(l["thk_um"]) < T_LAYER_MIN]
    if thin:
        bad.append("layer {:.4f} um < {:.3f} um".format(min(thin), T_LAYER_MIN))
    tot = stack_thickness_um(geom)
    if tot > STACK_MAX_UM:
        bad.append("stack {:.3f} um > {:.1f} um".format(tot, STACK_MAX_UM))
    return bad


def geometry_key(geom: dict) -> Tuple:
    """Hashable summary of a geometry, for dedup and labelling."""
    n_l = len(geom["mirrors"]["left"]) // 2
    n_r = len(geom["mirrors"]["right"]) // 2
    hi = geom["cavity"]["mat"]
    t_hi = next(float(l["thk_um"]) for l in geom["mirrors"]["left"] if l["mat"] == hi)
    t_lo = next(float(l["thk_um"]) for l in geom["mirrors"]["left"] if l["mat"] != hi)
    return (n_l, n_r, round(t_hi, 6), round(t_lo, 6), round(float(geom["cavity"]["L_um"]), 6))


def geometry_params(geom: dict) -> Dict[str, float]:
    n_l, n_r, t_hi, t_lo, L = geometry_key(geom)
    return {"n_left": n_l, "n_right": n_r, "t_hi": t_hi, "t_lo": t_lo, "L_cav": L,
            "stack_um": stack_thickness_um(geom)}


# ================================================================ TMM operating point === #
# The TMM engine (chi5_optimization/tmm.py) is used ONLY for what it is validated-accurate at:
# locating cavity modes (f0 to <0.7% vs FDTD) and their loaded Q.  It is NOT used to rank
# geometries or to pick the operating point -- the 2026-06 campaign established that the
# analytic chi5 FoM is anti-correlated with FDTD over geometry, and that FDTD must own the
# operating point.  Here TMM only proposes the candidate probe modes and pump centers.

def _tmm():
    import tmm as _t
    return _t


def probe_modes(geom: dict, sub_label: str = "SiO2",
                windows: Optional[Sequence[Tuple[float, float]]] = None) -> List[dict]:
    """Cavity modes inside the allowed probe windows, best (highest-Q) first.

    NOTE the ordering is by Q, which is NOT the same as by usefulness -- for several of these
    cavities the highest-Q probe mode sits at ~860 nm while the ~800 nm mode gives 3-6x more
    rotation (measured, 8/8 geometries).  Select by `windows`, not by taking the first entry.
    """
    t = _tmm()
    idx, layers = t.index_map(), t.build_layers(geom)
    ms: List[dict] = []
    for lo, hi in (PROBE_WINDOWS if windows is None else windows):
        ms += t.find_modes_in_band(layers, idx, 1.0 / hi, 1.0 / lo, sub_label)
    ms.sort(key=lambda m: m["Q"], reverse=True)
    return ms


def pump_modes(geom: dict, sub_label: str = "SiO2") -> List[dict]:
    """Cavity modes in the pump band, best (highest-Q = highest buildup) first."""
    t = _tmm()
    idx, layers = t.index_map(), t.build_layers(geom)
    ms = t.find_modes_in_band(layers, idx, 1.0 / PUMP_BAND[1], 1.0 / PUMP_BAND[0], sub_label)
    ms.sort(key=lambda m: m["Q"], reverse=True)
    return ms


def pump_centers(geom: dict, n_q: int = 2, n_span: int = 3,
                 sub_label: str = "SiO2") -> List[dict]:
    """Candidate pump centers: the n_q highest-Q modes PLUS n_span modes covering the band.

    WHY NOT JUST TOP-Q.  Every pump-band mode of these cavities has Q ~ 40-130, while the 100 fs
    pump can only resolve Q_cap = f/fwidth ~ 12.  All modes are therefore far broader than the
    pulse, intracavity buildup is saturated (the FDTD decomposition measured it FLAT), and Q
    carries essentially no information about which center is better.  Sorting by it is not just
    uninformative but actively harmful: the fabricated best_absolute has two anomalously high-Q
    modes at 1.75/1.87 um that crowd out its own design point at 1.547 um (Q~46, 5th by Q), so a
    top-2-by-Q grid denies the baseline the operating point it was designed around and makes any
    comparison against it meaningless.

    So we keep a couple of high-Q modes (cheap, occasionally right) and add modes nearest to
    frequencies spread uniformly across the pump band, which treats every geometry alike and
    guarantees the band is covered wherever the good physics happens to be.
    """
    pu = pump_modes(geom, sub_label)
    if not pu:
        return []
    chosen = list(pu[:n_q])
    have = {round(m["freq"], 6) for m in chosen}
    lo, hi = 1.0 / PUMP_BAND[1], 1.0 / PUMP_BAND[0]
    for target in np.linspace(lo, hi, n_span + 2)[1:-1]:
        m = min(pu, key=lambda mm: abs(mm["freq"] - target))
        if round(m["freq"], 6) not in have:
            have.add(round(m["freq"], 6))
            chosen.append(m)
    return chosen


def pump_pairs(geom: dict, max_pairs: int = 4, sub_label: str = "SiO2") -> List[Tuple[float, float]]:
    """(center, Delta) for pairs of pump modes whose spacing lies in DELTA_RANGE.

    This is the configuration the FABRICATED best_absolute actually uses: pump1 and pump2 each
    sit on their OWN cavity mode (0.6573 and 0.6353, Delta = 0.0219), and the center between them
    is not itself a mode.  A grid that only straddles single modes therefore cannot express the
    reference design, which is why it has to be offered separately.  Delta is DERIVED from the
    pair, not taken from the grid.  Ordered by smallest Delta first (the M5 lever)."""
    pu = pump_modes(geom, sub_label)
    out = []
    for i in range(len(pu)):
        for j in range(i + 1, len(pu)):
            f1, f2 = max(pu[i]["freq"], pu[j]["freq"]), min(pu[i]["freq"], pu[j]["freq"])
            d = f1 - f2
            if DELTA_RANGE[0] <= d <= DELTA_RANGE[1]:
                out.append((0.5 * (f1 + f2), d))
    out.sort(key=lambda cd: cd[1])
    return out[:max_pairs]


def operating_points(geom: dict, max_centers: int = 3, max_probes: int = 2,
                     deltas: Optional[Sequence[float]] = None,
                     sub_label: str = "SiO2", span_centers: int = 0,
                     probe_windows: Optional[Sequence[Tuple[float, float]]] = None,
                     max_pairs: int = 0) -> List[Dict[str, float]]:
    """Candidate operating points for the FDTD sweep:
        (top-Q probe modes) x (top-Q resonant pump centers) x (Delta grid).

    Pumps straddle a resonant center, f1,2 = center +- Delta/2 -- the placement the validated
    hybrid Stage B used, and the one the FDTD operating-point diagnostic endorsed ("pumps on
    resonant high-buildup modes, wherever they are, plus small Delta"; the f1+f2 ~ f_probe
    octave criterion was tested and found to be flat wrong).

    Several probe modes are offered because the probe is tunable in the lab, so the geometry is
    not obliged to use its highest-Q mode; restricting to that one can miss the optimum when a
    lower-Q mode places the +-Delta sidebands better (the M5 lever).
    """
    pr = probe_modes(geom, sub_label, windows=probe_windows)
    if span_centers:
        pu = pump_centers(geom, n_q=max_centers, n_span=span_centers, sub_label=sub_label)
    else:
        pu = pump_modes(geom, sub_label)[:max_centers]
    if not pr or not pu:
        return []
    out = []
    for pm in pr[:max_probes]:
        # (a) pumps straddling a single mode, Delta from the grid
        for m in pu:
            for d in (DELTA_GRID if deltas is None else deltas):
                out.append({"probe": float(pm["freq"]), "center": float(m["freq"]),
                            "pump1": float(m["freq"]) + 0.5 * d,
                            "pump2": float(m["freq"]) - 0.5 * d,
                            "delta": float(d), "kind": "straddle",
                            "probe_nm": 1000.0 / float(pm["freq"]),
                            "Q_probe": float(pm["Q"]), "Q_pump": float(m["Q"])})
        # (b) each pump on its own mode, Delta derived -- the fabricated design's configuration
        for c, d in pump_pairs(geom, max_pairs, sub_label) if max_pairs else []:
            out.append({"probe": float(pm["freq"]), "center": float(c),
                        "pump1": float(c) + 0.5 * d, "pump2": float(c) - 0.5 * d,
                        "delta": float(d), "kind": "pair",
                        "probe_nm": 1000.0 / float(pm["freq"]),
                        "Q_probe": float(pm["Q"]), "Q_pump": float("nan")})
    return out


# ====================================================================== FDTD plumbing === #
def modes_json(f_probe: float, f1: float, f2: float) -> dict:
    """cavity_modes.json payload for one operating point."""
    d = f1 - f2
    return {"probe": {"frequency": float(f_probe), "lambda_um": 1.0 / float(f_probe)},
            "pump1": {"frequency": float(f1), "lambda_um": 1.0 / float(f1)},
            "pump2": {"frequency": float(f2), "lambda_um": 1.0 / float(f2)},
            "sidebands": {"frequency_plus": float(f_probe + d),
                          "frequency_minus": float(f_probe - d),
                          "delta_frequency": float(d),
                          "pump_separation_um": abs(1.0 / f2 - 1.0 / f1)}}


def carrier_period_fs(f1: float) -> float:
    """Pump1 optical period T1 = lambda1 / c -- the period the sub-samples must span."""
    return 1.0 / (float(f1) * C0_UM_FS)


def fdtd_cmd(out: Path, res: int, decay: str, tau_fs: float, pad_fs: float,
             ellip_deg: float = 0.0, dim: int = 1,
             pump_intensity: float = PUMP_INTENSITY,
             probe_intensity: float = PROBE_INTENSITY,
             pulse_label_fs: float = PULSE_LABEL_FS,
             extra: Optional[Sequence[str]] = None) -> List[str]:
    """Argument list for one faraday_meep_fp_circ run (used directly in 1D, and to build the
    argument list for the MPI 3D case).  `pulse_label_fs` is the df_from_pulse_duration label,
    NOT the intensity FWHM -- see PULSE_LABEL_FS."""
    cmd = [sys.executable, "faraday_meep_fp_circ.py", "--dim", str(dim), "--mode", "full",
           *FDTD_FLAGS,
           "--geometry-file", str(out / "geometry.json"),
           "--cavity-modes-file", str(out / "cavity_modes.json"),
           "--pulse-duration-fs", "{:.6f}".format(pulse_label_fs),
           "--resolution", str(res), "--decay-threshold", str(decay),
           "--pump-intensity", str(pump_intensity),
           "--probe-intensity", str(probe_intensity),
           "--pump1-delay-fs", "{:.6f}".format(tau_fs),
           "--delay-pad-fs", "{:.6f}".format(pad_fs),
           "--probe-azimuth-deg", "45.0",
           "--probe-ellipticity-deg", "{:.6f}".format(ellip_deg),
           "--pump-imbalance", "1.0",
           "--output-dir", str(out)]
    if extra:
        cmd += list(extra)
    return cmd


def run_case(out: Path, geom: dict, freqs: Dict[str, float], tau_fs: float,
             res: int = RES_1D, decay: str = DECAY_1D, pad_fs: float = PAD_FS,
             ellip_deg: float = 0.0, pump_intensity: float = PUMP_INTENSITY,
             pulse_label_fs: float = PULSE_LABEL_FS,
             extra: Optional[Sequence[str]] = None) -> Path:
    """Run one 1D FDTD case.  Idempotent: an existing faraday_summary.json is left alone, so
    jobs can be farmed across nodes, overlap, and be re-run safely."""
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "faraday_summary.json"
    if summary.exists():
        return out
    json.dump(geom, open(out / "geometry.json", "w"))
    json.dump(modes_json(freqs["probe"], freqs["pump1"], freqs["pump2"]),
              open(out / "cavity_modes.json", "w"))
    cmd = fdtd_cmd(out, res, decay, tau_fs, pad_fs, ellip_deg,
                   pump_intensity=pump_intensity, pulse_label_fs=pulse_label_fs, extra=extra)
    env = dict(os.environ, OMP_NUM_THREADS="1", MPLBACKEND="Agg")
    with open(out / "run.log", "w") as lf:
        subprocess.run(cmd, cwd=str(MEEP), env=env, stdout=lf, stderr=subprocess.STDOUT)
    return out


def read_case(out: Path) -> Optional[Dict[str, float]]:
    """Both estimators from one run.  `S*` are the pulse-integrated (energy-weighted) Stokes
    components -- the lab observable; `legacy` is the tail-window azimuth kept only for
    continuity with the published 0.137 deg."""
    p = Path(out) / "faraday_summary.json"
    if not p.exists():
        return None
    try:
        d = json.load(open(p))
    except Exception:
        return None
    pi = d.get("probe_pulse_integrated") or {}
    if not pi or pi.get("S0") is None:
        return None
    return {"S0": float(pi["S0"]), "S1": float(pi["S1"]),
            "S2": float(pi["S2"]), "S3": float(pi["S3"]),
            "dolp": float(pi.get("dolp", float("nan"))),
            "legacy": float((d.get("probe_rotation_deg") or {}).get("final_relative_deg",
                                                                    float("nan")))}


# ==================================================================== Stokes / averaging === #
def stokes_to_angles(S0, S1, S2, S3):
    """(theta relative to the 45 deg launch azimuth, ellipticity chi, DoLP, normalized V-H).
    Identical to chi5_optimization.delay_physics.stokes_to_angles (validated to 0.0 against the
    simulator's own reported angles, including for an elliptical probe)."""
    theta = 0.5 * np.degrees(np.arctan2(S2, S1)) - 45.0
    theta = (theta + 90.0) % 180.0 - 90.0
    lin = np.sqrt(np.asarray(S1) ** 2 + np.asarray(S2) ** 2)
    chi = 0.5 * np.degrees(np.arctan2(S3, lin))
    return theta, chi, lin / np.maximum(S0, 1e-30), -np.asarray(S1) / np.maximum(S0, 1e-30)


def carrier_average(records: Sequence[Dict[str, float]]) -> Optional[Dict[str, float]]:
    """Average the Stokes vector over uniformly-spaced pump1 carrier phases.

    The sub-samples sit at phases phi_k = 2 pi k / N over one T1, so:
      * the MEAN Stokes vector kills every exp(i(n-m) w1 tau) term with 0 < |n-m| < N, leaving
        the rectified chi3/chi5 response -> `theta_chi5_deg`, the objective;
      * the FUNDAMENTAL fringe amplitude follows from the exact discrete Fourier projection
        2|sum_k y_k exp(-i phi_k)|/N.  Peak-to-peak would be biased (sqrt(2)A..2A for N=4,
        ~30% jitter) and is reported only as a cross-check.
    Averaging is done on the Stokes COMPONENTS for the pulse-integrated channel (the correct
    incoherent average) and on the angle for `legacy`, which has no detector analogue anyway.
    """
    recs = [r for r in records if r]
    n = len(recs)
    if n == 0:
        return None
    S = {k: float(np.mean([r[k] for r in recs])) for k in ("S0", "S1", "S2", "S3")}
    th, ch, dolp, vmh = stokes_to_angles(S["S0"], S["S1"], S["S2"], S["S3"])
    per = [stokes_to_angles(r["S0"], r["S1"], r["S2"], r["S3"]) for r in recs]
    th_sub = np.array([p[0] for p in per], dtype=float)
    vmh_sub = np.array([p[3] for p in per], dtype=float)
    leg = np.array([r["legacy"] for r in recs], dtype=float)
    phi = 2.0 * np.pi * np.arange(n) / n

    def fundamental(y):
        # A fringe amplitude needs at least 3 uniform phases to be identifiable; with n < 3 the
        # projection is not a measurement of the fundamental (n=1 would just return 2|y0|).
        if n < 3:
            return float("nan")
        y = np.asarray(y, dtype=float)
        if not np.all(np.isfinite(y)):
            return float("nan")
        return float(2.0 * np.abs(np.sum(y * np.exp(-1j * phi))) / n)

    return {"n_sub": n,
            # --- the objective ---
            "theta_chi5_deg": float(th),
            "vmh_chi5": float(vmh),
            "chi_deg": float(ch), "dolp": float(dolp),
            # --- the coherent artifact, kept as a diagnostic ---
            "theta_fringe_amp_deg": fundamental(th_sub),
            "vmh_fringe_amp": fundamental(vmh_sub),
            "theta_fringe_ptp_deg": float(np.ptp(th_sub)),
            # --- single-phase and legacy estimators, for the correlation study ---
            "theta_single_phase_deg": float(th_sub[0]),
            "theta_legacy_deg": float(np.mean(leg)),
            "theta_legacy_single_deg": float(leg[0]),
            "theta_legacy_fringe_amp_deg": fundamental(leg),
            # --- per-sub-sample series, so the fringe itself can be plotted ---
            "theta_sub_deg": [float(v) for v in th_sub],
            "vmh_sub": [float(v) for v in vmh_sub],
            "legacy_sub_deg": [float(v) for v in leg],
            "S0": S["S0"], "S1": S["S1"], "S2": S["S2"], "S3": S["S3"]}


def subsample_taus(f1: float, n_sub: int = SUBSAMPLES, tau0_fs: float = 0.0) -> List[float]:
    """Pump1 delays for the carrier average: one period T1 split into n_sub uniform phases."""
    T1 = carrier_period_fs(f1)
    return [float(tau0_fs + s * T1 / n_sub) for s in range(n_sub)]


# =========================================================================== reporting === #
def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    """Rank correlation without a scipy dependency in the hot path."""
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan")

    def rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        means = np.zeros(len(cnt))
        np.add.at(means, inv, r)
        means /= cnt
        return means[inv]

    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def load_base_geometry() -> dict:
    """SiN best_absolute -- the fabricated reference the campaign must beat."""
    return json.load(open(MEEP / "SiN_optimizations" / "best_absolute" / "geometry.json"))


def load_base_modes() -> dict:
    return json.load(open(MEEP / "SiN_optimizations" / "best_absolute" / "cavity_modes.json"))
