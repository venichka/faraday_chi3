"""Shared loader, constants and plot style for the SIN51 lock-in delay-scan analysis.

Datasets (repo-root data/, file names are acquisition times HHMMSS; see DATASETS):
  146  pump2-delay scan, 29.85 fs step, 68 points
  235  pump2-delay scan, 20 fs step, 101 points
  626  BOTH pumps locked, nothing delayed -- the reference (user, 2026-09-23/24)
Each CSV is one stage scan read out by a lock-in:
  delay_fs      nominal delay set point
  position_mm   delay-stage position read back at that point
  X_A, Y_A      lock-in in-phase / quadrature outputs (V)
  R_A, theta_A  lock-in magnitude (V) and phase (deg)
X/Y/R/theta are read sequentially, not as one sample (R != hypot(X, Y) point by point), so
X is the primary channel and R/theta are only cross-checks.

Lab configuration (user, 2026-09-23):
  * sample SIN51 = the fabricated SiN best_absolute cavity
  * pump1 and the probe travel together; PUMP2 is the delayed beam (tau) and is CHOPPED at 500 Hz
    on a 1 kHz laser, i.e. every other pump2 pulse is blocked. X is therefore the pump2-induced
    change in the balanced V-H signal (pump2-on shots minus pump2-off shots).
    NB: the simulations delay "pump1"; the fringe carrier here is the DELAYED pump's, pump2's.
  * pumps at 1560 and 1620 nm (user, 2026-09-23; NOT the 1521.5 / 1574.0 nm design point).
    The DELAYED pump2 is the 1620 nm one (user) -> fringe period 5.404 fs.
  * pulse duration ~100-150 fs
  * tau = 0 is NOMINAL, not a calibrated overlap
  * total probe signal S0 = 9 mV, measured on the V+H output, whose gain differs from the V-H
    output's (user); until the ratio is known it is taken as 1 (user, 2026-09-24)
  * probe 800 nm; pump peak intensity 1 TW/cm^2 = 1e12 W/cm^2 (user, 2026-09-25), which is also
    every simulation's reference intensity
  * the two pumps come from DIFFERENT OPAs (user, 2026-09-25): their relative optical path, hence
    the fringe phase, is not interferometrically stable
  * 626: "nothing moved" (user, 2026-09-25) -- yet its stage read-back spans the same range

Rotation calibration (assumptions, stated once and used everywhere via ROT_DEG_PER_V):
  For a 45-deg probe on a balanced detector, V - H = S0 sin(2 theta) ~ 2 S0 theta.
  Chopping every other shot of a pulse train makes the 500 Hz component of the detector output
  equal to Delta/2 in complex amplitude, where Delta is the on-minus-off difference of the
  time-averaged level; a lock-in reports the RMS of that component, Delta/sqrt(2).
  So theta = sqrt(2) X / (2 S0) * G_sum / G_diff = X / (sqrt(2) S0) * G_sum / G_diff,  with
    G_sum, G_diff  the transimpedance gains of the V+H and V-H outputs (they differ -- user),
    and valid if (a) S0 is the time-averaged V+H level, (b) the lock-in outputs RMS values
    (SR830-style), and (c) the detector response is flat from DC to 500 Hz.
  A pure square-wave model would instead give pi / 2 = 1.57x larger angles.
  GAIN_SUM_OVER_DIFF = 1 (user, provisional): theta_true = theta_quoted * G_sum / G_diff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA = REPO / "data"
FIGS = HERE / "figs"
RESULTS = HERE / "results"

DATASETS = {   # in acquisition order
    "146": {"file": "SIN51_lockin_180146.csv", "kind": "delay",
            "label": "delay scan, 29.85 fs step"},
    "235": {"file": "SIN51_lockin_180235.csv", "kind": "delay",
            "label": "delay scan, 20 fs step"},
    "626": {"file": "SIN51_lockin_180626.csv", "kind": "locked",
            "label": "both pumps locked (reference)"},
}
DELAY_TAGS = [k for k, v in DATASETS.items() if v["kind"] == "delay"]

# Common delay axis for every file: all scans start at the same stage position, which is
# pinned to the nominal first set point (-1500 fs). tau = X_REF_FS + 2 (x - X_REF_MM) / c.
X_REF_MM, X_REF_FS = 185.70530, -1500.0

C_UM_PER_FS = 0.299792458          # speed of light
DOUBLE_PASS = 2.0                  # delay = 2 * stage travel / c (verified in s0, 0.01%)

S0_V = 9.0e-3                      # total probe signal on the V+H output (user)
GAIN_SUM_OVER_DIFF = 1.0           # G(V+H) / G(V-H): they differ; 1 until measured (user)
ROT_DEG_PER_V = np.degrees(GAIN_SUM_OVER_DIFF / (np.sqrt(2.0) * S0_V))  # theta[deg] = X[V] * this
THETA = "θ"                        # axis-label symbol
GAIN_NOTE = "gain ratio G(V+H)/G(V−H) = {:g} (provisional — scales every angle)".format(
    GAIN_SUM_OVER_DIFF)
I_PEAK_W_CM2 = 1e12                # 1 TW/cm^2 peak intensity (user, confirmed 2026-09-25)
PROBE_NM = 800.0

LAM_LAB_NM = (1560.0, 1620.0)      # the lab's pumps (user)
LAM_DELAYED_NM = 1620.0            # the delayed, chopped pump2 (user); 1560 is kept as a control
LAM_DESIGN_NM = (1521.5, 1574.0)   # the best_absolute design point, for reference only
LAM_SCAN_NM = (1450.0, 1750.0)     # range searched when the delayed pump's wavelength is free


def fringe_period_fs(lam_nm):
    """Carrier period of the delayed pump: the fringe period in tau."""
    return lam_nm * 1e-3 / C_UM_PER_FS


def alias_period_fs(lam_nm, step_fs):
    """Apparent period of that fringe when sampled every `step_fs` (inf when frozen)."""
    cyc = step_fs / fringe_period_fs(lam_nm)
    frac = abs(cyc - np.round(cyc))
    return np.where(frac > 0, step_fs / np.maximum(frac, 1e-12), np.inf)

# plot style: reference categorical palette (light), recessive grid, one axis per panel
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"


def load(tag):
    """Arrays for one dataset. `tau` is the delay from the stage read-back on the common axis
    shared by all files; `tau_nom` is the set point; `pos_mm` the raw read-back."""
    d = np.genfromtxt(DATA / DATASETS[tag]["file"], delimiter=",", names=True)
    pos_um = (d["position_mm"] - X_REF_MM) * 1000.0
    return {"tag": tag, "tau_nom": d["delay_fs"].astype(float),
            "tau": X_REF_FS + DOUBLE_PASS * pos_um / C_UM_PER_FS,
            "pos_um": pos_um, "pos_mm": d["position_mm"],
            "X": d["X_A"], "Y": d["Y_A"], "R": d["R_A"], "phase": d["theta_A"]}


def dbr_harness():
    """chi5_dbr_design/common.py (TMM mode finder, FDTD runner), imported under its own module
    name: it is also called `common`, and a plain import would return this file instead."""
    import importlib.util
    import sys
    name = "chi5_dbr_common"
    if name in sys.modules:
        return sys.modules[name]
    path = HERE.parents[1] / "meep_project" / "chi5_dbr_design" / "common.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def step_fs(d):
    return float(np.mean(np.diff(d["tau"])))


def tags_from_argv(default=None):
    """Datasets named on the command line, else `default` (else all)."""
    import sys
    names = [a for a in sys.argv[1:] if a in DATASETS]
    return names or list(default or DATASETS)


def load_result(tag, stage):
    import json
    return json.load(open(RESULTS / tag / "{}_result.json".format(stage)))


def save_result(tag, stage, obj):
    import json
    p = RESULTS / tag
    p.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(p / "{}_result.json".format(stage), "w"), indent=2)


def title(tag):
    return "{} — {}".format(tag, DATASETS[tag]["label"])


def ar1_rho(resid):
    """Lag-1 autocorrelation of a residual series (the noise model used by `gls`)."""
    r = resid - resid.mean()
    return float(np.dot(r[:-1], r[1:]) / np.dot(r, r))


def whiten(v, rho):
    """Prais-Winsten transform: makes AR(1)-correlated noise white. Works on vectors and on
    design matrices (column-wise)."""
    v = np.asarray(v, float)
    out = np.empty_like(v)
    out[0] = np.sqrt(1.0 - rho ** 2) * v[0]
    out[1:] = v[1:] - rho * v[:-1]
    return out


def gls(y, A, rho):
    """Least squares under AR(1) noise. Returns (coef, chi2_unscaled, cov_unscaled):
    chi2 is the whitened residual sum of squares; divide by sigma_white^2 for a chi-square."""
    yw, Aw = whiten(y, rho), whiten(A, rho)
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    r = yw - Aw @ coef
    cov = np.linalg.pinv(Aw.T @ Aw)
    return coef, float(r @ r), cov


def gauss(t, t0, fwhm):
    return np.exp(-4.0 * np.log(2.0) * ((t - t0) / fwhm) ** 2)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "lines.markersize": 4, "font.size": 10,
        "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })
    return plt


def footnote(fig):
    """Stamp the calibration status on every figure that shows angles."""
    fig.text(0.01, -0.005, "Calibration: θ = X / (√2 S₀), S₀ = 9 mV; " + GAIN_NOTE + ".",
             fontsize=7.5, color=INK2, ha="left", va="top")


def save(fig, name, tag=None):
    d = FIGS / tag if tag else FIGS
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    fig.savefig(p)
    print("-> {}".format(p.relative_to(REPO)))
    return p
