#!/usr/bin/env python3
"""
Direct full-dynamics optimizer for DBR-like cavity geometry.

This version removes TMM prefiltering entirely.
Objective:
    maximize abs(final probe polarization rotation) from faraday_meep_fp_circ.py
    in quasi-1D with counter-rotating pumps.

The optimizer writes:
  - optimized_geometry.json
  - cavity_modes.json
  - optimize_report.json  (final-design-only summary)

Optional debug mode writes:
  - epsilon profile
  - reflectance spectrum with pump/probe/sideband markers
  - mode profiles at pump/probe/sideband frequencies
  - cavity-overlap matrix of these mode profiles
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from mode_targeting import get_cavity_materials, material_index_at_wavelength
from nonlinear_materials import (
    canonical_high_index_material,
    high_index_material_choices,
    resolve_high_index_index,
    resolve_high_index_kappa,
    resolve_high_index_n2,
)

try:
    from scipy.optimize import minimize  # type: ignore

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel  # type: ignore

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False


PUMP_MIN_UM = 1.3
PUMP_MAX_UM = 1.7
PROBE_EXACT_UM = 0.8
PROBE_BAND_MIN_UM = 0.85
PROBE_BAND_MAX_UM = 0.95
MIN_PUMP_SEP_UM = 0.02
PUMP_TARGET1_UM = 1.55
PUMP_TARGET2_UM = 1.65
PUMP_TARGET_CENTER_FREQ_INV_UM = 0.5 * (
    (1.0 / PUMP_TARGET1_UM) + (1.0 / PUMP_TARGET2_UM)
)
PUMP_TARGET_DELTA_FREQ_INV_UM = abs((1.0 / PUMP_TARGET1_UM) - (1.0 / PUMP_TARGET2_UM))


@dataclass
class Candidate:
    profile: str
    N_per: int
    t_sin_um: float
    t_sio2_um: float
    L_cav_um: float
    pump1_um: float
    pump2_um: float
    probe_um: float
    probe_reflectance: float
    pump1_reflectance: float
    pump2_reflectance: float
    rotation_deg: float
    abs_rotation_deg: float
    objective_summary: str
    score: float = float("nan")
    quality_factor: float = float("nan")
    quality_dolp_tail: float = float("nan")
    quality_theta_std_deg: float = float("nan")
    quality_s0_rel_max: float = float("nan")
    probe_q: float = float("nan")
    pump1_q: float = float("nan")
    pump2_q: float = float("nan")
    probe_depth: float = float("nan")
    pump1_depth: float = float("nan")
    pump2_depth: float = float("nan")


@dataclass
class SearchConfig:
    t_sin_qw: float
    t_sio2_qw: float
    bounds: List[Tuple[float, float]]


_WORKER_MATERIAL_CACHE: Dict[Tuple, Tuple[mp.Medium, mp.Medium]] = {}
_WORKER_SHARED_CONTEXT: Dict[str, object] = {}


def _configure_numeric_threads() -> None:
    # Outer process parallelism evaluates candidates concurrently; keep inner
    # numeric libraries single-threaded to avoid oversubscription.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def normalize_material_args(args: argparse.Namespace) -> argparse.Namespace:
    args.high_index_material = canonical_high_index_material(
        getattr(args, "high_index_material", "sin")
    )
    args.nH = resolve_high_index_index(getattr(args, "nH", None), args.high_index_material)
    args.kH = resolve_high_index_kappa(getattr(args, "kH", None), args.high_index_material)
    args.high_index_n2 = resolve_high_index_n2(
        getattr(args, "high_index_n2", None), args.high_index_material
    )
    if getattr(args, "nL", None) is None:
        args.nL = 1.45
    args.kappa_ref_lambda = float(max(getattr(args, "kappa_ref_lambda", 1.55), 1e-9))
    return args


def build_material_payload(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "materials": str(args.materials),
        "nH": float(args.nH),
        "kH": float(args.kH),
        "nL": float(args.nL),
        "high_index_material": str(args.high_index_material),
        "kappa_ref_lambda": float(args.kappa_ref_lambda),
        "sin_fit": str(args.sin_fit),
        "sio2_fit": str(args.sio2_fit),
        "fit_window": (int(args.fit_window[0]), int(args.fit_window[1])),
        "fit_poles": int(args.fit_poles),
    }


def _material_payload_key(payload: Dict[str, object]) -> Tuple:
    fw = payload.get("fit_window", (600, 2000))
    fw0 = int(fw[0]) if isinstance(fw, (tuple, list)) and len(fw) >= 1 else 600
    fw1 = int(fw[1]) if isinstance(fw, (tuple, list)) and len(fw) >= 2 else 2000
    return (
        str(payload.get("materials", "fit")),
        float(payload.get("nH", 2.0)),
        float(payload.get("kH", 0.0)),
        float(payload.get("nL", 1.45)),
        str(payload.get("high_index_material", "sin")),
        float(payload.get("kappa_ref_lambda", 1.55)),
        str(payload.get("sin_fit", "si3n4.csv")),
        str(payload.get("sio2_fit", "sio2.csv")),
        fw0,
        fw1,
        int(payload.get("fit_poles", 2)),
    )


def _get_worker_materials(payload: Dict[str, object]) -> Tuple[mp.Medium, mp.Medium]:
    key = _material_payload_key(payload)
    mats = _WORKER_MATERIAL_CACHE.get(key)
    if mats is not None:
        return mats

    model, nH, kH, nL, high_mat, kref, sin_csv, sio2_csv, fw0, fw1, fit_poles = key
    mat_sin, mat_sio2 = get_cavity_materials(
        model=str(model),
        index_high=float(nH),
        kappa_high=float(kH),
        index_low=float(nL),
        high_index_material=str(high_mat),
        kappa_ref_wavelength_um=float(kref),
        sin_csv=str(sin_csv),
        sio2_csv=str(sio2_csv),
        lam_min=int(fw0),
        lam_max=int(fw1),
        fit_poles=int(fit_poles),
    )
    _WORKER_MATERIAL_CACHE[key] = (mat_sin, mat_sio2)
    return mat_sin, mat_sio2


def init_objective_worker(shared_context: Dict[str, object]) -> None:
    global _WORKER_SHARED_CONTEXT
    _configure_numeric_threads()
    _WORKER_SHARED_CONTEXT = dict(shared_context)


def objective_run_worker(payload: Dict[str, object]) -> Tuple[int, Candidate]:
    global _WORKER_SHARED_CONTEXT
    if not _WORKER_SHARED_CONTEXT:
        raise RuntimeError("Worker shared context is not initialized.")
    args = argparse.Namespace(**dict(_WORKER_SHARED_CONTEXT["args_dict"]))
    mat_sin, mat_sio2 = _get_worker_materials(
        dict(_WORKER_SHARED_CONTEXT["material_payload"])
    )
    cand = objective_run(
        profile=str(payload["profile"]),
        n_per=int(payload["n_per"]),
        design=dict(payload["design"]),
        args=args,
        n_sin_ref=float(_WORKER_SHARED_CONTEXT["n_sin_ref"]),
        n_sio2_ref=float(_WORKER_SHARED_CONTEXT["n_sio2_ref"]),
        mat_sin=mat_sin,
        mat_sio2=mat_sio2,
        eval_root=Path(str(_WORKER_SHARED_CONTEXT["eval_root"])),
        eval_id=int(payload["eval_id"]),
    )
    return int(payload["eval_id"]), cand


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def format_duration(seconds: float) -> str:
    s = max(0.0, float(seconds))
    if s < 60.0:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60.0)
    if m < 60.0:
        return f"{int(m)}m {sec:.0f}s"
    h, m = divmod(m, 60.0)
    return f"{int(h)}h {int(m)}m"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Direct Meep-based optimizer for Faraday rotation objective."
    )

    ap.add_argument("--materials", choices=("library", "constant", "fit"), default="fit")
    ap.add_argument(
        "--high-index-material",
        choices=high_index_material_choices(),
        default="sin",
        help="High-index cavity/DBR material preset (default keeps existing SiN path).",
    )
    ap.add_argument(
        "--sin-fit",
        dest="sin_fit",
        type=str,
        default="si3n4.csv",
        help="CSV with wavelength_nm,n,k for selected high-index material when --materials fit.",
    )
    ap.add_argument("--sio2-fit", dest="sio2_fit", type=str, default="sio2.csv")
    ap.add_argument(
        "--fit-window",
        type=int,
        nargs=2,
        metavar=("lambda_min", "lambda_max"),
        default=(600, 2000),
    )
    ap.add_argument("--fit-poles", type=int, default=2)
    ap.add_argument("--nH", type=float, default=None)
    ap.add_argument("--kH", type=float, default=None)
    ap.add_argument("--nL", type=float, default=1.45)
    ap.add_argument(
        "--kappa-ref-lambda",
        type=float,
        default=1.55,
        help="Reference wavelength (um) used when mapping constant kappa to Meep conductivity.",
    )
    ap.add_argument(
        "--high-index-n2",
        type=float,
        default=None,
        help="Override high-index Kerr nonlinear index n2 (m^2/W).",
    )

    ap.add_argument("--probe-target-mode", choices=("exact", "band", "both"), default="both")
    ap.add_argument("--mirror-min", type=int, default=2)
    ap.add_argument("--mirror-max", type=int, default=6)

    ap.add_argument("--pump-intensity", type=float, default=1e12)

    # Optimization objective fidelity (reasonable-but-fast settings)
    ap.add_argument("--objective-resolution", type=int, default=50)
    ap.add_argument("--objective-decay-threshold", type=float, default=1e-4)
    ap.add_argument("--objective-mode", choices=("quick", "full"), default="quick")
    ap.add_argument(
        "--objective-metric",
        choices=("abs_rotation", "quality_weighted_abs_rotation"),
        default="abs_rotation",
        help=(
            "Optimization score: pure |rotation|, or |rotation| weighted by "
            "probe signal quality (DoLP, tail stability, normalized tail S0)."
        ),
    )
    ap.add_argument(
        "--quality-std-ref-deg",
        type=float,
        default=15.0,
        help=(
            "Reference width (deg) for the stability factor exp(-(std/std_ref)^2) "
            "in quality-weighted objective."
        ),
    )
    ap.add_argument(
        "--quality-pump-dom-ref",
        type=float,
        default=0.2,
        help=(
            "Reference scale for dominant pump monitor amplitude term in "
            "quality-weighted objective: term=dom/(dom+ref)."
        ),
    )
    ap.add_argument(
        "--quality-pump-balance-sigma-dec",
        type=float,
        default=0.35,
        help=(
            "Log10 ratio width for pump-balance term exp(-(log10(ratio)/sigma)^2) "
            "used in quality-weighted objective."
        ),
    )
    ap.add_argument(
        "--cavity-min-length",
        type=float,
        default=1.5,
        help="Hard minimum on central cavity layer length L_cav (um).",
    )
    ap.add_argument(
        "--cavity-max-length",
        type=float,
        default=4.5,
        help="Hard maximum on central cavity layer length L_cav (um).",
    )

    # Resonance enforcement (all chosen frequencies must come from reflectance dips)
    ap.add_argument(
        "--resonance-resolution",
        type=int,
        default=32,
        help="Resolution for coarse reflectance resonance finder during optimization.",
    )
    ap.add_argument(
        "--resonance-nfreq",
        type=int,
        default=320,
        help="Number of frequencies for coarse reflectance resonance finder.",
    )
    ap.add_argument(
        "--resonance-decay-threshold",
        type=float,
        default=1e-5,
        help="Field-decay threshold used in coarse resonance reflectance runs.",
    )
    ap.add_argument(
        "--resonance-max-R",
        type=float,
        default=0.35,
        help="Maximum allowed reflectance at selected probe/pump resonances.",
    )
    ap.add_argument(
        "--pump-min-q",
        type=float,
        default=30.0,
        help="Minimum estimated Q=lambda/FWHM required for each selected pump resonance (set <=0 to disable).",
    )
    ap.add_argument(
        "--pump-min-depth",
        type=float,
        default=0.01,
        help="Minimum reflectance dip depth for selected pump resonances (0 disables).",
    )
    ap.add_argument(
        "--probe-min-depth",
        type=float,
        default=0.0,
        help="Minimum reflectance dip depth for selected probe resonance (0 disables).",
    )
    ap.add_argument(
        "--resonance-linewidth-level",
        type=float,
        default=0.5,
        help="Relative dip-depth level used to estimate resonance linewidth (0.5 ~ FWHM-like width).",
    )
    ap.add_argument(
        "--probe-exact-tol",
        type=float,
        default=0.06,
        help="Allowed |lambda_probe_res - 0.8| for profile=exact (um).",
    )
    ap.add_argument(
        "--pump-local-q-check",
        choices=("off", "final", "strict"),
        default="final",
        help=(
            "Local high-resolution Q check around selected pump dips: "
            "'off' disables, 'final' checks only best geometry, "
            "'strict' enforces per objective evaluation."
        ),
    )
    ap.add_argument(
        "--local-q-window-um",
        type=float,
        default=0.14,
        help="Wavelength span (um) around selected pumps for local Q refinement.",
    )
    ap.add_argument(
        "--local-q-resolution",
        type=int,
        default=40,
        help="Resolution for local Q refinement reflectance run.",
    )
    ap.add_argument(
        "--local-q-nfreq",
        type=int,
        default=401,
        help="Number of frequencies for local Q refinement reflectance run.",
    )
    ap.add_argument(
        "--local-q-decay-threshold",
        type=float,
        default=1e-5,
        help="Field-decay threshold for local Q refinement reflectance run.",
    )

    # Search budget
    ap.add_argument(
        "--optimizer",
        choices=("bayes", "powell"),
        default="bayes",
        help="Local/global refinement strategy after seed evaluation.",
    )
    ap.add_argument(
        "--maxfev",
        type=int,
        default=5,
        help="Max objective calls for each local refine run (used by Powell/coordinate fallback).",
    )
    ap.add_argument(
        "--powell-mode",
        choices=("auto", "scipy", "pattern"),
        default="auto",
        help=(
            "Powell refinement backend: auto selects SciPy for serial runs and "
            "parallel pattern-search when workers>1."
        ),
    )
    ap.add_argument(
        "--top-mirrors-for-refine",
        type=int,
        default=1,
        help="How many mirror counts (best seed scores) to locally refine per profile.",
    )
    ap.add_argument(
        "--seed-variants",
        type=int,
        default=1,
        choices=(1, 2),
        help="Number of initial seeds per mirror count.",
    )
    ap.add_argument(
        "--bayes-init",
        type=int,
        default=6,
        help="Minimum observed points per mirror count before GP-guided BO iterations.",
    )
    ap.add_argument(
        "--bayes-iters",
        type=int,
        default=12,
        help="Number of GP acquisition/evaluation steps per selected mirror count.",
    )
    ap.add_argument(
        "--bayes-batch-size",
        type=int,
        default=1,
        help=(
            "Number of BO acquisition candidates to evaluate per iteration "
            "(>1 improves worker utilization)."
        ),
    )
    ap.add_argument(
        "--bayes-candidates",
        type=int,
        default=256,
        help="Number of sampled acquisition candidates per BO iteration.",
    )
    ap.add_argument(
        "--bayes-xi",
        type=float,
        default=0.01,
        help="Exploration parameter xi for expected improvement acquisition.",
    )
    ap.add_argument(
        "--bayes-gp-restarts",
        type=int,
        default=1,
        help="Number of hyperparameter optimizer restarts for the GP model.",
    )
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes for independent candidate evaluations "
            "(1 disables parallelism)."
        ),
    )

    # Geometry padding
    ap.add_argument("--pad-air", type=float, default=0.8)
    ap.add_argument("--pad-sub", type=float, default=0.8)
    ap.add_argument("--dpml", type=float, default=1.0)

    # Outputs
    ap.add_argument("--out-geom", type=str, default="optimized_geometry.json")
    ap.add_argument("--out-modes", type=str, default="cavity_modes.json")
    ap.add_argument("--out-report", type=str, default="optimize_report.json")

    # Debug artifacts
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug-prefix", type=str, default="optimize_debug")
    ap.add_argument("--debug-resolution", type=int, default=90)
    ap.add_argument("--debug-nfreq", type=int, default=700)

    ap.add_argument("--eval-root", type=str, default=".opt_eval_tmp")
    ap.add_argument("--keep-eval-artifacts", action="store_true")
    ap.add_argument(
        "--meep-verbosity",
        type=int,
        default=0,
        help="Meep verbosity during optimization/debug helper runs (0 = quiet).",
    )
    return ap.parse_args()


def build_geometry_spec(
    n_sin: float,
    n_sio2: float,
    dpml: float,
    pad_air: float,
    pad_sub: float,
    n_per: int,
    t_sin_um: float,
    t_sio2_um: float,
    L_cav_um: float,
    high_index_material: str = "sin",
) -> Dict:
    left: List[Dict[str, float]] = []
    right: List[Dict[str, float]] = []
    for _ in range(n_per):
        left += [
            {"mat": "SiN", "thk_um": float(t_sin_um)},
            {"mat": "SiO2", "thk_um": float(t_sio2_um)},
        ]
        right += [
            {"mat": "SiO2", "thk_um": float(t_sio2_um)},
            {"mat": "SiN", "thk_um": float(t_sin_um)},
        ]

    return {
        "materials": {
            "SiN": {"type": "Medium", "params": {"index": float(n_sin)}},
            "SiO2": {"type": "Medium", "params": {"index": float(n_sio2)}},
        },
        "pads": {
            "pml_um": float(dpml),
            "air_um": float(pad_air),
            "substrate_um": float(pad_sub),
        },
        "spacers": {"left_um": 0.0, "right_um": 0.0},
        "cavity": {"mat": "SiN", "L_um": float(L_cav_um)},
        "mirrors": {"left": left, "right": right},
        "meta": {
            "generated_on": utcnow_iso(),
            "generator": "optimize_cavity_geometry.py",
            "objective": "probe_rotation_optimization",
            "high_index_material": str(high_index_material),
        },
    }


def build_modes_spec(probe_um: float, pump1_um: float, pump2_um: float) -> Dict:
    f_probe = 1.0 / probe_um
    f_p1 = 1.0 / pump1_um
    f_p2 = 1.0 / pump2_um
    delta = abs(f_p1 - f_p2)
    f_sb_plus = f_probe + delta
    f_sb_minus = max(f_probe - delta, 0.0)
    return {
        "probe": {"frequency": float(f_probe), "lambda_um": float(probe_um)},
        "pump1": {"frequency": float(f_p1), "lambda_um": float(pump1_um)},
        "pump2": {"frequency": float(f_p2), "lambda_um": float(pump2_um)},
        "sidebands": {
            "frequency_plus": float(f_sb_plus),
            "frequency_minus": float(f_sb_minus),
            "delta_frequency": float(delta),
            "lambda_plus_um": float(1.0 / f_sb_plus),
            "lambda_minus_um": float(1.0 / f_sb_minus) if f_sb_minus > 0 else float("inf"),
        },
    }


def search_config(
    profile: str,
    n_sin: float,
    n_sio2: float,
    cavity_min_length: float,
    cavity_max_length: float,
) -> SearchConfig:
    lam_qw = 1.5
    t_sin_qw = lam_qw / (4.0 * n_sin)
    t_sio2_qw = lam_qw / (4.0 * n_sio2)

    lmin = float(max(cavity_min_length, 1.5))
    lmax = float(max(cavity_max_length, lmin + 0.05))

    bounds: List[Tuple[float, float]] = [
        (0.65 * t_sin_qw, 1.35 * t_sin_qw),  # t_sin
        (0.65 * t_sio2_qw, 1.35 * t_sio2_qw),  # t_sio2
        (lmin, lmax),  # L_cav
    ]
    return SearchConfig(t_sin_qw=t_sin_qw, t_sio2_qw=t_sio2_qw, bounds=bounds)


def clip_to_bounds(x: Sequence[float], bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    out = arr.copy()
    for i, (lo, hi) in enumerate(bounds):
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def decode_design_vector(x: Sequence[float], bounds: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    xx = clip_to_bounds(x, bounds)

    t_sin = float(xx[0])
    t_sio2 = float(xx[1])
    L_cav = float(xx[2])

    return {
        "t_sin_um": t_sin,
        "t_sio2_um": t_sio2,
        "L_cav_um": L_cav,
    }


def seed_vector(profile: str, cfg: SearchConfig, n_sin: float, n_per: int, variant: int) -> np.ndarray:
    probe_seed = PROBE_EXACT_UM if profile == "exact" else 0.90
    m_order = 5 if variant == 0 else 7
    L_cav = max(cfg.bounds[2][0], m_order * probe_seed / (2.0 * n_sin))

    # Two seed variants: lambda/4 and slightly detuned mirror periods.
    if variant == 0:
        t_sin = cfg.t_sin_qw
        t_sio2 = cfg.t_sio2_qw
    else:
        mirror_scale = 1.0 + 0.05 * ((n_per % 3) - 1)
        t_sin = cfg.t_sin_qw * mirror_scale
        t_sio2 = cfg.t_sio2_qw * (2.0 - mirror_scale)

    vec = [t_sin, t_sio2, L_cav]
    return clip_to_bounds(vec, cfg.bounds)


def candidate_tag(profile: str, n_per: int, eval_id: int) -> str:
    return f"{profile}_N{n_per}_eval{eval_id:04d}"


def random_vector_in_bounds(
    bounds: Sequence[Tuple[float, float]], rng: np.random.Generator
) -> np.ndarray:
    return np.array([rng.uniform(lo, hi) for (lo, hi) in bounds], dtype=float)


def normal_cdf(z: np.ndarray) -> np.ndarray:
    zz = np.asarray(z, dtype=float)
    if hasattr(np, "erf"):
        return 0.5 * (1.0 + np.erf(zz / np.sqrt(2.0)))
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(zz / np.sqrt(2.0)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma_safe = np.maximum(sigma, 1e-12)
    improvement = mu - float(best) - float(xi)
    z = improvement / sigma_safe
    cdf = normal_cdf(z)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    ei = improvement * cdf + sigma_safe * pdf
    ei = np.where(np.isfinite(ei), ei, 0.0)
    return np.maximum(ei, 0.0)


def normalize_to_unit_box(
    x: np.ndarray, bounds: Sequence[Tuple[float, float]]
) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    span = np.maximum(hi - lo, 1e-12)
    return (arr - lo) / span


def gp_predict_rbf_fallback(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    length_scale: float = 0.25,
    noise: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lightweight GP-like posterior with fixed isotropic RBF kernel.
    Used when sklearn is unavailable so Bayesian optimization still works.
    """
    xt = normalize_to_unit_box(np.asarray(x_train, dtype=float), bounds)
    xq = normalize_to_unit_box(np.asarray(x_query, dtype=float), bounds)
    yt = np.asarray(y_train, dtype=float)

    d2_tt = np.sum((xt[:, None, :] - xt[None, :, :]) ** 2, axis=2)
    k_tt = np.exp(-0.5 * d2_tt / max(length_scale * length_scale, 1e-12))
    k_tt = k_tt + (float(noise) + 1e-10) * np.eye(k_tt.shape[0], dtype=float)

    d2_tq = np.sum((xt[:, None, :] - xq[None, :, :]) ** 2, axis=2)
    k_tq = np.exp(-0.5 * d2_tq / max(length_scale * length_scale, 1e-12))

    try:
        alpha = np.linalg.solve(k_tt, yt)
        mu = k_tq.T @ alpha
        v = np.linalg.solve(k_tt, k_tq)
    except np.linalg.LinAlgError:
        k_tt = k_tt + 1e-6 * np.eye(k_tt.shape[0], dtype=float)
        alpha = np.linalg.solve(k_tt, yt)
        mu = k_tq.T @ alpha
        v = np.linalg.solve(k_tt, k_tq)

    # Prior variance is 1.0 for unit-amplitude kernel.
    var = 1.0 - np.sum(k_tq * v, axis=0)
    var = np.maximum(var, 1e-12)
    return np.asarray(mu, dtype=float), np.sqrt(var)


def _interp_x_at_y(x0: float, y0: float, x1: float, y1: float, y: float) -> float:
    dy = float(y1 - y0)
    if abs(dy) <= 1e-30:
        return 0.5 * (float(x0) + float(x1))
    t = (float(y) - float(y0)) / dy
    return float(x0) + t * (float(x1) - float(x0))


def estimate_dip_q(
    wl_um: np.ndarray,
    R: np.ndarray,
    idx: int,
    linewidth_level: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Estimate dip Q from local linewidth on a reflectance trace.

    Returns: (Q_est, width_um, depth)
      Q_est = lam_res / width_um
      width_um measured at R = R_min + linewidth_level * depth
      depth = min(local_shoulders) - R_min
    """
    wl = np.asarray(wl_um, dtype=float)
    rr = np.asarray(R, dtype=float)
    n = rr.size
    i = int(idx)
    if i <= 0 or i >= n - 1:
        return float("nan"), float("nan"), float("nan")

    lvl = float(np.clip(linewidth_level, 1e-3, 0.999))

    left_peak = i - 1
    while left_peak > 0 and rr[left_peak - 1] > rr[left_peak]:
        left_peak -= 1

    right_peak = i + 1
    while right_peak < (n - 1) and rr[right_peak + 1] > rr[right_peak]:
        right_peak += 1

    shoulder = min(float(rr[left_peak]), float(rr[right_peak]))
    depth = shoulder - float(rr[i])
    if depth <= 1e-9:
        return float("nan"), float("nan"), float(depth)

    target = float(rr[i]) + lvl * depth

    k = i
    while k > left_peak and rr[k] < target:
        k -= 1
    if rr[k] < target:
        return float("nan"), float("nan"), float(depth)
    k2 = min(k + 1, i)
    if k2 == k:
        return float("nan"), float("nan"), float(depth)
    lam_left = _interp_x_at_y(float(wl[k]), float(rr[k]), float(wl[k2]), float(rr[k2]), target)

    k = i
    while k < right_peak and rr[k] < target:
        k += 1
    if rr[k] < target:
        return float("nan"), float("nan"), float(depth)
    k1 = max(k - 1, i)
    if k1 == k:
        return float("nan"), float("nan"), float(depth)
    lam_right = _interp_x_at_y(float(wl[k1]), float(rr[k1]), float(wl[k]), float(rr[k]), target)

    width = float(lam_right - lam_left)
    if width <= 1e-12:
        return float("nan"), float("nan"), float(depth)
    q_est = float(wl[i] / width)
    return q_est, width, float(depth)


def find_reflectance_dips(
    wl_um: np.ndarray,
    R: np.ndarray,
    linewidth_level: float = 0.5,
) -> List[Dict[str, float]]:
    dips: List[Dict[str, float]] = []
    wl = np.asarray(wl_um, dtype=float)
    rr = np.asarray(R, dtype=float)
    if wl[0] > wl[-1]:
        wl = wl[::-1]
        rr = rr[::-1]
    for i in range(1, len(rr) - 1):
        if rr[i] <= rr[i - 1] and rr[i] < rr[i + 1]:
            q_est, width_um, depth = estimate_dip_q(
                wl_um=wl,
                R=rr,
                idx=i,
                linewidth_level=linewidth_level,
            )
            dips.append(
                {
                    "idx": int(i),
                    "lam": float(wl[i]),
                    "R": float(rr[i]),
                    "Q": float(q_est),
                    "width_um": float(width_um),
                    "depth": float(depth),
                }
            )
    return dips


def pick_resonant_modes_from_dips(
    profile: str,
    dips: Sequence[Dict[str, float]],
    probe_exact_tol: float,
    resonance_max_R: float,
    pump_min_q: float,
    pump_min_depth: float,
    probe_min_depth: float,
) -> Optional[Dict[str, float]]:
    if not dips:
        return None

    # Probe resonance
    if profile == "exact":
        d_probe = min(dips, key=lambda d: abs(d["lam"] - PROBE_EXACT_UM))
        if abs(d_probe["lam"] - PROBE_EXACT_UM) > probe_exact_tol:
            return None
    else:
        band = [d for d in dips if PROBE_BAND_MIN_UM <= d["lam"] <= PROBE_BAND_MAX_UM]
        if not band:
            return None
        d_probe = min(band, key=lambda d: (d["R"], abs(d["lam"] - 0.90)))
    if float(probe_min_depth) > 0.0 and float(d_probe.get("depth", 0.0)) < float(probe_min_depth):
        return None

    # Pump resonances
    def _freq_of(dip_entry: Dict[str, float]) -> float:
        lam = float(dip_entry["lam"])
        return float(1.0 / lam) if lam > 0 else float("nan")

    dip_freqs = np.array([_freq_of(d) for d in dips], dtype=float)
    dip_rs = np.array([float(d.get("R", 1.0)) for d in dips], dtype=float)
    f_probe = _freq_of(d_probe)

    pumps = []
    for d in dips:
        if not (PUMP_MIN_UM <= d["lam"] <= PUMP_MAX_UM):
            continue
        qv = float(d.get("Q", float("nan")))
        if float(pump_min_q) > 0.0 and (not np.isfinite(qv) or qv < float(pump_min_q)):
            continue
        if float(pump_min_depth) > 0.0 and float(d.get("depth", 0.0)) < float(pump_min_depth):
            continue
        pumps.append(d)
    if len(pumps) < 2:
        return None
    best_pair: Optional[Tuple[Dict[str, float], Dict[str, float], float]] = None
    for i in range(len(pumps) - 1):
        for j in range(i + 1, len(pumps)):
            d1 = pumps[i]
            d2 = pumps[j]
            lam1 = min(d1["lam"], d2["lam"])
            lam2 = max(d1["lam"], d2["lam"])
            if (lam2 - lam1) < MIN_PUMP_SEP_UM:
                continue

            f1 = _freq_of(d1)
            f2 = _freq_of(d2)
            if (not np.isfinite(f1)) or (not np.isfinite(f2)):
                continue
            f_center = 0.5 * (f1 + f2)
            f_detune = abs(f1 - f2)

            f_sb_plus = f_probe + f_detune
            f_sb_minus = max(f_probe - f_detune, 0.0)

            sb_plus_idx = int(np.argmin(np.abs(dip_freqs - f_sb_plus)))
            sb_minus_idx = int(np.argmin(np.abs(dip_freqs - f_sb_minus)))
            sb_plus_detune = float(abs(dip_freqs[sb_plus_idx] - f_sb_plus))
            sb_minus_detune = float(abs(dip_freqs[sb_minus_idx] - f_sb_minus))
            sb_plus_r = float(dip_rs[sb_plus_idx])
            sb_minus_r = float(dip_rs[sb_minus_idx])

            # Score in frequency-domain to match FWM sideband physics:
            #  • low pump reflectance
            #  • pump center/detune close to desired targets
            #  • sidebands close to existing cavity resonances and not strongly reflected
            score = float(
                d1["R"]
                + d2["R"]
                + 6.0 * abs(f_center - PUMP_TARGET_CENTER_FREQ_INV_UM)
                + 4.0 * abs(f_detune - PUMP_TARGET_DELTA_FREQ_INV_UM)
                + 8.0 * (sb_plus_detune + sb_minus_detune)
                + 0.5 * (sb_plus_r + sb_minus_r)
            )
            if best_pair is None or score < best_pair[2]:
                if d1["lam"] <= d2["lam"]:
                    best_pair = (d1, d2, score)
                else:
                    best_pair = (d2, d1, score)
    if best_pair is None:
        return None
    d_p1, d_p2, _ = best_pair

    if (
        d_probe["R"] > resonance_max_R
        or d_p1["R"] > resonance_max_R
        or d_p2["R"] > resonance_max_R
    ):
        return None

    return {
        "probe_um": float(d_probe["lam"]),
        "pump1_um": float(d_p1["lam"]),
        "pump2_um": float(d_p2["lam"]),
        "probe_R": float(d_probe["R"]),
        "pump1_R": float(d_p1["R"]),
        "pump2_R": float(d_p2["R"]),
        "probe_Q": float(d_probe.get("Q", float("nan"))),
        "pump1_Q": float(d_p1.get("Q", float("nan"))),
        "pump2_Q": float(d_p2.get("Q", float("nan"))),
        "probe_depth": float(d_probe.get("depth", float("nan"))),
        "pump1_depth": float(d_p1.get("depth", float("nan"))),
        "pump2_depth": float(d_p2.get("depth", float("nan"))),
        "pump_center_frequency_inv_um": float(
            0.5 * (_freq_of(d_p1) + _freq_of(d_p2))
        ),
        "pump_detune_frequency_inv_um": float(
            abs(_freq_of(d_p1) - _freq_of(d_p2))
        ),
    }


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _summary_float(mapping: Dict[str, Any], path: Sequence[str], default: float = float("nan")) -> float:
    node: Any = mapping
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return float(default)
        node = node[key]
    try:
        out = float(node)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _objective_from_summary_data(
    data: Dict,
    objective_metric: str = "abs_rotation",
    quality_std_ref_deg: float = 15.0,
    quality_pump_dom_ref: float = 0.2,
    quality_pump_balance_sigma_dec: float = 0.35,
) -> Tuple[float, float, float, Dict[str, float]]:
    pr = data.get("probe_rotation_deg", {})
    rot_raw = pr.get("final_relative_deg", float("nan"))
    rot_wrapped = pr.get("wrapped_final_relative_deg", float("nan"))
    rot = float(rot_raw)
    if (not np.isfinite(rot)) and np.isfinite(rot_wrapped):
        rot = float(rot_wrapped)
    # Guard against legacy summaries that reported unwrapped multi-turn angles.
    if np.isfinite(rot) and abs(rot) > 90.0 and np.isfinite(rot_wrapped):
        rot = float(rot_wrapped)
    abs_rot = float(abs(rot))

    tail = data.get("probe_stokes_dft", {}).get("tail_weighted", {})
    dolp_tail = float(tail.get("dolp", float("nan")))
    theta_std_deg = float(tail.get("theta_relative_std_deg", float("nan")))
    s0_rel_max = float(tail.get("S0_rel_max", float("nan")))

    dolp_term = _clip01(dolp_tail) if np.isfinite(dolp_tail) else 0.0
    s0_term = np.sqrt(_clip01(s0_rel_max)) if np.isfinite(s0_rel_max) else 0.0
    std_ref = max(float(quality_std_ref_deg), 1e-9)
    if np.isfinite(theta_std_deg):
        stability_term = float(np.exp(-((float(theta_std_deg) / std_ref) ** 2)))
    else:
        stability_term = 0.0
    probe_quality_factor = float(dolp_term * s0_term * stability_term)

    # Source-aware terms from the same nonlinear run (no extra simulations):
    # dominant pump circular components at output monitor.
    p1_dom = _summary_float(
        data,
        ["pump_monitor_metrics", "rms_integrated", "tail_weighted_abs", "pump1_dominant"],
        default=np.nan,
    )
    p2_dom = _summary_float(
        data,
        ["pump_monitor_metrics", "rms_integrated", "tail_weighted_abs", "pump2_dominant"],
        default=np.nan,
    )
    p1_purity = _summary_float(
        data,
        ["pump_monitor_metrics", "rms_integrated", "dominant_purity", "pump1_tail_weighted"],
        default=np.nan,
    )
    p2_purity = _summary_float(
        data,
        ["pump_monitor_metrics", "rms_integrated", "dominant_purity", "pump2_tail_weighted"],
        default=np.nan,
    )

    dom_geom = np.sqrt(max(p1_dom, 0.0) * max(p2_dom, 0.0)) if np.isfinite(p1_dom) and np.isfinite(p2_dom) else 0.0
    dom_ref = max(float(quality_pump_dom_ref), 1e-12)
    pump_dom_term = float(dom_geom / (dom_geom + dom_ref))

    pump_purity_term = float(np.sqrt(_clip01(p1_purity) * _clip01(p2_purity))) if (
        np.isfinite(p1_purity) and np.isfinite(p2_purity)
    ) else 0.0

    ratio = float(p2_dom / max(p1_dom, 1e-30)) if (np.isfinite(p1_dom) and np.isfinite(p2_dom) and p1_dom > 0.0) else np.nan
    sigma_dec = max(float(quality_pump_balance_sigma_dec), 1e-6)
    if np.isfinite(ratio) and ratio > 0.0:
        pump_balance_term = float(np.exp(-((np.log10(ratio) / sigma_dec) ** 2)))
    else:
        pump_balance_term = 0.0

    source_quality_factor = float(pump_dom_term * pump_purity_term * pump_balance_term)
    quality_factor = float(probe_quality_factor * source_quality_factor)

    metric = str(objective_metric).lower()
    if metric == "quality_weighted_abs_rotation":
        score = float(abs_rot * quality_factor) if np.isfinite(abs_rot) else -1.0
    else:
        # Keep pure-rotation objective unchanged.
        score = float(abs_rot) if np.isfinite(abs_rot) else -1.0

    details = {
        "dolp_tail": float(dolp_tail),
        "theta_std_deg": float(theta_std_deg),
        "s0_rel_max": float(s0_rel_max),
        "dolp_term": float(dolp_term),
        "s0_term": float(s0_term),
        "stability_term": float(stability_term),
        "probe_quality_factor": float(probe_quality_factor),
        "pump1_dom_tail": float(p1_dom),
        "pump2_dom_tail": float(p2_dom),
        "pump1_purity_tail": float(p1_purity),
        "pump2_purity_tail": float(p2_purity),
        "pump_dom_term": float(pump_dom_term),
        "pump_purity_term": float(pump_purity_term),
        "pump_balance_term": float(pump_balance_term),
        "source_quality_factor": float(source_quality_factor),
        "quality_factor": float(quality_factor),
    }
    return float(rot), float(abs_rot), float(score), details


def candidate_score(cand: Candidate) -> float:
    if np.isfinite(cand.score):
        return float(cand.score)
    if np.isfinite(cand.abs_rotation_deg):
        return float(cand.abs_rotation_deg)
    return -1.0


def objective_run(
    profile: str,
    n_per: int,
    design: Dict[str, float],
    args: argparse.Namespace,
    n_sin_ref: float,
    n_sio2_ref: float,
    mat_sin: mp.Medium,
    mat_sio2: mp.Medium,
    eval_root: Path,
    eval_id: int,
) -> Candidate:
    tag = candidate_tag(profile, n_per, eval_id)
    work = eval_root / tag
    work.mkdir(parents=True, exist_ok=True)

    geom_path = work / "optimized_geometry_eval.json"
    modes_path = work / "cavity_modes_eval.json"
    out_dir = work / "faraday_eval"

    geom_spec = build_geometry_spec(
        n_sin=n_sin_ref,
        n_sio2=n_sio2_ref,
        dpml=args.dpml,
        pad_air=args.pad_air,
        pad_sub=args.pad_sub,
        n_per=n_per,
        t_sin_um=design["t_sin_um"],
        t_sio2_um=design["t_sio2_um"],
        L_cav_um=design["L_cav_um"],
        high_index_material=str(args.high_index_material),
    )

    # Enforce resonance constraint from structure modes (reflectance dips).
    mats_for_refl = {"SiN": mat_sin, "SiO2": mat_sio2}
    try:
        wl_refl, R_refl = debug_reflectance(
            geom_spec,
            mats_for_refl,
            resolution=int(args.resonance_resolution),
            nfreq=int(args.resonance_nfreq),
            decay_threshold=float(args.resonance_decay_threshold),
        )
    except Exception as exc:
        return Candidate(
            profile=profile,
            N_per=n_per,
            t_sin_um=design["t_sin_um"],
            t_sio2_um=design["t_sio2_um"],
            L_cav_um=design["L_cav_um"],
            pump1_um=float("nan"),
            pump2_um=float("nan"),
            probe_um=float("nan"),
            probe_reflectance=float("nan"),
            pump1_reflectance=float("nan"),
            pump2_reflectance=float("nan"),
            rotation_deg=float("nan"),
            abs_rotation_deg=-1.0,
            objective_summary=f"resonance_eval_failed:{type(exc).__name__}",
            score=-1.0,
        )
    dips = find_reflectance_dips(
        wl_refl,
        R_refl,
        linewidth_level=float(args.resonance_linewidth_level),
    )
    selected = pick_resonant_modes_from_dips(
        profile=profile,
        dips=dips,
        probe_exact_tol=float(args.probe_exact_tol),
        resonance_max_R=float(args.resonance_max_R),
        pump_min_q=float(args.pump_min_q),
        pump_min_depth=float(args.pump_min_depth),
        probe_min_depth=float(args.probe_min_depth),
    )
    if selected is None:
        return Candidate(
            profile=profile,
            N_per=n_per,
            t_sin_um=design["t_sin_um"],
            t_sio2_um=design["t_sio2_um"],
            L_cav_um=design["L_cav_um"],
            pump1_um=float("nan"),
            pump2_um=float("nan"),
            probe_um=float("nan"),
            probe_reflectance=float("nan"),
            pump1_reflectance=float("nan"),
            pump2_reflectance=float("nan"),
            rotation_deg=float("nan"),
            abs_rotation_deg=-1.0,
            objective_summary="resonance_not_found",
        )

    # Optional strict local refinement of pump resonances.
    if str(args.pump_local_q_check).lower() == "strict":
        local = refine_local_resonances(
            spec=geom_spec,
            mats=mats_for_refl,
            targets_um={
                "pump1": float(selected["pump1_um"]),
                "pump2": float(selected["pump2_um"]),
            },
            linewidth_level=float(args.resonance_linewidth_level),
            resolution=int(args.local_q_resolution),
            nfreq=int(args.local_q_nfreq),
            decay_threshold=float(args.local_q_decay_threshold),
            window_um=float(args.local_q_window_um),
        )
        if "pump1" not in local or "pump2" not in local:
            return Candidate(
                profile=profile,
                N_per=n_per,
                t_sin_um=design["t_sin_um"],
                t_sio2_um=design["t_sio2_um"],
                L_cav_um=design["L_cav_um"],
                pump1_um=float("nan"),
                pump2_um=float("nan"),
                probe_um=float("nan"),
                probe_reflectance=float("nan"),
                pump1_reflectance=float("nan"),
                pump2_reflectance=float("nan"),
                rotation_deg=float("nan"),
                abs_rotation_deg=-1.0,
                objective_summary="local_q_not_found",
            )
        for key in ("pump1", "pump2"):
            qv = float(local[key].get("Q", float("nan")))
            depth = float(local[key].get("depth", float("nan")))
            if float(args.pump_min_q) > 0.0 and (not np.isfinite(qv) or qv < float(args.pump_min_q)):
                return Candidate(
                    profile=profile,
                    N_per=n_per,
                    t_sin_um=design["t_sin_um"],
                    t_sio2_um=design["t_sio2_um"],
                    L_cav_um=design["L_cav_um"],
                    pump1_um=float("nan"),
                    pump2_um=float("nan"),
                    probe_um=float("nan"),
                    probe_reflectance=float("nan"),
                    pump1_reflectance=float("nan"),
                    pump2_reflectance=float("nan"),
                    rotation_deg=float("nan"),
                    abs_rotation_deg=-1.0,
                    objective_summary="local_q_below_threshold",
                )
            if float(args.pump_min_depth) > 0.0 and (
                (not np.isfinite(depth)) or depth < float(args.pump_min_depth)
            ):
                return Candidate(
                    profile=profile,
                    N_per=n_per,
                    t_sin_um=design["t_sin_um"],
                    t_sio2_um=design["t_sio2_um"],
                    L_cav_um=design["L_cav_um"],
                    pump1_um=float("nan"),
                    pump2_um=float("nan"),
                    probe_um=float("nan"),
                    probe_reflectance=float("nan"),
                    pump1_reflectance=float("nan"),
                    pump2_reflectance=float("nan"),
                    rotation_deg=float("nan"),
                    abs_rotation_deg=-1.0,
                    objective_summary="local_depth_below_threshold",
                )
        selected["pump1_um"] = float(local["pump1"]["lam"])
        selected["pump2_um"] = float(local["pump2"]["lam"])
        selected["pump1_R"] = float(local["pump1"]["R"])
        selected["pump2_R"] = float(local["pump2"]["R"])
        selected["pump1_Q"] = float(local["pump1"]["Q"])
        selected["pump2_Q"] = float(local["pump2"]["Q"])
        selected["pump1_depth"] = float(local["pump1"]["depth"])
        selected["pump2_depth"] = float(local["pump2"]["depth"])

    modes_spec = build_modes_spec(
        probe_um=selected["probe_um"],
        pump1_um=selected["pump1_um"],
        pump2_um=selected["pump2_um"],
    )

    geom_path.write_text(json.dumps(geom_spec, indent=2), encoding="utf-8")
    modes_path.write_text(json.dumps(modes_spec, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        "faraday_meep_fp_circ.py",
        "--mode",
        args.objective_mode,
        "--dim",
        "1",
        "--resolution",
        str(int(args.objective_resolution)),
        "--materials",
        args.materials,
        "--high-index-material",
        str(args.high_index_material),
        "--pump-intensity",
        str(float(args.pump_intensity)),
        "--kappa-ref-lambda",
        str(float(args.kappa_ref_lambda)),
        "--high-index-n2",
        str(float(args.high_index_n2)),
        "--decay-threshold",
        str(float(args.objective_decay_threshold)),
        "--geometry-file",
        str(geom_path),
        "--cavity-modes-file",
        str(modes_path),
        "--output-dir",
        str(out_dir),
    ]
    if args.materials == "fit":
        cmd.extend(["--sin-fit", args.sin_fit, "--sio2-fit", args.sio2_fit])
        cmd.extend(["--fit-window", str(args.fit_window[0]), str(args.fit_window[1])])
        cmd.extend(["--fit-poles", str(int(args.fit_poles))])
    if args.nH is not None:
        cmd.extend(["--nH", str(float(args.nH))])
    if args.kH is not None:
        cmd.extend(["--kH", str(float(args.kH))])
    if args.nL is not None:
        cmd.extend(["--nL", str(float(args.nL))])

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parent),
        capture_output=True,
        text=True,
        env=env,
    )
    stdout_text = completed.stdout or ""
    stderr_text = completed.stderr or ""
    (work / "faraday_stdout.log").write_text(stdout_text, encoding="utf-8")
    (work / "faraday_stderr.log").write_text(stderr_text, encoding="utf-8")

    summary_path = out_dir / "faraday_summary.json"
    if completed.returncode != 0 or (not summary_path.exists()):
        err_tail = stderr_text.strip().splitlines()[-3:] if stderr_text.strip() else []
        err_brief = " | ".join(err_tail) if err_tail else "no stderr captured"
        print(
            "[warn]",
            f"objective eval failed: tag={tag}",
            f"returncode={completed.returncode}",
            f"reason={err_brief}",
        )
        return Candidate(
            profile=profile,
            N_per=n_per,
            t_sin_um=design["t_sin_um"],
            t_sio2_um=design["t_sio2_um"],
            L_cav_um=design["L_cav_um"],
            pump1_um=selected["pump1_um"],
            pump2_um=selected["pump2_um"],
            probe_um=selected["probe_um"],
            probe_reflectance=selected["probe_R"],
            pump1_reflectance=selected["pump1_R"],
            pump2_reflectance=selected["pump2_R"],
            rotation_deg=float("nan"),
            abs_rotation_deg=-1.0,
            objective_summary=f"failed returncode={completed.returncode}; see {work/'faraday_stderr.log'}",
            score=-1.0,
            probe_q=float(selected.get("probe_Q", float("nan"))),
            pump1_q=float(selected.get("pump1_Q", float("nan"))),
            pump2_q=float(selected.get("pump2_Q", float("nan"))),
            probe_depth=float(selected.get("probe_depth", float("nan"))),
            pump1_depth=float(selected.get("pump1_depth", float("nan"))),
            pump2_depth=float(selected.get("pump2_depth", float("nan"))),
        )

    try:
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("[warn]", f"failed to parse summary {summary_path}: {exc}")
        return Candidate(
            profile=profile,
            N_per=n_per,
            t_sin_um=design["t_sin_um"],
            t_sio2_um=design["t_sio2_um"],
            L_cav_um=design["L_cav_um"],
            pump1_um=selected["pump1_um"],
            pump2_um=selected["pump2_um"],
            probe_um=selected["probe_um"],
            probe_reflectance=selected["probe_R"],
            pump1_reflectance=selected["pump1_R"],
            pump2_reflectance=selected["pump2_R"],
            rotation_deg=float("nan"),
            abs_rotation_deg=-1.0,
            objective_summary=f"summary_parse_failed: {summary_path}",
            score=-1.0,
            probe_q=float(selected.get("probe_Q", float("nan"))),
            pump1_q=float(selected.get("pump1_Q", float("nan"))),
            pump2_q=float(selected.get("pump2_Q", float("nan"))),
            probe_depth=float(selected.get("probe_depth", float("nan"))),
            pump1_depth=float(selected.get("pump1_depth", float("nan"))),
            pump2_depth=float(selected.get("pump2_depth", float("nan"))),
        )

    rot, abs_rot, score, quality = _objective_from_summary_data(
        summary_data,
        objective_metric=str(getattr(args, "objective_metric", "abs_rotation")),
        quality_std_ref_deg=float(getattr(args, "quality_std_ref_deg", 15.0)),
        quality_pump_dom_ref=float(getattr(args, "quality_pump_dom_ref", 0.2)),
        quality_pump_balance_sigma_dec=float(
            getattr(args, "quality_pump_balance_sigma_dec", 0.35)
        ),
    )
    return Candidate(
        profile=profile,
        N_per=n_per,
        t_sin_um=design["t_sin_um"],
        t_sio2_um=design["t_sio2_um"],
        L_cav_um=design["L_cav_um"],
        pump1_um=selected["pump1_um"],
        pump2_um=selected["pump2_um"],
        probe_um=selected["probe_um"],
        probe_reflectance=selected["probe_R"],
        pump1_reflectance=selected["pump1_R"],
        pump2_reflectance=selected["pump2_R"],
        rotation_deg=float(rot),
        abs_rotation_deg=float(abs_rot),
        objective_summary=str(summary_path),
        score=float(score),
        quality_factor=float(quality.get("quality_factor", float("nan"))),
        quality_dolp_tail=float(quality.get("dolp_tail", float("nan"))),
        quality_theta_std_deg=float(quality.get("theta_std_deg", float("nan"))),
        quality_s0_rel_max=float(quality.get("s0_rel_max", float("nan"))),
        probe_q=float(selected.get("probe_Q", float("nan"))),
        pump1_q=float(selected.get("pump1_Q", float("nan"))),
        pump2_q=float(selected.get("pump2_Q", float("nan"))),
        probe_depth=float(selected.get("probe_depth", float("nan"))),
        pump1_depth=float(selected.get("pump1_depth", float("nan"))),
        pump2_depth=float(selected.get("pump2_depth", float("nan"))),
    )


def objective_search_profile(
    profile: str,
    args: argparse.Namespace,
    n_sin_ref: float,
    n_sio2_ref: float,
    mat_sin: mp.Medium,
    mat_sio2: mp.Medium,
    eval_root: Path,
    rng: np.random.Generator,
    progress_state: Optional[Dict[str, int]] = None,
    est_profile_evals: Optional[int] = None,
) -> Tuple[Optional[Candidate], Dict]:
    cfg = search_config(
        profile,
        n_sin_ref,
        n_sio2_ref,
        float(args.cavity_min_length),
        float(args.cavity_max_length),
    )
    profile_start_t = time.time()
    if est_profile_evals is not None:
        print(f"[progress] profile={profile} target_evals~{est_profile_evals}")

    eval_counter = 0
    eval_completed = 0
    cache: Dict[Tuple, Candidate] = {}
    workers = max(1, int(getattr(args, "workers", 1)))
    workers = min(workers, max(1, int(os.cpu_count() or 1)))
    args_dict = dict(vars(args))
    material_payload = build_material_payload(args)

    executor: Optional[cf.ProcessPoolExecutor] = None
    if workers > 1:
        print(f"[parallel] profile={profile} workers={workers}")
        try:
            shared_context = {
                "args_dict": args_dict,
                "material_payload": material_payload,
                "n_sin_ref": float(n_sin_ref),
                "n_sio2_ref": float(n_sio2_ref),
                "eval_root": str(eval_root),
            }
            executor = cf.ProcessPoolExecutor(
                max_workers=workers,
                initializer=init_objective_worker,
                initargs=(shared_context,),
            )
        except Exception as exc:
            print(
                "[warn]",
                f"profile={profile}",
                "parallel worker pool unavailable; falling back to serial execution.",
                f"reason={exc}",
            )
            workers = 1
            executor = None

    def cache_key(n_per: int, design: Dict[str, float]) -> Tuple:
        return (
            profile,
            int(n_per),
            round(float(design["t_sin_um"]), 7),
            round(float(design["t_sio2_um"]), 7),
            round(float(design["L_cav_um"]), 7),
        )

    def report_overall_progress() -> None:
        if progress_state is None:
            return
        progress_state["done"] = int(progress_state.get("done", 0)) + 1
        done = progress_state["done"]
        total_est = int(progress_state.get("total_est", 0))
        pct = (100.0 * done / total_est) if total_est > 0 else float("nan")
        overall_elapsed = time.time() - float(progress_state.get("start_ts", time.time()))
        overall_eta = float("nan")
        if done > 0 and total_est > done:
            overall_eta = overall_elapsed * (float(total_est - done) / float(done))
        print(
            f"[progress] overall {done}/{total_est} ({pct:.1f}%) "
            f"elapsed={format_duration(overall_elapsed)} "
            f"eta~{format_duration(overall_eta) if np.isfinite(overall_eta) else 'n/a'}"
        )

    def make_invalid_candidate(n_per: int, design: Dict[str, float], reason: str) -> Candidate:
        return Candidate(
            profile=profile,
            N_per=int(n_per),
            t_sin_um=float(design["t_sin_um"]),
            t_sio2_um=float(design["t_sio2_um"]),
            L_cav_um=float(design["L_cav_um"]),
            pump1_um=float("nan"),
            pump2_um=float("nan"),
            probe_um=float("nan"),
            probe_reflectance=float("nan"),
            pump1_reflectance=float("nan"),
            pump2_reflectance=float("nan"),
            rotation_deg=float("nan"),
            abs_rotation_deg=-1.0,
            objective_summary=reason,
            score=-1.0,
        )

    def log_candidate(cand: Candidate, n_per: int, eval_id: int, eval_elapsed: float) -> None:
        nonlocal eval_completed
        eval_completed += 1
        profile_elapsed = time.time() - profile_start_t
        eval_rate = profile_elapsed / max(eval_completed, 1)
        profile_eta = float("nan")
        if est_profile_evals is not None and est_profile_evals > eval_completed:
            profile_eta = eval_rate * float(est_profile_evals - eval_completed)
        print(
            "[obj]",
            f"profile={profile}",
            f"N={n_per}",
            f"eval={eval_id}",
            f"t_eval={format_duration(eval_elapsed)}",
            f"t_profile={format_duration(profile_elapsed)}",
            f"eta_profile~{format_duration(profile_eta) if np.isfinite(profile_eta) else 'n/a'}",
            f"score={candidate_score(cand):.6f}",
            f"abs_rot={cand.abs_rotation_deg:.6f}",
            f"Lc={cand.L_cav_um:.4f}",
            f"probe={cand.probe_um:.4f}",
            f"pumps=({cand.pump1_um:.4f},{cand.pump2_um:.4f})",
            f"Q=(probe:{cand.probe_q:.1f},p1:{cand.pump1_q:.1f},p2:{cand.pump2_q:.1f})",
            f"depth=(probe:{cand.probe_depth:.3f},p1:{cand.pump1_depth:.3f},p2:{cand.pump2_depth:.3f})",
            f"R=(p:{cand.probe_reflectance:.3f},p1:{cand.pump1_reflectance:.3f},p2:{cand.pump2_reflectance:.3f})",
        )

    def evaluate_batch(batch: Sequence[Tuple[int, Sequence[float]]]) -> List[Candidate]:
        nonlocal eval_counter
        if not batch:
            return []

        results: List[Optional[Candidate]] = [None] * len(batch)
        pending: List[Dict[str, object]] = []
        for idx, (n_per, x) in enumerate(batch):
            x_clip = clip_to_bounds(x, cfg.bounds)
            design = decode_design_vector(x_clip, cfg.bounds)
            key = cache_key(int(n_per), design)
            cached = cache.get(key)
            if cached is not None:
                results[idx] = cached
                continue

            eval_counter += 1
            eval_id = int(eval_counter)
            report_overall_progress()
            pending.append(
                {
                    "idx": int(idx),
                    "n_per": int(n_per),
                    "design": design,
                    "key": key,
                    "eval_id": eval_id,
                }
            )

        # Keep single-evaluation calls local to avoid process overhead in inner Powell loops.
        use_parallel = executor is not None and len(pending) > 1
        if use_parallel:
            fut_to_job: Dict[cf.Future, Dict[str, object]] = {}
            for job in pending:
                job["t_submit"] = time.time()
                payload = {
                    "profile": profile,
                    "n_per": int(job["n_per"]),
                    "design": dict(job["design"]),
                    "eval_id": int(job["eval_id"]),
                }
                fut = executor.submit(objective_run_worker, payload)
                fut_to_job[fut] = job
            for fut in cf.as_completed(fut_to_job):
                job = fut_to_job[fut]
                elapsed = time.time() - float(job["t_submit"])
                try:
                    _eid, cand = fut.result()
                except Exception as exc:
                    print(
                        "[warn]",
                        f"profile={profile}",
                        f"N={int(job['n_per'])}",
                        f"eval={int(job['eval_id'])}",
                        f"worker_exception={exc}",
                    )
                    cand = make_invalid_candidate(
                        int(job["n_per"]), dict(job["design"]), reason="worker_exception"
                    )
                cache[job["key"]] = cand
                results[int(job["idx"])] = cand
                log_candidate(cand, int(job["n_per"]), int(job["eval_id"]), elapsed)
        else:
            for job in pending:
                t0 = time.time()
                cand = objective_run(
                    profile=profile,
                    n_per=int(job["n_per"]),
                    design=dict(job["design"]),
                    args=args,
                    n_sin_ref=n_sin_ref,
                    n_sio2_ref=n_sio2_ref,
                    mat_sin=mat_sin,
                    mat_sio2=mat_sio2,
                    eval_root=eval_root,
                    eval_id=int(job["eval_id"]),
                )
                elapsed = time.time() - t0
                cache[job["key"]] = cand
                results[int(job["idx"])] = cand
                log_candidate(cand, int(job["n_per"]), int(job["eval_id"]), elapsed)

        out: List[Candidate] = []
        for idx, res in enumerate(results):
            if res is not None:
                out.append(res)
                continue
            # Defensive fallback: should not happen unless a worker failed before writing result.
            n_per, x = batch[idx]
            design = decode_design_vector(clip_to_bounds(x, cfg.bounds), cfg.bounds)
            key = cache_key(int(n_per), design)
            cached = cache.get(key)
            if cached is None:
                cached = make_invalid_candidate(int(n_per), design, reason="missing_result")
                cache[key] = cached
            out.append(cached)
        return out

    def evaluate_vector(n_per: int, x: Sequence[float]) -> Candidate:
        return evaluate_batch([(int(n_per), x)])[0]

    try:
        # Step 1: evaluate seed designs for each N_per.
        seed_jobs: List[Tuple[int, np.ndarray]] = []
        for n_per in range(int(args.mirror_min), int(args.mirror_max) + 1):
            for variant in range(int(args.seed_variants)):
                x0 = seed_vector(profile, cfg, n_sin_ref, n_per, variant=variant)
                # tiny random jitter to avoid symmetry lock.
                jitter = np.array([rng.normal(0.0, 0.01 * (hi - lo)) for (lo, hi) in cfg.bounds])
                x_seed = clip_to_bounds(x0 + jitter, cfg.bounds)
                seed_jobs.append((int(n_per), x_seed))

        seed_cands = evaluate_batch(seed_jobs)
        seeds: List[Tuple[float, int, np.ndarray, Candidate]] = []
        for (n_per, x_seed), cand in zip(seed_jobs, seed_cands):
            seeds.append(
                (candidate_score(cand), int(n_per), np.array(x_seed, dtype=float), cand)
            )

        if not seeds:
            return None, {"status": "no_seeds", "evaluations": 0}

        seeds.sort(key=lambda item: item[0], reverse=True)
        top = seeds[: max(1, int(args.top_mirrors_for_refine))]

        best = seeds[0][3]

        def observed_data_for_nper(n_per: int) -> Tuple[np.ndarray, np.ndarray]:
            xx: List[np.ndarray] = []
            yy: List[float] = []
            for cand in cache.values():
                if int(cand.N_per) != int(n_per):
                    continue
                score = candidate_score(cand)
                if not np.isfinite(score):
                    continue
                xx.append(np.array([cand.t_sin_um, cand.t_sio2_um, cand.L_cav_um], dtype=float))
                yy.append(float(score))
            if not xx:
                return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
            return np.vstack(xx), np.array(yy, dtype=float)

        def run_bayesian_refine_for_nper(n_per: int) -> None:
            init_min = max(int(args.bayes_init), len(cfg.bounds) + 1)
            while True:
                x_obs, _ = observed_data_for_nper(n_per)
                need = int(init_min - x_obs.shape[0])
                if need <= 0:
                    break
                boot_jobs = [(int(n_per), random_vector_in_bounds(cfg.bounds, rng)) for _ in range(need)]
                evaluate_batch(boot_jobs)

            for it in range(max(0, int(args.bayes_iters))):
                x_obs, y_obs = observed_data_for_nper(n_per)
                if x_obs.shape[0] < 2:
                    cand = evaluate_vector(n_per, random_vector_in_bounds(cfg.bounds, rng))
                    print(
                        "[bayes]",
                        f"profile={profile}",
                        f"N={n_per}",
                        f"iter={it + 1}/{int(args.bayes_iters)}",
                        f"score={candidate_score(cand):.6f}",
                        f"abs_rot={cand.abs_rotation_deg:.6f}",
                        "mode=random_bootstrap",
                    )
                    continue

                x_round = np.round(x_obs, 8)
                x_u, idx_u = np.unique(x_round, axis=0, return_index=True)
                y_u = y_obs[idx_u]
                if x_u.shape[0] < 2:
                    cand = evaluate_vector(n_per, random_vector_in_bounds(cfg.bounds, rng))
                    print(
                        "[bayes]",
                        f"profile={profile}",
                        f"N={n_per}",
                        f"iter={it + 1}/{int(args.bayes_iters)}",
                        f"score={candidate_score(cand):.6f}",
                        f"abs_rot={cand.abs_rotation_deg:.6f}",
                        "mode=random_unique",
                    )
                    continue

                n_pool = max(64, int(args.bayes_candidates))
                x_pool = np.array(
                    [random_vector_in_bounds(cfg.bounds, rng) for _ in range(n_pool)],
                    dtype=float,
                )
                x_best_obs = x_u[int(np.argmax(y_u))]
                span = np.array([hi - lo for (lo, hi) in cfg.bounds], dtype=float)
                local = x_best_obs[None, :] + rng.normal(
                    0.0, 0.08 * span, size=(max(8, n_pool // 4), len(cfg.bounds))
                )
                local = np.array([clip_to_bounds(row, cfg.bounds) for row in local], dtype=float)
                x_pool = np.vstack([x_pool, local, x_u])

                if HAVE_SKLEARN:
                    gp = GaussianProcessRegressor(
                        kernel=(
                            ConstantKernel(1.0, (1e-3, 1e3))
                            * Matern(length_scale=np.ones(len(cfg.bounds)), nu=2.5)
                            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1e-2))
                        ),
                        alpha=1e-9,
                        normalize_y=True,
                        n_restarts_optimizer=max(0, int(args.bayes_gp_restarts)),
                        random_state=int(args.random_seed) + int(n_per) * 1000 + it,
                    )
                    try:
                        gp.fit(x_u, y_u)
                        mu, sigma = gp.predict(x_pool, return_std=True)
                    except Exception:
                        cand = evaluate_vector(n_per, random_vector_in_bounds(cfg.bounds, rng))
                        print(
                            "[bayes]",
                            f"profile={profile}",
                            f"N={n_per}",
                            f"iter={it + 1}/{int(args.bayes_iters)}",
                            f"score={candidate_score(cand):.6f}",
                            f"abs_rot={cand.abs_rotation_deg:.6f}",
                            "mode=random_fit_fail",
                        )
                        continue
                else:
                    mu, sigma = gp_predict_rbf_fallback(
                        x_train=x_u,
                        y_train=y_u,
                        x_query=x_pool,
                        bounds=cfg.bounds,
                    )
                ei = expected_improvement(mu, sigma, best=float(np.max(y_u)), xi=float(args.bayes_xi))
                order = np.argsort(ei)[::-1]

                batch_sz = max(1, int(getattr(args, "bayes_batch_size", 1)))
                x_next_batch: List[np.ndarray] = []
                for idx in order:
                    x_try = clip_to_bounds(x_pool[int(idx)], cfg.bounds)
                    if np.min(np.linalg.norm(x_u - x_try[None, :], axis=1)) <= 1e-5:
                        continue
                    if x_next_batch:
                        d_batch = np.min(
                            [float(np.linalg.norm(x_try - x_prev)) for x_prev in x_next_batch]
                        )
                        if d_batch <= 1e-5:
                            continue
                    x_next_batch.append(x_try)
                    if len(x_next_batch) >= batch_sz:
                        break
                while len(x_next_batch) < batch_sz:
                    x_next_batch.append(random_vector_in_bounds(cfg.bounds, rng))

                cands = evaluate_batch([(n_per, x) for x in x_next_batch])
                best_ei = float(np.max(ei)) if ei.size else float("nan")
                for jj, cand in enumerate(cands, start=1):
                    print(
                        "[bayes]",
                        f"profile={profile}",
                        f"N={n_per}",
                        f"iter={it + 1}/{int(args.bayes_iters)}",
                        f"batch={jj}/{len(cands)}",
                        f"acq_ei={best_ei:.3e}",
                        f"score={candidate_score(cand):.6f}",
                        f"abs_rot={cand.abs_rotation_deg:.6f}",
                    )

        def candidate_to_vector(cand: Candidate) -> np.ndarray:
            return np.array([cand.t_sin_um, cand.t_sio2_um, cand.L_cav_um], dtype=float)

        def run_parallel_pattern_refine_for_nper(n_per: int, x_start: np.ndarray) -> Candidate:
            dim = len(cfg.bounds)
            span = np.array([hi - lo for (lo, hi) in cfg.bounds], dtype=float)
            x_curr = clip_to_bounds(np.array(x_start, dtype=float), cfg.bounds)
            cand_curr = evaluate_vector(n_per, x_curr)

            budget = max(1, int(args.maxfev))
            used = 0
            it = 0
            step = np.array([0.12 * s for s in span], dtype=float)
            step = np.maximum(step, 1e-3 * np.maximum(1.0, span))
            step_min = np.maximum(0.01 * span, 1e-4 * np.maximum(1.0, span))
            step_max = np.maximum(0.45 * span, step)
            target_batch = max(2 * dim, int(max(1, workers)))

            while used < budget and bool(np.any(step > step_min)):
                it += 1
                remaining = max(0, budget - used)
                if remaining <= 0:
                    break

                trial_vectors: List[np.ndarray] = []
                # Coordinate directions first (good local search signal).
                for j in range(dim):
                    for sign in (+1.0, -1.0):
                        if len(trial_vectors) >= remaining:
                            break
                        x_try = x_curr.copy()
                        x_try[j] += sign * step[j]
                        trial_vectors.append(clip_to_bounds(x_try, cfg.bounds))
                    if len(trial_vectors) >= remaining:
                        break

                # Fill the rest with random directions to keep workers utilized.
                want = min(target_batch, remaining)
                while len(trial_vectors) < want:
                    direction = rng.normal(size=dim)
                    nrm = float(np.linalg.norm(direction))
                    if (not np.isfinite(nrm)) or nrm <= 1e-12:
                        continue
                    direction /= nrm
                    x_try = clip_to_bounds(x_curr + direction * step, cfg.bounds)
                    if all(float(np.linalg.norm(x_try - prev)) > 1e-6 for prev in trial_vectors):
                        trial_vectors.append(x_try)
                        continue
                    x_jitter = clip_to_bounds(
                        x_try + rng.normal(0.0, 0.02, size=dim) * step, cfg.bounds
                    )
                    if all(float(np.linalg.norm(x_jitter - prev)) > 1e-6 for prev in trial_vectors):
                        trial_vectors.append(x_jitter)

                if not trial_vectors:
                    break

                cand_trials = evaluate_batch([(int(n_per), x_try) for x_try in trial_vectors])
                used += len(trial_vectors)
                best_trial = max(
                    cand_trials,
                    key=lambda c: candidate_score(c),
                )
                improved = (
                    np.isfinite(candidate_score(best_trial))
                    and np.isfinite(candidate_score(cand_curr))
                    and (candidate_score(best_trial) > candidate_score(cand_curr) + 1e-8)
                )

                if improved:
                    x_prev = x_curr.copy()
                    x_curr = clip_to_bounds(candidate_to_vector(best_trial), cfg.bounds)
                    cand_curr = best_trial
                    direction = x_curr - x_prev
                    dir_nrm = float(np.linalg.norm(direction))
                    if dir_nrm > 1e-9 and used < budget:
                        x_pattern = clip_to_bounds(x_curr + direction, cfg.bounds)
                        cand_pattern = evaluate_vector(n_per, x_pattern)
                        used += 1
                        if (
                            np.isfinite(candidate_score(cand_pattern))
                            and (candidate_score(cand_pattern) > candidate_score(cand_curr) + 1e-8)
                        ):
                            x_curr = clip_to_bounds(candidate_to_vector(cand_pattern), cfg.bounds)
                            cand_curr = cand_pattern
                            step = np.minimum(step * 1.25, step_max)
                        else:
                            step = np.minimum(step * 1.10, step_max)
                    else:
                        step = np.minimum(step * 1.10, step_max)
                else:
                    step *= 0.60

                print(
                    "[powell-pattern]",
                    f"profile={profile}",
                    f"N={n_per}",
                    f"iter={it}",
                    f"used={used}/{budget}",
                    f"score={candidate_score(cand_curr):.6f}",
                    f"abs_rot={cand_curr.abs_rotation_deg:.6f}",
                    f"step_max={float(np.max(step)):.5f}",
                )

            return cand_curr

        # Step 2: refinement on best mirror counts.
        refine_mode = str(args.optimizer).lower()
        if refine_mode == "bayes" and (not HAVE_SKLEARN):
            print("[info] sklearn unavailable; using built-in RBF GP backend for Bayesian refinement.")
        powell_mode_req = str(getattr(args, "powell_mode", "auto")).lower()
        powell_backends_used: List[str] = []

        for _, n_per, x_start, _ in top:
            if refine_mode == "bayes":
                run_bayesian_refine_for_nper(int(n_per))
                continue

            if powell_mode_req == "pattern":
                powell_backend = "pattern"
            elif powell_mode_req == "scipy":
                powell_backend = "scipy" if HAVE_SCIPY else "pattern"
            else:
                # Auto policy: SciPy in serial, parallel pattern-search with workers > 1.
                if workers > 1:
                    powell_backend = "pattern"
                elif HAVE_SCIPY:
                    powell_backend = "scipy"
                else:
                    powell_backend = "pattern"

            if (powell_backend == "pattern") and (powell_mode_req == "scipy") and (not HAVE_SCIPY):
                print("[warn] SciPy unavailable; falling back from --powell-mode=scipy to pattern.")
            if powell_backend not in powell_backends_used:
                powell_backends_used.append(powell_backend)

            print(
                "[powell]",
                f"profile={profile}",
                f"N={n_per}",
                f"backend={powell_backend}",
                f"workers={workers}",
                f"maxfev={int(args.maxfev)}",
            )

            if powell_backend == "scipy":

                def fun(xx: np.ndarray) -> float:
                    cand = evaluate_vector(n_per, xx)
                    score = candidate_score(cand)
                    if not np.isfinite(score) or score < 0:
                        return 1e6
                    return -float(score)

                res = minimize(
                    fun,
                    x_start,
                    method="Powell",
                    bounds=cfg.bounds,
                    options={
                        "maxfev": int(args.maxfev),
                        "xtol": 1e-2,
                        "ftol": 1e-3,
                        "disp": False,
                    },
                )
                cand_ref = evaluate_vector(n_per, res.x)
            else:
                cand_ref = run_parallel_pattern_refine_for_nper(int(n_per), x_start)

            if candidate_score(cand_ref) > candidate_score(best):
                best = cand_ref

        def classify_invalid_reason(summary: str) -> str:
            s = str(summary or "").strip()
            if not s:
                return "unknown_invalid"
            if s.startswith("failed returncode="):
                return "solver_runtime_failure"
            if s.startswith("summary_parse_failed"):
                return "summary_parse_failed"
            if s.startswith("worker_exception"):
                return "worker_exception"
            if s.startswith("missing_result"):
                return "missing_result"
            if s.startswith("resonance_not_found"):
                return "resonance_not_found"
            if s.startswith("local_q_not_found"):
                return "local_q_not_found"
            if s.startswith("local_q_below_threshold"):
                return "local_q_below_threshold"
            if s.startswith("local_depth_below_threshold"):
                return "local_depth_below_threshold"
            return s.split(";", 1)[0][:120]

        all_candidates = list(cache.values())
        all_scores = np.array([candidate_score(c) for c in all_candidates], dtype=float)
        valid_mask = np.isfinite(all_scores) & (all_scores >= 0.0)
        valid: List[Candidate] = [
            c for c, ok in zip(all_candidates, valid_mask.tolist()) if bool(ok)
        ]
        invalid_reasons = Counter(
            classify_invalid_reason(c.objective_summary)
            for c, ok in zip(all_candidates, valid_mask.tolist())
            if not bool(ok)
        )
        valid_scores = np.array([candidate_score(c) for c in valid], dtype=float)
        valid_abs_rot = np.array([float(c.abs_rotation_deg) for c in valid], dtype=float)
        seed_scores = np.array([float(item[0]) for item in seeds], dtype=float)
        seed_valid_scores = seed_scores[np.isfinite(seed_scores) & (seed_scores >= 0.0)]
        seed_best_score = (
            float(np.max(seed_valid_scores)) if seed_valid_scores.size else float("nan")
        )
        elapsed_s = float(time.time() - profile_start_t)
        final_best_score = (
            float(np.max(valid_scores)) if valid_scores.size else float("nan")
        )
        score_improvement = (
            float(final_best_score - seed_best_score)
            if (np.isfinite(final_best_score) and np.isfinite(seed_best_score))
            else float("nan")
        )
        score_improvement_rel = (
            float(score_improvement / max(abs(seed_best_score), 1e-12))
            if np.isfinite(score_improvement) and np.isfinite(seed_best_score)
            else float("nan")
        )
        invalid_reason_counts = {
            k: int(v) for k, v in sorted(invalid_reasons.items(), key=lambda kv: (-kv[1], kv[0]))
        }

        success_metrics = {
            "requested_eval_estimate": int(est_profile_evals) if est_profile_evals is not None else None,
            "evaluations_attempted": int(eval_counter),
            "unique_candidates": int(len(all_candidates)),
            "valid_candidates": int(len(valid)),
            "invalid_candidates": int(len(all_candidates) - len(valid)),
            "valid_fraction": (
                float(len(valid) / max(len(all_candidates), 1))
                if all_candidates
                else float("nan")
            ),
            "invalid_reason_counts": invalid_reason_counts,
            "seed_valid_count": int(seed_valid_scores.size),
            "seed_best_score": float(seed_best_score),
            "final_best_score": float(final_best_score),
            "score_improvement_from_seed_best": float(score_improvement),
            "relative_improvement_from_seed_best": float(score_improvement_rel),
            "best_improved_vs_seed": bool(
                np.isfinite(score_improvement) and (score_improvement > 1e-9)
            ),
            "valid_score_mean": (
                float(np.mean(valid_scores)) if valid_scores.size else float("nan")
            ),
            "valid_score_median": (
                float(np.median(valid_scores)) if valid_scores.size else float("nan")
            ),
            "valid_score_std": (
                float(np.std(valid_scores)) if valid_scores.size else float("nan")
            ),
            "valid_abs_rotation_mean_deg": (
                float(np.mean(valid_abs_rot)) if valid_abs_rot.size else float("nan")
            ),
            "valid_abs_rotation_median_deg": (
                float(np.median(valid_abs_rot)) if valid_abs_rot.size else float("nan")
            ),
            "elapsed_s": float(elapsed_s),
            "eval_rate_per_min": (
                float(eval_counter / max(elapsed_s / 60.0, 1e-9))
                if eval_counter > 0
                else 0.0
            ),
            "estimated_budget_utilization": (
                float(eval_counter / max(int(est_profile_evals), 1))
                if est_profile_evals is not None
                else float("nan")
            ),
        }

        if not valid:
            return None, {
                "status": "no_valid_candidates",
                "evaluations": int(eval_counter),
                "objective_metric": str(args.objective_metric),
                "best_profile": profile,
                "optimizer": str(args.optimizer).lower(),
                "workers_requested": int(args.workers),
                "workers_effective": int(workers),
                "parallel_enabled": bool(executor is not None),
                "elapsed_s": float(elapsed_s),
                "success_metrics": success_metrics,
            }
        best = max(valid, key=lambda c: candidate_score(c))

        diag = {
            "status": "ok",
            "evaluations": int(eval_counter),
            "best_score": float(candidate_score(best)),
            "best_abs_rotation_deg": float(best.abs_rotation_deg),
            "best_rotation_deg": float(best.rotation_deg),
            "objective_metric": str(args.objective_metric),
            "best_profile": profile,
            "optimizer": str(args.optimizer).lower(),
            "powell_mode_requested": powell_mode_req if refine_mode == "powell" else "n/a",
            "powell_backends_used": list(powell_backends_used),
            "workers_requested": int(args.workers),
            "workers_effective": int(workers),
            "parallel_enabled": bool(executor is not None),
            "elapsed_s": float(elapsed_s),
            "success_metrics": success_metrics,
        }
        return best, diag
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def build_1d_geometry_from_spec(
    spec: Dict,
    mats: Dict[str, mp.Medium],
    margin_z: float = 0.4,
) -> Tuple[List[mp.Block], float, Tuple[float, float]]:
    pad_air = float(spec["pads"]["air_um"])
    pad_sub = float(spec["pads"]["substrate_um"])
    dpml = float(spec["pads"]["pml_um"])
    left_layers = spec["mirrors"]["left"]
    right_layers = spec["mirrors"]["right"]
    cavity_thk = float(spec["cavity"]["L_um"])
    cavity_mat = mats[spec["cavity"]["mat"]]

    def sum_layers(layers: Sequence[Dict[str, float]]) -> float:
        return sum(float(layer["thk_um"]) for layer in layers)

    stack_len = pad_air + sum_layers(left_layers) + cavity_thk + sum_layers(right_layers) + pad_sub
    cell_z = stack_len + 2.0 * dpml + margin_z

    geom: List[mp.Block] = []
    z = -0.5 * cell_z + dpml
    z += pad_air

    def add_block(thk: float, mat: mp.Medium) -> None:
        nonlocal z
        c = z + 0.5 * thk
        geom.append(
            mp.Block(
                center=mp.Vector3(0, 0, c),
                size=mp.Vector3(mp.inf, mp.inf, thk),
                material=mat,
            )
        )
        z += thk

    for layer in left_layers:
        add_block(float(layer["thk_um"]), mats[layer["mat"]])

    cav_z0 = z
    add_block(cavity_thk, cavity_mat)
    cav_z1 = z

    for layer in right_layers:
        add_block(float(layer["thk_um"]), mats[layer["mat"]])

    add_block(pad_sub, mats.get("SiO2", list(mats.values())[0]))
    return geom, cell_z, (cav_z0, cav_z1)


def epsilon_profile(spec: Dict, n_sin: float, n_sio2: float) -> Tuple[np.ndarray, np.ndarray]:
    pad_air = float(spec["pads"]["air_um"])
    pad_sub = float(spec["pads"]["substrate_um"])
    left_layers = spec["mirrors"]["left"]
    right_layers = spec["mirrors"]["right"]
    L_cav = float(spec["cavity"]["L_um"])

    segments: List[Tuple[float, float, float]] = []
    z = 0.0
    z += pad_air

    for layer in left_layers:
        thk = float(layer["thk_um"])
        eps = n_sin * n_sin if layer["mat"] == "SiN" else n_sio2 * n_sio2
        segments.append((z, z + thk, eps))
        z += thk

    segments.append((z, z + L_cav, n_sin * n_sin))
    z += L_cav

    for layer in right_layers:
        thk = float(layer["thk_um"])
        eps = n_sin * n_sin if layer["mat"] == "SiN" else n_sio2 * n_sio2
        segments.append((z, z + thk, eps))
        z += thk

    segments.append((z, z + pad_sub, n_sio2 * n_sio2))
    z += pad_sub

    zz = np.linspace(0.0, z, 2200)
    eps = np.ones_like(zz)
    for z0, z1, ev in segments:
        m = (zz >= z0) & (zz <= z1)
        eps[m] = ev
    return zz, eps


def debug_reflectance(
    spec: Dict,
    mats: Dict[str, mp.Medium],
    resolution: int,
    nfreq: int,
    decay_threshold: float = 1e-7,
    wl_min: float = 0.6,
    wl_max: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    geom, cell_z, _ = build_1d_geometry_from_spec(spec, mats)
    dpml = float(spec["pads"]["pml_um"])

    wl_min = float(min(wl_min, wl_max))
    wl_max = float(max(wl_min + 1e-6, wl_max))
    fmin, fmax = 1.0 / wl_max, 1.0 / wl_min
    fcen, df = 0.5 * (fmin + fmax), (fmax - fmin)

    src_z = -0.5 * cell_z + dpml + 0.2
    refl_z = src_z + 0.1

    src = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ex,
            center=mp.Vector3(0, 0, src_z),
        )
    ]

    def make_sim(geometry: List[mp.Block]) -> mp.Simulation:
        return mp.Simulation(
            cell_size=mp.Vector3(0, 0, cell_z),
            geometry=geometry,
            sources=src,
            boundary_layers=[mp.PML(dpml)],
            default_material=mp.air,
            resolution=int(resolution),
            dimensions=1,
            force_complex_fields=True,
        )

    sim_ref = make_sim([])
    fr_ref = sim_ref.add_flux(fcen, df, int(nfreq), mp.FluxRegion(center=mp.Vector3(0, 0, refl_z)))
    sim_ref.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ex, mp.Vector3(0, 0, refl_z), float(decay_threshold)
        )
    )
    inc = np.asarray(mp.get_fluxes(fr_ref), dtype=float)
    freqs = np.asarray(mp.get_flux_freqs(fr_ref), dtype=float)
    ref_data = sim_ref.get_flux_data(fr_ref)

    sim = make_sim(geom)
    fr = sim.add_flux(fcen, df, int(nfreq), mp.FluxRegion(center=mp.Vector3(0, 0, refl_z)))
    sim.load_minus_flux_data(fr, ref_data)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ex, mp.Vector3(0, 0, refl_z), float(decay_threshold)
        )
    )
    refl = np.asarray(mp.get_fluxes(fr), dtype=float)

    den = np.where(np.abs(inc) > 1e-30, inc, np.nan)
    R = np.maximum(0.0, np.nan_to_num(-refl / den, nan=0.0, posinf=0.0, neginf=0.0))
    wl = 1.0 / freqs
    return wl, R


def refine_local_resonances(
    spec: Dict,
    mats: Dict[str, mp.Medium],
    targets_um: Dict[str, float],
    linewidth_level: float,
    resolution: int,
    nfreq: int,
    decay_threshold: float,
    window_um: float,
) -> Dict[str, Dict[str, float]]:
    if not targets_um:
        return {}
    vals = [float(v) for v in targets_um.values() if np.isfinite(v)]
    if not vals:
        return {}

    span = max(float(window_um), 0.03)
    half = 0.5 * span
    wl_lo = max(0.55, min(vals) - half)
    wl_hi = min(2.1, max(vals) + half)
    if wl_hi <= wl_lo + 1e-5:
        wl_hi = wl_lo + 0.05

    wl, rr = debug_reflectance(
        spec=spec,
        mats=mats,
        resolution=int(resolution),
        nfreq=int(nfreq),
        decay_threshold=float(decay_threshold),
        wl_min=float(wl_lo),
        wl_max=float(wl_hi),
    )
    dips = find_reflectance_dips(wl, rr, linewidth_level=float(linewidth_level))
    if not dips:
        return {}

    out: Dict[str, Dict[str, float]] = {}
    # Keep matching local to requested targets.
    max_detune = max(0.015, 0.5 * span)
    for key, lam0 in targets_um.items():
        lam0 = float(lam0)
        if not np.isfinite(lam0):
            continue
        d = min(dips, key=lambda x: abs(float(x["lam"]) - lam0))
        detune = abs(float(d["lam"]) - lam0)
        if detune > max_detune:
            continue
        out[key] = {
            "lam": float(d["lam"]),
            "R": float(d["R"]),
            "Q": float(d.get("Q", float("nan"))),
            "width_um": float(d.get("width_um", float("nan"))),
            "depth": float(d.get("depth", float("nan"))),
            "detune_um": float(detune),
        }
    return out


def debug_mode_profiles(
    spec: Dict,
    mats: Dict[str, mp.Medium],
    freqs: Dict[str, float],
    resolution: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Tuple[float, float]]:
    geom, cell_z, cav_bounds = build_1d_geometry_from_spec(spec, mats)
    dpml = float(spec["pads"]["pml_um"])

    fvals = np.array(list(freqs.values()), dtype=float)
    f0 = float(np.mean(fvals))
    fspan = float(np.max(fvals) - np.min(fvals))
    fwidth = max(0.08 * f0, 1.2 * fspan)

    src_z = -0.5 * cell_z + dpml + 0.2
    src = [
        mp.Source(
            mp.GaussianSource(f0, fwidth=fwidth),
            component=mp.Ex,
            center=mp.Vector3(0, 0, src_z),
        )
    ]

    sim = mp.Simulation(
        cell_size=mp.Vector3(0, 0, cell_z),
        geometry=geom,
        sources=src,
        boundary_layers=[mp.PML(dpml)],
        default_material=mp.air,
        resolution=int(resolution),
        dimensions=1,
        force_complex_fields=True,
    )

    monitor_len = cell_z - 2.0 * dpml - 0.02
    vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 0, monitor_len))
    order = list(freqs.keys())
    dft = sim.add_dft_fields([mp.Ex, mp.Ey], [freqs[k] for k in order], where=vol)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            60, mp.Ex, mp.Vector3(0, 0, 0.0), 1e-7
        )
    )

    ex0 = np.asarray(sim.get_dft_array(dft, mp.Ex, 0))
    z = np.linspace(-0.5 * monitor_len, 0.5 * monitor_len, ex0.size)

    profiles: Dict[str, np.ndarray] = {}
    for i, key in enumerate(order):
        ex = np.asarray(sim.get_dft_array(dft, mp.Ex, i))
        ey = np.asarray(sim.get_dft_array(dft, mp.Ey, i))
        profiles[key] = np.sqrt(np.abs(ex) ** 2 + np.abs(ey) ** 2)

    return z, profiles, cav_bounds


def cavity_overlaps(
    z: np.ndarray,
    profiles: Dict[str, np.ndarray],
    cavity_bounds: Tuple[float, float],
) -> Dict[str, Dict[str, float]]:
    z0, z1 = cavity_bounds
    m = (z >= z0) & (z <= z1)
    if not np.any(m):
        return {}

    zc = z[m]
    out: Dict[str, Dict[str, float]] = {}
    keys = list(profiles.keys())
    for ki in keys:
        out[ki] = {}
        ai = np.asarray(profiles[ki][m], dtype=float)
        n_ai = np.sqrt(np.trapezoid(ai * ai, zc))
        for kj in keys:
            bj = np.asarray(profiles[kj][m], dtype=float)
            n_bj = np.sqrt(np.trapezoid(bj * bj, zc))
            den = n_ai * n_bj
            if den <= 1e-30:
                out[ki][kj] = float("nan")
            else:
                out[ki][kj] = float(np.trapezoid(ai * bj, zc) / den)
    return out


def plot_overlap_matrix(overlaps: Dict[str, Dict[str, float]], out_path: Path) -> None:
    keys = list(overlaps.keys())
    if not keys:
        return
    M = np.zeros((len(keys), len(keys)), dtype=float)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = overlaps[ki].get(kj, np.nan)

    fig = plt.figure(figsize=(5.4, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right")
    ax.set_yticklabels(keys)
    ax.set_title("Mode overlap in cavity")

    for i in range(len(keys)):
        for j in range(len(keys)):
            val = M[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="w", fontsize=8)

    fig.colorbar(im, ax=ax, label="normalized overlap")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_debug_artifacts(
    spec: Dict,
    modes: Dict,
    mats: Dict[str, mp.Medium],
    n_sin_ref: float,
    n_sio2_ref: float,
    args: argparse.Namespace,
) -> Dict:
    prefix = args.debug_prefix
    debug_out: Dict[str, str] = {}

    # Epsilon profile
    z_eps, eps = epsilon_profile(spec, n_sin_ref, n_sio2_ref)
    eps_path = Path(f"{prefix}_epsilon_profile.png").resolve()
    fig = plt.figure(figsize=(8.5, 3.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(z_eps, eps, lw=1.4)
    ax.set_xlabel("z (um)")
    ax.set_ylabel("epsilon")
    ax.set_title("Optimized cavity epsilon profile")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(eps_path, bbox_inches="tight")
    plt.close(fig)
    debug_out["epsilon_profile"] = str(eps_path)

    # Reflectance (Meep, no TMM)
    wl, R = debug_reflectance(spec, mats, resolution=args.debug_resolution, nfreq=args.debug_nfreq)
    refl_path = Path(f"{prefix}_reflectance_marked.png").resolve()

    lam_probe = float(modes["probe"]["lambda_um"])
    lam_p1 = float(modes["pump1"]["lambda_um"])
    lam_p2 = float(modes["pump2"]["lambda_um"])
    f_sb_plus = float(modes["sidebands"]["frequency_plus"])
    f_sb_minus = float(modes["sidebands"]["frequency_minus"])
    lam_sb_plus = float(1.0 / f_sb_plus)
    lam_sb_minus = float(1.0 / f_sb_minus) if f_sb_minus > 1e-12 else float("inf")

    fig = plt.figure(figsize=(8.8, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(1e3 * wl, R, lw=1.4, label="Meep reflectance")

    markers = [
        (lam_p1, "pump1", "tab:blue"),
        (lam_p2, "pump2", "tab:orange"),
        (lam_probe, "probe", "tab:green"),
        (lam_sb_minus, "sb-", "tab:red"),
        (lam_sb_plus, "sb+", "tab:purple"),
    ]
    for lam, label, color in markers:
        if not np.isfinite(lam):
            continue
        ax.axvline(1e3 * lam, color=color, ls="--", lw=1.0, alpha=0.8)
        y = float(np.interp(lam, wl[::-1], R[::-1])) if wl[0] > wl[-1] else float(np.interp(lam, wl, R))
        ax.plot([1e3 * lam], [y], marker="o", color=color, ms=4)
        ax.text(1e3 * lam + 2.0, y + 0.01, label, color=color, fontsize=8)

    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Reflectance with pump/probe/sidebands")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(refl_path, bbox_inches="tight")
    plt.close(fig)
    debug_out["reflectance_marked"] = str(refl_path)

    # Mode profiles at pumps/probe/sidebands + overlap matrix.
    freq_map = {
        "pump1": float(modes["pump1"]["frequency"]),
        "pump2": float(modes["pump2"]["frequency"]),
        "probe": float(modes["probe"]["frequency"]),
        "sb_minus": float(modes["sidebands"]["frequency_minus"]),
        "sb_plus": float(modes["sidebands"]["frequency_plus"]),
    }
    z_mode, profiles, cav_bounds = debug_mode_profiles(
        spec,
        mats,
        freq_map,
        resolution=args.debug_resolution,
    )

    mode_path = Path(f"{prefix}_mode_profiles.png").resolve()
    fig = plt.figure(figsize=(9.2, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    for k, arr in profiles.items():
        ax.plot(z_mode, arr, lw=1.2, label=k)
    ax.axvspan(cav_bounds[0], cav_bounds[1], color="gray", alpha=0.15, label="cavity")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("|E| (DFT)")
    ax.set_title("Mode profiles at pump/probe/sideband frequencies")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(mode_path, bbox_inches="tight")
    plt.close(fig)
    debug_out["mode_profiles"] = str(mode_path)

    overlaps = cavity_overlaps(z_mode, profiles, cav_bounds)
    overlap_plot = Path(f"{prefix}_mode_overlap_matrix.png").resolve()
    plot_overlap_matrix(overlaps, overlap_plot)
    debug_out["mode_overlap_matrix"] = str(overlap_plot)

    return {
        "plots": debug_out,
        "overlaps": overlaps,
        "cavity_bounds_um": [float(cav_bounds[0]), float(cav_bounds[1])],
    }


def write_outputs(
    best: Candidate,
    args: argparse.Namespace,
    n_sin_ref: float,
    n_sio2_ref: float,
    profile_diags: Dict[str, Dict],
    evals_total: int,
    objective_summary_file: Optional[str] = None,
    debug_info: Optional[Dict] = None,
    local_q_refined: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict:
    geom_spec = build_geometry_spec(
        n_sin=n_sin_ref,
        n_sio2=n_sio2_ref,
        dpml=args.dpml,
        pad_air=args.pad_air,
        pad_sub=args.pad_sub,
        n_per=best.N_per,
        t_sin_um=best.t_sin_um,
        t_sio2_um=best.t_sio2_um,
        L_cav_um=best.L_cav_um,
        high_index_material=str(args.high_index_material),
    )
    modes_spec = build_modes_spec(best.probe_um, best.pump1_um, best.pump2_um)

    out_geom = Path(args.out_geom).resolve()
    out_modes = Path(args.out_modes).resolve()
    out_report = Path(args.out_report).resolve()

    out_geom.write_text(json.dumps(geom_spec, indent=2), encoding="utf-8")
    out_modes.write_text(json.dumps(modes_spec, indent=2), encoding="utf-8")

    optimizer_mode = str(args.optimizer).lower()
    if optimizer_mode == "bayes":
        search_desc = "discrete-mirror sweep + Bayesian optimization (GP + expected improvement)"
    else:
        search_desc = "discrete-mirror sweep + budgeted derivative-free local search (Powell)"

    profile_status = {
        str(name): str(diag.get("status", "unknown"))
        for name, diag in profile_diags.items()
    }
    profiles_ok = [name for name, status in profile_status.items() if status == "ok"]
    profiles_failed = [name for name, status in profile_status.items() if status != "ok"]
    per_profile_evals = {
        str(name): int(diag.get("evaluations", 0))
        for name, diag in profile_diags.items()
    }
    per_profile_best_score = {
        str(name): float(diag.get("best_score", float("nan")))
        for name, diag in profile_diags.items()
    }
    per_profile_success = {
        str(name): dict(diag.get("success_metrics", {}))
        for name, diag in profile_diags.items()
    }
    valid_counts = [
        float(metrics.get("valid_candidates", float("nan")))
        for metrics in per_profile_success.values()
        if isinstance(metrics, dict)
    ]
    invalid_counts = [
        float(metrics.get("invalid_candidates", float("nan")))
        for metrics in per_profile_success.values()
        if isinstance(metrics, dict)
    ]
    valid_count_total = int(np.nansum(valid_counts)) if valid_counts else 0
    invalid_count_total = int(np.nansum(invalid_counts)) if invalid_counts else 0
    overall_valid_fraction = (
        float(valid_count_total / max(valid_count_total + invalid_count_total, 1))
        if (valid_count_total + invalid_count_total) > 0
        else float("nan")
    )

    report = {
        "meta": {
            "generated_on": utcnow_iso(),
            "script": "optimize_cavity_geometry.py",
            "objective": str(args.objective_metric),
            "tmm_used": False,
            "search_algorithm": search_desc,
        },
        "selected": {
            "profile": best.profile,
            "N_per": int(best.N_per),
            "t_sin_um": float(best.t_sin_um),
            "t_sio2_um": float(best.t_sio2_um),
            "L_cav_um": float(best.L_cav_um),
            "probe_um": float(best.probe_um),
            "pump1_um": float(best.pump1_um),
            "pump2_um": float(best.pump2_um),
            "resonance_reflectance": {
                "probe_R": float(best.probe_reflectance),
                "pump1_R": float(best.pump1_reflectance),
                "pump2_R": float(best.pump2_reflectance),
            },
            "resonance_depth": {
                "probe_depth": float(best.probe_depth),
                "pump1_depth": float(best.pump1_depth),
                "pump2_depth": float(best.pump2_depth),
            },
            "resonance_q_est": {
                "probe_Q": float(best.probe_q),
                "pump1_Q": float(best.pump1_q),
                "pump2_Q": float(best.pump2_q),
            },
            "pump_handedness": {"pump1": "sigma_plus", "pump2": "sigma_minus"},
            "rotation_deg": float(best.rotation_deg),
            "abs_rotation_deg": float(best.abs_rotation_deg),
            "objective_score": float(candidate_score(best)),
            "objective_metric": str(args.objective_metric),
            "quality_metrics": {
                "quality_factor": float(best.quality_factor),
                "dolp_tail": float(best.quality_dolp_tail),
                "theta_std_deg": float(best.quality_theta_std_deg),
                "s0_rel_max": float(best.quality_s0_rel_max),
                "quality_std_ref_deg": float(args.quality_std_ref_deg),
            },
            "objective_summary_file": objective_summary_file,
        },
        # Keep legacy-compatible fields for scripts expecting load_params(report)
        "best_theta": {
            "t_SiN_um": float(best.t_sin_um),
            "t_SiO2_um": float(best.t_sio2_um),
            "L_cav_um": float(best.L_cav_um),
            "cell_margin_um": 0.4,
            "N_per": int(best.N_per),
        },
        "sim": {
            "resolution_px_per_um": int(args.objective_resolution),
            "dpml_um": float(args.dpml),
            "pad_air_um": float(args.pad_air),
            "pad_sub_um": float(args.pad_sub),
            "objective_decay_threshold": float(args.objective_decay_threshold),
            "pump_intensity_w_cm2": float(args.pump_intensity),
        },
        "optimization": {
            "profile_mode": args.probe_target_mode,
            "mirror_range": [int(args.mirror_min), int(args.mirror_max)],
            "evaluations_total": int(evals_total),
            "profiles_considered": list(profile_diags.keys()),
            "materials": args.materials,
            "high_index_material": str(args.high_index_material),
            "high_index_constant_n": float(args.nH),
            "high_index_constant_k": float(args.kH),
            "high_index_kappa_ref_lambda_um": float(args.kappa_ref_lambda),
            "high_index_n2_m2_per_w": float(args.high_index_n2),
            "optimizer": optimizer_mode,
            "optimizer_settings": {
                "workers": int(args.workers),
                "seed_variants": int(args.seed_variants),
                "top_mirrors_for_refine": int(args.top_mirrors_for_refine),
                "maxfev_powell": int(args.maxfev),
                "powell_mode": str(args.powell_mode),
                "bayes_init": int(args.bayes_init),
                "bayes_iters": int(args.bayes_iters),
                "bayes_batch_size": int(args.bayes_batch_size),
                "bayes_candidates": int(args.bayes_candidates),
                "bayes_xi": float(args.bayes_xi),
                "bayes_gp_restarts": int(args.bayes_gp_restarts),
                "sklearn_available": bool(HAVE_SKLEARN),
                "bayes_backend": ("sklearn_gp" if HAVE_SKLEARN else "builtin_rbf_gp"),
                "objective_metric": str(args.objective_metric),
                "quality_std_ref_deg": float(args.quality_std_ref_deg),
                "quality_pump_dom_ref": float(args.quality_pump_dom_ref),
                "quality_pump_balance_sigma_dec": float(args.quality_pump_balance_sigma_dec),
            },
            "constraints": {
                "cavity_min_length_um": float(args.cavity_min_length),
                "cavity_max_length_um": float(args.cavity_max_length),
                "pump_range_um": [PUMP_MIN_UM, PUMP_MAX_UM],
                "probe_exact_um": PROBE_EXACT_UM,
                "probe_band_um": [PROBE_BAND_MIN_UM, PROBE_BAND_MAX_UM],
                "resonance_max_reflectance": float(args.resonance_max_R),
                "pump_min_q_est": float(args.pump_min_q),
                "pump_min_depth": float(args.pump_min_depth),
                "probe_min_depth": float(args.probe_min_depth),
                "resonance_linewidth_level": float(args.resonance_linewidth_level),
                "pump_local_q_check": str(args.pump_local_q_check),
                "local_q_window_um": float(args.local_q_window_um),
                "local_q_resolution": int(args.local_q_resolution),
                "local_q_nfreq": int(args.local_q_nfreq),
                "local_q_decay_threshold": float(args.local_q_decay_threshold),
                "probe_exact_tolerance_um": float(args.probe_exact_tol),
                "min_pump_separation_um": MIN_PUMP_SEP_UM,
            },
            "profile_status": profile_status,
            "profile_diagnostics": profile_diags,
            "success_metrics": {
                "profiles_total": int(len(profile_diags)),
                "profiles_ok": int(len(profiles_ok)),
                "profiles_failed": int(len(profiles_failed)),
                "profile_names_ok": profiles_ok,
                "profile_names_failed": profiles_failed,
                "overall_success": bool(len(profiles_ok) > 0),
                "selected_profile": str(best.profile),
                "selected_score": float(candidate_score(best)),
                "selected_abs_rotation_deg": float(best.abs_rotation_deg),
                "per_profile_evaluations": per_profile_evals,
                "per_profile_best_score": per_profile_best_score,
                "overall_valid_candidates": int(valid_count_total),
                "overall_invalid_candidates": int(invalid_count_total),
                "overall_valid_fraction": float(overall_valid_fraction),
            },
        },
        "files": {
            "geometry_json": str(out_geom),
            "cavity_modes_json": str(out_modes),
        },
    }

    if local_q_refined is not None:
        report["selected"]["local_q_refined"] = local_q_refined

    if debug_info is not None:
        report["debug"] = debug_info

    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    _configure_numeric_threads()
    args = parse_args()
    args = normalize_material_args(args)
    mp.verbosity(int(args.meep_verbosity))

    if float(args.cavity_max_length) <= float(args.cavity_min_length):
        raise SystemExit("--cavity-max-length must be larger than --cavity-min-length.")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1.")
    if int(args.bayes_batch_size) < 1:
        raise SystemExit("--bayes-batch-size must be >= 1.")

    if args.materials == "fit" and (not args.sin_fit or not args.sio2_fit):
        raise SystemExit("For --materials fit provide both --sin-fit and --sio2-fit.")

    rng = np.random.default_rng(int(args.random_seed))

    mat_sin, mat_sio2 = get_cavity_materials(
        model=args.materials,
        index_high=float(args.nH),
        kappa_high=float(args.kH),
        index_low=float(args.nL),
        high_index_material=str(args.high_index_material),
        kappa_ref_wavelength_um=float(args.kappa_ref_lambda),
        sin_csv=args.sin_fit,
        sio2_csv=args.sio2_fit,
        lam_min=int(args.fit_window[0]),
        lam_max=int(args.fit_window[1]),
        fit_poles=int(args.fit_poles),
    )

    n_sin_ref = float(material_index_at_wavelength(mat_sin, 0.9))
    n_sio2_ref = float(material_index_at_wavelength(mat_sio2, 0.9))

    profiles = (
        [args.probe_target_mode]
        if args.probe_target_mode in ("exact", "band")
        else ["exact", "band"]
    )
    n_mirrors = int(args.mirror_max) - int(args.mirror_min) + 1
    seeds_per_profile = n_mirrors * int(args.seed_variants)
    top_count = max(1, int(args.top_mirrors_for_refine))
    if str(args.optimizer).lower() == "bayes":
        est_refine_per_top = (
            max(0, int(args.bayes_init) - int(args.seed_variants))
            + int(args.bayes_iters) * max(1, int(args.bayes_batch_size))
        )
    else:
        est_refine_per_top = int(args.maxfev) + 1
    est_profile_evals = seeds_per_profile + top_count * est_refine_per_top
    progress_state = {
        "done": 0,
        "total_est": max(1, len(profiles) * est_profile_evals),
        "start_ts": time.time(),
    }
    print(
        f"[progress] estimated objective evaluations: ~{progress_state['total_est']} "
        f"(per profile ~{est_profile_evals})"
    )

    eval_root = Path(args.eval_root).resolve()
    if eval_root.exists():
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    best_overall: Optional[Candidate] = None
    profile_diags: Dict[str, Dict] = {}
    total_evals = 0

    for profile in profiles:
        print(f"[profile] {profile}")
        best_prof, diag = objective_search_profile(
            profile=profile,
            args=args,
            n_sin_ref=n_sin_ref,
            n_sio2_ref=n_sio2_ref,
            mat_sin=mat_sin,
            mat_sio2=mat_sio2,
            eval_root=eval_root,
            rng=rng,
            progress_state=progress_state,
            est_profile_evals=est_profile_evals,
        )
        profile_diags[profile] = diag
        total_evals += int(diag.get("evaluations", 0))

        if best_prof is None:
            continue
        if (best_overall is None) or (candidate_score(best_prof) > candidate_score(best_overall)):
            best_overall = best_prof

        overall_elapsed = time.time() - float(progress_state["start_ts"])
        done = int(progress_state.get("done", 0))
        total_est = int(progress_state.get("total_est", 0))
        overall_eta = float("nan")
        if done > 0 and total_est > done:
            overall_eta = overall_elapsed * (float(total_est - done) / float(done))
        print(
            f"[progress] completed profile={profile} "
            f"elapsed={format_duration(overall_elapsed)} "
            f"eta~{format_duration(overall_eta) if np.isfinite(overall_eta) else 'n/a'}"
        )

    if best_overall is None:
        raise SystemExit("Optimization failed: no valid objective evaluations.")

    # Rebuild final specs and optional debug analysis from the final design.
    final_geom = build_geometry_spec(
        n_sin=n_sin_ref,
        n_sio2=n_sio2_ref,
        dpml=args.dpml,
        pad_air=args.pad_air,
        pad_sub=args.pad_sub,
        n_per=best_overall.N_per,
        t_sin_um=best_overall.t_sin_um,
        t_sio2_um=best_overall.t_sio2_um,
        L_cav_um=best_overall.L_cav_um,
        high_index_material=str(args.high_index_material),
    )
    final_modes = build_modes_spec(
        probe_um=best_overall.probe_um,
        pump1_um=best_overall.pump1_um,
        pump2_um=best_overall.pump2_um,
    )
    local_q_refined: Optional[Dict[str, Dict[str, float]]] = None
    if str(args.pump_local_q_check).lower() in ("final", "strict"):
        local_q_refined = refine_local_resonances(
            spec=final_geom,
            mats={"SiN": mat_sin, "SiO2": mat_sio2},
            targets_um={
                "probe": float(best_overall.probe_um),
                "pump1": float(best_overall.pump1_um),
                "pump2": float(best_overall.pump2_um),
            },
            linewidth_level=float(args.resonance_linewidth_level),
            resolution=int(args.local_q_resolution),
            nfreq=int(args.local_q_nfreq),
            decay_threshold=float(args.local_q_decay_threshold),
            window_um=float(args.local_q_window_um),
        )

    debug_info = None
    if args.debug:
        debug_mats = {"SiN": mat_sin, "SiO2": mat_sio2}
        debug_info = run_debug_artifacts(
            spec=final_geom,
            modes=final_modes,
            mats=debug_mats,
            n_sin_ref=n_sin_ref,
            n_sio2_ref=n_sio2_ref,
            args=args,
        )

    objective_summary_file: Optional[str] = None
    if best_overall.objective_summary:
        summary_src = Path(best_overall.objective_summary)
        if summary_src.exists():
            summary_dst = Path(args.out_report).resolve().with_name("optimized_objective_summary.json")
            shutil.copy2(summary_src, summary_dst)
            objective_summary_file = str(summary_dst)

    write_outputs(
        best=best_overall,
        args=args,
        n_sin_ref=n_sin_ref,
        n_sio2_ref=n_sio2_ref,
        profile_diags=profile_diags,
        evals_total=total_evals,
        objective_summary_file=objective_summary_file,
        debug_info=debug_info,
        local_q_refined=local_q_refined,
    )

    print("[done] profile:", best_overall.profile)
    print("[done] objective score:", f"{candidate_score(best_overall):.6f}")
    print("[done] abs rotation (deg):", f"{best_overall.abs_rotation_deg:.6f}")
    print("[done] objective summary:", best_overall.objective_summary)
    print("[done] geometry:", Path(args.out_geom).resolve())
    print("[done] modes:", Path(args.out_modes).resolve())
    print("[done] report:", Path(args.out_report).resolve())

    if not args.keep_eval_artifacts:
        shutil.rmtree(eval_root, ignore_errors=True)


if __name__ == "__main__":
    main()
