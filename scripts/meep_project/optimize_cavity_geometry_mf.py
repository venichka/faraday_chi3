#!/usr/bin/env python3
"""
Multi-fidelity optimizer for Faraday-rotation cavity design.

This module keeps output compatibility with optimize_cavity_geometry.py:
  - optimized_geometry.json
  - cavity_modes.json
  - optimize_report.json

Approach:
  Stage A (cheap): reflectance-based resonance screening + proxy score.
  Stage B (medium): optional local resonance refinement of top proxy candidates.
  Stage C (expensive): full objective runs only on top shortlist.

Compatibility:
  All original optimizer CLI options are accepted by forwarding unknown args to
  optimize_cavity_geometry.parse_args(). This module adds only multi-fidelity
  options (see --help section at the end).
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import optimize_cavity_geometry as base
from mode_targeting import get_cavity_materials, material_index_at_wavelength


@dataclass
class MFOptions:
    probe_epsilon: float
    stage1_per_n: int
    stage2_topk: int
    stage3_topk: int
    disable_stage2: bool


@dataclass
class ProxyEval:
    profile: str
    n_per: int
    design: Dict[str, float]
    geom_spec: Dict[str, Any]
    selected: Optional[Dict[str, float]]
    proxy_stage1: float
    proxy_stage2: float
    reason: str

    @property
    def is_valid(self) -> bool:
        return self.selected is not None and np.isfinite(self.proxy_stage1)


def _print_combined_help() -> None:
    # Print base help first.
    orig = list(sys.argv)
    try:
        sys.argv = [orig[0], "--help"]
        try:
            base.parse_args()
        except SystemExit:
            pass
    finally:
        sys.argv = orig

    print("\nMulti-Fidelity Options:")
    print("  --probe-epsilon FLOAT")
    print("      Enforce/score exact probe resonance around 0.8 um in [0.8-eps, 0.8+eps].")
    print("  --mf-stage1-per-n INT")
    print("      Number of cheap proxy candidates per mirror count.")
    print("  --mf-stage2-topk INT")
    print("      Number of stage-A candidates to keep for stage-B refinement per profile.")
    print("  --mf-stage3-topk INT")
    print("      Number of stage-B candidates to evaluate with full objective per profile.")
    print("  --mf-disable-stage2")
    print("      Skip local-resonance refinement stage and rank directly by stage-A proxy.")


def parse_args() -> Tuple[argparse.Namespace, MFOptions]:
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        _print_combined_help()
        raise SystemExit(0)

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--probe-epsilon", type=float, default=0.02)
    ap.add_argument("--mf-stage1-per-n", type=int, default=6)
    ap.add_argument("--mf-stage2-topk", type=int, default=3)
    ap.add_argument("--mf-stage3-topk", type=int, default=2)
    ap.add_argument("--mf-disable-stage2", action="store_true")
    mf_ns, remaining = ap.parse_known_args()

    orig = list(sys.argv)
    try:
        sys.argv = [orig[0], *remaining]
        args = base.parse_args()
    finally:
        sys.argv = orig

    if float(mf_ns.probe_epsilon) <= 0.0:
        raise SystemExit("--probe-epsilon must be > 0.")
    if int(mf_ns.mf_stage1_per_n) < 1:
        raise SystemExit("--mf-stage1-per-n must be >= 1.")
    if int(mf_ns.mf_stage2_topk) < 1:
        raise SystemExit("--mf-stage2-topk must be >= 1.")
    if int(mf_ns.mf_stage3_topk) < 1:
        raise SystemExit("--mf-stage3-topk must be >= 1.")

    # If exact probe mode is used, enforce the explicit neighborhood around 0.8 um.
    if str(args.probe_target_mode) in ("exact", "both"):
        args.probe_exact_tol = min(
            float(args.probe_exact_tol), float(mf_ns.probe_epsilon)
        )

    mf = MFOptions(
        probe_epsilon=float(mf_ns.probe_epsilon),
        stage1_per_n=int(mf_ns.mf_stage1_per_n),
        stage2_topk=int(mf_ns.mf_stage2_topk),
        stage3_topk=int(mf_ns.mf_stage3_topk),
        disable_stage2=bool(mf_ns.mf_disable_stage2),
    )
    return args, mf


def probe_window_score(probe_um: float, profile: str, epsilon: float) -> float:
    p = float(probe_um)
    eps = max(float(epsilon), 1e-9)
    if str(profile) == "exact":
        delta = abs(p - base.PROBE_EXACT_UM)
        if delta >= eps:
            return 0.0
        u = delta / eps
        return float((1.0 - u * u) ** 2)

    # band profile: emphasize center of [0.85, 0.95] for robustness.
    center = 0.5 * (base.PROBE_BAND_MIN_UM + base.PROBE_BAND_MAX_UM)
    half = max(0.5 * (base.PROBE_BAND_MAX_UM - base.PROBE_BAND_MIN_UM), 1e-9)
    delta = abs(p - center)
    if delta >= half:
        return 0.0
    u = delta / half
    return float((1.0 - u * u) ** 2)


def _pump_frequency_alignment_score(selected: Dict[str, float]) -> float:
    f_center = float(selected.get("pump_center_frequency_inv_um", float("nan")))
    f_detune = float(selected.get("pump_detune_frequency_inv_um", float("nan")))
    if (not np.isfinite(f_center)) or (not np.isfinite(f_detune)):
        return 0.0
    dc = abs(f_center - base.PUMP_TARGET_CENTER_FREQ_INV_UM) / max(
        abs(base.PUMP_TARGET_CENTER_FREQ_INV_UM), 1e-9
    )
    dd = abs(f_detune - base.PUMP_TARGET_DELTA_FREQ_INV_UM) / max(
        abs(base.PUMP_TARGET_DELTA_FREQ_INV_UM), 1e-9
    )
    return float(np.exp(-(2.0 * dc + 1.5 * dd)))


def proxy_score_from_selected(
    selected: Dict[str, float],
    profile: str,
    probe_epsilon: float,
) -> float:
    probe_term = probe_window_score(
        float(selected.get("probe_um", float("nan"))), profile, float(probe_epsilon)
    )
    if probe_term <= 0.0:
        return 0.0

    rvals = np.array(
        [
            float(selected.get("probe_R", 1.0)),
            float(selected.get("pump1_R", 1.0)),
            float(selected.get("pump2_R", 1.0)),
        ],
        dtype=float,
    )
    rvals = np.clip(np.nan_to_num(rvals, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
    reflect_term = float(np.clip(1.0 - np.mean(rvals), 0.0, 1.0))

    depths = np.array(
        [
            float(selected.get("probe_depth", 0.0)),
            float(selected.get("pump1_depth", 0.0)),
            float(selected.get("pump2_depth", 0.0)),
        ],
        dtype=float,
    )
    depths = np.clip(np.nan_to_num(depths, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    depth_term = float(np.cbrt(np.prod(depths)))

    qvals = np.array(
        [
            float(selected.get("probe_Q", 0.0)),
            float(selected.get("pump1_Q", 0.0)),
            float(selected.get("pump2_Q", 0.0)),
        ],
        dtype=float,
    )
    qvals = np.nan_to_num(qvals, nan=0.0, posinf=0.0, neginf=0.0)
    q_term = float(
        np.mean(np.clip(np.log1p(np.clip(qvals, 0.0, None)) / np.log1p(150.0), 0.0, 1.0))
    )

    pump_sep = abs(
        float(selected.get("pump2_um", float("nan")))
        - float(selected.get("pump1_um", float("nan")))
    )
    if np.isfinite(pump_sep):
        sep_term = float(np.clip((pump_sep - base.MIN_PUMP_SEP_UM) / 0.25, 0.0, 1.0))
    else:
        sep_term = 0.0

    freq_term = _pump_frequency_alignment_score(selected)

    # Weighted proxy in [0, 1].
    base_score = (
        0.30 * reflect_term
        + 0.25 * depth_term
        + 0.20 * q_term
        + 0.15 * sep_term
        + 0.10 * freq_term
    )
    return float(np.clip(probe_term * base_score, 0.0, 1.0))


def _candidate_vectors_for_n(
    profile: str,
    cfg: base.SearchConfig,
    n_sin_ref: float,
    n_per: int,
    rng: np.random.Generator,
    seed_variants: int,
    count: int,
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    n_seed = min(int(seed_variants), max(1, int(count)))
    for variant in range(n_seed):
        x0 = base.seed_vector(profile, cfg, n_sin_ref, n_per, variant=variant)
        jitter = np.array(
            [rng.normal(0.0, 0.01 * (hi - lo)) for (lo, hi) in cfg.bounds], dtype=float
        )
        out.append(base.clip_to_bounds(x0 + jitter, cfg.bounds))
    remaining = int(count) - len(out)
    if remaining > 0:
        sobol_seed = int(rng.integers(0, 2**31))
        sobol_pts = base.sobol_vectors_in_bounds(cfg.bounds, remaining, seed=sobol_seed)
        out.extend(sobol_pts)

    uniq: List[np.ndarray] = []
    seen = set()
    for v in out:
        key = tuple(np.round(np.asarray(v, dtype=float), 7).tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(np.asarray(v, dtype=float))
    return uniq


def _evaluate_stage1_proxy(
    profile: str,
    n_per: int,
    design: Dict[str, float],
    args: argparse.Namespace,
    n_sin_ref: float,
    n_sio2_ref: float,
    mat_sin: Any,
    mat_sio2: Any,
    probe_epsilon: float,
) -> ProxyEval:
    geom_spec = base.build_geometry_spec(
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
    mats = {"SiN": mat_sin, "SiO2": mat_sio2}
    try:
        wl_refl, r_refl = base.debug_reflectance(
            geom_spec,
            mats,
            resolution=int(args.resonance_resolution),
            nfreq=int(args.resonance_nfreq),
            decay_threshold=float(args.resonance_decay_threshold),
        )
    except Exception as exc:
        return ProxyEval(
            profile=profile,
            n_per=int(n_per),
            design=dict(design),
            geom_spec=geom_spec,
            selected=None,
            proxy_stage1=float("-inf"),
            proxy_stage2=float("-inf"),
            reason=f"stage1_reflectance_failed:{type(exc).__name__}",
        )

    dips = base.find_reflectance_dips(
        wl_refl,
        r_refl,
        linewidth_level=float(args.resonance_linewidth_level),
    )
    probe_tol = (
        min(float(args.probe_exact_tol), float(probe_epsilon))
        if str(profile) == "exact"
        else float(args.probe_exact_tol)
    )
    selected = base.pick_resonant_modes_from_dips(
        profile=profile,
        dips=dips,
        probe_exact_tol=float(probe_tol),
        resonance_max_R=float(args.resonance_max_R),
        pump_min_q=float(args.pump_min_q),
        pump_min_depth=float(args.pump_min_depth),
        probe_min_depth=float(args.probe_min_depth),
    )
    if selected is None:
        return ProxyEval(
            profile=profile,
            n_per=int(n_per),
            design=dict(design),
            geom_spec=geom_spec,
            selected=None,
            proxy_stage1=float("-inf"),
            proxy_stage2=float("-inf"),
            reason="stage1_resonance_not_found",
        )

    proxy = proxy_score_from_selected(selected, profile=profile, probe_epsilon=probe_epsilon)
    return ProxyEval(
        profile=profile,
        n_per=int(n_per),
        design=dict(design),
        geom_spec=geom_spec,
        selected=dict(selected),
        proxy_stage1=float(proxy),
        proxy_stage2=float(proxy),
        reason="ok",
    )


def _apply_stage2_local_refinement(
    item: ProxyEval,
    args: argparse.Namespace,
    mats: Dict[str, Any],
    probe_epsilon: float,
) -> ProxyEval:
    if not item.is_valid or item.selected is None:
        return item
    try:
        local = base.refine_local_resonances(
            spec=item.geom_spec,
            mats=mats,
            targets_um={
                "probe": float(item.selected["probe_um"]),
                "pump1": float(item.selected["pump1_um"]),
                "pump2": float(item.selected["pump2_um"]),
            },
            linewidth_level=float(args.resonance_linewidth_level),
            resolution=int(args.local_q_resolution),
            nfreq=int(args.local_q_nfreq),
            decay_threshold=float(args.local_q_decay_threshold),
            window_um=float(args.local_q_window_um),
        )
    except Exception as exc:
        out = copy.deepcopy(item)
        out.reason = f"stage2_refine_failed:{type(exc).__name__}"
        return out

    out = copy.deepcopy(item)
    if "probe" in local:
        out.selected["probe_um"] = float(local["probe"].get("lam", out.selected["probe_um"]))
        out.selected["probe_R"] = float(local["probe"].get("R", out.selected["probe_R"]))
        out.selected["probe_Q"] = float(local["probe"].get("Q", out.selected.get("probe_Q", float("nan"))))
        out.selected["probe_depth"] = float(local["probe"].get("depth", out.selected.get("probe_depth", float("nan"))))
    for key in ("pump1", "pump2"):
        if key not in local:
            continue
        out.selected[f"{key}_um"] = float(local[key].get("lam", out.selected[f"{key}_um"]))
        out.selected[f"{key}_R"] = float(local[key].get("R", out.selected[f"{key}_R"]))
        out.selected[f"{key}_Q"] = float(local[key].get("Q", out.selected.get(f"{key}_Q", float("nan"))))
        out.selected[f"{key}_depth"] = float(local[key].get("depth", out.selected.get(f"{key}_depth", float("nan"))))

    p1 = float(out.selected.get("pump1_um", float("nan")))
    p2 = float(out.selected.get("pump2_um", float("nan")))
    if np.isfinite(p1) and np.isfinite(p2) and p1 > 0.0 and p2 > 0.0:
        f1, f2 = 1.0 / p1, 1.0 / p2
        out.selected["pump_center_frequency_inv_um"] = float(0.5 * (f1 + f2))
        out.selected["pump_detune_frequency_inv_um"] = float(abs(f1 - f2))

    proxy2 = proxy_score_from_selected(
        out.selected, profile=out.profile, probe_epsilon=probe_epsilon
    )
    out.proxy_stage2 = float(0.5 * out.proxy_stage1 + 0.5 * proxy2)
    out.reason = "ok_stage2"
    return out


def _profile_status_from_reasons(reasons: Sequence[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in reasons:
        counts[str(r)] = int(counts.get(str(r), 0) + 1)
    return counts


def objective_search_profile_mf(
    profile: str,
    args: argparse.Namespace,
    mf: MFOptions,
    n_sin_ref: float,
    n_sio2_ref: float,
    mat_sin: Any,
    mat_sio2: Any,
    eval_root: Path,
    rng: np.random.Generator,
) -> Tuple[Optional[base.Candidate], Dict[str, Any]]:
    t0 = time.time()
    cfg = base.search_config(
        profile=profile,
        n_sin=n_sin_ref,
        n_sio2=n_sio2_ref,
        cavity_min_length=float(args.cavity_min_length),
        cavity_max_length=float(args.cavity_max_length),
    )

    stage1_all: List[ProxyEval] = []
    for n_per in range(int(args.mirror_min), int(args.mirror_max) + 1):
        x_list = _candidate_vectors_for_n(
            profile=profile,
            cfg=cfg,
            n_sin_ref=n_sin_ref,
            n_per=int(n_per),
            rng=rng,
            seed_variants=int(args.seed_variants),
            count=int(mf.stage1_per_n),
        )
        for x in x_list:
            design = base.decode_design_vector(base.clip_to_bounds(x, cfg.bounds), cfg.bounds)
            st1 = _evaluate_stage1_proxy(
                profile=profile,
                n_per=int(n_per),
                design=design,
                args=args,
                n_sin_ref=n_sin_ref,
                n_sio2_ref=n_sio2_ref,
                mat_sin=mat_sin,
                mat_sio2=mat_sio2,
                probe_epsilon=float(mf.probe_epsilon),
            )
            stage1_all.append(st1)

    stage1_valid = [it for it in stage1_all if it.is_valid]
    stage1_valid.sort(key=lambda it: float(it.proxy_stage1), reverse=True)

    if not stage1_valid:
        return None, {
            "status": "no_valid_stage1",
            "evaluations": 0,
            "best_profile": profile,
            "objective_metric": str(args.objective_metric),
            "workers_requested": int(args.workers),
            "workers_effective": 1,
            "parallel_enabled": False,
            "elapsed_s": float(time.time() - t0),
            "success_metrics": {
                "stage1_total": int(len(stage1_all)),
                "stage1_valid": 0,
                "stage1_invalid": int(len(stage1_all)),
                "stage1_reason_counts": _profile_status_from_reasons([it.reason for it in stage1_all]),
            },
        }

    stage2_pool = stage1_valid[: max(1, int(mf.stage2_topk))]
    if not mf.disable_stage2:
        mats = {"SiN": mat_sin, "SiO2": mat_sio2}
        stage2_pool = [
            _apply_stage2_local_refinement(
                it,
                args=args,
                mats=mats,
                probe_epsilon=float(mf.probe_epsilon),
            )
            for it in stage2_pool
        ]

    stage2_pool.sort(key=lambda it: float(it.proxy_stage2), reverse=True)
    stage3_pool = stage2_pool[: max(1, int(mf.stage3_topk))]

    full_candidates: List[base.Candidate] = []
    full_eval_count = 0
    for it in stage3_pool:
        full_eval_count += 1
        cand = base.objective_run(
            profile=profile,
            n_per=int(it.n_per),
            design=dict(it.design),
            args=args,
            n_sin_ref=n_sin_ref,
            n_sio2_ref=n_sio2_ref,
            mat_sin=mat_sin,
            mat_sio2=mat_sio2,
            eval_root=eval_root,
            eval_id=int(full_eval_count),
        )
        full_candidates.append(cand)
        print(
            "[mf-full]",
            f"profile={profile}",
            f"eval={full_eval_count}/{len(stage3_pool)}",
            f"score={base.candidate_score(cand):.6f}",
            f"abs_rot={cand.abs_rotation_deg:.6f}",
            f"proxy_stage2={it.proxy_stage2:.4f}",
            flush=True,
        )

    valid_full = [
        c
        for c in full_candidates
        if np.isfinite(base.candidate_score(c)) and base.candidate_score(c) >= 0.0
    ]
    if not valid_full:
        return None, {
            "status": "no_valid_full_objective",
            "evaluations": int(full_eval_count),
            "best_profile": profile,
            "objective_metric": str(args.objective_metric),
            "workers_requested": int(args.workers),
            "workers_effective": 1,
            "parallel_enabled": False,
            "elapsed_s": float(time.time() - t0),
            "success_metrics": {
                "stage1_total": int(len(stage1_all)),
                "stage1_valid": int(len(stage1_valid)),
                "stage1_invalid": int(len(stage1_all) - len(stage1_valid)),
                "stage1_reason_counts": _profile_status_from_reasons([it.reason for it in stage1_all]),
                "stage2_count": int(len(stage2_pool)),
                "stage3_full_count": int(len(stage3_pool)),
                "full_valid_count": 0,
            },
        }

    best = max(valid_full, key=lambda c: base.candidate_score(c))
    success = {
        "stage1_total": int(len(stage1_all)),
        "stage1_valid": int(len(stage1_valid)),
        "stage1_invalid": int(len(stage1_all) - len(stage1_valid)),
        "stage1_reason_counts": _profile_status_from_reasons([it.reason for it in stage1_all]),
        "stage1_best_proxy": float(stage1_valid[0].proxy_stage1),
        "stage2_count": int(len(stage2_pool)),
        "stage2_best_proxy": float(stage2_pool[0].proxy_stage2) if stage2_pool else float("nan"),
        "stage3_full_count": int(len(stage3_pool)),
        "full_valid_count": int(len(valid_full)),
    }
    return best, {
        "status": "ok",
        "evaluations": int(full_eval_count),
        "best_score": float(base.candidate_score(best)),
        "best_abs_rotation_deg": float(best.abs_rotation_deg),
        "best_rotation_deg": float(best.rotation_deg),
        "objective_metric": str(args.objective_metric),
        "best_profile": profile,
        "optimizer": "multi_fidelity_proxy_shortlist",
        "workers_requested": int(args.workers),
        "workers_effective": 1,
        "parallel_enabled": False,
        "elapsed_s": float(time.time() - t0),
        "success_metrics": success,
    }


def main() -> None:
    base._configure_numeric_threads()
    args, mf = parse_args()
    args = base.normalize_material_args(args)
    base.mp.verbosity(int(args.meep_verbosity))

    if float(args.cavity_max_length) <= float(args.cavity_min_length):
        raise SystemExit("--cavity-max-length must be larger than --cavity-min-length.")
    if int(args.workers) < 1:
        raise SystemExit("--workers must be >= 1.")

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

    eval_root = Path(args.eval_root).resolve()
    if eval_root.exists():
        shutil.rmtree(eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)

    best_overall: Optional[base.Candidate] = None
    profile_diags: Dict[str, Dict[str, Any]] = {}
    total_objective_evals = 0

    t_all = time.time()
    for profile in profiles:
        print(f"[mf-profile] {profile}", flush=True)
        best_prof, diag = objective_search_profile_mf(
            profile=profile,
            args=args,
            mf=mf,
            n_sin_ref=n_sin_ref,
            n_sio2_ref=n_sio2_ref,
            mat_sin=mat_sin,
            mat_sio2=mat_sio2,
            eval_root=eval_root,
            rng=rng,
        )
        profile_diags[profile] = diag
        total_objective_evals += int(diag.get("evaluations", 0))
        if best_prof is None:
            continue
        if (best_overall is None) or (
            base.candidate_score(best_prof) > base.candidate_score(best_overall)
        ):
            best_overall = best_prof

    if best_overall is None:
        raise SystemExit("Optimization failed: no valid full-objective candidates.")

    final_geom = base.build_geometry_spec(
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
    final_modes = base.build_modes_spec(
        probe_um=best_overall.probe_um,
        pump1_um=best_overall.pump1_um,
        pump2_um=best_overall.pump2_um,
    )

    local_q_refined: Optional[Dict[str, Dict[str, float]]] = None
    if str(args.pump_local_q_check).lower() in ("final", "strict"):
        local_q_refined = base.refine_local_resonances(
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
        debug_info = base.run_debug_artifacts(
            spec=final_geom,
            modes=final_modes,
            mats={"SiN": mat_sin, "SiO2": mat_sio2},
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

    report = base.write_outputs(
        best=best_overall,
        args=args,
        n_sin_ref=n_sin_ref,
        n_sio2_ref=n_sio2_ref,
        profile_diags=profile_diags,
        evals_total=total_objective_evals,
        objective_summary_file=objective_summary_file,
        debug_info=debug_info,
        local_q_refined=local_q_refined,
    )

    report["meta"]["script"] = "optimize_cavity_geometry_mf.py"
    report["meta"]["search_algorithm"] = (
        "multi-fidelity resonance proxy screening + local refinement + full objective shortlist"
    )
    report.setdefault("optimization", {})
    report["optimization"]["multi_fidelity"] = {
        "probe_epsilon_um": float(mf.probe_epsilon),
        "stage1_per_n": int(mf.stage1_per_n),
        "stage2_topk": int(mf.stage2_topk),
        "stage3_topk": int(mf.stage3_topk),
        "stage2_enabled": bool(not mf.disable_stage2),
        "total_wall_elapsed_s": float(time.time() - t_all),
    }
    out_report = Path(args.out_report).resolve()
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[done] profile:", best_overall.profile)
    print("[done] objective score:", f"{base.candidate_score(best_overall):.6f}")
    print("[done] abs rotation (deg):", f"{best_overall.abs_rotation_deg:.6f}")
    print("[done] geometry:", Path(args.out_geom).resolve())
    print("[done] modes:", Path(args.out_modes).resolve())
    print("[done] report:", out_report)

    if not args.keep_eval_artifacts:
        shutil.rmtree(eval_root, ignore_errors=True)


if __name__ == "__main__":
    main()
