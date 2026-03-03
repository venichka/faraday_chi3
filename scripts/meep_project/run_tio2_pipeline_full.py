#!/usr/bin/env python3
"""Run full TiO2 pipeline: 3 optimizers + sims + sweeps + consolidated report."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_stage(
    name: str,
    cmd: List[str],
    run_root: Path,
    cwd: Path,
    env: Dict[str, str],
    stage_times: Dict[str, Any],
) -> None:
    stage_dir = run_root / "logs"
    stage_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = stage_dir / f"{name}.stdout.log"
    stderr_path = stage_dir / f"{name}.stderr.log"
    print(f"[stage] {name}", flush=True)
    print("[cmd]", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
    )
    dt = float(time.time() - t0)
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    stage_times[name] = {
        "seconds": dt,
        "returncode": int(proc.returncode),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "cmd": cmd,
    }
    if proc.returncode != 0:
        tail = (proc.stderr or "").splitlines()[-10:]
        err = "\n".join(tail) if tail else f"return code {proc.returncode}"
        raise RuntimeError(f"Stage '{name}' failed.\n{err}")
    print(f"[done] {name} in {dt:.1f}s", flush=True)


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt(v: Any, nd: int = 6) -> str:
    try:
        x = float(v)
    except Exception:
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}g}"


def _generate_legacy_modes(
    legacy_geom: Path,
    out_modes: Path,
    out_plot: Path,
    out_diag: Path,
    n_high: float,
    n_low: float,
    k_high: float,
    decay_threshold: float,
) -> None:
    import optimize_cavity_geometry as ocg
    from geometry_io import read_json
    from mode_targeting import get_cavity_materials

    spec = read_json(str(legacy_geom))
    mat_h, mat_l = get_cavity_materials(
        model="constant",
        index_high=float(n_high),
        index_low=float(n_low),
        high_index_material="tio2",
        kappa_high=float(k_high),
        kappa_ref_wavelength_um=1.55,
    )
    mats = {"SiN": mat_h, "SiO2": mat_l}
    wl, rr = ocg.debug_reflectance(
        spec,
        mats,
        resolution=80,
        nfreq=900,
        decay_threshold=float(decay_threshold),
    )
    dips = ocg.find_reflectance_dips(wl, rr, linewidth_level=0.5)
    selected = ocg.pick_resonant_modes_from_dips(
        profile="exact",
        dips=dips,
        probe_exact_tol=0.10,
        resonance_max_R=1.0,
        pump_min_q=0.0,
        pump_min_depth=0.0,
        probe_min_depth=0.0,
    )

    if selected is None:
        if len(dips) >= 3:
            d_probe = min(dips, key=lambda d: abs(float(d["lam"]) - 0.8))
            pumps = [d for d in dips if 1.3 <= float(d["lam"]) <= 1.7]
            if len(pumps) < 2:
                p1 = 1.55
                p2 = 1.65
            else:
                pumps = sorted(pumps, key=lambda d: float(d["R"]))
                p1 = float(min(pumps[0]["lam"], pumps[1]["lam"]))
                p2 = float(max(pumps[0]["lam"], pumps[1]["lam"]))
            selected = {
                "probe_um": float(d_probe["lam"]),
                "pump1_um": float(p1),
                "pump2_um": float(p2),
                "probe_R": float(d_probe.get("R", np.nan)),
                "pump1_R": float("nan"),
                "pump2_R": float("nan"),
            }
        else:
            selected = {"probe_um": 0.8, "pump1_um": 1.55, "pump2_um": 1.65}

    modes = ocg.build_modes_spec(
        probe_um=float(selected["probe_um"]),
        pump1_um=float(selected["pump1_um"]),
        pump2_um=float(selected["pump2_um"]),
    )
    out_modes.write_text(json.dumps(modes, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(1e3 * wl, rr, lw=1.4, color="k")
    ax.invert_xaxis()
    for label, lam in (
        ("probe", float(selected["probe_um"])),
        ("pump1", float(selected["pump1_um"])),
        ("pump2", float(selected["pump2_um"])),
    ):
        r_here = float(np.interp(lam, wl, rr))
        ax.scatter([1e3 * lam], [r_here], s=22, label=f"{label}: {lam:.4f} um")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("Reflectance")
    ax.set_title("Legacy mode selection from reflectance dips")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)

    diag = {
        "selected": selected,
        "n_dips": int(len(dips)),
        "plot": str(out_plot),
        "modes_json": str(out_modes),
    }
    out_diag.write_text(json.dumps(diag, indent=2), encoding="utf-8")


def _report(
    run_root: Path,
    stage_times: Dict[str, Any],
    branch_dirs: Dict[str, Path],
    sim_dirs: Dict[str, Dict[int, Path]],
    sweep_dirs: Dict[str, Path],
) -> Path:
    out_report = run_root / "pipeline_tio2_report.md"
    plot_dir = run_root / "report_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    optim_rows: List[Dict[str, Any]] = []
    sim_rows: List[Dict[str, Any]] = []
    sweep_rows: List[Dict[str, Any]] = []

    for key in ("new", "mf", "legacy"):
        bdir = branch_dirs[key]
        opt_report = _safe_json(bdir / "optimize_report.json")
        selected = opt_report.get("selected", {})
        best_theta = opt_report.get("best_theta", {})
        optim_rows.append(
            {
                "name": key,
                "time_s": float(stage_times.get(f"opt_{key}", {}).get("seconds", np.nan)),
                "objective": float(selected.get("abs_rotation_deg", np.nan)),
                "score": float(selected.get("objective_score", opt_report.get("score", np.nan))),
                "N_per": int(selected.get("N_per", best_theta.get("N_per", -1))),
                "t_SiN_um": float(selected.get("t_sin_um", best_theta.get("t_SiN_um", np.nan))),
                "t_SiO2_um": float(selected.get("t_sio2_um", best_theta.get("t_SiO2_um", np.nan))),
                "L_cav_um": float(selected.get("L_cav_um", best_theta.get("L_cav_um", np.nan))),
                "probe_um": float(selected.get("probe_um", np.nan)),
                "pump1_um": float(selected.get("pump1_um", np.nan)),
                "pump2_um": float(selected.get("pump2_um", np.nan)),
                "report_path": str(bdir / "optimize_report.json"),
            }
        )

        for dim in (1, 3):
            summary = _safe_json(sim_dirs[key][dim] / "faraday_summary.json")
            pr = summary.get("probe_rotation_deg", {})
            sim_rows.append(
                {
                    "name": key,
                    "dim": dim,
                    "time_s": float(stage_times.get(f"sim_{key}_dim{dim}", {}).get("seconds", np.nan)),
                    "final_rel_deg": float(pr.get("final_relative_deg", np.nan)),
                    "final_rel_wrapped_deg": float(pr.get("wrapped_final_relative_deg", np.nan)),
                    "std_tail_deg": float(
                        summary.get("probe_stokes_dft", {})
                        .get("tail_weighted", {})
                        .get("theta_relative_std_deg", np.nan)
                    ),
                    "summary_path": str(sim_dirs[key][dim] / "faraday_summary.json"),
                    "plot_probe_rotation": str(sim_dirs[key][dim] / "probe_polarization.png"),
                    "plot_probe_zoom": str(sim_dirs[key][dim] / "probe_polarization_zoom.png"),
                }
            )

        sw = _safe_json(sweep_dirs[key] / "pump_intensity_sweep_report.json")
        dim1 = sw.get("dimension_reports", {}).get("1", sw if isinstance(sw, dict) else {})
        agg = dim1.get("aggregate_metrics", {})
        best = dim1.get("best_point", {})
        sweep_rows.append(
            {
                "name": key,
                "time_s": float(stage_times.get(f"sweep_{key}", {}).get("seconds", np.nan)),
                "max_abs_deg": float(agg.get("max_abs_final_relative_deg", np.nan)),
                "mean_abs_deg": float(agg.get("mean_abs_final_relative_deg", np.nan)),
                "best_intensity": float(best.get("pump_intensity_w_cm2", np.nan)),
                "best_final_deg": float(best.get("final_relative_deg", np.nan)),
                "report_path": str(sweep_dirs[key] / "pump_intensity_sweep_report.json"),
                "plot_rotation": str(
                    Path(
                        dim1.get("plot_paths", {}).get(
                            "rotation_vs_intensity",
                            sweep_dirs[key] / "dim1" / "rotation_vs_intensity.png",
                        )
                    )
                ),
            }
        )

    # Runtime plot.
    labels = ["opt_new", "opt_mf", "opt_legacy", "sim_new", "sim_mf", "sim_legacy", "sweep_new", "sweep_mf", "sweep_legacy"]
    vals = [
        stage_times.get("opt_new", {}).get("seconds", np.nan),
        stage_times.get("opt_mf", {}).get("seconds", np.nan),
        stage_times.get("opt_legacy", {}).get("seconds", np.nan),
        np.nansum([stage_times.get("sim_new_dim1", {}).get("seconds", np.nan), stage_times.get("sim_new_dim3", {}).get("seconds", np.nan)]),
        np.nansum([stage_times.get("sim_mf_dim1", {}).get("seconds", np.nan), stage_times.get("sim_mf_dim3", {}).get("seconds", np.nan)]),
        np.nansum([stage_times.get("sim_legacy_dim1", {}).get("seconds", np.nan), stage_times.get("sim_legacy_dim3", {}).get("seconds", np.nan)]),
        stage_times.get("sweep_new", {}).get("seconds", np.nan),
        stage_times.get("sweep_mf", {}).get("seconds", np.nan),
        stage_times.get("sweep_legacy", {}).get("seconds", np.nan),
    ]
    fig = plt.figure(figsize=(10.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(len(labels)), vals)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("seconds")
    ax.set_title("TiO2 pipeline runtime by stage")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p_runtime = plot_dir / "runtime_breakdown.png"
    fig.savefig(p_runtime, bbox_inches="tight")
    plt.close(fig)

    # Final rotation comparison plot.
    names = ["new", "mf", "legacy"]
    d1 = [next((r["final_rel_deg"] for r in sim_rows if r["name"] == n and r["dim"] == 1), np.nan) for n in names]
    d3 = [next((r["final_rel_deg"] for r in sim_rows if r["name"] == n and r["dim"] == 3), np.nan) for n in names]
    xx = np.arange(len(names))
    w = 0.36
    fig = plt.figure(figsize=(7.2, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(xx - 0.5 * w, d1, width=w, label="1D (res=100)")
    ax.bar(xx + 0.5 * w, d3, width=w, label="3D (res=30)")
    ax.set_xticks(xx)
    ax.set_xticklabels(names)
    ax.set_ylabel("final relative rotation (deg)")
    ax.set_title("Final probe rotation by optimizer")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    p_rot = plot_dir / "final_rotation_compare.png"
    fig.savefig(p_rot, bbox_inches="tight")
    plt.close(fig)

    # Sweep max |rotation| comparison.
    max_abs = [next((r["max_abs_deg"] for r in sweep_rows if r["name"] == n), np.nan) for n in names]
    fig = plt.figure(figsize=(6.6, 3.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(names, max_abs)
    ax.set_ylabel("max |final rotation| in sweep (deg)")
    ax.set_title("Sweep peak response comparison (1D)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p_sw = plot_dir / "sweep_max_abs_compare.png"
    fig.savefig(p_sw, bbox_inches="tight")
    plt.close(fig)

    # Conclusion heuristic.
    best_sim = max(
        [r for r in sim_rows if r["dim"] == 3 and np.isfinite(r["final_rel_deg"])],
        key=lambda r: abs(float(r["final_rel_deg"])),
        default=None,
    )
    best_sweep = max(
        [r for r in sweep_rows if np.isfinite(r["max_abs_deg"])],
        key=lambda r: float(r["max_abs_deg"]),
        default=None,
    )

    lines: List[str] = []
    lines.append("# TiO2 Full Pipeline Report")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("- High-index material: `TiO2` (constant n/k approximation)")
    lines.append("- Material parameters: `nH=2.31`, `kH=8e-6`, `n2=2.3e-18 m^2/W`, `kappa_ref_lambda=1.55 um`")
    lines.append("- Pump intensity for optimizers and sims: `1e12 W/cm^2`")
    lines.append("- Simulations: `1D(res=100)` and `3D(res=30)`, `decay_threshold=1e-4`")
    lines.append("- Sweeps: `1D`, `6` log-spaced points in `[1e8, 2e12] W/cm^2`, `res=100`, `decay_threshold=1e-4`")
    lines.append("")
    lines.append("## Runtime Plots")
    lines.append(f"![runtime](./{p_runtime.relative_to(run_root)})")
    lines.append(f"![final rotation](./{p_rot.relative_to(run_root)})")
    lines.append(f"![sweep max](./{p_sw.relative_to(run_root)})")
    lines.append("")
    lines.append("## Optimizer Results")
    lines.append("")
    lines.append("| Optimizer | Runtime (s) | Objective | Score | N_per | t_H (um) | t_L (um) | L_cav (um) | probe (um) | pump1 (um) | pump2 (um) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in optim_rows:
        lines.append(
            f"| {r['name']} | {_fmt(r['time_s'])} | {_fmt(r['objective'])} | {_fmt(r['score'])} | "
            f"{r['N_per']} | {_fmt(r['t_SiN_um'])} | {_fmt(r['t_SiO2_um'])} | {_fmt(r['L_cav_um'])} | "
            f"{_fmt(r['probe_um'])} | {_fmt(r['pump1_um'])} | {_fmt(r['pump2_um'])} |"
        )
    lines.append("")
    lines.append("## Simulation Results")
    lines.append("")
    lines.append("| Optimizer | Dim | Runtime (s) | Final rel (deg) | Wrapped final rel (deg) | Tail std (deg) | Summary |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in sim_rows:
        rel_summary = Path(r["summary_path"]).relative_to(run_root)
        lines.append(
            f"| {r['name']} | {r['dim']} | {_fmt(r['time_s'])} | {_fmt(r['final_rel_deg'])} | "
            f"{_fmt(r['final_rel_wrapped_deg'])} | {_fmt(r['std_tail_deg'])} | `{rel_summary}` |"
        )
    lines.append("")
    lines.append("## Sweep Results (1D)")
    lines.append("")
    lines.append("| Optimizer | Runtime (s) | Max |theta| (deg) | Mean |theta| (deg) | Best I (W/cm^2) | theta(best) (deg) | Report |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for r in sweep_rows:
        rel_report = Path(r["report_path"]).relative_to(run_root)
        lines.append(
            f"| {r['name']} | {_fmt(r['time_s'])} | {_fmt(r['max_abs_deg'])} | {_fmt(r['mean_abs_deg'])} | "
            f"{_fmt(r['best_intensity'])} | {_fmt(r['best_final_deg'])} | `{rel_report}` |"
        )
    lines.append("")
    lines.append("## Key Artifacts")
    lines.append("")
    for key in ("new", "mf", "legacy"):
        lines.append(f"### {key}")
        lines.append(f"- Optimizer report: `{(branch_dirs[key] / 'optimize_report.json').relative_to(run_root)}`")
        lines.append(f"- Geometry: `{(branch_dirs[key] / 'optimized_geometry.json').relative_to(run_root)}`")
        lines.append(f"- Modes: `{(branch_dirs[key] / 'cavity_modes.json').relative_to(run_root)}`")
        lines.append(f"- 1D summary: `{(sim_dirs[key][1] / 'faraday_summary.json').relative_to(run_root)}`")
        lines.append(f"- 3D summary: `{(sim_dirs[key][3] / 'faraday_summary.json').relative_to(run_root)}`")
        lines.append(f"- Sweep report: `{(sweep_dirs[key] / 'pump_intensity_sweep_report.json').relative_to(run_root)}`")
        lines.append("")
    lines.append("## Conclusion")
    if best_sim is not None:
        lines.append(
            f"- Strongest 3D final rotation magnitude came from `{best_sim['name']}`: "
            f"{_fmt(best_sim['final_rel_deg'])} deg."
        )
    if best_sweep is not None:
        lines.append(
            f"- Largest sweep peak |rotation| came from `{best_sweep['name']}`: "
            f"{_fmt(best_sweep['max_abs_deg'])} deg."
        )
    lines.append(
        "- Differences between optimizer objective value and final 1D/3D rotations remain expected, "
        "because mode selection, dimensionality, and nonlinear full-wave dynamics differ."
    )
    out_report.write_text("\n".join(lines), encoding="utf-8")
    return out_report


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full TiO2 pipeline and write consolidated report.")
    ap.add_argument("--run-root", type=str, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sweep-workers", type=int, default=6)
    ap.add_argument("--optimizer-pump-intensity", type=float, default=1e12)
    ap.add_argument("--objective-resolution", type=int, default=60)
    ap.add_argument("--objective-mode", choices=("quick", "full"), default="full")
    ap.add_argument("--objective-decay-threshold", type=float, default=1e-4)
    ap.add_argument("--sim-decay-threshold", type=float, default=1e-4)
    ap.add_argument("--sweep-decay-threshold", type=float, default=1e-4)
    ap.add_argument("--sweep-i-min", type=float, default=1e8)
    ap.add_argument("--sweep-i-max", type=float, default=2e12)
    ap.add_argument("--sweep-points", type=int, default=6)
    ap.add_argument("--legacy-maxiter", type=int, default=150)
    ap.add_argument("--seed-geometry", type=str, default="optimized_geometry_bayes_w6_signal_new.json")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent
    run_root = Path(args.run_root).resolve() if args.run_root else (project_root / f"pipeline_tio2_{_now_tag()}").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"[run-root] {run_root}", flush=True)

    # TiO2 constants for this run.
    n_high = 2.31
    n_low = 1.45
    k_high = 8e-6
    n2_high = 2.3e-18

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")

    py = sys.executable
    stage_times: Dict[str, Any] = {}

    branch_dirs = {
        "new": run_root / "optimizers" / "new",
        "mf": run_root / "optimizers" / "mf",
        "legacy": run_root / "optimizers" / "legacy",
    }
    for bdir in branch_dirs.values():
        bdir.mkdir(parents=True, exist_ok=True)

    material_args = [
        "--materials",
        "constant",
        "--high-index-material",
        "tio2",
        "--nH",
        str(n_high),
        "--kH",
        str(k_high),
        "--high-index-n2",
        str(n2_high),
        "--kappa-ref-lambda",
        "1.55",
        "--nL",
        str(n_low),
    ]

    # 1) Optimizers
    _run_stage(
        "opt_new",
        [
            py,
            "optimize_cavity_geometry.py",
            "--optimizer",
            "bayes",
            "--workers",
            str(int(args.workers)),
            "--objective-mode",
            str(args.objective_mode),
            "--objective-resolution",
            str(int(args.objective_resolution)),
            "--objective-decay-threshold",
            str(float(args.objective_decay_threshold)),
            "--pump-intensity",
            str(float(args.optimizer_pump_intensity)),
            "--debug",
            "--debug-prefix",
            str(branch_dirs["new"] / "optimize_debug"),
            "--out-geom",
            str(branch_dirs["new"] / "optimized_geometry.json"),
            "--out-modes",
            str(branch_dirs["new"] / "cavity_modes.json"),
            "--out-report",
            str(branch_dirs["new"] / "optimize_report.json"),
            "--eval-root",
            str(branch_dirs["new"] / ".opt_eval_tmp"),
            *material_args,
        ],
        run_root,
        project_root,
        env,
        stage_times,
    )

    _run_stage(
        "opt_mf",
        [
            py,
            "optimize_cavity_geometry_mf.py",
            "--optimizer",
            "bayes",
            "--workers",
            str(int(args.workers)),
            "--objective-mode",
            str(args.objective_mode),
            "--objective-resolution",
            str(int(args.objective_resolution)),
            "--objective-decay-threshold",
            str(float(args.objective_decay_threshold)),
            "--pump-intensity",
            str(float(args.optimizer_pump_intensity)),
            "--mf-stage1-per-n",
            "8",
            "--mf-stage2-topk",
            "4",
            "--mf-stage3-topk",
            "3",
            "--debug",
            "--debug-prefix",
            str(branch_dirs["mf"] / "optimize_debug"),
            "--out-geom",
            str(branch_dirs["mf"] / "optimized_geometry.json"),
            "--out-modes",
            str(branch_dirs["mf"] / "cavity_modes.json"),
            "--out-report",
            str(branch_dirs["mf"] / "optimize_report.json"),
            "--eval-root",
            str(branch_dirs["mf"] / ".opt_eval_tmp"),
            *material_args,
        ],
        run_root,
        project_root,
        env,
        stage_times,
    )

    _run_stage(
        "opt_legacy",
        [
            py,
            "optimize_cavity_geometry_legacy.py",
            "--in-json",
            str(Path(args.seed_geometry).resolve()),
            "--materials",
            "constant",
            "--nH",
            str(n_high),
            "--nL",
            str(n_low),
            "--maxiter",
            str(int(args.legacy_maxiter)),
            "--out-geom",
            str(branch_dirs["legacy"] / "optimized_geometry.json"),
            "--outfile",
            str(branch_dirs["legacy"] / "optimize_report.json"),
            "--plot",
            str(branch_dirs["legacy"] / "reflectance_plot.png"),
        ],
        run_root,
        project_root,
        env,
        stage_times,
    )

    # Legacy optimizer doesn't emit cavity_modes.json; derive one from reflectance dips.
    t0 = time.time()
    _generate_legacy_modes(
        legacy_geom=branch_dirs["legacy"] / "optimized_geometry.json",
        out_modes=branch_dirs["legacy"] / "cavity_modes.json",
        out_plot=branch_dirs["legacy"] / "legacy_modes_reflectance.png",
        out_diag=branch_dirs["legacy"] / "legacy_modes_selection.json",
        n_high=n_high,
        n_low=n_low,
        k_high=k_high,
        decay_threshold=float(args.sim_decay_threshold),
    )
    stage_times["legacy_mode_selection"] = {"seconds": float(time.time() - t0), "returncode": 0}

    # 2) 1D/3D simulations
    sim_dirs: Dict[str, Dict[int, Path]] = {k: {} for k in branch_dirs}
    for name in ("new", "mf", "legacy"):
        for dim, res in ((1, 100), (3, 30)):
            out_dir = run_root / "sims" / name / f"dim{dim}"
            out_dir.mkdir(parents=True, exist_ok=True)
            sim_dirs[name][dim] = out_dir
            _run_stage(
                f"sim_{name}_dim{dim}",
                [
                    py,
                    "faraday_meep_fp_circ.py",
                    "--mode",
                    "full",
                    "--dim",
                    str(dim),
                    "--resolution",
                    str(res),
                    "--decay-threshold",
                    str(float(args.sim_decay_threshold)),
                    "--pump-intensity",
                    str(float(args.optimizer_pump_intensity)),
                    "--geometry-file",
                    str(branch_dirs[name] / "optimized_geometry.json"),
                    "--cavity-modes-file",
                    str(branch_dirs[name] / "cavity_modes.json"),
                    "--output-dir",
                    str(out_dir),
                    *material_args,
                ],
                run_root,
                project_root,
                env,
                stage_times,
            )

    # 3) Sweeps
    sweep_dirs: Dict[str, Path] = {}
    for name in ("new", "mf", "legacy"):
        out_root = run_root / "sweeps" / name
        out_root.mkdir(parents=True, exist_ok=True)
        sweep_dirs[name] = out_root
        _run_stage(
            f"sweep_{name}",
            [
                py,
                "pump_intensity_sweep.py",
                "--dim",
                "1",
                "--intensity-range",
                str(float(args.sweep_i_min)),
                str(float(args.sweep_i_max)),
                str(int(args.sweep_points)),
                "--range-scale",
                "log",
                "--workers",
                str(int(args.sweep_workers)),
                "--mode",
                "full",
                "--resolution",
                "100",
                "--decay-threshold",
                str(float(args.sweep_decay_threshold)),
                "--geometry-file",
                str(branch_dirs[name] / "optimized_geometry.json"),
                "--cavity-modes-file",
                str(branch_dirs[name] / "cavity_modes.json"),
                "--output-root",
                str(out_root),
                *material_args,
            ],
            run_root,
            project_root,
            env,
            stage_times,
        )

    # 4) Report
    report = _report(
        run_root=run_root,
        stage_times=stage_times,
        branch_dirs=branch_dirs,
        sim_dirs=sim_dirs,
        sweep_dirs=sweep_dirs,
    )
    (run_root / "pipeline_tio2_stage_times.json").write_text(
        json.dumps(stage_times, indent=2), encoding="utf-8"
    )
    print(f"[report] {report}", flush=True)


if __name__ == "__main__":
    main()
