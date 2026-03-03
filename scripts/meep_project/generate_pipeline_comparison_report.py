#!/usr/bin/env python3
"""Generate a consolidated optimizer/simulation/sweep comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def fmt_float(value: Optional[float], digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


@dataclass
class OptimizerInfo:
    label: str
    profile: str
    objective_score: Optional[float]
    abs_rotation_deg: Optional[float]
    evals: Optional[int]
    elapsed_s: Optional[float]
    n_per: Optional[int]
    t_sin_um: Optional[float]
    t_sio2_um: Optional[float]
    l_cav_um: Optional[float]
    pump_intensity_w_cm2: Optional[float]
    report_path: Path
    geom_path: Path
    modes_path: Path


@dataclass
class SimInfo:
    label: str
    dim: int
    final_dft_deg: Optional[float]
    final_td_deg: Optional[float]
    summary_path: Path
    plot_paths: Dict[str, str]


@dataclass
class SweepInfo:
    label: str
    intensities: List[float]
    final_deg: List[float]
    csv_path: Path
    summary_md: Path
    report_path: Path
    plot_rotation: Optional[str]
    plot_dft_traces: Optional[str]
    plot_td_traces: Optional[str]


@dataclass
class RuntimeInfo:
    optimizer_old_s: Optional[float]
    optimizer_new_s: Optional[float]
    sim_old_1d_s: Optional[float]
    sim_new_1d_s: Optional[float]
    sim_old_3d_s: Optional[float]
    sim_new_3d_s: Optional[float]
    sweep_old_1d_s: Optional[float]
    sweep_new_1d_s: Optional[float]


def read_optimizer(run_root: Path, branch: str, label: str) -> OptimizerInfo:
    report_path = run_root / branch / "optimize_report.json"
    report = load_json(report_path)
    selected = report.get("selected", {})
    optim = report.get("optimization", {})
    sim = report.get("sim", {})
    profile = str(selected.get("profile", "exact"))
    prof_diag = (optim.get("profile_diagnostics") or {}).get(profile, {})
    success = optim.get("success_metrics") or {}
    evals = (success.get("per_profile_evaluations") or {}).get(profile)
    if evals is None:
        evals = prof_diag.get("evaluations")
    files = report.get("files", {})
    return OptimizerInfo(
        label=label,
        profile=profile,
        objective_score=parse_float(selected.get("objective_score")),
        abs_rotation_deg=parse_float(selected.get("abs_rotation_deg")),
        evals=int(evals) if evals is not None else None,
        elapsed_s=parse_float(prof_diag.get("elapsed_s")),
        n_per=int(selected["N_per"]) if selected.get("N_per") is not None else None,
        t_sin_um=parse_float(selected.get("t_sin_um")),
        t_sio2_um=parse_float(selected.get("t_sio2_um")),
        l_cav_um=parse_float(selected.get("L_cav_um")),
        pump_intensity_w_cm2=parse_float(sim.get("pump_intensity_w_cm2")),
        report_path=report_path,
        geom_path=Path(files.get("geometry_json", run_root / branch / "optimized_geometry.json")),
        modes_path=Path(files.get("cavity_modes_json", run_root / branch / "cavity_modes.json")),
    )


def read_sim(run_root: Path, branch: str, label: str, dim: int) -> Optional[SimInfo]:
    summary_path = run_root / "sims" / branch / f"dim{dim}" / "faraday_summary.json"
    if not summary_path.exists():
        return None
    summary = load_json(summary_path)
    rot = summary.get("probe_rotation_deg", {})
    td = rot.get("time_domain_reference", {}) or {}
    plot_paths = summary.get("plot_paths", {}) or {}
    return SimInfo(
        label=label,
        dim=dim,
        final_dft_deg=parse_float(rot.get("final_relative_deg")),
        final_td_deg=parse_float(td.get("final_relative_deg")),
        summary_path=summary_path,
        plot_paths={k: str(v) for k, v in plot_paths.items()},
    )


def read_sweep(run_root: Path, branch: str, label: str) -> SweepInfo:
    csv_path = run_root / "sweeps" / branch / "dim1" / "rotation_vs_intensity_points.csv"
    report_path = run_root / "sweeps" / branch / "pump_intensity_sweep_report.json"
    summary_md = run_root / "sweeps" / branch / "dim1" / "pump_intensity_sweep_summary.md"
    intensities: List[float] = []
    final_deg: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intensities.append(float(row["pump_intensity_w_cm2"]))
            final_deg.append(float(row["final_relative_deg"]))
    report = load_json(report_path)
    dim_report = (report.get("dimension_reports") or {}).get("1", {})
    plots = dim_report.get("plot_paths", {}) or {}
    return SweepInfo(
        label=label,
        intensities=intensities,
        final_deg=final_deg,
        csv_path=csv_path,
        summary_md=summary_md,
        report_path=report_path,
        plot_rotation=plots.get("rotation_vs_intensity"),
        plot_dft_traces=plots.get("dft_traces"),
        plot_td_traces=plots.get("time_domain_traces"),
    )


def make_plot_optimizer(out_dir: Path, new: OptimizerInfo, old: OptimizerInfo) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = [old.label, new.label]
    rot_vals = [old.abs_rotation_deg or 0.0, new.abs_rotation_deg or 0.0]
    rt_vals = [old.elapsed_s or 0.0, new.elapsed_s or 0.0]
    ax[0].bar(labels, rot_vals, color=["#d95f02", "#1b9e77"])
    ax[0].set_ylabel("Abs rotation (deg)")
    ax[0].set_title("Optimizer result")
    ax[1].bar(labels, rt_vals, color=["#d95f02", "#1b9e77"])
    ax[1].set_ylabel("Runtime (s)")
    ax[1].set_title("Optimizer runtime")
    fig.tight_layout()
    out = out_dir / "optimizer_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def make_plot_sim(out_dir: Path, sims: List[SimInfo]) -> Optional[Path]:
    by_dim: Dict[int, Dict[str, float]] = {}
    for sim in sims:
        if sim.final_dft_deg is None:
            continue
        by_dim.setdefault(sim.dim, {})[sim.label] = sim.final_dft_deg
    if not by_dim:
        return None
    dims = sorted(by_dim.keys())
    labels = sorted({k for d in by_dim.values() for k in d.keys()})
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = list(range(len(dims)))
    for idx, label in enumerate(labels):
        vals = [by_dim[d].get(label, float("nan")) for d in dims]
        offs = [v + (idx - 0.5) * width for v in x]
        ax.bar(offs, vals, width=width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}D" for d in dims])
    ax.set_ylabel("Final DFT relative rotation (deg)")
    ax.set_title("Single-run rotation comparison")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "simulation_rotation_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def make_plot_sweep(out_dir: Path, new: SweepInfo, old: SweepInfo) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(old.intensities, old.final_deg, "o-", label=old.label, color="#d95f02")
    ax.plot(new.intensities, new.final_deg, "o-", label=new.label, color="#1b9e77")
    ax.set_xscale("log")
    ax.set_xlabel("Pump intensity (W/cm^2)")
    ax.set_ylabel("Final DFT relative rotation (deg)")
    ax.set_title("Sweep comparison")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "sweep_rotation_overlay.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def sim_lookup(sims: List[SimInfo], label: str, dim: int) -> Optional[SimInfo]:
    for sim in sims:
        if sim.label == label and sim.dim == dim:
            return sim
    return None


def best_abs_sweep(sweep: SweepInfo) -> tuple[float, float]:
    pairs = [(abs(v), i, v) for i, v in enumerate(sweep.final_deg)]
    pairs.sort(reverse=True)
    best = pairs[0]
    return sweep.intensities[best[1]], best[2]


def read_runtime_info(run_root: Path, old_opt: OptimizerInfo, new_opt: OptimizerInfo) -> RuntimeInfo:
    runtime_path = run_root / "run_times.json"
    payload: Dict[str, Any] = {}
    if runtime_path.exists():
        payload = load_json(runtime_path)
    optimizer = payload.get("optimizer", {}) if isinstance(payload, dict) else {}
    simulations = payload.get("simulations", {}) if isinstance(payload, dict) else {}
    sweeps = payload.get("sweeps", {}) if isinstance(payload, dict) else {}
    return RuntimeInfo(
        optimizer_old_s=parse_float(optimizer.get("old_s")) or old_opt.elapsed_s,
        optimizer_new_s=parse_float(optimizer.get("new_s")) or new_opt.elapsed_s,
        sim_old_1d_s=parse_float(simulations.get("old_dim1_s")),
        sim_new_1d_s=parse_float(simulations.get("new_dim1_s")),
        sim_old_3d_s=parse_float(simulations.get("old_dim3_s")),
        sim_new_3d_s=parse_float(simulations.get("new_dim3_s")),
        sweep_old_1d_s=parse_float(sweeps.get("old_dim1_s")),
        sweep_new_1d_s=parse_float(sweeps.get("new_dim1_s")),
    )


def make_plot_runtime(out_dir: Path, runtime: RuntimeInfo) -> Optional[Path]:
    labels = ["optimizer", "sim 1D", "sim 3D", "sweep 1D"]
    old_vals = [
        runtime.optimizer_old_s,
        runtime.sim_old_1d_s,
        runtime.sim_old_3d_s,
        runtime.sweep_old_1d_s,
    ]
    new_vals = [
        runtime.optimizer_new_s,
        runtime.sim_new_1d_s,
        runtime.sim_new_3d_s,
        runtime.sweep_new_1d_s,
    ]
    if not any(v is not None for v in old_vals + new_vals):
        return None
    old_plot = [v if v is not None else 0.0 for v in old_vals]
    new_plot = [v if v is not None else 0.0 for v in new_vals]
    x = list(range(len(labels)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar([i - width / 2 for i in x], old_plot, width=width, label="old", color="#d95f02")
    ax.bar([i + width / 2 for i in x], new_plot, width=width, label="new", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime comparison")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = out_dir / "runtime_comparison.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_report(
    run_root: Path,
    out_path: Path,
    new_opt: OptimizerInfo,
    old_opt: OptimizerInfo,
    sims: List[SimInfo],
    new_sweep: SweepInfo,
    old_sweep: SweepInfo,
    runtime: RuntimeInfo,
    plot_optimizer: Path,
    plot_runtime: Optional[Path],
    plot_sim: Optional[Path],
    plot_sweep: Path,
) -> None:
    sim_old_1d = sim_lookup(sims, old_opt.label, 1)
    sim_new_1d = sim_lookup(sims, new_opt.label, 1)
    sim_old_3d = sim_lookup(sims, old_opt.label, 3)
    sim_new_3d = sim_lookup(sims, new_opt.label, 3)
    best_i_new, best_rot_new = best_abs_sweep(new_sweep)
    best_i_old, best_rot_old = best_abs_sweep(old_sweep)

    lines: List[str] = []
    lines.append("# Pipeline Comparison Report")
    lines.append("")
    lines.append(f"- Run root: `{run_root.resolve()}`")
    lines.append("- Decay threshold used: `1e-4` (simulation + calibration)")
    lines.append("- Sweep intensity range: `1e8 .. 2e12` W/cm^2, `6` points (log scale)")
    lines.append("- Note: requested sweep parallelism (`workers=6`) is blocked by system semaphore restrictions in this environment; sweeps were run with `workers=1`.")
    lines.append("")
    lines.append("## Optimizer Comparison")
    lines.append("")
    lines.append("| Metric | Old optimizer | New optimizer |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Objective score | {fmt_float(old_opt.objective_score)} | {fmt_float(new_opt.objective_score)} |")
    lines.append(f"| Abs rotation at optimum (deg) | {fmt_float(old_opt.abs_rotation_deg)} | {fmt_float(new_opt.abs_rotation_deg)} |")
    lines.append(f"| Evaluations | {old_opt.evals if old_opt.evals is not None else 'n/a'} | {new_opt.evals if new_opt.evals is not None else 'n/a'} |")
    lines.append(f"| Runtime (s) | {fmt_float(old_opt.elapsed_s)} | {fmt_float(new_opt.elapsed_s)} |")
    lines.append(f"| N periods | {old_opt.n_per if old_opt.n_per is not None else 'n/a'} | {new_opt.n_per if new_opt.n_per is not None else 'n/a'} |")
    lines.append(f"| t_SiN (um) | {fmt_float(old_opt.t_sin_um)} | {fmt_float(new_opt.t_sin_um)} |")
    lines.append(f"| t_SiO2 (um) | {fmt_float(old_opt.t_sio2_um)} | {fmt_float(new_opt.t_sio2_um)} |")
    lines.append(f"| L_cav (um) | {fmt_float(old_opt.l_cav_um)} | {fmt_float(new_opt.l_cav_um)} |")
    lines.append(f"| Pump intensity in objective (W/cm^2) | {fmt_float(old_opt.pump_intensity_w_cm2)} | {fmt_float(new_opt.pump_intensity_w_cm2)} |")
    lines.append("")
    lines.append(f"- Optimizer comparison plot: ![optimizer]({rel_or_abs(plot_optimizer, run_root)})")
    lines.append("")
    lines.append("## Runtime Summary")
    lines.append("")
    lines.append("| Task | Old (s) | New (s) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Optimizer | {fmt_float(runtime.optimizer_old_s)} | {fmt_float(runtime.optimizer_new_s)} |")
    lines.append(f"| Single sim 1D | {fmt_float(runtime.sim_old_1d_s)} | {fmt_float(runtime.sim_new_1d_s)} |")
    lines.append(f"| Single sim 3D | {fmt_float(runtime.sim_old_3d_s)} | {fmt_float(runtime.sim_new_3d_s)} |")
    lines.append(f"| Sweep 1D (6 pts) | {fmt_float(runtime.sweep_old_1d_s)} | {fmt_float(runtime.sweep_new_1d_s)} |")
    if plot_runtime is not None:
        lines.append("")
        lines.append(f"- Runtime comparison plot: ![runtime]({rel_or_abs(plot_runtime, run_root)})")
    lines.append("")
    lines.append("## Single-Run Simulations (Pump=1e12 W/cm^2)")
    lines.append("")
    lines.append("| Case | DFT final rel. rotation (deg) | TD final rel. rotation (deg) | Summary |")
    lines.append("|---|---:|---:|---|")

    def row(name: str, sim: Optional[SimInfo]) -> str:
        if sim is None:
            return f"| {name} | n/a | n/a | n/a |"
        return (
            f"| {name} | {fmt_float(sim.final_dft_deg)} | {fmt_float(sim.final_td_deg)} | "
            f"`{rel_or_abs(sim.summary_path, run_root)}` |"
        )

    lines.append(row(f"{old_opt.label} 1D", sim_old_1d))
    lines.append(row(f"{new_opt.label} 1D", sim_new_1d))
    lines.append(row(f"{old_opt.label} 3D", sim_old_3d))
    lines.append(row(f"{new_opt.label} 3D", sim_new_3d))
    lines.append("")

    if plot_sim is not None:
        lines.append(f"- Single-run comparison plot: ![sim]({rel_or_abs(plot_sim, run_root)})")
    lines.append("")
    lines.append("### Zoomed stabilized-angle plots")
    lines.append("")

    def zoom_lines(title: str, sim: Optional[SimInfo]) -> List[str]:
        if sim is None:
            return [f"- {title}: n/a"]
        p_dft = sim.plot_paths.get("probe_rotation_zoom")
        p_td = sim.plot_paths.get("probe_rotation_td_zoom")
        out = [f"- {title}:"]
        out.append(f"  - DFT zoom: `{p_dft}`" if p_dft else "  - DFT zoom: n/a")
        out.append(f"  - TD zoom: `{p_td}`" if p_td else "  - TD zoom: n/a")
        return out

    lines.extend(zoom_lines(f"{old_opt.label} 1D", sim_old_1d))
    lines.extend(zoom_lines(f"{new_opt.label} 1D", sim_new_1d))
    lines.extend(zoom_lines(f"{old_opt.label} 3D", sim_old_3d))
    lines.extend(zoom_lines(f"{new_opt.label} 3D", sim_new_3d))
    lines.append("")
    lines.append("## Pump-Intensity Sweep (1D)")
    lines.append("")
    lines.append("| Pump intensity (W/cm^2) | Old final rel. rotation (deg) | New final rel. rotation (deg) |")
    lines.append("|---:|---:|---:|")
    for i, i_old in enumerate(old_sweep.intensities):
        i_new = new_sweep.intensities[i]
        i_ref = i_old
        if abs(i_old - i_new) > 1e-6 * max(i_old, 1.0):
            i_ref = 0.5 * (i_old + i_new)
        lines.append(
            f"| {i_ref:.6g} | {old_sweep.final_deg[i]:.9g} | {new_sweep.final_deg[i]:.9g} |"
        )
    lines.append("")
    lines.append(f"- Max |rotation| old sweep: `{abs(best_rot_old):.6g} deg` at `{best_i_old:.6g} W/cm^2`")
    lines.append(f"- Max |rotation| new sweep: `{abs(best_rot_new):.6g} deg` at `{best_i_new:.6g} W/cm^2`")
    lines.append("")
    lines.append(f"- Sweep overlay plot: ![sweep]({rel_or_abs(plot_sweep, run_root)})")
    lines.append(f"- Old sweep curve: `{rel_or_abs(Path(old_sweep.plot_rotation or ''), run_root)}`")
    lines.append(f"- New sweep curve: `{rel_or_abs(Path(new_sweep.plot_rotation or ''), run_root)}`")
    lines.append("")
    lines.append("## Key observations")
    lines.append("")
    lines.append(
        f"- In this run, the old optimizer found a higher objective optimum ({fmt_float(old_opt.abs_rotation_deg)}) than the new optimizer ({fmt_float(new_opt.abs_rotation_deg)})."
    )
    lines.append(
        f"- Sweep behavior diverges strongly at high intensity: old reaches |rotation| > 0.1 deg at 2e12 W/cm^2 ({best_rot_old:.6g} deg), while new remains near {best_rot_new:.6g} deg peak."
    )
    lines.append(
        "- TD final-angle values remain unreliable as an absolute metric; DFT-based final relative angle should be used for optimization and comparisons."
    )
    lines.append(
        "- The stricter decay stop (1e-4) significantly increases runtime for long-ringdown designs (especially old geometry)."
    )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate pipeline comparison report.")
    ap.add_argument("--run-root", type=Path, required=True, help="Pipeline run root directory.")
    ap.add_argument(
        "--out-report",
        type=Path,
        default=None,
        help="Output markdown report path (default: <run-root>/pipeline_comparison_report.md).",
    )
    args = ap.parse_args()

    run_root = args.run_root.resolve()
    out_report = args.out_report.resolve() if args.out_report else (run_root / "pipeline_comparison_report.md")
    plot_dir = run_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    new_opt = read_optimizer(run_root, "new", "new")
    old_opt = read_optimizer(run_root, "old", "old")
    sims: List[SimInfo] = []
    for branch, label in (("new", "new"), ("old", "old")):
        for dim in (1, 3):
            sim = read_sim(run_root, branch, label, dim)
            if sim is not None:
                sims.append(sim)
    new_sweep = read_sweep(run_root, "new", "new")
    old_sweep = read_sweep(run_root, "old", "old")
    runtime = read_runtime_info(run_root, old_opt=old_opt, new_opt=new_opt)

    p_opt = make_plot_optimizer(plot_dir, new_opt, old_opt)
    p_rt = make_plot_runtime(plot_dir, runtime)
    p_sim = make_plot_sim(plot_dir, sims)
    p_sweep = make_plot_sweep(plot_dir, new_sweep, old_sweep)

    write_report(
        run_root=run_root,
        out_path=out_report,
        new_opt=new_opt,
        old_opt=old_opt,
        sims=sims,
        new_sweep=new_sweep,
        old_sweep=old_sweep,
        runtime=runtime,
        plot_optimizer=p_opt,
        plot_runtime=p_rt,
        plot_sim=p_sim,
        plot_sweep=p_sweep,
    )
    print(f"Report written: {out_report}")


if __name__ == "__main__":
    main()
