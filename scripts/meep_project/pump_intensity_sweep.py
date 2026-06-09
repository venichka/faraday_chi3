#!/usr/bin/env python3
"""
Sweep pump intensity for faraday_meep_fp_circ.py and aggregate diagnostics.

Features:
  * Supports quasi-1D and full-3D sweeps (single dim or both dims in one run).
  * Supports serial or parallel execution across sweep points via workers.
  * Stores per-run outputs and aggregate plots/reports per dimension.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from nonlinear_materials import high_index_material_choices

C0 = 299792458.0
FS_PER_MEEP = (1e-6 / C0) * 1e15

_RUN_SIMULATION = None


def _configure_numeric_threads() -> None:
    # Process-level sweep parallelism should not multiply inner BLAS/OpenMP threads.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _get_run_simulation():
    global _RUN_SIMULATION
    if _RUN_SIMULATION is None:
        from faraday_meep_fp_circ import run_simulation as _run_simulation

        _RUN_SIMULATION = _run_simulation
    return _RUN_SIMULATION


@dataclass
class SweepPoint:
    pump_intensity_w_cm2: float
    final_deg: float
    min_deg: float
    max_deg: float
    output_dir: str
    summary_path: str
    trace_bundle_path: str


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _update_point_progress(run_dir: Path, stage: str, **kwargs: Any) -> None:
    payload: Dict[str, Any] = {
        "stage": str(stage),
        "ts_unix": float(time.time()),
    }
    payload.update(kwargs)
    _write_json_atomic(run_dir / "progress.json", payload)


def _read_point_stage(run_dir: Path) -> str:
    path = run_dir / "progress.json"
    if not path.exists():
        return "unknown"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        stage = data.get("stage", "unknown")
        return str(stage)
    except Exception:
        return "unknown"


def _load_existing_point(run_dir: Path) -> Dict[str, Any] | None:
    summary_path = run_dir / "faraday_summary.json"
    trace_bundle_path = run_dir / "sweep_trace_bundle.npz"
    if not (summary_path.exists() and trace_bundle_path.exists()):
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        # Ensure the cached trace bundle is readable; interrupted writes can
        # leave a corrupt .npz that would otherwise break final aggregation.
        with np.load(str(trace_bundle_path)) as npz:
            required = {
                "dft_time",
                "dft_abs_eplus",
                "dft_abs_eminus",
                "td_time",
                "td_abs_eplus",
                "td_abs_eminus",
                "theta_deg_rel",
            }
            if not required.issubset(set(npz.files)):
                return None
        probe = summary.get("probe_rotation_deg", {})
        run_params = summary.get("run_params", {})
        return {
            "pump_intensity_w_cm2": float(run_params.get("pump_intensity_w_cm2", np.nan)),
            "final_deg": float(probe.get("final_relative_deg", np.nan)),
            "min_deg": float(probe.get("min_relative_deg", np.nan)),
            "max_deg": float(probe.get("max_relative_deg", np.nan)),
            "output_dir": str(run_dir),
            "summary_path": str(summary_path),
            "trace_bundle_path": str(trace_bundle_path),
        }
    except Exception:
        return None


def _write_trace_bundle(result, output_dir: Path) -> Path:
    path = output_dir / "sweep_trace_bundle.npz"
    np.savez_compressed(
        path,
        dft_time=np.asarray(result.dft_traces.time, dtype=float),
        dft_freqs=np.asarray(result.dft_traces.freqs, dtype=float),
        dft_abs_eplus=np.asarray(result.dft_traces.abs_eplus, dtype=float),
        dft_abs_eminus=np.asarray(result.dft_traces.abs_eminus, dtype=float),
        td_time=np.asarray(result.time_domain_traces.time, dtype=float),
        td_freqs=np.asarray(result.time_domain_traces.freqs, dtype=float),
        td_abs_eplus=np.asarray(result.time_domain_traces.abs_eplus, dtype=float),
        td_abs_eminus=np.asarray(result.time_domain_traces.abs_eminus, dtype=float),
        theta_deg_rel=np.asarray(result.probe_rotation.theta_deg_rel, dtype=float),
        theta_total_deg_rel=np.asarray(
            getattr(result.probe_rotation, "theta_total_deg_rel", None)
            if getattr(result.probe_rotation, "theta_total_deg_rel", None) is not None
            else result.probe_rotation.theta_deg_rel,
            dtype=float,
        ),
    )
    return path


def _load_summary_cached(point: Dict[str, Any]) -> Dict[str, Any]:
    cached = point.get("_summary")
    if isinstance(cached, dict):
        return cached
    with Path(str(point["summary_path"])).open("r", encoding="utf-8") as f:
        data = json.load(f)
    point["_summary"] = data
    return data


def _load_trace_bundle_cached(point: Dict[str, Any]) -> Dict[str, np.ndarray]:
    cached = point.get("_trace_bundle")
    if isinstance(cached, dict):
        return cached
    with np.load(str(point["trace_bundle_path"])) as npz:
        data = {key: np.asarray(npz[key]) for key in npz.files}
    point["_trace_bundle"] = data
    return data


def _parse_intensity_list(raw: str) -> List[float]:
    entries = [token.strip() for token in raw.split(",")]
    values = [float(token) for token in entries if token]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one pump intensity value.")
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("Pump intensities must be > 0.")
    return values


def _parse_dims(raw: str) -> List[int]:
    entries = [token.strip() for token in raw.split(",") if token.strip()]
    if not entries:
        raise argparse.ArgumentTypeError("Provide at least one dimension value.")
    dims = []
    for token in entries:
        value = int(token)
        if value not in (1, 3):
            raise argparse.ArgumentTypeError("Dimensions must be 1 and/or 3.")
        if value not in dims:
            dims.append(value)
    return dims


def _format_intensity_label(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2f}e{exponent}"


def _resolve_intensities(args: argparse.Namespace) -> List[float]:
    if args.intensity_range is not None:
        start, stop, count_raw = args.intensity_range
        count = int(count_raw)
        if count < 2:
            raise argparse.ArgumentTypeError("Range count must be >= 2.")
        if start <= 0 or stop <= 0:
            raise argparse.ArgumentTypeError("Range intensities must be > 0.")
        if args.range_scale == "log":
            values = np.geomspace(start, stop, count)
        else:
            values = np.linspace(start, stop, count)
        return [float(v) for v in values]

    if args.intensities is None:
        raise argparse.ArgumentTypeError("Provide --intensities or --intensity-range.")
    return _parse_intensity_list(args.intensities)


def _build_args(
    namespace: argparse.Namespace,
    dim: int,
    pump_intensity: float,
    output_dir: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        dim=int(dim),
        mode=namespace.mode,
        materials=namespace.materials,
        high_index_material=namespace.high_index_material,
        nH=namespace.nH,
        kH=namespace.kH,
        nL=namespace.nL,
        kappa_ref_lambda=namespace.kappa_ref_lambda,
        high_index_n2=namespace.high_index_n2,
        sin_fit=namespace.sin_fit,
        sio2_fit=namespace.sio2_fit,
        fit_window=tuple(namespace.fit_window),
        fit_poles=namespace.fit_poles,
        pump_intensity=float(pump_intensity),
        geometry_file=namespace.geometry_file,
        cavity_modes_file=namespace.cavity_modes_file,
        output_dir=str(output_dir),
        until_time=namespace.until_time,
        decay_threshold=namespace.decay_threshold,
        resolution=namespace.resolution,
        calibrate_sources=bool(namespace.calibrate_sources),
        calibration_decay_threshold=namespace.calibration_decay_threshold,
        pump1_frequency=None,
        pump2_frequency=None,
    )


def _run_single_job(job: Tuple[int, float, Dict[str, object]]) -> Tuple[int, float, Dict[str, Any]]:
    _configure_numeric_threads()
    dim, intensity, args_dict = job
    sim_args = argparse.Namespace(**args_dict)
    point_dir = Path(str(sim_args.output_dir))
    _update_point_progress(
        point_dir,
        "job_started",
        dim=int(dim),
        pump_intensity_w_cm2=float(intensity),
    )

    _update_point_progress(point_dir, "main_simulation_running")
    run_simulation = _get_run_simulation()
    result = run_simulation(sim_args)

    out_dir = Path(str(result.output_dir))
    trace_bundle = _write_trace_bundle(result, out_dir)
    point = SweepPoint(
        pump_intensity_w_cm2=float(result.pump_intensity_w_cm2),
        final_deg=float(result.probe_rotation.final_deg),
        min_deg=float(result.probe_rotation.min_deg),
        max_deg=float(result.probe_rotation.max_deg),
        output_dir=str(result.output_dir),
        summary_path=str(result.summary_path),
        trace_bundle_path=str(trace_bundle),
    )
    _update_point_progress(
        point_dir,
        "completed",
        final_theta_rel_deg=float(point.final_deg),
        summary_path=str(point.summary_path),
    )
    return int(dim), float(intensity), dict(point.__dict__)


def _fit_linear_vs_log_intensity(
    intensities: Sequence[float], values: Sequence[float]
) -> Dict[str, float] | None:
    i_arr = np.asarray(intensities, dtype=float)
    y_arr = np.asarray(values, dtype=float)
    valid = np.isfinite(i_arr) & np.isfinite(y_arr) & (i_arr > 0)
    if np.count_nonzero(valid) < 2:
        return None
    x = np.log10(i_arr[valid])
    y = y_arr[valid]
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return {
        "model": "theta = a*log10(I) + b",
        "a": float(slope),
        "b": float(intercept),
        "r2": r2,
        "n_points": int(np.count_nonzero(valid)),
    }


def _fit_power_law_abs_rotation(
    intensities: Sequence[float],
    values: Sequence[float],
    min_abs_theta: float = 1e-9,
) -> Dict[str, float] | None:
    i_arr = np.asarray(intensities, dtype=float)
    y_arr = np.asarray(values, dtype=float)
    abs_y = np.abs(y_arr)
    valid = (
        np.isfinite(i_arr)
        & np.isfinite(abs_y)
        & (i_arr > 0)
        & (abs_y > float(min_abs_theta))
    )
    if np.count_nonzero(valid) < 2:
        return None
    x = np.log10(i_arr[valid])
    z = np.log10(abs_y[valid])
    slope, intercept = np.polyfit(x, z, 1)
    z_pred = slope * x + intercept
    ss_res = float(np.sum((z - z_pred) ** 2))
    ss_tot = float(np.sum((z - np.mean(z)) ** 2))
    r2_log = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    a_pref = float(10.0 ** intercept)
    return {
        "model": "|theta| = A*I^p",
        "A": a_pref,
        "p": float(slope),
        "r2_log": r2_log,
        "min_abs_theta_used_deg": float(min_abs_theta),
        "n_points": int(np.count_nonzero(valid)),
    }


def _predict_linear_vs_log_fit(
    fit: Dict[str, float] | None, intensities: Sequence[float]
) -> np.ndarray | None:
    if not fit:
        return None
    i_arr = np.asarray(intensities, dtype=float)
    valid = np.isfinite(i_arr) & (i_arr > 0)
    if not np.any(valid):
        return None
    out = np.full_like(i_arr, np.nan, dtype=float)
    out[valid] = float(fit["a"]) * np.log10(i_arr[valid]) + float(fit["b"])
    return out


def _predict_power_fit(
    fit: Dict[str, float] | None, intensities: Sequence[float]
) -> np.ndarray | None:
    if not fit:
        return None
    i_arr = np.asarray(intensities, dtype=float)
    valid = np.isfinite(i_arr) & (i_arr > 0)
    if not np.any(valid):
        return None
    out = np.full_like(i_arr, np.nan, dtype=float)
    out[valid] = float(fit["A"]) * np.power(i_arr[valid], float(fit["p"]))
    return out


def _write_points_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _nested_get(mapping: Dict[str, Any], path: Sequence[str], default: Any = np.nan) -> Any:
    node: Any = mapping
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _write_dim_markdown_summary(
    dim_root: Path,
    dim: int,
    dim_report: Dict[str, Any],
    global_config: Dict[str, Any],
) -> Path:
    md_path = dim_root / "pump_intensity_sweep_summary.md"
    intensities = dim_report["intensities_w_cm2"]
    final_rot = dim_report["rotation_final_deg"]
    min_rot = dim_report["rotation_min_deg"]
    max_rot = dim_report["rotation_max_deg"]
    fit_linear = dim_report.get("fit_models", {}).get("linear_vs_log_intensity")
    fit_power = dim_report.get("fit_models", {}).get("power_law_abs_rotation")
    plots = dim_report["plot_paths"]
    csv_path = dim_report.get("points_csv_path")
    best_point = dim_report.get("best_point", {})
    aggregate_metrics = dim_report.get("aggregate_metrics", {})
    wavelengths_um = dim_report.get("wavelengths_um", {})
    run_params = dim_report.get("run_params", {})

    lines: List[str] = []
    lines.append(f"# Pump Intensity Sweep Summary (dim={dim})")
    lines.append("")
    lines.append("## Configuration")
    lines.append(
        f"- Sweep range: `{global_config['intensity_min_w_cm2']:.3e}` to "
        f"`{global_config['intensity_max_w_cm2']:.3e}` W/cm^2"
    )
    lines.append(f"- Number of points: `{global_config['num_points']}`")
    lines.append(f"- Spacing: `{global_config['range_scale']}`")
    lines.append(f"- Workers: `{global_config['workers_effective']}`")
    lines.append(f"- Mode: `{global_config['mode']}`")
    lines.append(f"- Materials: `{global_config['materials']}`")
    lines.append(f"- High-index material: `{global_config.get('high_index_material')}`")
    lines.append(
        f"- High-index n/k/n2: "
        f"`{global_config.get('high_index_n')}` / "
        f"`{global_config.get('high_index_k')}` / "
        f"`{global_config.get('high_index_n2_m2_per_w')}`"
    )
    lines.append(f"- Geometry file: `{global_config['geometry_file']}`")
    lines.append(f"- Cavity modes file: `{global_config['cavity_modes_file']}`")
    lines.append(f"- Decay threshold: `{global_config['decay_threshold']}`")
    lines.append("")
    lines.append("## Final Rotation Data")
    lines.append("")
    lines.append("| Pump intensity (W/cm^2) | Final wrapped (deg) | Min (deg) | Max (deg) |")
    lines.append("|---:|---:|---:|---:|")
    for idx, (i_val, f_val, mn_val, mx_val) in enumerate(zip(intensities, final_rot, min_rot, max_rot)):
        lines.append(
            f"| {float(i_val):.6e} | {float(f_val):.9f} | {float(mn_val):.9f} | {float(mx_val):.9f} |"
        )
    lines.append("")
    lines.append("## Key Metrics")
    lines.append(
        f"- Best `|theta_final|`: `{float(best_point.get('abs_final_relative_deg', float('nan'))):.6g}` deg "
        f"at `I={float(best_point.get('pump_intensity_w_cm2', float('nan'))):.6e}` W/cm^2"
    )
    lines.append(
        f"- Mean `|theta_final|`: `{float(aggregate_metrics.get('mean_abs_final_relative_deg', float('nan'))):.6g}` deg"
    )
    lines.append(
        f"- Max `|theta_final|`: `{float(aggregate_metrics.get('max_abs_final_relative_deg', float('nan'))):.6g}` deg"
    )
    lines.append(
        f"- Mean signed `theta_final`: `{float(aggregate_metrics.get('mean_relative_deg', float('nan'))):.6g}` deg"
    )
    lines.append("")
    if wavelengths_um:
        lines.append("## Wavelength Targets")
        lines.append(f"- Pump1: `{float(wavelengths_um.get('pump1', float('nan'))):.6f}` um")
        lines.append(f"- Pump2: `{float(wavelengths_um.get('pump2', float('nan'))):.6f}` um")
        lines.append(f"- Probe: `{float(wavelengths_um.get('probe', float('nan'))):.6f}` um")
        lines.append("")
    if run_params:
        lines.append("## Run Parameters")
        lines.append(f"- Resolution: `{run_params.get('resolution')}`")
        lines.append(f"- Pulse duration: `{run_params.get('pulse_duration_fs')}` fs")
        lines.append(f"- Probe intensity: `{run_params.get('probe_intensity_w_cm2')}` W/cm^2")
        lines.append(f"- Pump cutoff: `{run_params.get('pump_cutoff')}`")
        lines.append("")
    lines.append("## Fit Models")
    if fit_linear:
        lines.append(
            f"- Linear vs log-intensity: "
            f"`theta = {fit_linear['a']:.6g}*log10(I) + {fit_linear['b']:.6g}`, "
            f"`R^2={fit_linear['r2']:.5f}`"
        )
    else:
        lines.append("- Linear vs log-intensity fit: not available.")
    if fit_power:
        lines.append(
            f"- Power-law on |theta|: "
            f"`|theta| = {fit_power['A']:.6g}*I^{fit_power['p']:.6g}`, "
            f"`R^2(log)={fit_power['r2_log']:.5f}`"
        )
    else:
        lines.append("- Power-law fit on |theta|: not available.")
    lines.append("")
    lines.append("## Plots")
    lines.append("")
    lines.append(f"![rotation summary](./{Path(plots['rotation_vs_intensity']).name})")
    lines.append(f"![dft traces](./{Path(plots['dft_traces']).name})")
    lines.append(f"![time-domain traces](./{Path(plots['time_domain_traces']).name})")
    lines.append("")
    if csv_path:
        lines.append(f"- CSV points file: `{Path(csv_path).name}`")
    report_json_rel = os.path.relpath(Path(dim_report["report_json_path"]), start=dim_root)
    lines.append(f"- JSON sweep report: `{report_json_rel}`")
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def _plot_rotation_vs_intensity(
    intensities: Sequence[float],
    final_deg: Sequence[float],
    output_path: Path,
    title: str,
    fit_power: Dict[str, float] | None = None,
    companions: Sequence[Tuple[str, Sequence[float], str]] | None = None,
) -> None:
    intensities_arr = np.asarray(intensities, dtype=float)
    final_arr = np.asarray(final_deg, dtype=float)
    abs_theta = np.abs(final_arr)

    power_pred = _predict_power_fit(fit_power, intensities_arr)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    eps = 1e-12
    y_main = np.clip(abs_theta, eps, None)

    ax.plot(
        intensities_arr,
        y_main,
        "o-",
        color="tab:blue",
        lw=2.0,
        ms=5.5,
        label="|theta| forward-isolated, coherent (objective)",
    )
    # Companion readings (forward incoherent, raw total-field) for comparison.
    for label, values, color in companions or []:
        comp = np.abs(np.asarray(values, dtype=float))
        if not np.any(np.isfinite(comp)):
            continue
        ax.plot(
            intensities_arr,
            np.clip(comp, eps, None),
            "s--",
            color=color,
            lw=1.4,
            ms=4.0,
            alpha=0.85,
            label=label,
        )
    if power_pred is not None:
        ax.plot(
            intensities_arr,
            np.clip(power_pred, eps, None),
            "--",
            color="tab:orange",
            lw=2.0,
            label="fit: |theta| = A*I^p",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Pump intensity (W/cm^2)")
    ax.set_ylabel("|theta_rel| (deg)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    fit_lines: List[str] = []
    if fit_power is not None:
        fit_lines.append(
            f"power fit: A={fit_power['A']:.3g}, p={fit_power['p']:.3g}, R^2(log)={fit_power['r2_log']:.3f}"
        )
    if fit_lines:
        ax.text(
            0.01,
            0.02,
            "\n".join(fit_lines),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rotation_multi_dim(
    results_by_dim: Dict[int, Sequence[Dict[str, Any]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for dim in sorted(results_by_dim.keys()):
        ordered = sorted(results_by_dim[dim], key=lambda r: float(r["pump_intensity_w_cm2"]))
        if not ordered:
            continue
        intensities = np.array([float(r["pump_intensity_w_cm2"]) for r in ordered], dtype=float)
        final_rot = np.array([float(r["final_deg"]) for r in ordered], dtype=float)
        ax.plot(intensities, final_rot, "o-", label=f"dim={dim}")

    ax.set_xscale("log")
    ax.set_xlabel("Pump intensity (W/cm^2)")
    ax.set_ylabel("theta_rel (deg)")
    ax.set_title("Faraday rotation vs pump intensity (1D vs 3D)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_traces(
    results: Sequence[Dict[str, Any]],
    output_path: Path,
    trace_attr: str,
    title_suffix: str,
) -> None:
    """Render heatmaps of |E|(t, I_pump) for pumps, probe, and sidebands."""
    if not results:
        return

    ordered = sorted(results, key=lambda r: float(r["pump_intensity_w_cm2"]))
    bundles = [_load_trace_bundle_cached(point) for point in ordered]
    prefix = "dft" if str(trace_attr) == "dft_traces" else "td"

    time_arrays = [np.asarray(b[f"{prefix}_time"], dtype=float) for b in bundles]
    reference_time = time_arrays[0]
    if all(np.array_equal(t, reference_time) for t in time_arrays[1:]):
        master_time = reference_time
    else:
        master_time = np.unique(np.concatenate(time_arrays))
    master_time_fs = master_time * FS_PER_MEEP
    intensities = np.array([float(point["pump_intensity_w_cm2"]) for point in ordered], dtype=float)
    n_rows = len(intensities)
    sweep_rows = np.arange(n_rows, dtype=float)

    channel_specs = [
        ("Pump1 e-", "minus", 0),
        ("Pump2 e+", "plus", 1),
        ("Probe e+", "plus", 2),
        ("Probe e-", "minus", 2),
        ("Sideband - (e-)", "minus", 3),
        ("Sideband + (e+)", "plus", 4),
    ]

    ncols = 3
    nrows = int(np.ceil(len(channel_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    def _resample(values_arr: np.ndarray, trace_time: np.ndarray) -> np.ndarray:
        values = np.asarray(values_arr, dtype=float)
        tvals = np.asarray(trace_time, dtype=float)
        if len(trace_time) == len(master_time) and np.allclose(trace_time, master_time):
            return values
        valid = np.isfinite(tvals) & np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            return np.full_like(master_time, 0.0, dtype=float)
        t_fit = tvals[valid]
        v_fit = values[valid]
        # Fill outside each run's recorded interval with finite values to avoid
        # NaN "white gaps" in heatmaps when runs terminate at different times.
        return np.interp(master_time, t_fit, v_fit, left=float(v_fit[0]), right=0.0)

    for ax_index, (ax, (label, handedness, idx)) in enumerate(zip(axes, channel_specs)):
        arr_key = f"{prefix}_abs_eplus" if handedness == "plus" else f"{prefix}_abs_eminus"
        rows = [
            _resample(np.asarray(bundle[arr_key], dtype=float)[:, int(idx)], t_arr)
            for bundle, t_arr in zip(bundles, time_arrays)
        ]
        data = np.vstack(rows)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = 0.0
            vmax = float(np.nanpercentile(finite, 99.5))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = float(np.nanmax(finite))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
        pcm = ax.imshow(
            data,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=(float(master_time_fs[0]), float(master_time_fs[-1]), -0.5, n_rows - 0.5),
        )
        ax.set_title(f"{label} ({title_suffix})")
        ax.set_xlabel("time (fs)")
        ax.set_yticks(sweep_rows)
        ax.set_yticklabels([f"{val:.1e}" for val in intensities], fontsize=8)
        if (ax_index % ncols) == 0:
            ax.set_ylabel("pump intensity (W/cm^2)")
        fig.colorbar(pcm, ax=ax, fraction=0.047, pad=0.02, label="|E|")

    for ax in axes[len(channel_specs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_global_markdown_summary(output_root: Path, sweep_report: Dict[str, Any]) -> Path:
    md_path = output_root / "pump_intensity_sweep_summary.md"
    dim_reports: Dict[str, Dict[str, Any]] = sweep_report.get("dimension_reports", {})

    def _safe_rel(path_like: str) -> str:
        path = Path(path_like)
        try:
            return str(path.resolve().relative_to(output_root.resolve()))
        except Exception:
            return str(path)

    lines: List[str] = []
    lines.append("# Pump Intensity Sweep - Global Summary")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Dimensions: `{sweep_report.get('dims', [])}`")
    lines.append(f"- Workers requested/effective: `{sweep_report.get('workers_requested')}` / `{sweep_report.get('workers_effective')}`")
    lines.append(f"- Range scale: `{sweep_report.get('range_scale')}`")
    lines.append(
        f"- Intensity range: `{float(sweep_report.get('intensity_min_w_cm2', float('nan'))):.3e}` to "
        f"`{float(sweep_report.get('intensity_max_w_cm2', float('nan'))):.3e}` W/cm^2"
    )
    lines.append(f"- Number of points: `{sweep_report.get('num_points')}`")
    lines.append(f"- Mode/materials: `{sweep_report.get('mode')}` / `{sweep_report.get('materials')}`")
    lines.append(
        f"- High-index material n/k/n2: "
        f"`{sweep_report.get('high_index_material')}` / "
        f"`{sweep_report.get('high_index_n')}` / "
        f"`{sweep_report.get('high_index_k')}` / "
        f"`{sweep_report.get('high_index_n2_m2_per_w')}`"
    )
    lines.append("")

    multi_plot = sweep_report.get("plot_paths", {}).get("rotation_vs_intensity_by_dim")
    if multi_plot:
        lines.append("## Cross-Dimension Rotation")
        lines.append(f"![rotation by dim](./{Path(multi_plot).name})")
        lines.append("")

    lines.append("## Dimension Comparison")
    lines.append("")
    lines.append("| dim | best |theta| (deg) | intensity at best (W/cm^2) | linear slope a (deg/dec) | power exponent p |")
    lines.append("|---:|---:|---:|---:|---:|")
    for dim_key in sorted(dim_reports.keys(), key=lambda x: int(x)):
        rep = dim_reports[dim_key]
        best = rep.get("best_point", {})
        fit_linear = rep.get("fit_models", {}).get("linear_vs_log_intensity") or {}
        fit_power = rep.get("fit_models", {}).get("power_law_abs_rotation") or {}
        lines.append(
            f"| {int(dim_key)} | "
            f"{float(best.get('abs_final_relative_deg', float('nan'))):.6g} | "
            f"{float(best.get('pump_intensity_w_cm2', float('nan'))):.6e} | "
            f"{float(fit_linear.get('a', float('nan'))):.6g} | "
            f"{float(fit_power.get('p', float('nan'))):.6g} |"
        )
    lines.append("")

    lines.append("## Per-Dimension Artifacts")
    for dim_key in sorted(dim_reports.keys(), key=lambda x: int(x)):
        rep = dim_reports[dim_key]
        lines.append(f"- dim={int(dim_key)}")
        if rep.get("markdown_summary_path"):
            lines.append(f"  - Markdown summary: `{_safe_rel(rep['markdown_summary_path'])}`")
        if rep.get("points_csv_path"):
            lines.append(f"  - CSV points: `{_safe_rel(rep['points_csv_path'])}`")
        plot_paths = rep.get("plot_paths", {})
        if plot_paths.get("rotation_vs_intensity"):
            lines.append(f"  - Rotation plot: `{_safe_rel(plot_paths['rotation_vs_intensity'])}`")
        if plot_paths.get("dft_traces"):
            lines.append(f"  - DFT traces: `{_safe_rel(plot_paths['dft_traces'])}`")
        if plot_paths.get("time_domain_traces"):
            lines.append(f"  - TD traces: `{_safe_rel(plot_paths['time_domain_traces'])}`")
    lines.append("")

    lines.append("## Embedded Plots")
    lines.append("")
    for dim_key in sorted(dim_reports.keys(), key=lambda x: int(x)):
        rep = dim_reports[dim_key]
        plot_paths = rep.get("plot_paths", {})
        lines.append(f"### dim={int(dim_key)}")
        if plot_paths.get("rotation_vs_intensity"):
            lines.append(f"![dim{int(dim_key)} rotation](./{_safe_rel(plot_paths['rotation_vs_intensity'])})")
        if plot_paths.get("dft_traces"):
            lines.append(f"![dim{int(dim_key)} dft traces](./{_safe_rel(plot_paths['dft_traces'])})")
        if plot_paths.get("time_domain_traces"):
            lines.append(f"![dim{int(dim_key)} td traces](./{_safe_rel(plot_paths['time_domain_traces'])})")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def main() -> None:
    _configure_numeric_threads()
    parser = argparse.ArgumentParser(description="Sweep pump intensity for the Faraday rotation simulation.")
    parser.add_argument(
        "--intensities",
        type=str,
        default=None,
        help="Comma-separated pump intensities in W/cm^2.",
    )
    parser.add_argument(
        "--intensity-range",
        type=float,
        nargs=3,
        metavar=("I_MIN", "I_MAX", "COUNT"),
        default=None,
        help="Generate sweep values from [I_MIN, I_MAX] using COUNT points.",
    )
    parser.add_argument(
        "--range-scale",
        choices=("log", "linear"),
        default="log",
        help="Spacing for --intensity-range (default: %(default)s).",
    )
    parser.add_argument(
        "--dim",
        type=int,
        choices=(1, 3),
        default=1,
        help="Single simulation dimensionality if --dims is not provided.",
    )
    parser.add_argument(
        "--dims",
        type=str,
        default=None,
        help="Comma-separated dimensions to run (e.g. '1,3'). Overrides --dim.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for sweep jobs.")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick", help="Simulation preset.")
    parser.add_argument(
        "--materials",
        choices=("library", "constant", "fit"),
        default="library",
        help="Material model forwarded to the simulation.",
    )
    parser.add_argument(
        "--high-index-material",
        choices=high_index_material_choices(),
        default="sin",
        help="High-index material preset forwarded to the simulation.",
    )
    parser.add_argument("--nH", type=float, default=None, help="Override high-index value when --materials constant.")
    parser.add_argument("--kH", type=float, default=None, help="Override high-index extinction coefficient k.")
    parser.add_argument("--nL", type=float, default=None, help="Override low-index value when --materials constant.")
    parser.add_argument(
        "--kappa-ref-lambda",
        type=float,
        default=1.55,
        help="Reference wavelength (um) for mapping constant k to conductivity.",
    )
    parser.add_argument(
        "--high-index-n2",
        type=float,
        default=None,
        help="Override Kerr nonlinear index n2 (m^2/W) for high-index material.",
    )
    parser.add_argument("--sin-fit", dest="sin_fit", type=str, default=None, help="CSV for SiN when --materials fit.")
    parser.add_argument("--sio2-fit", dest="sio2_fit", type=str, default=None, help="CSV for SiO2 when --materials fit.")
    parser.add_argument(
        "--fit-window",
        type=int,
        nargs=2,
        metavar=("lambda_min", "lambda_max"),
        default=(600, 2000),
        help="Wavelength window (nm) forwarded to faraday_meep_fp_circ.",
    )
    parser.add_argument("--fit-poles", type=int, default=2, help="Number of poles if --materials fit.")
    parser.add_argument("--geometry-file", type=str, default="optimized_geometry.json", help="Geometry JSON.")
    parser.add_argument("--cavity-modes-file", type=str, default="cavity_modes.json", help="Cavity-modes JSON.")
    parser.add_argument("--until-time", type=float, default=None, help="Forwarded until-time override.")
    parser.add_argument(
        "--decay-threshold",
        type=float,
        default=1e-4,
        help="Field-decay threshold when --until-time is not provided. Default 1e-4 "
        "(use 1e-3 for a fast sweep, 1e-6 for converged).",
    )
    parser.add_argument("--resolution", type=int, default=None, help="Resolution override for each run.")
    parser.add_argument("--calibrate-sources", action="store_true", help="Forward source calibration flag.")
    parser.add_argument(
        "--calibration-decay-threshold",
        type=float,
        default=1e-7,
        help="Decay threshold for source calibration helper runs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="pump_intensity_sweep_outputs",
        help="Directory where per-intensity runs and aggregate plots are stored.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing per-point outputs and rerun simulations before aggregation.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=20.0,
        help="Seconds between parent heartbeat progress lines while workers are running.",
    )
    args = parser.parse_args()

    intensities = _resolve_intensities(args)
    dims = _parse_dims(args.dims) if args.dims is not None else [int(args.dim)]
    workers = max(1, int(args.workers))
    workers = min(workers, max(1, int(os.cpu_count() or 1)))

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reuse_existing = not bool(args.force_rerun)
    jobs: List[Tuple[int, float, Dict[str, object]]] = []
    results_by_dim: Dict[int, List[Dict[str, Any]]] = {int(dim): [] for dim in dims}
    for dim in dims:
        dim_root = output_root / f"dim{dim}"
        dim_root.mkdir(parents=True, exist_ok=True)
        for idx, intensity in enumerate(intensities):
            run_dir = dim_root / f"I_{idx:02d}_{_format_intensity_label(float(intensity))}"
            run_dir.mkdir(parents=True, exist_ok=True)
            reused = _load_existing_point(run_dir) if reuse_existing else None
            if reused is not None:
                # Keep the exact requested intensity in the index, even if summary metadata is missing.
                reused["pump_intensity_w_cm2"] = float(intensity)
                results_by_dim[int(dim)].append(reused)
                _update_point_progress(
                    run_dir,
                    "reused_existing",
                    dim=int(dim),
                    pump_intensity_w_cm2=float(intensity),
                )
                continue
            _update_point_progress(
                run_dir,
                "queued",
                dim=int(dim),
                pump_intensity_w_cm2=float(intensity),
            )
            sim_args = _build_args(args, dim, float(intensity), run_dir)
            jobs.append((int(dim), float(intensity), dict(vars(sim_args))))

    total_jobs = len(jobs)
    reused_count = sum(len(v) for v in results_by_dim.values())
    print(
        f"[sweep] dims={dims} points={len(intensities)} total_jobs={total_jobs} reused={reused_count} workers={workers}",
        flush=True,
    )

    if total_jobs == 0:
        print("[sweep] no pending jobs; aggregating existing outputs only.", flush=True)
    elif workers == 1:
        for done, job in enumerate(jobs, start=1):
            dim, intensity, point = _run_single_job(job)
            results_by_dim[dim].append(point)
            print(
                "[sweep]",
                f"done={done}/{total_jobs}",
                f"dim={dim}",
                f"I={intensity:.3e}",
                f"final_theta={float(point['final_deg']):.6f}",
                flush=True,
            )
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_job = {executor.submit(_run_single_job, job): job for job in jobs}
            done = 0
            pending = set(future_to_job.keys())
            interval = max(1.0, float(args.progress_interval))
            last_heartbeat = float(time.time())
            while pending:
                finished, pending = cf.wait(
                    pending,
                    timeout=interval,
                    return_when=cf.FIRST_COMPLETED,
                )
                for fut in finished:
                    job = future_to_job[fut]
                    dim, intensity, _ = job
                    try:
                        dim_out, intensity_out, point = fut.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"Sweep job failed for dim={dim}, intensity={intensity:.3e}: {exc}"
                        ) from exc
                    results_by_dim[dim_out].append(point)
                    done += 1
                    print(
                        "[sweep]",
                        f"done={done}/{total_jobs}",
                        f"dim={dim_out}",
                        f"I={intensity_out:.3e}",
                        f"final_theta={float(point['final_deg']):.6f}",
                        flush=True,
                    )
                now = float(time.time())
                if pending and (now - last_heartbeat) >= interval:
                    stage_counts: Dict[str, int] = {}
                    for fut in pending:
                        _, _, args_pending = future_to_job[fut]
                        run_dir = Path(str(args_pending.get("output_dir", "")))
                        stage = _read_point_stage(run_dir)
                        stage_counts[stage] = int(stage_counts.get(stage, 0) + 1)
                    stage_text = ", ".join(
                        f"{k}:{v}" for k, v in sorted(stage_counts.items(), key=lambda kv: kv[0])
                    )
                    print(
                        "[sweep]",
                        f"heartbeat done={done}/{total_jobs}",
                        f"pending={len(pending)}",
                        f"stages={stage_text or 'unknown'}",
                        flush=True,
                    )
                    last_heartbeat = now

    report_path = output_root / "pump_intensity_sweep_report.json"
    global_config: Dict[str, Any] = {
        "intensity_min_w_cm2": float(min(intensities)),
        "intensity_max_w_cm2": float(max(intensities)),
        "num_points": int(len(intensities)),
        "range_scale": str(args.range_scale),
        "workers_effective": int(workers),
        "workers_requested": int(args.workers),
        "mode": str(args.mode),
        "materials": str(args.materials),
        "high_index_material": str(args.high_index_material),
        "high_index_n": (float(args.nH) if args.nH is not None else None),
        "high_index_k": (float(args.kH) if args.kH is not None else None),
        "high_index_n2_m2_per_w": (
            float(args.high_index_n2) if args.high_index_n2 is not None else None
        ),
        "kappa_ref_lambda_um": float(args.kappa_ref_lambda),
        "geometry_file": str(args.geometry_file),
        "cavity_modes_file": str(args.cavity_modes_file),
        "decay_threshold": float(args.decay_threshold),
        "resolution_override": int(args.resolution) if args.resolution is not None else None,
        "dimensions": [int(d) for d in dims],
    }

    dimension_reports: Dict[str, Dict[str, Any]] = {}
    for dim in sorted(results_by_dim.keys()):
        dim_results = sorted(results_by_dim[dim], key=lambda r: float(r["pump_intensity_w_cm2"]))
        if not dim_results:
            continue

        dim_root = output_root / f"dim{dim}"
        sweep_intensities = [float(res["pump_intensity_w_cm2"]) for res in dim_results]
        final_rot = [float(res["final_deg"]) for res in dim_results]
        min_rot = [float(res["min_deg"]) for res in dim_results]
        max_rot = [float(res["max_deg"]) for res in dim_results]
        var_rot = []
        for res in dim_results:
            bundle = _load_trace_bundle_cached(res)
            theta_series = np.asarray(bundle.get("theta_deg_rel", np.array([])), dtype=float)
            theta_valid = theta_series[np.isfinite(theta_series)]
            if theta_valid.size:
                tail_count = max(5, int(np.ceil(0.2 * theta_valid.size)))
                theta_tail = theta_valid[-tail_count:]
                var_rot.append(float(np.var(theta_tail)))
            else:
                var_rot.append(float("nan"))
        abs_final = [abs(v) for v in final_rot]
        tuned_p1_freq: List[float] = []
        tuned_p2_freq: List[float] = []
        cold_p1_freq: List[float] = []
        cold_p2_freq: List[float] = []
        for res in dim_results:
            summary = _load_summary_cached(res)
            freqs = _nested_get(summary, ["frequencies_inv_um"], {})
            p1_used = float(_nested_get(freqs, ["pump1"], np.nan))
            p2_used = float(_nested_get(freqs, ["pump2"], np.nan))
            p1_cold = float(_nested_get(freqs, ["pump1_cold"], p1_used))
            p2_cold = float(_nested_get(freqs, ["pump2_cold"], p2_used))
            tuned_p1_freq.append(p1_used)
            tuned_p2_freq.append(p2_used)
            cold_p1_freq.append(p1_cold)
            cold_p2_freq.append(p2_cold)

        fit_linear = _fit_linear_vs_log_intensity(sweep_intensities, final_rot)
        fit_power = _fit_power_law_abs_rotation(sweep_intensities, final_rot)

        fwd_incoherent_rot = [
            float(
                _nested_get(
                    _load_summary_cached(res),
                    ["probe_stokes_dft", "tail_weighted", "theta_relative_deg"],
                    np.nan,
                )
            )
            for res in dim_results
        ]
        total_field_rot = [
            float(
                _nested_get(
                    _load_summary_cached(res),
                    ["probe_stokes_total", "tail_weighted", "theta_relative_deg"],
                    np.nan,
                )
            )
            for res in dim_results
        ]

        rotation_plot = dim_root / "rotation_vs_intensity.png"
        _plot_rotation_vs_intensity(
            sweep_intensities,
            final_rot,
            rotation_plot,
            title=f"Faraday rotation vs pump intensity (dim={dim})",
            fit_power=fit_power,
            companions=[
                ("|theta| forward-isolated, incoherent", fwd_incoherent_rot, "tab:green"),
                ("|theta| total-field (no fwd/bwd split)", total_field_rot, "tab:red"),
            ],
        )

        dft_plot = dim_root / "dft_traces_vs_intensity.png"
        _plot_traces(dim_results, dft_plot, trace_attr="dft_traces", title_suffix=f"DFT |E| dim={dim}")

        td_plot = dim_root / "time_domain_traces_vs_intensity.png"
        _plot_traces(
            dim_results,
            td_plot,
            trace_attr="time_domain_traces",
            title_suffix=f"TD |E| dim={dim}",
        )

        points_rows: List[Dict[str, Any]] = []
        theta_vs_i = [
            {
                "pump_intensity_w_cm2": float(res["pump_intensity_w_cm2"]),
                "final_relative_deg": float(res["final_deg"]),
                "summary_path": str(res["summary_path"]),
            }
            for res in dim_results
        ]

        for idx, res in enumerate(dim_results):
            summary = _load_summary_cached(res)
            points_rows.append(
                {
                    "pump_intensity_w_cm2": float(res["pump_intensity_w_cm2"]),
                    # Primary metric: forward-isolated, coherent (= optimizer objective).
                    "final_relative_deg": float(res["final_deg"]),
                    "abs_final_relative_deg": float(abs(float(res["final_deg"]))),
                    # Companion readings (forward-isolated incoherent; raw total-field).
                    "forward_incoherent_final_relative_deg": float(
                        _nested_get(
                            summary,
                            ["probe_stokes_dft", "tail_weighted", "theta_relative_deg"],
                            np.nan,
                        )
                    ),
                    "total_field_final_relative_deg": float(
                        _nested_get(
                            summary,
                            ["probe_stokes_total", "tail_weighted", "theta_relative_deg"],
                            np.nan,
                        )
                    ),
                    "total_field_dolp_tail": float(
                        _nested_get(
                            summary,
                            ["probe_stokes_total", "tail_weighted", "dolp"],
                            np.nan,
                        )
                    ),
                    "min_relative_deg": float(res["min_deg"]),
                    "max_relative_deg": float(res["max_deg"]),
                    "span_relative_deg": float(float(res["max_deg"]) - float(res["min_deg"])),
                    "variance_relative_deg2": float(var_rot[idx]),
                    "pump1_frequency_inv_um": float(tuned_p1_freq[idx]),
                    "pump2_frequency_inv_um": float(tuned_p2_freq[idx]),
                    "pump1_cold_frequency_inv_um": float(cold_p1_freq[idx]),
                    "pump2_cold_frequency_inv_um": float(cold_p2_freq[idx]),
                    "summary_final_relative_unwrapped_deg": float(
                        _nested_get(summary, ["probe_rotation_deg", "final_relative_unwrapped_deg"], np.nan)
                    ),
                    "summary_mean_relative_deg": float(
                        _nested_get(summary, ["probe_rotation_deg", "mean_relative_deg"], np.nan)
                    ),
                    "summary_td_final_relative_deg": float(
                        _nested_get(
                            summary,
                            ["probe_rotation_deg", "time_domain_reference", "final_relative_deg"],
                            np.nan,
                        )
                    ),
                    "summary_td_final_relative_unwrapped_deg": float(
                        _nested_get(
                            summary,
                            ["probe_rotation_deg", "time_domain_reference", "final_relative_unwrapped_deg"],
                            np.nan,
                        )
                    ),
                    "pump_ratio_tail_weighted": float(
                        _nested_get(
                            summary,
                            ["pump_monitor_metrics", "rms_integrated", "ratio_p2_over_p1", "tail_weighted"],
                            np.nan,
                        )
                    ),
                    "pump_ratio_final": float(
                        _nested_get(
                            summary,
                            ["pump_monitor_metrics", "rms_integrated", "ratio_p2_over_p1", "final"],
                            np.nan,
                        )
                    ),
                    "pump1_purity_tail": float(
                        _nested_get(
                            summary,
                            [
                                "pump_monitor_metrics",
                                "rms_integrated",
                                "dominant_purity",
                                "pump1_tail_weighted",
                            ],
                            np.nan,
                        )
                    ),
                    "pump2_purity_tail": float(
                        _nested_get(
                            summary,
                            [
                                "pump_monitor_metrics",
                                "rms_integrated",
                                "dominant_purity",
                                "pump2_tail_weighted",
                            ],
                            np.nan,
                        )
                    ),
                    "summary_path": str(res["summary_path"]),
                }
            )

        points_csv = dim_root / "rotation_vs_intensity_points.csv"
        _write_points_csv(points_csv, points_rows)

        finite_abs = np.asarray(abs_final, dtype=float)
        finite_abs_mask = np.isfinite(finite_abs)
        if np.any(finite_abs_mask):
            valid_indices = np.where(finite_abs_mask)[0]
            best_idx = int(valid_indices[int(np.argmax(finite_abs[valid_indices]))])
            best_point = {
                "pump_intensity_w_cm2": float(sweep_intensities[best_idx]),
                "final_relative_deg": float(final_rot[best_idx]),
                "abs_final_relative_deg": float(abs_final[best_idx]),
                "min_relative_deg": float(min_rot[best_idx]),
                "max_relative_deg": float(max_rot[best_idx]),
                "summary_path": str(dim_results[best_idx]["summary_path"]),
            }
            max_abs_final = float(np.max(finite_abs[finite_abs_mask]))
            mean_abs_final = float(np.mean(finite_abs[finite_abs_mask]))
        else:
            best_point = {
                "pump_intensity_w_cm2": float("nan"),
                "final_relative_deg": float("nan"),
                "abs_final_relative_deg": float("nan"),
                "min_relative_deg": float("nan"),
                "max_relative_deg": float("nan"),
                "summary_path": "",
            }
            max_abs_final = float("nan")
            mean_abs_final = float("nan")

        first_summary = _load_summary_cached(dim_results[0])
        dimension_reports[str(dim)] = {
            "intensities_w_cm2": sweep_intensities,
            "rotation_final_deg": final_rot,
            "rotation_min_deg": min_rot,
            "rotation_max_deg": max_rot,
            "rotation_variance_deg2": var_rot,
            "rotation_abs_final_deg": abs_final,
            "fit_models": {
                "linear_vs_log_intensity": fit_linear,
                "power_law_abs_rotation": fit_power,
            },
            "best_point": best_point,
            "aggregate_metrics": {
                "max_abs_final_relative_deg": max_abs_final,
                "mean_abs_final_relative_deg": mean_abs_final,
                "mean_relative_deg": float(np.nanmean(np.asarray(final_rot, dtype=float))),
                "std_relative_deg": float(np.nanstd(np.asarray(final_rot, dtype=float))),
            },
            "plot_paths": {
                "rotation_vs_intensity": str(rotation_plot),
                "dft_traces": str(dft_plot),
                "time_domain_traces": str(td_plot),
            },
            "points_csv_path": str(points_csv),
            "run_summary_paths": [str(res["summary_path"]) for res in dim_results],
            "theta_deg_rel_I": theta_vs_i,
            "wavelengths_um": first_summary.get("wavelengths_um", {}),
            "frequencies_inv_um": first_summary.get("frequencies_inv_um", {}),
            "run_params": first_summary.get("run_params", {}),
            "report_json_path": str(report_path),
        }

        dim_md = _write_dim_markdown_summary(
            dim_root=dim_root,
            dim=dim,
            dim_report=dimension_reports[str(dim)],
            global_config=global_config,
        )
        dimension_reports[str(dim)]["markdown_summary_path"] = str(dim_md)

    multi_dim_plot = None
    if len(dimension_reports) > 1:
        multi_dim_plot = output_root / "rotation_vs_intensity_by_dim.png"
        _plot_rotation_multi_dim(results_by_dim, multi_dim_plot)

    sweep_report: Dict[str, Any] = {
        "dims": [int(d) for d in sorted(results_by_dim.keys())],
        "mode": str(args.mode),
        "materials": str(args.materials),
        "high_index_material": str(args.high_index_material),
        "high_index_n": (float(args.nH) if args.nH is not None else None),
        "high_index_k": (float(args.kH) if args.kH is not None else None),
        "high_index_n2_m2_per_w": (
            float(args.high_index_n2) if args.high_index_n2 is not None else None
        ),
        "kappa_ref_lambda_um": float(args.kappa_ref_lambda),
        "geometry_file": str(args.geometry_file),
        "cavity_modes_file": str(args.cavity_modes_file),
        "range_scale": str(args.range_scale),
        "workers_requested": int(args.workers),
        "workers_effective": int(workers),
        "intensities_requested_w_cm2": [float(v) for v in intensities],
        "intensity_min_w_cm2": float(min(intensities)),
        "intensity_max_w_cm2": float(max(intensities)),
        "num_points": int(len(intensities)),
        "dimension_reports": dimension_reports,
        "plot_paths": {
            "rotation_vs_intensity_by_dim": str(multi_dim_plot) if multi_dim_plot is not None else None,
        },
    }

    # Backward-compatible single-dim fields.
    if len(dimension_reports) == 1:
        only_dim = next(iter(dimension_reports.values()))
        sweep_report.update(only_dim)

    global_md = _write_global_markdown_summary(output_root, sweep_report)
    sweep_report["markdown_summary_path"] = str(global_md)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(sweep_report, f, indent=2)
    print(f"Sweep complete. Aggregate report written to {report_path}", flush=True)


if __name__ == "__main__":
    main()
