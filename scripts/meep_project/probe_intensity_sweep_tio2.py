#!/usr/bin/env python3
"""
Sweep probe intensity for TiO2 optimizer outputs and plot DFT final rotation.

This script runs 1D `faraday_meep_fp_circ.py` simulations using geometry and mode
artifacts from optimizer subfolders (e.g., `new`, `mf`) under a pipeline run root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from faraday_meep_fp_circ import run_simulation
from nonlinear_materials import high_index_material_choices


def _configure_numeric_threads() -> None:
    # Keep BLAS/OpenMP single-threaded since sweep already loops over points.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _parse_optimizers(raw: str) -> List[str]:
    allowed = {"new", "mf", "legacy"}
    values = [token.strip().lower() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one optimizer name.")
    out: List[str] = []
    bad: List[str] = []
    for name in values:
        if name not in allowed:
            bad.append(name)
        elif name not in out:
            out.append(name)
    if bad:
        raise argparse.ArgumentTypeError(
            f"Invalid optimizer(s): {', '.join(bad)}. Allowed: {', '.join(sorted(allowed))}."
        )
    return out


def _format_intensity_label(value: float) -> str:
    if value <= 0:
        return "0"
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / (10.0**exponent)
    return f"{mantissa:.2f}e{exponent}"


def _resolve_probe_intensities(
    values: str | None,
    i_min: float,
    i_max: float,
    count: int,
) -> List[float]:
    if values:
        parsed = [float(token.strip()) for token in values.split(",") if token.strip()]
        if not parsed:
            raise argparse.ArgumentTypeError("No valid probe intensities parsed from --probe-intensities.")
        if any(v <= 0 for v in parsed):
            raise argparse.ArgumentTypeError("Probe intensities must be > 0.")
        return parsed

    if i_min <= 0 or i_max <= 0:
        raise argparse.ArgumentTypeError("--probe-intensity-min/max must be > 0.")
    if count < 2:
        raise argparse.ArgumentTypeError("--probe-intensity-count must be >= 2.")
    return [float(v) for v in np.geomspace(float(i_min), float(i_max), int(count))]


def _extract_final_rotation_deg(summary: Dict[str, Any]) -> float:
    probe = summary.get("probe_rotation_deg", {})
    direct = float(probe.get("final_relative_deg", float("nan")))
    if np.isfinite(direct):
        return direct
    coherent = probe.get("coherent_window_estimate", {})
    return float(coherent.get("theta_relative_deg", float("nan")))


def _extract_coherent_snr_db(summary: Dict[str, Any]) -> float:
    probe = summary.get("probe_rotation_deg", {})
    coherent = probe.get("coherent_window_estimate", {})
    return float(coherent.get("snr_db", float("nan")))


def _extract_coherent_theta_std_deg(summary: Dict[str, Any]) -> float:
    probe = summary.get("probe_rotation_deg", {})
    coherent = probe.get("coherent_window_estimate", {})
    return float(coherent.get("theta_relative_std_deg", float("nan")))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_points_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "optimizer",
        "probe_intensity_w_cm2",
        "pump_intensity_w_cm2",
        "final_rotation_deg_dft",
        "abs_final_rotation_deg_dft",
        "rotation_theta_std_deg",
        "rotation_snr_db",
        "output_dir",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_sim_args(
    *,
    geometry_file: Path,
    cavity_modes_file: Path,
    output_dir: Path,
    pump_intensity: float,
    probe_intensity: float,
    resolution: int,
    high_index_material: str,
    decay_threshold: float,
    calibrate_sources: bool,
    calibration_decay_threshold: float,
    probe_rotation_tail_points: int,
    probe_rotation_window_fs: float | None,
    disable_strength_validity: bool,
    ) -> argparse.Namespace:
    return argparse.Namespace(
        dim=1,
        mode="quick",
        materials="constant",
        high_index_material=str(high_index_material),
        nH=None,
        kH=None,
        nL=None,
        sin_fit=None,
        sio2_fit=None,
        fit_window=(600, 2000),
        fit_poles=2,
        kappa_ref_lambda=1.55,
        high_index_n2=None,
        pump_intensity=float(pump_intensity),
        probe_intensity=float(probe_intensity),
        pump1_frequency=None,
        pump2_frequency=None,
        geometry_file=str(geometry_file),
        cavity_modes_file=str(cavity_modes_file),
        output_dir=str(output_dir),
        until_time=None,
        decay_threshold=float(decay_threshold),
        resolution=int(resolution),
        calibrate_sources=bool(calibrate_sources),
        calibration_decay_threshold=float(calibration_decay_threshold),
        probe_rotation_tail_points=int(probe_rotation_tail_points),
        probe_rotation_window_fs=(
            float(probe_rotation_window_fs)
            if probe_rotation_window_fs is not None
            else None
        ),
        disable_strength_validity=bool(disable_strength_validity),
    )


def _run_one_point(task: Dict[str, Any]) -> Dict[str, Any]:
    _configure_numeric_threads()

    run_dir = Path(str(task["run_dir"])).resolve()
    summary_path = run_dir / "faraday_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.exists() and not bool(task["force_rerun"]):
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        script_dir = Path(__file__).resolve().parent
        faraday_script = script_dir / "faraday_meep_fp_circ.py"
        cmd = [
            str(sys.executable),
            str(faraday_script),
            "--dim",
            "1",
            "--mode",
            "quick",
            "--materials",
            "constant",
            "--high-index-material",
            str(task["high_index_material"]),
            "--geometry-file",
            str(task["geometry_file"]),
            "--cavity-modes-file",
            str(task["cavity_modes_file"]),
            "--output-dir",
            str(run_dir),
            "--pump-intensity",
            str(float(task["pump_intensity"])),
            "--probe-intensity",
            str(float(task["probe_intensity"])),
            "--resolution",
            str(int(task["resolution"])),
            "--decay-threshold",
            str(float(task["decay_threshold"])),
            "--calibration-decay-threshold",
            str(float(task["calibration_decay_threshold"])),
            "--probe-rotation-tail-points",
            str(int(task["probe_rotation_tail_points"])),
        ]
        if bool(task["calibrate_sources"]):
            cmd.append("--calibrate-sources")
        if bool(task["disable_strength_validity"]):
            cmd.append("--disable-strength-validity")
        if task["probe_rotation_window_fs"] is not None:
            cmd.extend(
                [
                    "--probe-rotation-window-fs",
                    str(float(task["probe_rotation_window_fs"])),
                ]
            )

        worker_log = run_dir / "worker.log"
        with worker_log.open("w", encoding="utf-8") as f_log:
            subprocess.run(
                cmd,
                cwd=str(script_dir),
                stdout=f_log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

    final_deg = _extract_final_rotation_deg(summary)
    theta_std_deg = _extract_coherent_theta_std_deg(summary)
    snr_db = _extract_coherent_snr_db(summary)
    return {
        "optimizer": str(task["optimizer"]),
        "probe_intensity_w_cm2": float(task["probe_intensity"]),
        "pump_intensity_w_cm2": float(task["pump_intensity"]),
        "final_rotation_deg_dft": float(final_deg),
        "abs_final_rotation_deg_dft": float(abs(final_deg))
        if np.isfinite(final_deg)
        else float("nan"),
        "rotation_theta_std_deg": float(theta_std_deg),
        "rotation_snr_db": float(snr_db),
        "output_dir": str(run_dir),
        "summary_path": str(summary_path),
    }


def _plot_rotation_vs_probe_intensity(
    rows: Sequence[Dict[str, Any]],
    optimizers: Sequence[str],
    output_path: Path,
    pump_intensity: float,
    resolution: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for opt in optimizers:
        subset = [r for r in rows if str(r["optimizer"]) == str(opt)]
        subset_sorted = sorted(subset, key=lambda r: float(r["probe_intensity_w_cm2"]))
        if not subset_sorted:
            continue
        x = np.array([float(r["probe_intensity_w_cm2"]) for r in subset_sorted], dtype=float)
        y = np.array([float(r["final_rotation_deg_dft"]) for r in subset_sorted], dtype=float)
        yerr = np.array(
            [float(r.get("rotation_theta_std_deg", float("nan"))) for r in subset_sorted],
            dtype=float,
        )
        yerr_plot = np.where(np.isfinite(yerr), yerr, 0.0)
        ax.errorbar(
            x,
            y,
            yerr=yerr_plot,
            fmt="o-",
            lw=2.0,
            ms=5.5,
            capsize=3.5,
            elinewidth=1.2,
            label=f"{str(opt)} (mean ± std)",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Probe intensity (W/cm^2)")
    ax.set_ylabel("Final probe rotation (deg, DFT)")
    ax.set_title(
        "1D TiO2: final DFT probe rotation vs probe intensity (mean ± std)\n"
        f"(pump={float(pump_intensity):.2e} W/cm^2, resolution={int(resolution)})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


@dataclass
class SweepResult:
    rows: List[Dict[str, Any]]
    report_path: Path
    plot_path: Path
    csv_path: Path


def run_probe_sweep(args: argparse.Namespace) -> SweepResult:
    _configure_numeric_threads()

    pipeline_dir = Path(args.pipeline_dir).resolve()
    optimizers = _parse_optimizers(args.optimizers)
    probe_intensities = _resolve_probe_intensities(
        values=args.probe_intensities,
        i_min=float(args.probe_intensity_min),
        i_max=float(args.probe_intensity_max),
        count=int(args.probe_intensity_count),
    )
    probe_intensities = sorted(float(v) for v in probe_intensities)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_root is None:
        output_root = Path(f"probe_intensity_sweep_tio2_{ts}").resolve()
    else:
        output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    tasks: List[Dict[str, Any]] = []
    for optimizer in optimizers:
        opt_root = pipeline_dir / "optimizers" / optimizer
        geom = opt_root / "optimized_geometry.json"
        modes = opt_root / "cavity_modes.json"
        if not geom.exists() or not modes.exists():
            raise FileNotFoundError(
                f"Missing optimizer artifacts for '{optimizer}'. "
                f"Expected: {geom} and {modes}."
            )
        opt_out = output_root / optimizer
        opt_out.mkdir(parents=True, exist_ok=True)

        for idx, probe_i in enumerate(probe_intensities):
            run_dir = opt_out / f"I_{idx:02d}_{_format_intensity_label(float(probe_i))}"
            run_dir.mkdir(parents=True, exist_ok=True)
            tasks.append(
                {
                    "optimizer": str(optimizer),
                    "probe_intensity": float(probe_i),
                    "pump_intensity": float(args.pump_intensity),
                    "run_dir": str(run_dir),
                    "geometry_file": str(geom),
                    "cavity_modes_file": str(modes),
                    "resolution": int(args.resolution),
                    "high_index_material": str(args.high_index_material),
                    "decay_threshold": float(args.decay_threshold),
                    "calibrate_sources": bool(args.calibrate_sources),
                    "calibration_decay_threshold": float(args.calibration_decay_threshold),
                    "probe_rotation_tail_points": int(args.probe_rotation_tail_points),
                    "probe_rotation_window_fs": (
                        float(args.probe_rotation_window_fs)
                        if args.probe_rotation_window_fs is not None
                        else None
                    ),
                    "disable_strength_validity": bool(args.disable_strength_validity),
                    "force_rerun": bool(args.force_rerun),
                }
            )

    all_rows: List[Dict[str, Any]] = []
    workers = max(1, int(getattr(args, "workers", 1)))
    if workers <= 1:
        for task in tasks:
            all_rows.append(_run_one_point(task))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_run_one_point, task) for task in tasks]
            for fut in as_completed(futures):
                all_rows.append(dict(fut.result()))

    all_rows.sort(key=lambda r: (str(r["optimizer"]), float(r["probe_intensity_w_cm2"])))
    csv_path = output_root / "probe_intensity_sweep_points.csv"
    _write_points_csv(csv_path, all_rows)

    plot_path = output_root / "rotation_vs_probe_intensity_dft.png"
    _plot_rotation_vs_probe_intensity(
        rows=all_rows,
        optimizers=optimizers,
        output_path=plot_path,
        pump_intensity=float(args.pump_intensity),
        resolution=int(args.resolution),
    )

    report_path = output_root / "probe_intensity_sweep_report.json"
    _write_json(
        report_path,
        {
            "pipeline_dir": str(pipeline_dir),
            "optimizers": list(optimizers),
            "probe_intensities_w_cm2": [float(v) for v in probe_intensities],
            "pump_intensity_w_cm2": float(args.pump_intensity),
            "dimension": 1,
            "mode": "quick",
            "materials": "constant",
            "high_index_material": str(args.high_index_material),
            "resolution": int(args.resolution),
            "decay_threshold": float(args.decay_threshold),
            "probe_rotation_tail_points": int(args.probe_rotation_tail_points),
            "probe_rotation_window_fs": (
                float(args.probe_rotation_window_fs)
                if args.probe_rotation_window_fs is not None
                else None
            ),
            "workers": int(workers),
            "disable_strength_validity": bool(args.disable_strength_validity),
            "calibrate_sources": bool(args.calibrate_sources),
            "plot_rotation_vs_probe_intensity": str(plot_path),
            "points_csv": str(csv_path),
            "points": all_rows,
        },
    )

    return SweepResult(
        rows=all_rows,
        report_path=report_path,
        plot_path=plot_path,
        csv_path=csv_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 1D TiO2 probe-intensity sweeps using optimizer artifacts and "
            "plot final DFT probe rotation vs probe intensity."
        )
    )
    parser.add_argument(
        "--pipeline-dir",
        type=str,
        default="pipeline_tio2_20260302_162215",
        help="Pipeline run directory containing optimizers/<name> artifacts.",
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default="new,mf",
        help="Comma-separated optimizer names (subset of: new,mf,legacy).",
    )
    parser.add_argument(
        "--probe-intensities",
        type=str,
        default=None,
        help="Optional explicit comma-separated probe intensities in W/cm^2.",
    )
    parser.add_argument(
        "--probe-intensity-min",
        type=float,
        default=1e3,
        help="Probe sweep minimum intensity in W/cm^2 (used if --probe-intensities is absent).",
    )
    parser.add_argument(
        "--probe-intensity-max",
        type=float,
        default=1e8,
        help="Probe sweep maximum intensity in W/cm^2 (used if --probe-intensities is absent).",
    )
    parser.add_argument(
        "--probe-intensity-count",
        type=int,
        default=6,
        help="Number of log-spaced points between min and max (inclusive).",
    )
    parser.add_argument(
        "--pump-intensity",
        type=float,
        default=1e12,
        help="Fixed pump intensity in W/cm^2.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=40,
        help="Simulation resolution in px/um.",
    )
    parser.add_argument(
        "--high-index-material",
        choices=high_index_material_choices(),
        default="tio2",
        help="High-index cavity material preset forwarded to simulation.",
    )
    parser.add_argument(
        "--decay-threshold",
        type=float,
        default=1e-3,
        help="Field-decay threshold forwarded to simulation.",
    )
    parser.add_argument(
        "--probe-rotation-tail-points",
        type=int,
        default=64,
        help="Number of final valid points used to compute reported final rotation.",
    )
    parser.add_argument(
        "--probe-rotation-window-fs",
        type=float,
        default=None,
        help=(
            "Optional final averaging-window width in fs. "
            "If set, it overrides --probe-rotation-tail-points."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for parallel sweep execution.",
    )
    parser.add_argument(
        "--disable-strength-validity",
        action="store_true",
        help="Use finite-only validity mask (disable S0-threshold validity criterion).",
    )
    parser.add_argument(
        "--calibrate-sources",
        action="store_true",
        help="Enable source intensity calibration before each run.",
    )
    parser.add_argument(
        "--calibration-decay-threshold",
        type=float,
        default=1e-7,
        help="Decay threshold for source calibration helper runs.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Output directory root (default: auto timestamped).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore existing per-point summaries and rerun all points.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_probe_sweep(args)
    print(f"[done] report: {result.report_path}")
    print(f"[done] csv: {result.csv_path}")
    print(f"[done] plot: {result.plot_path}")


if __name__ == "__main__":
    main()
