#!/usr/bin/env python3
"""
Sweep the pump intensity for faraday_meep_fp_circ.py and aggregate diagnostics.

For each specified intensity the script:
  * runs the cavity simulation (reusing faraday_meep_fp_circ.run_simulation),
  * collects the relative Faraday rotation time series,
  * aggregates fixed-frequency DFT and time-domain demodulated traces,
  * writes summary plots showing how rotation and field magnitudes depend on
    pump intensity.

The script also emits a sweep_report.json file containing theta_deg_rel(I) so
downstream notebooks can reuse the data without re-running MEEP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from faraday_meep_fp_circ import SimulationResult, run_simulation


def _parse_intensity_list(raw: str) -> List[float]:
    entries = [token.strip() for token in raw.split(",")]
    values = [float(token) for token in entries if token]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one pump intensity value.")
    return values


def _format_intensity_label(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return f"{mantissa:.2f}e{exponent}"


def _build_args(namespace: argparse.Namespace, pump_intensity: float, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        mode=namespace.mode,
        materials=namespace.materials,
        nH=namespace.nH,
        nL=namespace.nL,
        sin_fit=namespace.sin_fit,
        sio2_fit=namespace.sio2_fit,
        fit_window=tuple(namespace.fit_window),
        fit_poles=namespace.fit_poles,
        pump_intensity=pump_intensity,
        output_dir=str(output_dir),
    )


def _plot_rotation_vs_intensity(intensities: Sequence[float],
                                final_deg: Sequence[float],
                                output_path: Path) -> None:
    intensities = np.asarray(intensities, dtype=float)
    final_deg = np.asarray(final_deg, dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(intensities, final_deg, "o-", label="final θ_rel")
    ax.set_xscale("log")
    ax.set_xlabel("Pump intensity (W/cm²)")
    ax.set_ylabel("θ_rel (deg)")
    ax.set_title("Faraday rotation vs pump intensity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_traces(results: Sequence["SimulationResult"], output_path: Path, trace_attr: str, title_suffix: str) -> None:
    """Render heatmaps of |E|(t, I_pump) for pumps, probe, and sidebands."""
    if not results:
        return

    ordered = sorted(results, key=lambda r: r.pump_intensity_w_cm2)
    traces = [getattr(res, trace_attr) for res in ordered]

    time_arrays = [np.asarray(tr.time, dtype=float) for tr in traces]
    reference_time = time_arrays[0]
    if all(np.array_equal(t, reference_time) for t in time_arrays[1:]):
        master_time = reference_time
    else:
        master_time = np.unique(np.concatenate(time_arrays))
    intensities = np.array([res.pump_intensity_w_cm2 for res in ordered], dtype=float)

    channel_specs = [
        ("Pump1 e-", lambda tr: tr.abs_eminus[:, 0]),
        ("Pump2 e+", lambda tr: tr.abs_eplus[:, 1]),
        ("Probe e+", lambda tr: tr.abs_eplus[:, 2]),
        ("Probe e-", lambda tr: tr.abs_eminus[:, 2]),
        ("Sideband − (e-)", lambda tr: tr.abs_eminus[:, 3]),
        ("Sideband + (e+)", lambda tr: tr.abs_eplus[:, 4]),
    ]

    ncols = 3
    nrows = int(np.ceil(len(channel_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    def _resample(trace, trace_time, extractor):
        values = np.asarray(extractor(trace), dtype=float)
        if len(trace_time) == len(master_time) and np.allclose(trace_time, master_time):
            return values
        return np.interp(master_time, trace_time, values, left=np.nan, right=np.nan)

    T, I = np.meshgrid(master_time, intensities)

    for ax_index, (ax, (label, extractor)) in enumerate(zip(axes, channel_specs)):
        data = np.vstack(
            [_resample(tr, t_arr, extractor) for tr, t_arr in zip(traces, time_arrays)]
        )
        pcm = ax.pcolormesh(T, I, data, shading="auto", cmap="magma")
        ax.set_title(f"{label} ({title_suffix})")
        ax.set_xlabel("time (Meep units)")
        if (ax_index % ncols) == 0:
            ax.set_ylabel("pump intensity (W/cm²)")
        # if np.all(intensities > 0):
        #     ax.set_yscale("log")
        fig.colorbar(pcm, ax=ax, fraction=0.047, pad=0.02, label="|E|")

    for ax in axes[len(channel_specs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep pump intensity for the Faraday rotation simulation.")
    parser.add_argument(
        "--intensities",
        type=str,
        default="1e8,5e8,1e9",
        help="Comma-separated pump intensities in W/cm^2 (default: %(default)s).",
    )
    parser.add_argument("--mode", choices=("quick", "full"), default="quick", help="Simulation preset for each sweep point.")
    parser.add_argument(
        "--materials",
        choices=("library", "constant", "fit"),
        default="library",
        help="Material model forwarded to the underlying simulation.",
    )
    parser.add_argument("--nH", type=float, default=None, help="Override high-index value when --materials constant.")
    parser.add_argument("--nL", type=float, default=None, help="Override low-index value when --materials constant.")
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
    parser.add_argument(
        "--output-root",
        type=str,
        default="pump_intensity_sweep_outputs",
        help="Directory where per-intensity runs and aggregate plots will be stored.",
    )
    args = parser.parse_args()

    intensities = _parse_intensity_list(args.intensities)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Run sweeps
    sweep_results: List[SimulationResult] = []
    for idx, intensity in enumerate(intensities):
        run_dir = output_root / f"I_{idx:02d}_{_format_intensity_label(intensity)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        sim_args = _build_args(args, intensity, run_dir)
        result = run_simulation(sim_args)
        sweep_results.append(result)

    sweep_intensities = [res.pump_intensity_w_cm2 for res in sweep_results]

    # Aggregated plots
    final_rot = [res.probe_rotation.final_deg for res in sweep_results]
    min_rot = [res.probe_rotation.min_deg for res in sweep_results]
    max_rot = [res.probe_rotation.max_deg for res in sweep_results]

    rotation_plot = output_root / "rotation_vs_intensity.png"
    _plot_rotation_vs_intensity(sweep_intensities, final_rot, rotation_plot)

    dft_plot = output_root / "dft_traces_vs_intensity.png"
    _plot_traces(sweep_results, dft_plot, trace_attr="dft_traces", title_suffix="DFT |E|")

    td_plot = output_root / "time_domain_traces_vs_intensity.png"
    _plot_traces(sweep_results, td_plot, trace_attr="time_domain_traces", title_suffix="TD |E|")

    # Aggregate report with theta_deg_rel(I)
    theta_vs_i = [
        {
            "pump_intensity_w_cm2": res.pump_intensity_w_cm2,
            "final_relative_deg": res.probe_rotation.final_deg,
            "summary_path": str(res.summary_path),
        }
        for res in sweep_results
    ]

    sweep_report = {
        "intensities_w_cm2": sweep_intensities,
        "rotation_final_deg": final_rot,
        "rotation_min_deg": min_rot,
        "rotation_max_deg": max_rot,
        "plot_paths": {
            "rotation_vs_intensity": str(rotation_plot),
            "dft_traces": str(dft_plot),
            "time_domain_traces": str(td_plot),
        },
        "run_summary_paths": [str(res.summary_path) for res in sweep_results],
        "theta_deg_rel_I": theta_vs_i,
    }

    report_path = output_root / "pump_intensity_sweep_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(sweep_report, f, indent=2)
    print(f"Sweep complete. Aggregate report written to {report_path}")


if __name__ == "__main__":
    main()
