#!/usr/bin/env python3
"""
1D/3D pump–probe simulation of Faraday rotation in an optimized DBR cavity.

This script keeps the modular structure of the original faraday_meep_fp_circ.py
but follows the computation sequence (DFT monitoring, demodulated envelopes,
plotting) implemented in faraday_rotation_tutorial.py. Geometry and modal
frequencies continue to be imported from JSON artifacts produced by the
optimization workflow.

Outputs (saved to --output-dir):
    - Pumps / probe / sidebands traces (DFT and demodulated time-domain).
    - Probe-band spectrogram and polarization rotation plot.
    - Optional X–Z field snapshots.
    - JSON report summarizing parameters plus probe-rotation relative to the
      input polarization.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from geometry_io import material_factory, read_json as load_geometry_json
from mode_targeting import get_cavity_materials, material_index_at_wavelength

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})


# --------------------------------------------------------------------------- #
# Physical constants (SI units)
# --------------------------------------------------------------------------- #
EPS0 = 8.854187817e-12
C0 = 299792458.0
UM = 1.0  # μm as Meep length unit
SCALE_E = 1.0 / (1e-6 * EPS0 * C0)  # converts Meep E-field (μm units) → SI
FS_PER_MEEP = (1e-6 / C0) * 1e15  # 1 meep-time unit (um/c) in femtoseconds
INIT_PROBE_POLARIZATION_DEG = 45.0


# --------------------------------------------------------------------------- #
# Helpers for field/intensity conversions
# --------------------------------------------------------------------------- #
def intensity_to_meep_amplitude(intensity_w_cm2: float, n_lin: float) -> float:
    """Convert plane-wave intensity (W/cm²) to Meep electric-field amplitude."""
    intensity_si = intensity_w_cm2 * 1e4  # cm² → m²
    e_si = np.sqrt(2.0 * intensity_si / (n_lin * EPS0 * C0))
    return float(e_si / SCALE_E)


def meep_field_to_intensity(field: np.ndarray, n_lin: float) -> np.ndarray:
    """Return intensity (W/cm²) from complex field envelope (Meep units)."""
    e_si = np.abs(field) * SCALE_E
    intensity_si = 0.5 * n_lin * EPS0 * C0 * e_si**2
    return intensity_si / 1e4


def df_from_bandwidth(lam_um: float, dlam_nm: float) -> float:
    """Gaussian fwidth parameter from bandwidth (nm) centered at lam_um."""
    return (dlam_nm * 1e-3) / (lam_um * lam_um)


def df_from_pulse_duration(pulse_duration: float) -> float:
    """Gaussian fwidth parameter from pulse duration (fs)."""
    width_fs = (pulse_duration / (2.0*np.log(2)))
    width_meep = width_fs * (C0 / 1e-6 * 1e-15)  # [fs * um / fs]
    return 1.0 / width_meep


def unwrap_linear_polarization_deg(theta_deg: np.ndarray) -> np.ndarray:
    """Unwrap linear-polarization angle using 2*theta periodicity."""
    arr = np.asarray(theta_deg, dtype=float)
    if arr.size == 0:
        return arr.copy()
    return np.degrees(0.5 * np.unwrap(np.radians(2.0 * arr)))


def wrap_linear_polarization_deg(theta_deg: np.ndarray | float) -> np.ndarray:
    """Map linear-polarization angle to principal branch [-90, 90)."""
    arr = np.asarray(theta_deg, dtype=float)
    return (arr + 90.0) % 180.0 - 90.0


def weighted_linear_mean_deg(
    theta_deg: np.ndarray, weights: np.ndarray | None = None
) -> float:
    """Weighted circular mean for linear polarization (period 180 deg)."""
    th = np.asarray(theta_deg, dtype=float)
    if th.size == 0:
        return float("nan")
    if weights is None:
        w = np.ones_like(th)
    else:
        w = np.asarray(weights, dtype=float)
    valid = np.isfinite(th) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        vv = wrap_linear_polarization_deg(th[np.isfinite(th)])
        return float(np.mean(vv)) if vv.size else float("nan")
    phi = np.radians(2.0 * th[valid])
    c = float(np.average(np.cos(phi), weights=w[valid]))
    s = float(np.average(np.sin(phi), weights=w[valid]))
    return float(np.degrees(0.5 * np.arctan2(s, c)))


# --------------------------------------------------------------------------- #
# Dataclasses for run configuration
# --------------------------------------------------------------------------- #
@dataclass
class RunParams:
    name: str
    resolution: int
    span_xy: float
    dpml_xy: float
    dpml_z: float
    src_buffer: float
    runtime_factor: float
    pulse_duration_fs: float
    pump_band_nm: float
    probe_band_nm: float
    pump_intensity_w_cm2: float
    probe_intensity_w_cm2: float
    nonlinear_scale: float
    sample_dt: float
    lp_tau: float
    capture_fields: bool
    pump_cutoff: float


def quick_params() -> RunParams:
    return RunParams(
        name="quick",
        resolution=30,
        span_xy=0.8,
        dpml_xy=1.0,
        dpml_z=1.0,
        src_buffer=0.25,
        runtime_factor=0.35,
        pulse_duration_fs=100.0,
        pump_band_nm=10.0,
        probe_band_nm=30.0,
        pump_intensity_w_cm2=1.0e12,
        probe_intensity_w_cm2=5.0e7,
        nonlinear_scale=1.0,
        sample_dt=0.05,
        lp_tau=0.8,
        capture_fields=True,
        pump_cutoff=4.0,
    )


def full_params() -> RunParams:
    return RunParams(
        name="full",
        resolution=96,
        span_xy=3.0,
        dpml_xy=1.0,
        dpml_z=1.0,
        src_buffer=0.5,
        runtime_factor=6.0,
        pulse_duration_fs=100.0,
        pump_band_nm=30.0,
        probe_band_nm=10.0,
        pump_intensity_w_cm2=1.0e12,
        probe_intensity_w_cm2=5.0e7,
        nonlinear_scale=1.0,
        sample_dt=0.05,
        lp_tau=0.8,
        capture_fields=True,
        pump_cutoff=4.0,
    )


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class FieldTrace:
    time: np.ndarray
    freqs: np.ndarray
    abs_eplus: np.ndarray
    abs_eminus: np.ndarray


@dataclass
class ProbeBandTrace:
    time: np.ndarray
    freqs: np.ndarray
    abs_field: np.ndarray


@dataclass
class ProbeRotationTrace:
    time: np.ndarray
    theta_deg_rel: np.ndarray
    final_deg: float
    min_deg: float
    max_deg: float
    time_domain_time: np.ndarray | None = None
    time_domain_theta_deg_rel: np.ndarray | None = None


@dataclass
class SimulationResult:
    run_mode: str
    pump_intensity_w_cm2: float
    probe_rotation: ProbeRotationTrace
    dft_traces: FieldTrace
    time_domain_traces: FieldTrace
    probe_band_trace: ProbeBandTrace
    plot_paths: Dict[str, str]
    output_dir: str
    summary: Dict[str, Any]
    summary_path: Path
    metadata: Dict[str, Any]

    def summary_dict(self) -> Dict[str, Any]:
        return self.summary


# --------------------------------------------------------------------------- #
# Geometry loaders
# --------------------------------------------------------------------------- #
def load_cavity_modes(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run fp_cavity_modes_spectrum.py first to generate it."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_geometry_from_spec(
    spec: Dict,
    mats: Dict[str, mp.Medium],
    core_span_xy: float,
    dpml_z: float,
    dimension: int = 3,
    margin_z: float = 0.4,
) -> Tuple[List[mp.Block], float, float]:
    """Create a dimension-aware geometry list from the optimized geometry JSON."""
    pad_air = float(spec["pads"]["air_um"])
    pad_sub = float(spec["pads"]["substrate_um"])
    dpml_geom = float(spec["pads"]["pml_um"])
    dpml_z = max(dpml_z, dpml_geom)

    left_layers = spec["mirrors"]["left"]
    right_layers = spec["mirrors"]["right"]
    cavity_thk = float(spec["cavity"]["L_um"])
    cavity_mat = mats[spec["cavity"]["mat"]]
    spacer_left = float(spec.get("spacers", {}).get("left_um", 0.0))
    spacer_right = float(spec.get("spacers", {}).get("right_um", 0.0))

    def layer_sum(layers: Sequence[Dict[str, float]]) -> float:
        return sum(layer["thk_um"] for layer in layers)

    stack_len = (
        pad_air
        + layer_sum(left_layers)
        + spacer_left
        + cavity_thk
        + spacer_right
        + layer_sum(right_layers)
        + pad_sub
    )

    cell_z = stack_len + 2 * dpml_z + margin_z
    geometry: List[mp.Block] = []
    is_1d = int(dimension) == 1
    xy_size = mp.inf if is_1d else core_span_xy

    def add_block(z_start: float, thickness: float, mat: mp.Medium) -> float:
        center_z = z_start + 0.5 * thickness
        geometry.append(
            mp.Block(
                center=mp.Vector3(0, 0, center_z),
                size=mp.Vector3(xy_size, xy_size, thickness),
                material=mat,
            )
        )
        return z_start + thickness

    z = -0.5 * cell_z + dpml_z
    z += pad_air

    for layer in left_layers:
        z = add_block(z, layer["thk_um"], mats[layer["mat"]])

    if spacer_left > 0:
        z = add_block(z, spacer_left, mats["SiO2"])

    cavity_start = z
    z = add_block(z, cavity_thk, cavity_mat)
    cavity_center = cavity_start + 0.5 * cavity_thk

    if spacer_right > 0:
        z = add_block(z, spacer_right, mats["SiO2"])

    for layer in right_layers:
        z = add_block(z, layer["thk_um"], mats[layer["mat"]])

    add_block(z, pad_sub, mats["SiO2"])

    return geometry, cell_z, cavity_center


def load_geometry(
    path: Path,
    core_span_xy: float,
    dpml_z: float,
    dimension: int,
) -> Tuple[List[mp.Block], float, float, Dict[str, mp.Medium], Dict]:
    spec = load_geometry_json(str(path))
    mats = {
        name: material_factory(name, entry, mp)
        for name, entry in spec["materials"].items()
    }
    geometry, cell_z, cavity_center = build_geometry_from_spec(
        spec, mats, core_span_xy, dpml_z, dimension=dimension
    )
    return geometry, cell_z, cavity_center, mats, spec


# --------------------------------------------------------------------------- #
# Source builders
# --------------------------------------------------------------------------- #
def circular_sources(
    frequency: float,
    fwidth: float,
    cutoff: float,
    amplitude: float,
    handedness: str,
    src_center: mp.Vector3,
    src_span: float,
    include_ey: bool = True,
) -> List[mp.Source]:
    """Return (Ex, Ey) sources for circular polarization."""
    phase = 1.0j if handedness == "plus" else -1.0j
    amp = amplitude / np.sqrt(2.0)
    base = mp.GaussianSource(frequency=frequency, fwidth=fwidth, cutoff=cutoff)
    size = mp.Vector3() if src_span <= 0 else mp.Vector3(src_span, src_span, 0)
    if not include_ey:
        return [mp.Source(src=base, component=mp.Ex, center=src_center, size=size, amplitude=amp)]
    return [
        mp.Source(src=base, component=mp.Ex, center=src_center, size=size, amplitude=amp),
        mp.Source(
            src=base,
            component=mp.Ey,
            center=src_center,
            size=size,
            amplitude=amp * phase,
        ),
    ]


def linear_sources_45deg(
    frequency: float,
    fwidth: float,
    cutoff: float,
    amplitude: float,
    src_center: mp.Vector3,
    src_span: float,
    include_ey: bool = True,
) -> List[mp.Source]:
    amp = amplitude / np.sqrt(2.0)
    base = mp.GaussianSource(frequency=frequency, fwidth=fwidth, cutoff=cutoff)
    size = mp.Vector3() if src_span <= 0 else mp.Vector3(src_span, src_span, 0)
    if not include_ey:
        return [mp.Source(src=base, component=mp.Ex, center=src_center, size=size, amplitude=amp)]
    return [
        mp.Source(src=base, component=mp.Ex, center=src_center, size=size, amplitude=amp),
        mp.Source(src=base, component=mp.Ey, center=src_center, size=size, amplitude=amp),
    ]


def calibrate_source_to_intensity(
    *,
    label: str,
    target_intensity_w_cm2: float,
    frequency: float,
    monitor_volume: mp.Volume,
    monitor_area_um2: float,
    simulation_dimensions: int,
    cell: mp.Vector3,
    boundary_layers: Sequence[mp.PML],
    resolution: int,
    build_sources,
    decay_threshold: float = 1e-7,
) -> Tuple[float, Dict[str, float]]:
    """
    Calibrate source amplitude using an empty-cell run.

    The calibration uses DFT fields at the monitor plane to estimate launched
    intensity in air, and reports flux density as a secondary diagnostic.
    """
    sim = mp.Simulation(
        cell_size=cell,
        geometry=[],
        sources=build_sources(1.0),
        boundary_layers=list(boundary_layers),
        resolution=int(resolution),
        dimensions=int(simulation_dimensions),
        default_material=mp.air,
        force_complex_fields=True,
    )
    dft = sim.add_dft_fields([mp.Ex, mp.Ey], [float(frequency)], where=monitor_volume)
    flux = sim.add_flux(
        float(frequency),
        0.0,
        1,
        mp.FluxRegion(center=monitor_volume.center, size=monitor_volume.size),
    )
    probe_point = mp.Vector3(0, 0, monitor_volume.center.z)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            15, mp.Ex, probe_point, float(decay_threshold)
        )
    )

    ex = np.asarray(sim.get_dft_array(dft, mp.Ex, 0))
    ey = np.asarray(sim.get_dft_array(dft, mp.Ey, 0))
    s0 = float(np.mean(np.abs(ex) ** 2 + np.abs(ey) ** 2))
    e_rms = float(np.sqrt(max(s0, 0.0)))
    intensity_per_amp1 = float(meep_field_to_intensity(np.array([e_rms]), n_lin=1.0)[0])
    amp = float(np.sqrt(max(target_intensity_w_cm2, 0.0) / max(intensity_per_amp1, 1e-30)))

    flux_val = float(np.asarray(mp.get_fluxes(flux), dtype=float)[0])
    flux_density = flux_val / max(float(monitor_area_um2), 1e-30)
    diag = {
        "target_intensity_w_cm2": float(target_intensity_w_cm2),
        "intensity_per_amp1_w_cm2": float(intensity_per_amp1),
        "amplitude_scale": float(amp),
        "flux_per_amp1_meep": float(flux_val),
        "flux_density_per_amp1_meep_per_um2": float(flux_density),
        "frequency_inv_um": float(frequency),
    }
    print(
        "[source-cal]",
        f"{label}: amp={amp:.6g}",
        f"I_target={target_intensity_w_cm2:.3e} W/cm^2",
        f"I_per_amp1={intensity_per_amp1:.3e} W/cm^2",
    )
    return amp, diag


# --------------------------------------------------------------------------- #
# Plot utilities
# --------------------------------------------------------------------------- #
def save_figure(fig: plt.Figure, filename: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main simulation driver
# --------------------------------------------------------------------------- #
def run_simulation(args: argparse.Namespace | None = None) -> SimulationResult:
    if args is None:
        args = parse_args()

    run = quick_params() if args.mode == "quick" else full_params()
    if args.resolution is not None:
        run.resolution = int(args.resolution)
    dimension = int(getattr(args, "dim", 1))
    if dimension not in (1, 3):
        raise ValueError("--dim must be either 1 or 3.")
    is_quasi_1d = dimension == 1
    # In quasi-1D mode we still use vector fields (Ex/Ey) in a collapsed transverse cell.
    simulation_dimensions = 3
    track_ey = True
    capture_spatial_fields = run.capture_fields and not is_quasi_1d
    output_dir = Path(args.output_dir or f"faraday_{run.name}_outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = load_cavity_modes(Path(args.cavity_modes_file))
    spec_path = Path(args.geometry_file)
    if not spec_path.exists():
        raise FileNotFoundError(f"{spec_path} not found.")

    core_span = run.span_xy
    geometry, cell_z, cavity_center, materials, spec = load_geometry(
        spec_path,
        core_span_xy=core_span,
        dpml_z=run.dpml_z,
        dimension=dimension,
    )

    default_high = float(getattr(materials.get("SiN"), "index", 2.0))
    default_low = float(getattr(materials.get("SiO2"), "index", 1.45))
    mat_sin, mat_sio2 = get_cavity_materials(
        model=args.materials,
        index_high=args.nH if args.nH is not None else default_high,
        index_low=args.nL if args.nL is not None else default_low,
        sin_csv=args.sin_fit,
        sio2_csv=args.sio2_fit,
        lam_min=args.fit_window[0],
        lam_max=args.fit_window[1],
        fit_poles=args.fit_poles,
    )
    materials["SiN"] = mat_sin
    materials["SiO2"] = mat_sio2
    geometry, cell_z, cavity_center = build_geometry_from_spec(
        spec, materials, core_span, run.dpml_z, dimension=dimension
    )

    freq_probe = float(modes["probe"]["frequency"])
    lam_probe = float(modes["probe"]["lambda_um"])
    freq_p1_cold = float(modes["pump1"]["frequency"])
    freq_p2_cold = float(modes["pump2"]["frequency"])
    freq_p1 = float(
        args.pump1_frequency if getattr(args, "pump1_frequency", None) is not None else freq_p1_cold
    )
    freq_p2 = float(
        args.pump2_frequency if getattr(args, "pump2_frequency", None) is not None else freq_p2_cold
    )
    lam_p1 = float(1.0 / freq_p1)
    lam_p2 = float(1.0 / freq_p2)
    delta_omega = abs(freq_p1 - freq_p2)
    freq_sb_plus = freq_probe + delta_omega
    freq_sb_minus = max(freq_probe - delta_omega, 0.0)

    # Nonlinear response for SiN scaled as requested
    n2_sin = 2.0*2.5e-19  # m²/W
    n_linear_probe = material_index_at_wavelength(mat_sin, lam_probe)
    n_linear_p1 = material_index_at_wavelength(mat_sin, lam_p1)
    n_linear_p2 = material_index_at_wavelength(mat_sin, lam_p2)
    kerr_xpm_factor = float(getattr(args, "kerr_xpm_factor", 1.0))
    kerr_intensity_metric = str(getattr(args, "kerr_intensity_metric", "p95")).lower()
    chi3_si = (4.0 / 3.0) * n2_sin * (n_linear_probe**2) * EPS0 * C0
    e_chi3_meep = chi3_si * (SCALE_E**2) * run.nonlinear_scale
    mat_sin.E_chi3_diag = mp.Vector3(e_chi3_meep, e_chi3_meep, e_chi3_meep)

    n_source_medium = 1.0  # sources are injected in air

    if args.pump_intensity is not None:
        run.pump_intensity_w_cm2 = args.pump_intensity
    pump_amp1 = intensity_to_meep_amplitude(run.pump_intensity_w_cm2, n_source_medium)
    pump_amp2 = intensity_to_meep_amplitude(run.pump_intensity_w_cm2, n_source_medium)
    probe_amp = intensity_to_meep_amplitude(run.probe_intensity_w_cm2, n_source_medium)

    # df_probe = df_from_bandwidth(lam_probe, run.probe_band_nm)
    # df_pump1 = df_from_bandwidth(lam_p1, run.pump_band_nm)
    # df_pump2 = df_from_bandwidth(lam_p2, run.pump_band_nm)
    df_probe = df_from_pulse_duration(run.pulse_duration_fs)
    df_pump1 = df_from_pulse_duration(run.pulse_duration_fs)
    df_pump2 = df_from_pulse_duration(run.pulse_duration_fs)

    boundary_layers: List[mp.PML] = [mp.PML(run.dpml_z, direction=mp.Z)]
    if (not is_quasi_1d) and run.dpml_xy > 0:
        boundary_layers.extend(
            [mp.PML(run.dpml_xy, direction=mp.X), mp.PML(run.dpml_xy, direction=mp.Y)]
        )

    if is_quasi_1d:
        cell = mp.Vector3(0, 0, cell_z)
        src_span = 0.0
    else:
        cell = mp.Vector3(
            run.span_xy + 2 * run.dpml_xy, run.span_xy + 2 * run.dpml_xy, cell_z
        )
        src_span = run.span_xy + 2 * run.dpml_xy
    src_z = -0.5 * cell_z + run.dpml_z + run.src_buffer
    src_center = mp.Vector3(0, 0, src_z)

    pulse_duration_meep = run.pulse_duration_fs / (1e9 / C0)
    stop_time = run.runtime_factor * pulse_duration_meep
    snapshot_time = 0.6 * stop_time

    # Monitors
    z_tr = 0.5 * cell_z - run.dpml_z - 0.2
    if is_quasi_1d:
        monitor_span = 0.0
        dft_plane_xy = mp.Volume(center=mp.Vector3(0, 0, z_tr), size=mp.Vector3())
        dft_plane_xz = None
        td_plane_size = mp.Vector3()
        monitor_area_um2 = 1.0
    else:
        monitor_span = 0.95 * run.span_xy
        td_plane_size = mp.Vector3(monitor_span, monitor_span, 0)
        dft_plane_xy = mp.Volume(
            center=mp.Vector3(0, 0, z_tr),
            size=td_plane_size,
        )
        dft_plane_xz = mp.Volume(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(monitor_span, 0, cell_z - 2.05 * run.dpml_z),
        )
        monitor_area_um2 = monitor_span * monitor_span

    dft_freqs = [freq_p1, freq_p2, freq_probe, freq_sb_minus, freq_sb_plus]
    fixed_freqs = np.array(dft_freqs, dtype=float)
    nfreq_probe = 15
    probe_freqs = np.linspace(
        freq_probe - 0.5 * df_probe, freq_probe + 0.5 * df_probe, nfreq_probe
    )
    k_probe_center = nfreq_probe // 2

    def build_pump1_sources(amp: float) -> List[mp.Source]:
        return circular_sources(
            freq_p1, df_pump1, run.pump_cutoff, amp, "plus", src_center, src_span,
            include_ey=track_ey,
        )

    def build_pump2_sources(amp: float) -> List[mp.Source]:
        return circular_sources(
            freq_p2, df_pump2, run.pump_cutoff, amp, "minus", src_center, src_span,
            include_ey=track_ey,
        )

    def build_probe_sources(amp: float) -> List[mp.Source]:
        return linear_sources_45deg(
            freq_probe, df_probe, run.pump_cutoff, amp, src_center, src_span,
            include_ey=track_ey,
        )

    source_calibration: Dict[str, Dict[str, float]] = {}
    if args.calibrate_sources:
        pump_amp1, source_calibration["pump1"] = calibrate_source_to_intensity(
            label="pump1",
            target_intensity_w_cm2=float(run.pump_intensity_w_cm2),
            frequency=float(freq_p1),
            monitor_volume=dft_plane_xy,
            monitor_area_um2=float(monitor_area_um2),
            simulation_dimensions=simulation_dimensions,
            cell=cell,
            boundary_layers=boundary_layers,
            resolution=int(run.resolution),
            build_sources=build_pump1_sources,
            decay_threshold=float(args.calibration_decay_threshold),
        )
        pump_amp2, source_calibration["pump2"] = calibrate_source_to_intensity(
            label="pump2",
            target_intensity_w_cm2=float(run.pump_intensity_w_cm2),
            frequency=float(freq_p2),
            monitor_volume=dft_plane_xy,
            monitor_area_um2=float(monitor_area_um2),
            simulation_dimensions=simulation_dimensions,
            cell=cell,
            boundary_layers=boundary_layers,
            resolution=int(run.resolution),
            build_sources=build_pump2_sources,
            decay_threshold=float(args.calibration_decay_threshold),
        )
        probe_amp, source_calibration["probe"] = calibrate_source_to_intensity(
            label="probe",
            target_intensity_w_cm2=float(run.probe_intensity_w_cm2),
            frequency=float(freq_probe),
            monitor_volume=dft_plane_xy,
            monitor_area_um2=float(monitor_area_um2),
            simulation_dimensions=simulation_dimensions,
            cell=cell,
            boundary_layers=boundary_layers,
            resolution=int(run.resolution),
            build_sources=build_probe_sources,
            decay_threshold=float(args.calibration_decay_threshold),
        )

    sources: List[mp.Source] = []
    sources += build_pump1_sources(pump_amp1)
    sources += build_pump2_sources(pump_amp2)
    sources += build_probe_sources(probe_amp)

    simulation = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=boundary_layers,
        resolution=run.resolution,
        dimensions=simulation_dimensions,
        default_material=mp.air,
        force_complex_fields=True,
    )

    monitor_components = [mp.Ex, mp.Ey]
    dft_fields = simulation.add_dft_fields(
        monitor_components, dft_freqs, where=dft_plane_xy
    )
    trans_monitor = simulation.add_dft_fields(
        monitor_components, freq_probe, df_probe, nfreq_probe, where=dft_plane_xy
    )
    cavity_len_um = float(spec.get("cavity", {}).get("L_um", 0.0))
    cavity_line_size = mp.Vector3(0, 0, max(cavity_len_um, 1.0 / max(run.resolution, 1)))
    cavity_line_volume = mp.Volume(
        center=mp.Vector3(0, 0, cavity_center),
        size=cavity_line_size,
    )
    cavity_line_dft = simulation.add_dft_fields(
        monitor_components,
        [freq_p1, freq_p2],
        where=cavity_line_volume,
    )
    dft_fields_xz = (
        simulation.add_dft_fields(monitor_components, dft_freqs, where=dft_plane_xz)
        if capture_spatial_fields and dft_plane_xz is not None
        else None
    )

    if not is_quasi_1d:
        simulation.plot2D(output_plane=mp.Volume(center=mp.Vector3(),
                                                 size=mp.Vector3(
                                                     run.span_xy + 2*run.dpml_xy,
                                                     0,
                                                     cell_z
                                                     )
                                                )
        )
        plt.savefig(output_dir / "cavity.pdf")

    # Storage
    time_trace = {
        "t": [],
        "fixed": {
            "Ex": [],
            "Ey": [],
            "absE": [],
            "Eplus_rms": [],
            "Eminus_rms": [],
        },
        "probe_band": {
            "Ex": [],
            "Ey": [],
            "absE": [],
            "Eplus_rms": [],
            "Eminus_rms": [],
        },
        "probe_pol": {
            "theta_deg": [],
            "Ix": [],
            "Iy": [],
            "S0": [],
            "S1": [],
            "S2": [],
            "S3": [],
            "chi_deg": [],
            "dolp": [],
            "docp": [],
        },
    }
    xz_snapshot = {
        "taken": False,
        "t": None,
        "freqs": fixed_freqs,
        "Ex_maps": {},
        "Ey_maps": {},
    }
    xz_td_snapshot = {
        "taken": False,
        "t": None,
        "Ex": None,
        "Ey": None
    }
    td_env = {"t": [], "Eplus": [], "Eminus": [], "theta_deg_t": []}
    _env_plus = np.zeros(len(fixed_freqs), dtype=complex)
    _env_minus = np.zeros(len(fixed_freqs), dtype=complex)
    plane_size = td_plane_size
    plane_center = mp.Vector3(0, 0, z_tr)

    def plane_avg_mag(arr_ex: np.ndarray, arr_ey: np.ndarray) -> float:
        mag = np.sqrt(np.abs(arr_ex) ** 2 + np.abs(arr_ey) ** 2)
        return float(np.mean(mag))

    def circular_components(
        ex_arr: np.ndarray | complex, ey_arr: np.ndarray | complex
    ) -> Tuple[np.ndarray, np.ndarray]:
        ex = np.asarray(ex_arr)
        ey = np.asarray(ey_arr)
        eplus = (ex + 1j * ey) / np.sqrt(2.0)
        eminus = (ex - 1j * ey) / np.sqrt(2.0)
        return eplus, eminus

    def circular_rms(ex_arr: np.ndarray, ey_arr: np.ndarray) -> Tuple[float, float]:
        eplus, eminus = circular_components(ex_arr, ey_arr)
        eplus_rms = float(np.sqrt(np.mean(np.abs(eplus) ** 2)))
        eminus_rms = float(np.sqrt(np.mean(np.abs(eminus) ** 2)))
        return eplus_rms, eminus_rms

    def stokes_metrics(ex_arr: np.ndarray, ey_arr: np.ndarray) -> Dict[str, float]:
        ex = np.asarray(ex_arr)
        ey = np.asarray(ey_arr)
        s0 = float(np.mean(np.abs(ex) ** 2 + np.abs(ey) ** 2))
        s1 = float(np.mean(np.abs(ex) ** 2 - np.abs(ey) ** 2))
        s2 = float(2.0 * np.mean(np.real(ex * np.conjugate(ey))))
        s3 = float(-2.0 * np.mean(np.imag(ex * np.conjugate(ey))))
        theta = 0.5 * np.arctan2(s2, s1)
        chi = 0.5 * np.arctan2(s3, np.sqrt(max(s1 * s1 + s2 * s2, 0.0)))
        denom = max(s0, 1e-30)
        dolp = np.sqrt(max(s1 * s1 + s2 * s2, 0.0)) / denom
        docp = s3 / denom
        return {
            "S0": s0,
            "S1": s1,
            "S2": s2,
            "S3": s3,
            "theta_deg": float(np.degrees(theta)),
            "chi_deg": float(np.degrees(chi)),
            "dolp": float(dolp),
            "docp": float(docp),
        }

    def stokes_theta_deg(ex_arr: np.ndarray, ey_arr: np.ndarray) -> float:
        return stokes_metrics(ex_arr, ey_arr)["theta_deg"]

    def sample_callback(sim: mp.Simulation) -> None:
        t = sim.meep_time()
        time_trace["t"].append(t)

        # Fixed-frequency DFT monitors
        ex_vals, ey_vals, abs_vals = [], [], []
        eplus_rms_vals, eminus_rms_vals = [], []
        for idx in range(len(fixed_freqs)):
            ex_arr = np.asarray(sim.get_dft_array(dft_fields, mp.Ex, idx))
            ey_arr = np.asarray(sim.get_dft_array(dft_fields, mp.Ey, idx))
            ex_vals.append(complex(np.mean(ex_arr)))
            ey_vals.append(complex(np.mean(ey_arr)))
            abs_vals.append(plane_avg_mag(ex_arr, ey_arr))
            eplus_rms, eminus_rms = circular_rms(ex_arr, ey_arr)
            eplus_rms_vals.append(eplus_rms)
            eminus_rms_vals.append(eminus_rms)
        time_trace["fixed"]["Ex"].append(np.array(ex_vals))
        time_trace["fixed"]["Ey"].append(np.array(ey_vals))
        time_trace["fixed"]["absE"].append(np.array(abs_vals))
        time_trace["fixed"]["Eplus_rms"].append(np.array(eplus_rms_vals))
        time_trace["fixed"]["Eminus_rms"].append(np.array(eminus_rms_vals))

        # Probe-band DFT monitor
        ex_pb, ey_pb, abs_pb = [], [], []
        eplus_pb_rms, eminus_pb_rms = [], []
        for k in range(nfreq_probe):
            ex_arr = np.asarray(sim.get_dft_array(trans_monitor, mp.Ex, k))
            ey_arr = np.asarray(sim.get_dft_array(trans_monitor, mp.Ey, k))
            ex_pb.append(complex(np.mean(ex_arr)))
            ey_pb.append(complex(np.mean(ey_arr)))
            abs_pb.append(plane_avg_mag(ex_arr, ey_arr))
            eplus_rms, eminus_rms = circular_rms(ex_arr, ey_arr)
            eplus_pb_rms.append(eplus_rms)
            eminus_pb_rms.append(eminus_rms)
        time_trace["probe_band"]["Ex"].append(np.array(ex_pb))
        time_trace["probe_band"]["Ey"].append(np.array(ey_pb))
        time_trace["probe_band"]["absE"].append(np.array(abs_pb))
        time_trace["probe_band"]["Eplus_rms"].append(np.array(eplus_pb_rms))
        time_trace["probe_band"]["Eminus_rms"].append(np.array(eminus_pb_rms))

        # Probe polarization angle at center frequency
        ex_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Ex, k_probe_center))
        ey_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Ey, k_probe_center))
        stokes_c = stokes_metrics(ex_c, ey_c)
        theta_deg = stokes_c["theta_deg"]
        ix_c = float(np.mean(np.abs(ex_c) ** 2))
        iy_c = float(np.mean(np.abs(ey_c) ** 2))
        time_trace["probe_pol"]["theta_deg"].append(theta_deg)
        time_trace["probe_pol"]["Ix"].append(ix_c)
        time_trace["probe_pol"]["Iy"].append(iy_c)
        time_trace["probe_pol"]["S0"].append(stokes_c["S0"])
        time_trace["probe_pol"]["S1"].append(stokes_c["S1"])
        time_trace["probe_pol"]["S2"].append(stokes_c["S2"])
        time_trace["probe_pol"]["S3"].append(stokes_c["S3"])
        time_trace["probe_pol"]["chi_deg"].append(stokes_c["chi_deg"])
        time_trace["probe_pol"]["dolp"].append(stokes_c["dolp"])
        time_trace["probe_pol"]["docp"].append(stokes_c["docp"])

        # Time-domain plane averages and demodulation
        ex_td = np.asarray(
            sim.get_array(
                center=plane_center, size=plane_size, component=mp.Ex
            )
        )
        ey_td = np.asarray(
            sim.get_array(
                center=plane_center, size=plane_size, component=mp.Ey
            )
        )
        ex_mean = complex(np.mean(ex_td))
        ey_mean = complex(np.mean(ey_td))
        eplus_td = (ex_mean + 1j * ey_mean) / np.sqrt(2.0)
        eminus_td = (ex_mean - 1j * ey_mean) / np.sqrt(2.0)

        dt = run.sample_dt
        alpha = dt / max(run.lp_tau, 1e-9)
        eplus_env = np.zeros(len(fixed_freqs), dtype=complex)
        eminus_env = np.zeros(len(fixed_freqs), dtype=complex)
        for i, freq in enumerate(fixed_freqs):
            rot = np.exp(-2j * np.pi * freq * t)
            _env_plus[i] = (1 - alpha) * _env_plus[i] + alpha * (eplus_td * rot)
            _env_minus[i] = (1 - alpha) * _env_minus[i] + alpha * (eminus_td * rot)
            eplus_env[i] = _env_plus[i]
            eminus_env[i] = _env_minus[i]
        td_env["t"].append(t)
        td_env["Eplus"].append(eplus_env)
        td_env["Eminus"].append(eminus_env)
        # Probe-angle diagnostic from demodulated probe envelopes.
        td_env["theta_deg_t"].append(stokes_theta_deg(
            (eplus_env[2] + eminus_env[2]) / np.sqrt(2),
            1j * (- eplus_env[2] + eminus_env[2]) / np.sqrt(2)))

        # Optional snapshot near snapshot_time
        if (
            capture_spatial_fields
            and (not xz_snapshot["taken"])
            and (t >= snapshot_time)
            and dft_fields_xz is not None
        ):
            for i, freq in enumerate(fixed_freqs):
                ex_map = np.asarray(sim.get_dft_array(dft_fields_xz, mp.Ex, i))
                ey_map = np.asarray(sim.get_dft_array(dft_fields_xz, mp.Ey, i))
                xz_snapshot["Ex_maps"][float(freq)] = ex_map
                xz_snapshot["Ey_maps"][float(freq)] = ey_map
            xz_snapshot["taken"] = True
            xz_snapshot["t"] = float(t)
        # ---- One-time time-domain XZ snapshot at snapshot_time ----
        if (
            capture_spatial_fields
            and (not xz_td_snapshot["taken"])
            and (t >= snapshot_time)
        ):
            # time-domain instantaneous fields on the same XZ plane
            Ex_map_td = np.asarray(sim.get_array(
                center=mp.Vector3(0,0,0),
                size=mp.Vector3(run.span_xy, 0, cell_z - 2*run.dpml_z),
                component=mp.Ex,
            ))
            Ey_map_td = np.asarray(sim.get_array(
                center=mp.Vector3(0,0,0),
                size=mp.Vector3(run.span_xy, 0, cell_z - 2*run.dpml_z),
                component=mp.Ey,
            ))
            xz_td_snapshot["Ex"] = Ex_map_td
            xz_td_snapshot["Ey"] = Ey_map_td
            xz_td_snapshot["taken"] = True
            xz_td_snapshot["t"] = float(t)

    def build_decay_stop_condition():
        points = [mp.Vector3(0, 0, z_tr), mp.Vector3(0, 0, cavity_center)]
        components = [mp.Ex, mp.Ey]
        check_interval = max(float(run.sample_dt), 0.05)
        consecutive_needed = 8
        state = {"peak": 0.0, "below": 0, "last_t": -1e30}

        def _stop(sim: mp.Simulation) -> bool:
            t_now = float(sim.meep_time())
            if (t_now - float(state["last_t"])) < check_interval:
                return False
            state["last_t"] = t_now
            cur = 0.0
            for pt in points:
                for comp in components:
                    val = sim.get_field_point(comp, pt)
                    cur = max(cur, abs(complex(val)))
            state["peak"] = max(float(state["peak"]), float(cur))
            peak = float(state["peak"])
            if peak <= 1e-30:
                return False
            if cur <= float(args.decay_threshold) * peak:
                state["below"] = int(state["below"]) + 1
            else:
                state["below"] = 0
            return int(state["below"]) >= consecutive_needed

        return _stop

    if args.until_time is not None:
        simulation.run(
            mp.at_every(run.sample_dt, sample_callback),
            until=float(args.until_time),
        )
    else:
        simulation.run(
            mp.at_every(run.sample_dt, sample_callback),
            until_after_sources=build_decay_stop_condition(),
        )

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #
    t_arr = np.array(time_trace["t"])
    fixed_ex = np.vstack(time_trace["fixed"]["Ex"])
    fixed_ey = np.vstack(time_trace["fixed"]["Ey"])
    fixed_abs = np.vstack(time_trace["fixed"]["absE"])
    fixed_eplus_rms = np.vstack(time_trace["fixed"]["Eplus_rms"])
    fixed_eminus_rms = np.vstack(time_trace["fixed"]["Eminus_rms"])

    probe_ex = np.vstack(time_trace["probe_band"]["Ex"])
    probe_ey = np.vstack(time_trace["probe_band"]["Ey"])
    probe_abs = np.vstack(time_trace["probe_band"]["absE"])
    probe_eplus_rms = np.vstack(time_trace["probe_band"]["Eplus_rms"])
    probe_eminus_rms = np.vstack(time_trace["probe_band"]["Eminus_rms"])
    theta_deg_wrapped = np.array(time_trace["probe_pol"]["theta_deg"])
    theta_deg_unwrapped = unwrap_linear_polarization_deg(theta_deg_wrapped)
    theta_deg_rel_unwrapped = theta_deg_unwrapped - INIT_PROBE_POLARIZATION_DEG
    theta_deg_rel = wrap_linear_polarization_deg(
        theta_deg_wrapped - INIT_PROBE_POLARIZATION_DEG
    )
    probe_ix = np.array(time_trace["probe_pol"]["Ix"])
    probe_iy = np.array(time_trace["probe_pol"]["Iy"])
    probe_s0 = np.array(time_trace["probe_pol"]["S0"])
    probe_s1 = np.array(time_trace["probe_pol"]["S1"])
    probe_s2 = np.array(time_trace["probe_pol"]["S2"])
    probe_s3 = np.array(time_trace["probe_pol"]["S3"])
    probe_chi_deg = np.array(time_trace["probe_pol"]["chi_deg"])
    probe_dolp = np.array(time_trace["probe_pol"]["dolp"])
    probe_docp = np.array(time_trace["probe_pol"]["docp"])

    t_td = np.array(td_env["t"])
    epl_td = np.vstack(td_env["Eplus"])
    emi_td = np.vstack(td_env["Eminus"])
    theta_deg_t_wrapped = np.array(td_env["theta_deg_t"])
    theta_deg_t_unwrapped = unwrap_linear_polarization_deg(theta_deg_t_wrapped)
    theta_deg_t_rel_unwrapped = theta_deg_t_unwrapped - INIT_PROBE_POLARIZATION_DEG
    theta_deg_t_rel = wrap_linear_polarization_deg(
        theta_deg_t_wrapped - INIT_PROBE_POLARIZATION_DEG
    )

    epl_dft = (fixed_ex + 1j * fixed_ey) / np.sqrt(2.0)
    emi_dft = (fixed_ex - 1j * fixed_ey) / np.sqrt(2.0)
    abs_epl_dft_coherent = np.abs(epl_dft)
    abs_emi_dft_coherent = np.abs(emi_dft)
    abs_epl_dft_rms = fixed_eplus_rms
    abs_emi_dft_rms = fixed_eminus_rms

    i_p1, i_p2, i_probe, i_sb_minus, i_sb_plus = range(5)
    t_arr_fs = t_arr * FS_PER_MEEP
    t_td_fs = t_td * FS_PER_MEEP

    # Rotation at output, relative to input polarization angle, from probe DFT monitor.
    probe_s0_dft = probe_s0
    intensity_threshold_dft = (
        0.01 * float(np.max(probe_s0_dft)) if probe_s0_dft.size else 0.0
    )
    valid_probe_dft = probe_s0_dft > intensity_threshold_dft
    if np.any(valid_probe_dft):
        theta_probe_valid_dft = theta_deg_rel[valid_probe_dft]
        theta_probe_valid_dft_unwrapped = theta_deg_rel_unwrapped[valid_probe_dft]
        probe_rotation_final_rel = float(theta_probe_valid_dft[-1])
        probe_rotation_min_rel = float(np.min(theta_probe_valid_dft))
        probe_rotation_max_rel = float(np.max(theta_probe_valid_dft))
        probe_rotation_mean_rel = weighted_linear_mean_deg(
            theta_probe_valid_dft, probe_s0_dft[valid_probe_dft]
        )
        probe_rotation_final_rel_unwrapped = float(theta_probe_valid_dft_unwrapped[-1])
    else:
        probe_rotation_final_rel = (
            float(theta_deg_rel[-1]) if theta_deg_rel.size else float("nan")
        )
        probe_rotation_min_rel = (
            float(np.min(theta_deg_rel)) if theta_deg_rel.size else float("nan")
        )
        probe_rotation_max_rel = (
            float(np.max(theta_deg_rel)) if theta_deg_rel.size else float("nan")
        )
        probe_rotation_mean_rel = weighted_linear_mean_deg(theta_deg_rel)
        probe_rotation_final_rel_unwrapped = (
            float(theta_deg_rel_unwrapped[-1])
            if theta_deg_rel_unwrapped.size
            else float("nan")
        )

    # Keep the TD envelope estimate for diagnostics.
    probe_eplus_td = epl_td[:, i_probe]
    probe_eminus_td = emi_td[:, i_probe]
    probe_ex_td = (probe_eplus_td + probe_eminus_td) / np.sqrt(2.0)
    probe_ey_td = 1j * (-probe_eplus_td + probe_eminus_td) / np.sqrt(2.0)
    probe_s0_td = np.abs(probe_ex_td)**2 + np.abs(probe_ey_td)**2
    intensity_threshold_td = (
        0.01 * float(np.max(probe_s0_td)) if probe_s0_td.size else 0.0
    )
    valid_probe_td = probe_s0_td > intensity_threshold_td
    if np.any(valid_probe_td):
        theta_probe_valid_td = theta_deg_t_rel[valid_probe_td]
        theta_probe_valid_td_unwrapped = theta_deg_t_rel_unwrapped[valid_probe_td]
        probe_rotation_final_rel_td = float(theta_probe_valid_td[-1])
        probe_rotation_min_rel_td = float(np.min(theta_probe_valid_td))
        probe_rotation_max_rel_td = float(np.max(theta_probe_valid_td))
        probe_rotation_mean_rel_td = weighted_linear_mean_deg(
            theta_probe_valid_td, probe_s0_td[valid_probe_td]
        )
        probe_rotation_final_rel_td_unwrapped = float(theta_probe_valid_td_unwrapped[-1])
    else:
        probe_rotation_final_rel_td = (
            float(theta_deg_t_rel[-1]) if theta_deg_t_rel.size else float("nan")
        )
        probe_rotation_min_rel_td = (
            float(np.min(theta_deg_t_rel)) if theta_deg_t_rel.size else float("nan")
        )
        probe_rotation_max_rel_td = (
            float(np.max(theta_deg_t_rel)) if theta_deg_t_rel.size else float("nan")
        )
        probe_rotation_mean_rel_td = weighted_linear_mean_deg(theta_deg_t_rel)
        probe_rotation_final_rel_td_unwrapped = (
            float(theta_deg_t_rel_unwrapped[-1])
            if theta_deg_t_rel_unwrapped.size
            else float("nan")
        )

    def safe_ratio(num: float, den: float) -> float:
        return float(num / den) if abs(den) > 1e-30 else float("nan")

    def cavity_field_metrics(freq_idx: int, n_lin: float) -> Dict[str, float]:
        ex_line = np.ravel(np.asarray(simulation.get_dft_array(cavity_line_dft, mp.Ex, freq_idx)))
        ey_line = np.ravel(np.asarray(simulation.get_dft_array(cavity_line_dft, mp.Ey, freq_idx)))
        if ex_line.size == 0 or ey_line.size == 0:
            return {
                "samples": 0,
                "i_peak_w_cm2": float("nan"),
                "i_p95_w_cm2": float("nan"),
                "i_mean_w_cm2": float("nan"),
                "i_rms_w_cm2": float("nan"),
                "e_peak_meep": float("nan"),
            }
        e_mag = np.sqrt(np.abs(ex_line) ** 2 + np.abs(ey_line) ** 2)
        i_line = meep_field_to_intensity(e_mag, n_lin=max(float(n_lin), 1e-9))
        i_line = np.asarray(i_line, dtype=float)
        i_finite = i_line[np.isfinite(i_line)]
        if i_finite.size == 0:
            return {
                "samples": int(e_mag.size),
                "i_peak_w_cm2": float("nan"),
                "i_p95_w_cm2": float("nan"),
                "i_mean_w_cm2": float("nan"),
                "i_rms_w_cm2": float("nan"),
                "e_peak_meep": float(np.nanmax(np.asarray(e_mag, dtype=float))),
            }
        return {
            "samples": int(e_mag.size),
            "i_peak_w_cm2": float(np.max(i_finite)),
            "i_p95_w_cm2": float(np.percentile(i_finite, 95)),
            "i_mean_w_cm2": float(np.mean(i_finite)),
            "i_rms_w_cm2": float(np.sqrt(np.mean(i_finite**2))),
            "e_peak_meep": float(np.max(np.asarray(e_mag, dtype=float))),
        }

    def pick_kerr_intensity(metric: Dict[str, float], mode: str) -> float:
        key = {
            "peak": "i_peak_w_cm2",
            "p95": "i_p95_w_cm2",
            "mean": "i_mean_w_cm2",
            "rms": "i_rms_w_cm2",
        }.get(mode, "i_p95_w_cm2")
        try:
            out = float(metric.get(key, float("nan")))
            return out if np.isfinite(out) else float("nan")
        except Exception:
            return float("nan")

    def weighted_tail_mean(
        values: np.ndarray, weights: np.ndarray, tail_fraction: float = 0.2
    ) -> float:
        if values.size == 0:
            return float("nan")
        i0 = max(0, int((1.0 - tail_fraction) * values.size))
        v = np.asarray(values[i0:], dtype=float)
        w = np.asarray(weights[i0:], dtype=float)
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(valid):
            return float(np.mean(v)) if v.size else float("nan")
        return float(np.average(v[valid], weights=w[valid]))

    def weighted_tail_std(
        values: np.ndarray, weights: np.ndarray, tail_fraction: float = 0.2
    ) -> float:
        if values.size == 0:
            return float("nan")
        i0 = max(0, int((1.0 - tail_fraction) * values.size))
        v = np.asarray(values[i0:], dtype=float)
        w = np.asarray(weights[i0:], dtype=float)
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(valid):
            return float(np.std(v)) if v.size else float("nan")
        vv = v[valid]
        ww = w[valid]
        mu = np.average(vv, weights=ww)
        var = np.average((vv - mu) ** 2, weights=ww)
        return float(np.sqrt(max(var, 0.0)))

    def weighted_tail_linear_mean(
        values: np.ndarray, weights: np.ndarray, tail_fraction: float = 0.2
    ) -> float:
        if values.size == 0:
            return float("nan")
        i0 = max(0, int((1.0 - tail_fraction) * values.size))
        v = np.asarray(values[i0:], dtype=float)
        w = np.asarray(weights[i0:], dtype=float)
        return weighted_linear_mean_deg(v, w)

    def weighted_tail_linear_std(
        values: np.ndarray, weights: np.ndarray, tail_fraction: float = 0.2
    ) -> float:
        if values.size == 0:
            return float("nan")
        i0 = max(0, int((1.0 - tail_fraction) * values.size))
        v = np.asarray(values[i0:], dtype=float)
        w = np.asarray(weights[i0:], dtype=float)
        mu = weighted_linear_mean_deg(v, w)
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(valid):
            dv = wrap_linear_polarization_deg(v - mu)
            return float(np.std(dv)) if dv.size else float("nan")
        dv = wrap_linear_polarization_deg(v[valid] - mu)
        var = np.average(dv * dv, weights=w[valid])
        return float(np.sqrt(max(var, 0.0)))

    # Coherent-mean monitor traces (legacy) and RMS/integrated traces (new primary).
    pump1_dom_coh = abs_emi_dft_coherent[:, i_p1]
    pump2_dom_coh = abs_epl_dft_coherent[:, i_p2]
    pump1_orth_coh = abs_epl_dft_coherent[:, i_p1]
    pump2_orth_coh = abs_emi_dft_coherent[:, i_p2]

    pump1_dom_rms = abs_emi_dft_rms[:, i_p1]
    pump2_dom_rms = abs_epl_dft_rms[:, i_p2]
    pump1_orth_rms = abs_epl_dft_rms[:, i_p1]
    pump2_orth_rms = abs_emi_dft_rms[:, i_p2]

    pump_weights_coh = pump1_dom_coh + pump2_dom_coh
    pump_weights_rms = pump1_dom_rms + pump2_dom_rms

    pump1_dom_coh_tail = weighted_tail_mean(pump1_dom_coh, pump_weights_coh)
    pump2_dom_coh_tail = weighted_tail_mean(pump2_dom_coh, pump_weights_coh)
    pump1_orth_coh_tail = weighted_tail_mean(pump1_orth_coh, pump_weights_coh)
    pump2_orth_coh_tail = weighted_tail_mean(pump2_orth_coh, pump_weights_coh)
    pump1_dom_rms_tail = weighted_tail_mean(pump1_dom_rms, pump_weights_rms)
    pump2_dom_rms_tail = weighted_tail_mean(pump2_dom_rms, pump_weights_rms)
    pump1_orth_rms_tail = weighted_tail_mean(pump1_orth_rms, pump_weights_rms)
    pump2_orth_rms_tail = weighted_tail_mean(pump2_orth_rms, pump_weights_rms)

    pump_ratio_coh_final = (
        safe_ratio(pump2_dom_coh[-1], pump1_dom_coh[-1])
        if pump1_dom_coh.size
        else float("nan")
    )
    pump_ratio_rms_final = (
        safe_ratio(pump2_dom_rms[-1], pump1_dom_rms[-1])
        if pump1_dom_rms.size
        else float("nan")
    )
    pump_ratio_coh_tail = safe_ratio(pump2_dom_coh_tail, pump1_dom_coh_tail)
    pump_ratio_rms_tail = safe_ratio(pump2_dom_rms_tail, pump1_dom_rms_tail)

    pump1_purity_coh_final = (
        safe_ratio(
            pump1_dom_coh[-1] ** 2,
            pump1_dom_coh[-1] ** 2 + pump1_orth_coh[-1] ** 2,
        )
        if pump1_dom_coh.size
        else float("nan")
    )
    pump2_purity_coh_final = (
        safe_ratio(
            pump2_dom_coh[-1] ** 2,
            pump2_dom_coh[-1] ** 2 + pump2_orth_coh[-1] ** 2,
        )
        if pump2_dom_coh.size
        else float("nan")
    )
    pump1_purity_rms_final = (
        safe_ratio(
            pump1_dom_rms[-1] ** 2,
            pump1_dom_rms[-1] ** 2 + pump1_orth_rms[-1] ** 2,
        )
        if pump1_dom_rms.size
        else float("nan")
    )
    pump2_purity_rms_final = (
        safe_ratio(
            pump2_dom_rms[-1] ** 2,
            pump2_dom_rms[-1] ** 2 + pump2_orth_rms[-1] ** 2,
        )
        if pump2_dom_rms.size
        else float("nan")
    )
    pump1_purity_coh_tail = safe_ratio(
        pump1_dom_coh_tail ** 2, pump1_dom_coh_tail ** 2 + pump1_orth_coh_tail ** 2
    )
    pump2_purity_coh_tail = safe_ratio(
        pump2_dom_coh_tail ** 2, pump2_dom_coh_tail ** 2 + pump2_orth_coh_tail ** 2
    )
    pump1_purity_rms_tail = safe_ratio(
        pump1_dom_rms_tail ** 2, pump1_dom_rms_tail ** 2 + pump1_orth_rms_tail ** 2
    )
    pump2_purity_rms_tail = safe_ratio(
        pump2_dom_rms_tail ** 2, pump2_dom_rms_tail ** 2 + pump2_orth_rms_tail ** 2
    )

    probe_theta_tail_rel = weighted_tail_linear_mean(theta_deg_rel, probe_s0_dft)
    probe_theta_tail_std_rel = weighted_tail_linear_std(theta_deg_rel, probe_s0_dft)
    probe_chi_tail = weighted_tail_mean(probe_chi_deg, probe_s0_dft)
    probe_docp_tail = weighted_tail_mean(probe_docp, probe_s0_dft)
    probe_dolp_tail = weighted_tail_mean(probe_dolp, probe_s0_dft)
    probe_s0_tail = weighted_tail_mean(probe_s0_dft, np.ones_like(probe_s0_dft))
    probe_s0_max = float(np.max(probe_s0_dft)) if probe_s0_dft.size else float("nan")
    probe_s0_tail_rel_max = safe_ratio(probe_s0_tail, probe_s0_max)

    cavity_hotspot_p1 = cavity_field_metrics(0, n_linear_p1)
    cavity_hotspot_p2 = cavity_field_metrics(1, n_linear_p2)
    i_eff_p1_w_cm2 = pick_kerr_intensity(cavity_hotspot_p1, kerr_intensity_metric)
    i_eff_p2_w_cm2 = pick_kerr_intensity(cavity_hotspot_p2, kerr_intensity_metric)
    i_eff_p1_si = max(i_eff_p1_w_cm2, 0.0) * 1e4
    i_eff_p2_si = max(i_eff_p2_w_cm2, 0.0) * 1e4
    delta_n_p1 = float(n2_sin * (i_eff_p1_si + kerr_xpm_factor * i_eff_p2_si))
    delta_n_p2 = float(n2_sin * (i_eff_p2_si + kerr_xpm_factor * i_eff_p1_si))
    delta_rel_p1 = float(-delta_n_p1 / max(float(n_linear_p1), 1e-12))
    delta_rel_p2 = float(-delta_n_p2 / max(float(n_linear_p2), 1e-12))
    freq_p1_kerr = float(max(1e-9, freq_p1 * (1.0 + delta_rel_p1)))
    freq_p2_kerr = float(max(1e-9, freq_p2 * (1.0 + delta_rel_p2)))

    plot_paths: Dict[str, str] = {}

    fig = plt.figure(figsize=(7.2, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        t_arr_fs,
        abs_emi_dft_coherent[:, i_p1],
        label=f"pump1 DFT |e-| (f={fixed_freqs[i_p1]:.3f})",
    )
    ax.plot(
        t_arr_fs,
        abs_epl_dft_coherent[:, i_p2],
        label=f"pump2 DFT |e+| (f={fixed_freqs[i_p2]:.3f})",
    )
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("|E| (coherent mean, DFT)")
    ax.set_title("Pumps (DFT monitors, coherent mean reference)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_paths["pumps_dft_coherent"] = str(
        save_figure(fig, "pumps_dft_coherent.png", output_dir)
    )

    fig = plt.figure(figsize=(7.2, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        t_arr_fs,
        abs_emi_dft_rms[:, i_p1],
        label=f"pump1 DFT RMS |e-| (f={fixed_freqs[i_p1]:.3f})",
    )
    ax.plot(
        t_arr_fs,
        abs_epl_dft_rms[:, i_p2],
        label=f"pump2 DFT RMS |e+| (f={fixed_freqs[i_p2]:.3f})",
    )
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("|E| (RMS integrated, DFT)")
    ax.set_title("Pumps (DFT monitors, RMS integrated circular)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_paths["pumps_dft"] = str(save_figure(fig, "pumps_dft.png", output_dir))

    fig = plt.figure(figsize=(7.2, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_td_fs, np.abs(emi_td[:, i_p1]), label="pump1 TD |e-|")
    ax.plot(t_td_fs, np.abs(epl_td[:, i_p2]), label="pump2 TD |e+|")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("|E| (plane avg, TD demod)")
    ax.set_title("Pumps (time-domain demod)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_paths["pumps_td"] = str(save_figure(fig, "pumps_td.png", output_dir))

    def plot_pair_coherent(ax_, label: str, idx: int) -> None:
        ax_.plot(
            t_arr_fs, abs_epl_dft_coherent[:, idx], "-", label=f"{label} DFT |e+|"
        )
        ax_.plot(
            t_arr_fs, abs_emi_dft_coherent[:, idx], "--", label=f"{label} DFT |e-|"
        )

    def plot_pair_rms(ax_, label: str, idx: int) -> None:
        ax_.plot(
            t_arr_fs,
            abs_epl_dft_rms[:, idx],
            "-",
            label=f"{label} DFT RMS |e+|",
        )
        ax_.plot(
            t_arr_fs,
            abs_emi_dft_rms[:, idx],
            "--",
            label=f"{label} DFT RMS |e-|",
        )

    fig = plt.figure(figsize=(7.6, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_pair_coherent(ax, "probe", i_probe)
    plot_pair_coherent(ax, "sb-", i_sb_minus)
    plot_pair_coherent(ax, "sb+", i_sb_plus)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("|E| (coherent mean, DFT)")
    ax.set_title("Probe & sidebands (DFT monitors, coherent mean reference)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plot_paths["probe_dft_coherent"] = str(
        save_figure(fig, "probe_dft_coherent.png", output_dir)
    )

    fig = plt.figure(figsize=(7.6, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_pair_rms(ax, "probe", i_probe)
    plot_pair_rms(ax, "sb-", i_sb_minus)
    plot_pair_rms(ax, "sb+", i_sb_plus)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("|E| (RMS integrated, DFT)")
    ax.set_title("Probe & sidebands (DFT monitors, RMS integrated circular)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plot_paths["probe_dft"] = str(save_figure(fig, "probe_dft.png", output_dir))

    def plot_pair_td(ax_, label: str, idx: int) -> None:
        ax_.plot(t_td_fs, np.abs(epl_td[:, idx]), "-", label=f"{label} TD |e+|")
        ax_.plot(t_td_fs, np.abs(emi_td[:, idx]), "--", label=f"{label} TD |e-|")

    fig = plt.figure(figsize=(7.6, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_pair_td(ax, "probe", i_probe)
    plot_pair_td(ax, "sb-", i_sb_minus)
    plot_pair_td(ax, "sb+", i_sb_plus)
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("|E| (plane avg, TD demod)")
    ax.set_title("Probe & sidebands (time-domain demod)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plot_paths["probe_td"] = str(save_figure(fig, "probe_td.png", output_dir))

    probe_abs_rms = np.sqrt(probe_eplus_rms**2 + probe_eminus_rms**2)

    fig = plt.figure(figsize=(7.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        np.abs(probe_abs.T),
        aspect="auto",
        origin="lower",
        extent=[t_arr_fs.min(), t_arr_fs.max(), probe_freqs.min(), probe_freqs.max()],
    )
    fig.colorbar(im, ax=ax, label=r"$\langle |E| \rangle$")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("frequency (1/μm)")
    ax.set_title("Probe-band plane-avg |E| vs (f, t) - coherent reference")
    plot_paths["probe_band_heatmap_coherent"] = str(
        save_figure(fig, "probe_band_heatmap_coherent.png", output_dir)
    )

    fig = plt.figure(figsize=(7.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        np.abs(probe_abs_rms.T),
        aspect="auto",
        origin="lower",
        extent=[t_arr_fs.min(), t_arr_fs.max(), probe_freqs.min(), probe_freqs.max()],
    )
    fig.colorbar(im, ax=ax, label=r"$\sqrt{\langle |E_+|^2+|E_-|^2 \rangle}$")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("frequency (1/μm)")
    ax.set_title("Probe-band RMS integrated |E| vs (f, t)")
    plot_paths["probe_band_heatmap"] = str(
        save_figure(fig, "probe_band_heatmap.png", output_dir)
    )

    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_arr_fs, theta_deg_rel, "k-")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("polarization rotation (deg)")
    ax.set_title("Probe polarization angle vs time (relative to input)")
    ax.grid(True, alpha=0.3)
    plot_paths["probe_rotation"] = str(
        save_figure(fig, "probe_polarization.png", output_dir)
    )

    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_td_fs, theta_deg_t_rel, "k-")
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("polarization rotation (deg)")
    ax.set_title("Probe polarization angle vs time (relative to input) in time-domain")
    ax.grid(True, alpha=0.3)
    plot_paths["probe_rotation_td"] = str(
        save_figure(fig, "probe_polarization_td.png", output_dir)
    )

    if capture_spatial_fields and xz_snapshot["taken"]:
        x_half = 0.5 * run.span_xy
        z_half = 0.5 * (cell_z - 2*run.dpml_z)
        extent_xz = (-x_half, x_half, -z_half, z_half)

        freqs_to_show = list(xz_snapshot["Ex_maps"].keys())
        ncols = min(3, len(freqs_to_show))
        nrows = int(np.ceil(len(freqs_to_show) / max(ncols, 1)))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 5 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax_subplot, freq in zip(axes, freqs_to_show):
            ex_map = xz_snapshot["Ex_maps"][freq]
            ey_map = xz_snapshot["Ey_maps"][freq]
            emap = np.sqrt(np.abs(ex_map) ** 2 + np.abs(ey_map) ** 2)
            im = ax_subplot.imshow(
                np.abs(emap).T,
                origin="lower",
                aspect="auto",
                cmap="magma",
                extent=extent_xz,
            )
            ax_subplot.set_title(
                f"|E|(x,z) at f={freq:.3f}, t≈{float(xz_snapshot['t']) * FS_PER_MEEP:.2f} fs"
            )
            ax_subplot.set_xlabel("x index")
            ax_subplot.set_ylabel("z index")
        for ax_subplot in axes[len(freqs_to_show) :]:
            ax_subplot.axis("off")
        fig.colorbar(im, ax=axes.tolist(), shrink=0.9, label="|E|")
        # fig.tight_layout()
        plot_paths["xz_snapshot"] = str(
            save_figure(fig, "xz_snapshot.png", output_dir)
        )
    # --- (E2) X–Z spatial map from **time-domain** snapshot (instantaneous |E|) ---
    if capture_spatial_fields and xz_td_snapshot["taken"]:
        x_half = 0.5 * run.span_xy
        z_half = 0.5 * (cell_z - 2*run.dpml_z)
        extent_xz = (-x_half, x_half, -z_half, z_half)

        Ex_td = np.asarray(xz_td_snapshot["Ex"])
        Ey_td = np.asarray(xz_td_snapshot["Ey"])
        Emag_td = np.sqrt(np.abs(Ex_td)**2 + np.abs(Ey_td)**2)

        fig = plt.figure(figsize=(5.4, 8.0))
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(
            Emag_td.T,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=extent_xz,
        )
        fig.colorbar(im, ax=ax, label="|E| (instantaneous)")
        ax.set_xlabel("x (μm)")
        ax.set_ylabel("z (μm)")
        ax.set_title(
            f"Instantaneous |E|(x,z) at t≈{float(xz_td_snapshot['t']) * FS_PER_MEEP:.2f} fs"
        )
        fig.tight_layout()
        plot_paths["xz_td_snapshot"] = str(
            save_figure(fig, "xz_td_snapshot.png", output_dir)
        )

    # ------------------------------------------------------------------ #
    # Structured result + compact summary
    # ------------------------------------------------------------------ #
    run_params_dict = asdict(run)
    frequencies = {
        "pump1": float(freq_p1),
        "pump2": float(freq_p2),
        "probe": float(freq_probe),
        "sb_minus": float(freq_sb_minus),
        "sb_plus": float(freq_sb_plus),
        "pump1_cold": float(freq_p1_cold),
        "pump2_cold": float(freq_p2_cold),
    }
    wavelengths = {
        "pump1": float(lam_p1),
        "pump2": float(lam_p2),
        "probe": float(lam_probe),
    }

    abs_epl_dft = abs_epl_dft_rms
    abs_emi_dft = abs_emi_dft_rms
    abs_epl_td = np.abs(epl_td)
    abs_emi_td = np.abs(emi_td)
    probe_band_abs = np.abs(probe_abs_rms)

    probe_theta_final_rel = (
        float(theta_deg_rel[-1]) if theta_deg_rel.size else float("nan")
    )
    probe_chi_final = (
        float(probe_chi_deg[-1]) if probe_chi_deg.size else float("nan")
    )
    probe_dolp_final = float(probe_dolp[-1]) if probe_dolp.size else float("nan")
    probe_docp_final = float(probe_docp[-1]) if probe_docp.size else float("nan")
    probe_s0_final = float(probe_s0[-1]) if probe_s0.size else float("nan")
    probe_s1_final = float(probe_s1[-1]) if probe_s1.size else float("nan")
    probe_s2_final = float(probe_s2[-1]) if probe_s2.size else float("nan")
    probe_s3_final = float(probe_s3[-1]) if probe_s3.size else float("nan")

    probe_rotation_trace = ProbeRotationTrace(
        time=t_arr.copy(),
        theta_deg_rel=theta_deg_rel.copy(),
        final_deg=probe_rotation_final_rel,
        min_deg=probe_rotation_min_rel,
        max_deg=probe_rotation_max_rel,
        time_domain_time=t_td.copy(),
        time_domain_theta_deg_rel=theta_deg_t_rel.copy(),
    )

    dft_trace = FieldTrace(
        time=t_arr.copy(),
        freqs=fixed_freqs.copy(),
        abs_eplus=abs_epl_dft,
        abs_eminus=abs_emi_dft,
    )
    td_trace = FieldTrace(
        time=t_td.copy(),
        freqs=fixed_freqs.copy(),
        abs_eplus=abs_epl_td,
        abs_eminus=abs_emi_td,
    )
    probe_band_trace = ProbeBandTrace(
        time=t_arr.copy(),
        freqs=probe_freqs.copy(),
        abs_field=probe_band_abs,
    )

    summary_data = {
        "run_mode": run.name,
        "dimension": int(dimension),
        "run_params": run_params_dict,
        "geometry_file": str(spec_path),
        "materials_model": args.materials,
        "cell_size_um": {"x": float(cell.x), "y": float(cell.y), "z": float(cell_z)},
        "cavity_center_um": float(cavity_center),
        "frequencies_inv_um": frequencies,
        "wavelengths_um": wavelengths,
        "intensities_w_cm2": {
            "pump": float(run.pump_intensity_w_cm2),
            "probe": float(run.probe_intensity_w_cm2),
        },
        "source_amplitudes_meep": {
            "pump1": float(pump_amp1),
            "pump2": float(pump_amp2),
            "probe": float(probe_amp),
        },
        "source_calibration": {
            "enabled": bool(args.calibrate_sources),
            "monitor_area_um2": float(monitor_area_um2),
            "calibration_decay_threshold": (
                float(args.calibration_decay_threshold)
                if args.calibrate_sources
                else None
            ),
            "per_source": source_calibration,
        },
        "probe_band_frequencies_inv_um": probe_freqs.tolist(),
        "monitor_plane_z_um": float(z_tr),
        "plot_time_units": {
            "unit": "fs",
            "fs_per_meep_time": float(FS_PER_MEEP),
        },
        "probe_rotation_deg": {
            "initial_deg": INIT_PROBE_POLARIZATION_DEG,
            "final_relative_deg": probe_rotation_final_rel,
            "max_relative_deg": probe_rotation_max_rel,
            "min_relative_deg": probe_rotation_min_rel,
            "mean_relative_deg": probe_rotation_mean_rel,
            "final_relative_unwrapped_deg": probe_rotation_final_rel_unwrapped,
            "wrapped_final_relative_deg": (
                float(theta_deg_rel[-1])
                if theta_deg_rel.size
                else float("nan")
            ),
            "raw_final_relative_deg": (
                float(theta_deg_wrapped[-1] - INIT_PROBE_POLARIZATION_DEG)
                if theta_deg_wrapped.size
                else float("nan")
            ),
            "method": "dft_monitor_center_frequency_at_output_principal_linear_angle",
            "intensity_threshold_rel": 0.01,
            "time_domain_reference": {
                "final_relative_deg": probe_rotation_final_rel_td,
                "max_relative_deg": probe_rotation_max_rel_td,
                "min_relative_deg": probe_rotation_min_rel_td,
                "mean_relative_deg": probe_rotation_mean_rel_td,
                "final_relative_unwrapped_deg": probe_rotation_final_rel_td_unwrapped,
                "wrapped_final_relative_deg": (
                    float(theta_deg_t_rel[-1])
                    if theta_deg_t_rel.size
                    else float("nan")
                ),
                "raw_final_relative_deg": (
                    float(theta_deg_t_wrapped[-1] - INIT_PROBE_POLARIZATION_DEG)
                    if theta_deg_t_wrapped.size
                    else float("nan")
                ),
                "method": "time_domain_probe_envelope_at_output_principal_linear_angle",
                "intensity_threshold_rel": 0.01,
            },
        },
        "probe_stokes_dft": {
            "final": {
                "theta_relative_deg": probe_theta_final_rel,
                "chi_deg": probe_chi_final,
                "dolp": probe_dolp_final,
                "docp": probe_docp_final,
                "S0": probe_s0_final,
                "S1": probe_s1_final,
                "S2": probe_s2_final,
                "S3": probe_s3_final,
            },
            "tail_weighted": {
                "theta_relative_deg": float(probe_theta_tail_rel),
                "theta_relative_std_deg": float(probe_theta_tail_std_rel),
                "chi_deg": float(probe_chi_tail),
                "dolp": float(probe_dolp_tail),
                "docp": float(probe_docp_tail),
                "S0": float(probe_s0_tail),
                "S0_rel_max": float(probe_s0_tail_rel_max),
                "tail_fraction": 0.2,
                "weights": "S0",
            },
        },
        "pump_monitor_metrics": {
            "definition": {
                "pump1_dominant_component": "|e-| at f_p1",
                "pump1_orthogonal_component": "|e+| at f_p1",
                "pump2_dominant_component": "|e+| at f_p2",
                "pump2_orthogonal_component": "|e-| at f_p2",
                "ratio_reported_as": "pump2_dominant / pump1_dominant",
            },
            "coherent_reference": {
                "final_abs": {
                    "pump1_dominant": float(pump1_dom_coh[-1]) if pump1_dom_coh.size else float("nan"),
                    "pump2_dominant": float(pump2_dom_coh[-1]) if pump2_dom_coh.size else float("nan"),
                    "pump1_orthogonal": float(pump1_orth_coh[-1]) if pump1_orth_coh.size else float("nan"),
                    "pump2_orthogonal": float(pump2_orth_coh[-1]) if pump2_orth_coh.size else float("nan"),
                },
                "tail_weighted_abs": {
                    "pump1_dominant": float(pump1_dom_coh_tail),
                    "pump2_dominant": float(pump2_dom_coh_tail),
                    "pump1_orthogonal": float(pump1_orth_coh_tail),
                    "pump2_orthogonal": float(pump2_orth_coh_tail),
                    "tail_fraction": 0.2,
                    "weights": "pump1_dominant + pump2_dominant",
                },
                "ratio_p2_over_p1": {
                    "final": float(pump_ratio_coh_final),
                    "tail_weighted": float(pump_ratio_coh_tail),
                },
                "dominant_purity": {
                    "pump1_final": float(pump1_purity_coh_final),
                    "pump2_final": float(pump2_purity_coh_final),
                    "pump1_tail_weighted": float(pump1_purity_coh_tail),
                    "pump2_tail_weighted": float(pump2_purity_coh_tail),
                },
            },
            "rms_integrated": {
                "final_abs": {
                    "pump1_dominant": float(pump1_dom_rms[-1]) if pump1_dom_rms.size else float("nan"),
                    "pump2_dominant": float(pump2_dom_rms[-1]) if pump2_dom_rms.size else float("nan"),
                    "pump1_orthogonal": float(pump1_orth_rms[-1]) if pump1_orth_rms.size else float("nan"),
                    "pump2_orthogonal": float(pump2_orth_rms[-1]) if pump2_orth_rms.size else float("nan"),
                },
                "tail_weighted_abs": {
                    "pump1_dominant": float(pump1_dom_rms_tail),
                    "pump2_dominant": float(pump2_dom_rms_tail),
                    "pump1_orthogonal": float(pump1_orth_rms_tail),
                    "pump2_orthogonal": float(pump2_orth_rms_tail),
                    "tail_fraction": 0.2,
                    "weights": "pump1_dominant + pump2_dominant",
                },
                "ratio_p2_over_p1": {
                    "final": float(pump_ratio_rms_final),
                    "tail_weighted": float(pump_ratio_rms_tail),
                },
                "dominant_purity": {
                    "pump1_final": float(pump1_purity_rms_final),
                    "pump2_final": float(pump2_purity_rms_final),
                    "pump1_tail_weighted": float(pump1_purity_rms_tail),
                    "pump2_tail_weighted": float(pump2_purity_rms_tail),
                },
            },
        },
        "kerr_shift_estimate": {
            "model": "delta_omega_over_omega = -delta_n/neff",
            "n2_m2_per_w": float(n2_sin),
            "xpm_factor": float(kerr_xpm_factor),
            "intensity_metric": str(kerr_intensity_metric),
            "neff": {
                "pump1": float(n_linear_p1),
                "pump2": float(n_linear_p2),
            },
            "hotspot_monitor": {
                "type": "on_axis_z_line_in_cavity",
                "center_z_um": float(cavity_center),
                "length_um": float(max(cavity_len_um, 0.0)),
                "samples_pump1": int(cavity_hotspot_p1["samples"]),
                "samples_pump2": int(cavity_hotspot_p2["samples"]),
            },
            "local_intensity_w_cm2": {
                "pump1": cavity_hotspot_p1,
                "pump2": cavity_hotspot_p2,
                "used_for_prediction": {
                    "pump1": float(i_eff_p1_w_cm2),
                    "pump2": float(i_eff_p2_w_cm2),
                },
            },
            "predicted_frequency_shift": {
                "pump1_delta_n": float(delta_n_p1),
                "pump2_delta_n": float(delta_n_p2),
                "pump1_delta_omega_over_omega": float(delta_rel_p1),
                "pump2_delta_omega_over_omega": float(delta_rel_p2),
                "pump1_frequency_new_inv_um": float(freq_p1_kerr),
                "pump2_frequency_new_inv_um": float(freq_p2_kerr),
            },
        },
        "plot_paths": plot_paths,
        "theta_deg_rel_I": {
            "pump_intensity_w_cm2": float(run.pump_intensity_w_cm2),
            "final_relative_deg": probe_rotation_final_rel,
        },
        "objective_quality": {
            "abs_rotation_deg": float(abs(probe_rotation_final_rel)),
            "probe_dolp_tail": float(probe_dolp_tail),
            "probe_theta_tail_std_deg": float(probe_theta_tail_std_rel),
            "probe_s0_tail": float(probe_s0_tail),
            "probe_s0_tail_rel_max": float(probe_s0_tail_rel_max),
        },
    }

    summary_path = output_dir / "faraday_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    result = SimulationResult(
        run_mode=run.name,
        pump_intensity_w_cm2=float(run.pump_intensity_w_cm2),
        probe_rotation=probe_rotation_trace,
        dft_traces=dft_trace,
        time_domain_traces=td_trace,
        probe_band_trace=probe_band_trace,
        plot_paths=plot_paths,
        output_dir=str(output_dir),
        summary=summary_data,
        summary_path=summary_path,
        metadata={
            "frequencies_inv_um": frequencies,
            "wavelengths_um": wavelengths,
            "monitor_plane_z_um": float(z_tr),
        },
    )

    print(f"Simulation complete. Summary written to {summary_path}")
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D/3D Faraday rotation simulation.")
    parser.add_argument(
        "--dim",
        type=int,
        choices=(1, 3),
        default=1,
        help="Simulation mode: 1 = quasi-1D (collapsed transverse cell with Ex/Ey), 3 = full transverse cell.",
    )
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="Select quick sanity run or full-resolution simulation.",
    )
    parser.add_argument(
        "--materials",
        choices=("library", "constant", "fit"),
        default="library",
        help="Material model for SiN/SiO2.",
    )
    parser.add_argument(
        "--nH",
        type=float,
        default=None,
        help="Override high-index value when --materials constant.",
    )
    parser.add_argument(
        "--nL",
        type=float,
        default=None,
        help="Override low-index value when --materials constant.",
    )
    parser.add_argument(
        "--sin-fit",
        dest="sin_fit",
        type=str,
        default=None,
        help="CSV with wavelength_nm,n,k for SiN when --materials fit.",
    )
    parser.add_argument(
        "--sio2-fit",
        dest="sio2_fit",
        type=str,
        default=None,
        help="CSV with wavelength_nm,n,k for SiO2 when --materials fit.",
    )
    parser.add_argument(
        "--fit-window",
        type=int,
        nargs=2,
        metavar=("lambda_min", "lambda_max"),
        default=(600, 2000),
        help="Lower and upper wavelength limits for fitting epsilon.",
    )
    parser.add_argument(
        "--fit-poles",
        type=int,
        default=2,
        help="Number of Lorentz/Drude poles when fitting dispersive materials.",
    )
    parser.add_argument(
        "--pump-intensity",
        type=float,
        default=None,
        help="Pump intensity in W/cm^2.",
    )
    parser.add_argument(
        "--kerr-xpm-factor",
        type=float,
        default=1.0,
        help="Cross-phase weighting in Kerr shift estimate: delta_n1~n2*(I1+xpm*I2).",
    )
    parser.add_argument(
        "--kerr-intensity-metric",
        choices=("peak", "p95", "mean", "rms"),
        default="p95",
        help="Cavity-line intensity statistic used for Kerr shift prediction.",
    )
    parser.add_argument(
        "--pump1-frequency",
        type=float,
        default=None,
        help="Override pump1 frequency (1/um).",
    )
    parser.add_argument(
        "--pump2-frequency",
        type=float,
        default=None,
        help="Override pump2 frequency (1/um).",
    )
    parser.add_argument(
        "--geometry-file",
        type=str,
        default="optimized_geometry.json",
        help="Geometry JSON file (schema compatible with geometry_io).",
    )
    parser.add_argument(
        "--cavity-modes-file",
        type=str,
        default="cavity_modes.json",
        help="Cavity-mode summary JSON with probe/pump frequencies.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store plots and the summary JSON.",
    )
    parser.add_argument(
        "--until-time",
        type=float,
        default=None,
        help="If provided, run until this Meep time instead of field-decay stopping.",
    )
    parser.add_argument(
        "--decay-threshold",
        type=float,
        default=1e-9,
        help="Field-decay stopping threshold used when --until-time is not provided.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override default resolution (px/um) for the selected mode.",
    )
    parser.add_argument(
        "--calibrate-sources",
        action="store_true",
        help="Run empty-cell source calibration so source amplitudes match target intensities at the monitor plane.",
    )
    parser.add_argument(
        "--calibration-decay-threshold",
        type=float,
        default=1e-7,
        help="Field-decay threshold for the source calibration helper runs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if (
        cli_args.materials == "fit"
        and (cli_args.sin_fit is None or cli_args.sio2_fit is None)
    ):
        raise SystemExit(
            "For --materials fit provide both --sin-fit and --sio2-fit CSV paths."
        )
    test = run_simulation(cli_args)
