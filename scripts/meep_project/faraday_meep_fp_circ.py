#!/usr/bin/env python3
"""
3D pump–probe simulation of Faraday rotation in an optimized DBR cavity.

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
        probe_intensity_w_cm2=1.0e4,
        nonlinear_scale=1.0,
        sample_dt=0.015,
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
        probe_intensity_w_cm2=1.0e7,
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
    margin_z: float = 0.4,
) -> Tuple[List[mp.Block], float, float]:
    """Create a 3D geometry list from the optimized geometry JSON."""
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
    xy_size = core_span_xy

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
) -> Tuple[List[mp.Block], float, float, Dict[str, mp.Medium], Dict]:
    spec = load_geometry_json(str(path))
    mats = {
        name: material_factory(name, entry, mp)
        for name, entry in spec["materials"].items()
    }
    geometry, cell_z, cavity_center = build_geometry_from_spec(
        spec, mats, core_span_xy, dpml_z
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
) -> List[mp.Source]:
    """Return (Ex, Ey) sources for circular polarization."""
    phase = 1.0j if handedness == "plus" else -1.0j
    amp = amplitude / np.sqrt(2.0)
    base = mp.GaussianSource(frequency=frequency, fwidth=fwidth, cutoff=cutoff)
    size = mp.Vector3(src_span, src_span, 0)
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
) -> List[mp.Source]:
    amp = amplitude / np.sqrt(2.0)
    base = mp.GaussianSource(frequency=frequency, fwidth=fwidth, cutoff=cutoff)
    size = mp.Vector3(src_span, src_span, 0)
    return [
        mp.Source(src=base, component=mp.Ex, center=src_center, size=size, amplitude=amp),
        mp.Source(src=base, component=mp.Ey, center=src_center, size=size, amplitude=amp),
    ]


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
    output_dir = Path(args.output_dir or f"faraday_{run.name}_outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = load_cavity_modes(Path("cavity_modes_experiment.json"))
    spec_path = Path("optimized_geometry_experiment.json")
    if not spec_path.exists():
        raise FileNotFoundError("optimized_geometry.json not found.")

    core_span = run.span_xy
    geometry, cell_z, cavity_center, materials, spec = load_geometry(
        spec_path,
        core_span_xy=core_span,
        dpml_z=run.dpml_z,
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
        spec, materials, core_span, run.dpml_z
    )

    freq_probe = modes["probe"]["frequency"]
    lam_probe = modes["probe"]["lambda_um"]
    freq_p1 = modes["pump1"]["frequency"]
    freq_p2 = modes["pump2"]["frequency"]
    lam_p1 = modes["pump1"]["lambda_um"]
    lam_p2 = modes["pump2"]["lambda_um"]
    delta_omega = abs(freq_p1 - freq_p2)
    freq_sb_plus = freq_probe + delta_omega
    freq_sb_minus = max(freq_probe - delta_omega, 0.0)

    # Nonlinear response for SiN scaled as requested
    n2_sin = 2.5e-19  # m²/W
    n_linear_probe = material_index_at_wavelength(mat_sin, lam_probe)
    chi3_si = (4.0 / 3.0) * n2_sin * (n_linear_probe**2) * EPS0 * C0
    e_chi3_meep = chi3_si * (SCALE_E**3) * run.nonlinear_scale
    mat_sin.E_chi3_diag = mp.Vector3(e_chi3_meep, e_chi3_meep, e_chi3_meep)

    n_pump1 = material_index_at_wavelength(mat_sin, lam_p1)
    n_pump2 = material_index_at_wavelength(mat_sin, lam_p2)
    n_probe_lin = n_linear_probe

    if args.pump_intensity is None:
        pump_amp1 = intensity_to_meep_amplitude(run.pump_intensity_w_cm2,
                                                n_pump1)
        pump_amp2 = intensity_to_meep_amplitude(run.pump_intensity_w_cm2,
                                                n_pump2)
    else:
        run.pump_intensity_w_cm2 = args.pump_intensity
        pump_amp1 = intensity_to_meep_amplitude(args.pump_intensity,
                                                n_pump1)
        pump_amp2 = intensity_to_meep_amplitude(args.pump_intensity,
                                                n_pump2)
    probe_amp = intensity_to_meep_amplitude(run.probe_intensity_w_cm2, n_probe_lin)

    # df_probe = df_from_bandwidth(lam_probe, run.probe_band_nm)
    # df_pump1 = df_from_bandwidth(lam_p1, run.pump_band_nm)
    # df_pump2 = df_from_bandwidth(lam_p2, run.pump_band_nm)
    df_probe = df_from_pulse_duration(run.pulse_duration_fs)
    df_pump1 = df_from_pulse_duration(run.pulse_duration_fs)
    df_pump2 = df_from_pulse_duration(run.pulse_duration_fs)

    boundary_layers: List[mp.PML] = [mp.PML(run.dpml_z, direction=mp.Z)]
    if run.dpml_xy > 0:
        boundary_layers.extend(
            [mp.PML(run.dpml_xy, direction=mp.X), mp.PML(run.dpml_xy, direction=mp.Y)]
        )

    cell = mp.Vector3(
        run.span_xy + 2 * run.dpml_xy, run.span_xy + 2 * run.dpml_xy, cell_z
    )
    src_z = -0.5 * cell_z + run.dpml_z + run.src_buffer
    src_center = mp.Vector3(0, 0, src_z)
    src_span = run.span_xy + 2 * run.dpml_xy

    sources: List[mp.Source] = []
    sources += circular_sources(
        freq_p1, df_pump1, run.pump_cutoff, pump_amp1, "plus", src_center, src_span
    )
    sources += circular_sources(
        freq_p2, df_pump2, run.pump_cutoff, pump_amp2, "minus", src_center, src_span
    )
    sources += linear_sources_45deg(
        freq_probe, df_probe, run.pump_cutoff, probe_amp, src_center, src_span
    )

    simulation = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=boundary_layers,
        resolution=run.resolution,
        force_complex_fields=True,
        default_material=mp.air,
    )

    pulse_duration_meep = run.pulse_duration_fs / (1e9 / C0)
    stop_time = run.runtime_factor * pulse_duration_meep
    snapshot_time = 0.6 * stop_time

    # Monitors
    monitor_span = 0.95*run.span_xy
    z_tr = 0.5 * cell_z - run.dpml_z - 0.2
    dft_plane_xy = mp.Volume(
        center=mp.Vector3(0, 0, z_tr),
        size=mp.Vector3(monitor_span, monitor_span, 0)
    )
    dft_plane_xz = mp.Volume(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(monitor_span, 0, cell_z - 2.05 * run.dpml_z)
    )

    dft_freqs = [freq_p1, freq_p2, freq_probe, freq_sb_minus, freq_sb_plus]
    fixed_freqs = np.array(dft_freqs, dtype=float)
    nfreq_probe = 15
    probe_freqs = np.linspace(
        freq_probe - 0.5 * df_probe, freq_probe + 0.5 * df_probe, nfreq_probe
    )
    k_probe_center = nfreq_probe // 2

    dft_fields = simulation.add_dft_fields(
        [mp.Ex, mp.Ey], dft_freqs, where=dft_plane_xy
    )
    trans_monitor = simulation.add_dft_fields(
        [mp.Ex, mp.Ey], freq_probe, df_probe, nfreq_probe, where=dft_plane_xy
    )
    dft_fields_xz = (
        simulation.add_dft_fields([mp.Ex, mp.Ey], dft_freqs, where=dft_plane_xz)
        if run.capture_fields
        else None
    )

    # TODO: visualization
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
        "fixed": {"Ex": [], "Ey": [], "absE": []},
        "probe_band": {"Ex": [], "Ey": [], "absE": []},
        "probe_pol": {"theta_deg": [], "Ix": [], "Iy": [], "theta_deg_t": []},
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
    plane_size = mp.Vector3(src_span, src_span, 0)
    plane_center = mp.Vector3(0, 0, z_tr)

    def plane_avg_mag(arr_ex: np.ndarray, arr_ey: np.ndarray) -> float:
        mag = np.sqrt(np.abs(arr_ex) ** 2 + np.abs(arr_ey) ** 2)
        return float(np.mean(mag))

    def stokes_theta_deg(ex_arr: np.ndarray, ey_arr: np.ndarray) -> float:
        s1 = np.mean(np.abs(ex_arr) ** 2 - np.abs(ey_arr) ** 2)
        s2 = 2.0 * np.mean(np.real(ex_arr * np.conjugate(ey_arr)))
        theta = 0.5 * np.arctan2(s2, s1)
        return float(np.degrees(theta))

    def sample_callback(sim: mp.Simulation) -> None:
        t = sim.meep_time()
        time_trace["t"].append(t)

        # Fixed-frequency DFT monitors
        ex_vals, ey_vals, abs_vals = [], [], []
        for idx in range(len(fixed_freqs)):
            ex_arr = np.asarray(sim.get_dft_array(dft_fields, mp.Ex, idx))
            ey_arr = np.asarray(sim.get_dft_array(dft_fields, mp.Ey, idx))
            ex_vals.append(np.mean(ex_arr))
            ey_vals.append(np.mean(ey_arr))
            abs_vals.append(plane_avg_mag(ex_arr, ey_arr))
        time_trace["fixed"]["Ex"].append(np.array(ex_vals))
        time_trace["fixed"]["Ey"].append(np.array(ey_vals))
        time_trace["fixed"]["absE"].append(np.array(abs_vals))

        # Probe-band DFT monitor
        ex_pb, ey_pb, abs_pb = [], [], []
        for k in range(nfreq_probe):
            ex_arr = np.asarray(sim.get_dft_array(trans_monitor, mp.Ex, k))
            ey_arr = np.asarray(sim.get_dft_array(trans_monitor, mp.Ey, k))
            ex_pb.append(np.mean(ex_arr))
            ey_pb.append(np.mean(ey_arr))
            abs_pb.append(plane_avg_mag(ex_arr, ey_arr))
        time_trace["probe_band"]["Ex"].append(np.array(ex_pb))
        time_trace["probe_band"]["Ey"].append(np.array(ey_pb))
        time_trace["probe_band"]["absE"].append(np.array(abs_pb))

        # Probe polarization angle at center frequency
        ex_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Ex, k_probe_center))
        ey_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Ey, k_probe_center))
        theta_deg = stokes_theta_deg(ex_c, ey_c)
        time_trace["probe_pol"]["theta_deg"].append(theta_deg)
        time_trace["probe_pol"]["Ix"].append(float(np.mean(np.abs(ex_c) ** 2)))
        time_trace["probe_pol"]["Iy"].append(float(np.mean(np.abs(ey_c) ** 2)))

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
        # TODO
        td_env["theta_deg_t"].append(stokes_theta_deg(
            (eplus_env[2] + eminus_env[2]) / np.sqrt(2),
            1j * (- eplus_env[2] + eminus_env[2]) / np.sqrt(2)))

        # Optional snapshot near snapshot_time
        if (
            run.capture_fields
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
            run.capture_fields
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

    probe_sample_point = mp.Vector3(0, 0, z_tr + 0.1)
    # simulation.run(
    #     mp.at_every(run.sample_dt, sample_callback),
    #     until_after_sources=mp.stop_when_fields_decayed(
    #         50, mp.Ex, probe_sample_point, 1e-9
    #     ),
    # )
    simulation.run(
        mp.at_every(run.sample_dt, sample_callback), until=stop_time
    )

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #
    t_arr = np.array(time_trace["t"])
    fixed_ex = np.vstack(time_trace["fixed"]["Ex"])
    fixed_ey = np.vstack(time_trace["fixed"]["Ey"])
    fixed_abs = np.vstack(time_trace["fixed"]["absE"])

    probe_ex = np.vstack(time_trace["probe_band"]["Ex"])
    probe_ey = np.vstack(time_trace["probe_band"]["Ey"])
    probe_abs = np.vstack(time_trace["probe_band"]["absE"])
    theta_deg = np.array(time_trace["probe_pol"]["theta_deg"])
    theta_deg_rel = theta_deg - INIT_PROBE_POLARIZATION_DEG

    t_td = np.array(td_env["t"])
    epl_td = np.vstack(td_env["Eplus"])
    emi_td = np.vstack(td_env["Eminus"])
    theta_deg_t = np.array(td_env["theta_deg_t"])
    theta_deg_t_rel = theta_deg_t - INIT_PROBE_POLARIZATION_DEG

    epl_dft = (fixed_ex + 1j * fixed_ey) / np.sqrt(2.0)
    emi_dft = (fixed_ex - 1j * fixed_ey) / np.sqrt(2.0)

    i_p1, i_p2, i_probe, i_sb_minus, i_sb_plus = range(5)

    plot_paths: Dict[str, str] = {}

    fig = plt.figure(figsize=(7.2, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_arr, np.abs(emi_dft[:, i_p1]), label=f"pump1 DFT |e-| (f={fixed_freqs[i_p1]:.3f})")
    ax.plot(t_arr, np.abs(epl_dft[:, i_p2]), label=f"pump2 DFT |e+| (f={fixed_freqs[i_p2]:.3f})")
    ax.set_xlabel("time (Meep units)")
    ax.set_ylabel("|E| (plane avg, DFT)")
    ax.set_title("Pumps (DFT monitors)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_paths["pumps_dft"] = str(save_figure(fig, "pumps_dft.png", output_dir))

    fig = plt.figure(figsize=(7.2, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_td, np.abs(emi_td[:, i_p1]), label="pump1 TD |e-|")
    ax.plot(t_td, np.abs(epl_td[:, i_p2]), label="pump2 TD |e+|")
    ax.set_xlabel("time (Meep units)")
    ax.set_ylabel("|E| (plane avg, TD demod)")
    ax.set_title("Pumps (time-domain demod)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plot_paths["pumps_td"] = str(save_figure(fig, "pumps_td.png", output_dir))

    def plot_pair(ax_, label: str, idx: int) -> None:
        ax_.plot(t_arr, np.abs(epl_dft[:, idx]), "-", label=f"{label} DFT |e+|")
        ax_.plot(t_arr, np.abs(emi_dft[:, idx]), "--", label=f"{label} DFT |e-|")

    fig = plt.figure(figsize=(7.6, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_pair(ax, "probe", i_probe)
    plot_pair(ax, "sb-", i_sb_minus)
    plot_pair(ax, "sb+", i_sb_plus)
    ax.set_xlabel("time (Meep units)")
    ax.set_ylabel("|E| (plane avg, DFT)")
    ax.set_title("Probe & sidebands (DFT monitors)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plot_paths["probe_dft"] = str(save_figure(fig, "probe_dft.png", output_dir))

    def plot_pair_td(ax_, label: str, idx: int) -> None:
        ax_.plot(t_td, np.abs(epl_td[:, idx]), "-", label=f"{label} TD |e+|")
        ax_.plot(t_td, np.abs(emi_td[:, idx]), "--", label=f"{label} TD |e-|")

    fig = plt.figure(figsize=(7.6, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    plot_pair_td(ax, "probe", i_probe)
    plot_pair_td(ax, "sb-", i_sb_minus)
    plot_pair_td(ax, "sb+", i_sb_plus)
    ax.set_xlabel("time (Meep units)")
    ax.set_ylabel("|E| (plane avg, TD demod)")
    ax.set_title("Probe & sidebands (time-domain demod)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    plot_paths["probe_td"] = str(save_figure(fig, "probe_td.png", output_dir))

    fig = plt.figure(figsize=(7.4, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        np.abs(probe_abs.T),
        aspect="auto",
        origin="lower",
        extent=[t_arr.min(), t_arr.max(), probe_freqs.min(), probe_freqs.max()],
    )
    fig.colorbar(im, ax=ax, label=r"$\langle |E| \rangle$")
    ax.set_xlabel("time (Meep units)")
    ax.set_ylabel("frequency (1/μm)")
    ax.set_title("Probe-band plane-avg |E| vs (f, t)")
    plot_paths["probe_band_heatmap"] = str(
        save_figure(fig, "probe_band_heatmap.png", output_dir)
    )

    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_arr, theta_deg_rel, "k-")
    ax.set_xlabel("time (Meep units)")
    ax.set_ylabel("polarization rotation (deg)")
    ax.set_title("Probe polarization angle vs time (relative to input)")
    ax.grid(True, alpha=0.3)
    plot_paths["probe_rotation"] = str(
        save_figure(fig, "probe_polarization.png", output_dir)
    )

    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_arr, theta_deg_t_rel, "k-")
    ax.set_xlabel("time (Meep units)")
    ax.set_ylabel("polarization rotation (deg)")
    ax.set_title("Probe polarization angle vs time (relative to input) in time-domain")
    ax.grid(True, alpha=0.3)
    plot_paths["probe_rotation_td"] = str(
        save_figure(fig, "probe_polarization_td.png", output_dir)
    )

    if run.capture_fields and xz_snapshot["taken"]:
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
            ax_subplot.set_title(f"|E|(x,z) at f={freq:.3f}, t≈{xz_snapshot['t']:.2f}")
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
    if run.capture_fields and xz_td_snapshot["taken"]:
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
        ax.set_title(f"Instantaneous |E|(x,z) at t≈{xz_td_snapshot['t']:.2f}")
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
    }
    wavelengths = {
        "pump1": float(lam_p1),
        "pump2": float(lam_p2),
        "probe": float(lam_probe),
    }

    abs_epl_dft = np.abs(epl_dft)
    abs_emi_dft = np.abs(emi_dft)
    abs_epl_td = np.abs(epl_td)
    abs_emi_td = np.abs(emi_td)
    probe_band_abs = np.abs(probe_abs)

    probe_rotation_trace = ProbeRotationTrace(
        time=t_arr.copy(),
        theta_deg_rel=theta_deg_rel.copy(),
        final_deg=float(theta_deg_rel[-1]),
        min_deg=float(theta_deg_rel.min()),
        max_deg=float(theta_deg_rel.max()),
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
        "probe_band_frequencies_inv_um": probe_freqs.tolist(),
        "monitor_plane_z_um": float(z_tr),
        "probe_rotation_deg": {
            "initial_deg": INIT_PROBE_POLARIZATION_DEG,
            "final_relative_deg": float(theta_deg_rel[-1]),
            "max_relative_deg": float(theta_deg_rel.max()),
            "min_relative_deg": float(theta_deg_rel.min()),
        },
        "plot_paths": plot_paths,
        "theta_deg_rel_I": {
            "pump_intensity_w_cm2": float(run.pump_intensity_w_cm2),
            "final_relative_deg": float(theta_deg_rel[-1]),
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
    parser = argparse.ArgumentParser(description="3D Faraday rotation simulation.")
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
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store plots and the summary JSON.",
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
