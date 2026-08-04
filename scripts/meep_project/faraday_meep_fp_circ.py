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
from nonlinear_materials import (
    canonical_high_index_material,
    chi3_si_to_meep_e_chi3,
    get_high_index_preset,
    high_index_material_choices,
    n2_to_chi3_si,
    resolve_high_index_index,
    resolve_high_index_kappa,
    resolve_high_index_n2,
)

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
    theta_total_deg_rel: np.ndarray | None = None


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
    start_time: float = 0.0,
) -> List[mp.Source]:
    """Return (Ex, Ey) sources for circular polarization.

    ``start_time`` (Meep time units) turns the pulse on later; the Gaussian peak sits at
    ``start_time + cutoff/fwidth``, so a shift of the start time shifts the peak by the
    same amount. Used for pump-probe delay scans.
    """
    phase = 1.0j if handedness == "plus" else -1.0j
    amp = amplitude / np.sqrt(2.0)
    base = mp.GaussianSource(
        frequency=frequency, fwidth=fwidth, cutoff=cutoff, start_time=float(start_time)
    )
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


def probe_jones_vector(azimuth_deg: float, ellipticity_deg: float) -> Tuple[complex, complex]:
    """Unit-norm Jones vector (Ex, Ey) for an ellipse of azimuth psi and ellipticity chi.

    E = R(psi) . (cos chi, i sin chi), so |Ex|^2 + |Ey|^2 = 1 and, in the Stokes
    convention used by ``stokes_metrics``, the launched state has theta = psi and
    chi = chi. ``(45, 0)`` reproduces the historical 45-degree linear probe exactly.
    """
    psi = np.radians(float(azimuth_deg))
    chi = np.radians(float(ellipticity_deg))
    ex = np.cos(psi) * np.cos(chi) - 1.0j * np.sin(psi) * np.sin(chi)
    ey = np.sin(psi) * np.cos(chi) + 1.0j * np.cos(psi) * np.sin(chi)
    return complex(ex), complex(ey)


def linear_sources_45deg(
    frequency: float,
    fwidth: float,
    cutoff: float,
    amplitude: float,
    src_center: mp.Vector3,
    src_span: float,
    include_ey: bool = True,
    start_time: float = 0.0,
    azimuth_deg: float = INIT_PROBE_POLARIZATION_DEG,
    ellipticity_deg: float = 0.0,
) -> List[mp.Source]:
    """Probe sources. Defaults give the 45-degree linear probe used historically;
    ``azimuth_deg``/``ellipticity_deg`` launch a general elliptical state instead."""
    jx, jy = probe_jones_vector(azimuth_deg, ellipticity_deg)
    base = mp.GaussianSource(
        frequency=frequency, fwidth=fwidth, cutoff=cutoff, start_time=float(start_time)
    )
    size = mp.Vector3() if src_span <= 0 else mp.Vector3(src_span, src_span, 0)
    if not include_ey:
        return [
            mp.Source(
                src=base, component=mp.Ex, center=src_center, size=size,
                amplitude=amplitude * jx,
            )
        ]
    return [
        mp.Source(
            src=base, component=mp.Ex, center=src_center, size=size,
            amplitude=amplitude * jx,
        ),
        mp.Source(
            src=base, component=mp.Ey, center=src_center, size=size,
            amplitude=amplitude * jy,
        ),
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
    if getattr(args, "pulse_duration_fs", None) is not None:
        run.pulse_duration_fs = float(args.pulse_duration_fs)
    dimension = int(getattr(args, "dim", 1))
    if dimension not in (1, 3):
        raise ValueError("--dim must be either 1 or 3.")
    is_quasi_1d = dimension == 1
    # Meep strict 1D does not support the dual-polarization source set used here.
    # Keep an effective 1D model by collapsing transverse cell extents while using
    # the 3D field solver so Ex/Ey (and Hx/Hy for forward-wave separation) remain available.
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

    high_index_material = canonical_high_index_material(
        getattr(args, "high_index_material", "sin")
    )
    # The cavity slot and the mirror high-index slot are the same label for every geometry that
    # predates the 2026-08 SiC samples. They differ when the cavity is a distinct material
    # (SiC cavity inside SiN/SiO2 mirrors), so resolve them separately: the mirror high-index
    # layers are the mirror layers that are not the low-index (SiO2) material.
    cav_slot = str(spec.get("cavity", {}).get("mat", "SiN"))
    _mirror_mats = {
        str(layer["mat"])
        for side in ("left", "right")
        for layer in (spec.get("mirrors", {}) or {}).get(side, []) or []
    }
    _mirror_high = sorted(m for m in _mirror_mats if m != "SiO2")
    mirror_high_slot = _mirror_high[0] if len(_mirror_high) == 1 else cav_slot
    high_slot = mirror_high_slot
    default_high = float(getattr(materials.get(high_slot), "index", np.nan))
    if (not np.isfinite(default_high)) or default_high <= 0.0:
        default_high = resolve_high_index_index(None, high_index_material)
    if high_index_material != "sin" and getattr(args, "nH", None) is None:
        default_high = resolve_high_index_index(None, high_index_material)
    n_high = resolve_high_index_index(getattr(args, "nH", None), high_index_material)
    k_high = resolve_high_index_kappa(getattr(args, "kH", None), high_index_material)
    n2_high = resolve_high_index_n2(getattr(args, "high_index_n2", None), high_index_material)

    default_low = float(getattr(materials.get("SiO2"), "index", 1.45))
    n_low = float(args.nL if args.nL is not None else default_low)
    mat_sin, mat_sio2 = get_cavity_materials(
        model=args.materials,
        index_high=n_high if args.nH is not None else default_high,
        kappa_high=k_high,
        index_low=n_low,
        high_index_material=high_index_material,
        kappa_ref_wavelength_um=float(args.kappa_ref_lambda),
        sin_csv=args.sin_fit,
        sio2_csv=args.sio2_fit,
        lam_min=args.fit_window[0],
        lam_max=args.fit_window[1],
        fit_poles=args.fit_poles,
    )
    # Cavity medium: the same object as the mirrors unless --cavity-material asks otherwise.
    cavity_material_arg = getattr(args, "cavity_material", None)
    if cavity_material_arg is None:
        cavity_material = high_index_material
        mat_cav = mat_sin
        n2_cav = n2_high
    else:
        cavity_material = canonical_high_index_material(cavity_material_arg)
        default_cav = float(getattr(materials.get(cav_slot), "index", np.nan))
        if (not np.isfinite(default_cav)) or default_cav <= 0.0:
            default_cav = resolve_high_index_index(None, cavity_material)
        if cavity_material != "sin" and getattr(args, "n_cav", None) is None:
            default_cav = resolve_high_index_index(None, cavity_material)
        n_cav = resolve_high_index_index(getattr(args, "n_cav", None), cavity_material)
        k_cav = resolve_high_index_kappa(getattr(args, "k_cav", None), cavity_material)
        n2_cav = resolve_high_index_n2(getattr(args, "cavity_n2", None), cavity_material)
        mat_cav, _ = get_cavity_materials(
            model=args.materials,
            index_high=n_cav if getattr(args, "n_cav", None) is not None else default_cav,
            kappa_high=k_cav,
            index_low=n_low,
            high_index_material=cavity_material,
            kappa_ref_wavelength_um=float(args.kappa_ref_lambda),
            sin_csv=(getattr(args, "cavity_fit", None) or args.sin_fit),
            sio2_csv=args.sio2_fit,
            lam_min=args.fit_window[0],
            lam_max=args.fit_window[1],
            fit_poles=args.fit_poles,
        )
    if "SiN" in materials:
        materials["SiN"] = mat_sin
    materials[high_slot] = mat_sin
    materials["SiO2"] = mat_sio2
    # Written last so a distinct cavity material wins; a no-op when cav_slot == high_slot.
    materials[cav_slot] = mat_cav
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

    # Kerr nonlinearity for selected high-index material.
    n_linear_probe = material_index_at_wavelength(mat_sin, lam_probe)
    chi3_si = n2_to_chi3_si(n2_high, n_linear_probe)
    e_chi3_meep = chi3_si_to_meep_e_chi3(
        chi3_si, scale_e=SCALE_E, nonlinear_scale=run.nonlinear_scale
    )
    mat_sin.E_chi3_diag = mp.Vector3(e_chi3_meep, e_chi3_meep, e_chi3_meep)
    # A distinct cavity medium gets its OWN chi3 from its own n2. The mirror high-index layers
    # stay nonlinear either way -- in the fabricated SiC samples the SiN mirrors still have
    # n2 = 5e-19, they are just ~10x weaker than the SiC cavity. Mutating the Medium after
    # build_geometry_from_spec is fine: the mp.Blocks hold references to these same objects.
    if mat_cav is not mat_sin:
        n_linear_cav = material_index_at_wavelength(mat_cav, lam_probe)
        chi3_si_cav = n2_to_chi3_si(n2_cav, n_linear_cav)
        e_chi3_cav = chi3_si_to_meep_e_chi3(
            chi3_si_cav, scale_e=SCALE_E, nonlinear_scale=run.nonlinear_scale
        )
        mat_cav.E_chi3_diag = mp.Vector3(e_chi3_cav, e_chi3_cav, e_chi3_cav)
    else:
        n_linear_cav, chi3_si_cav, e_chi3_cav = n_linear_probe, chi3_si, e_chi3_meep
    high_preset = get_high_index_preset(high_index_material)
    n_monitor_medium = float(material_index_at_wavelength(materials["SiO2"], lam_probe))

    n_source_medium = 1.0  # sources are injected in air

    if args.pump_intensity is not None:
        run.pump_intensity_w_cm2 = args.pump_intensity
    if getattr(args, "probe_intensity", None) is not None:
        run.probe_intensity_w_cm2 = float(args.probe_intensity)
    # Probe polarization state and pump-probe delay (all default to the historical setup:
    # 45-degree linear probe, all three pulses coincident, balanced pumps).
    init_pol_deg = float(getattr(args, "probe_azimuth_deg", INIT_PROBE_POLARIZATION_DEG))
    probe_ellipticity_deg = float(getattr(args, "probe_ellipticity_deg", 0.0) or 0.0)
    # NB: not `... or 1.0` -- 0.0 is falsy, and 0.0 is a legitimate value here (it switches
    # pump2 off, which is exactly the single-pump control needed to test whether the balanced
    # sigma+/sigma- pair really cancels the direct chi3 term).
    _pump_imbalance = getattr(args, "pump_imbalance", None)
    pump_imbalance = 1.0 if _pump_imbalance is None else float(_pump_imbalance)

    # Delay of pump1 relative to (pump2, probe), which stay locked together. Both branches
    # are shifted to keep every source causal (start_time >= 0) while preserving tau.
    fs_to_meep = C0 / 1e9  # 1 fs of light travel, in Meep time units (a = 1 um)
    pump1_delay_fs = float(getattr(args, "pump1_delay_fs", 0.0) or 0.0)
    tau_meep = pump1_delay_fs * fs_to_meep
    delay_pad_fs = getattr(args, "delay_pad_fs", None)
    if delay_pad_fs is None:
        # Legacy fallback: absorb a negative delay by shifting pump2+probe instead. This
        # preserves the relative TIMING but not the relative optical PHASE (pump2/probe pick
        # up exp(i w2 |tau|), exp(i ws |tau|) instead of pump1 picking up exp(-i w1 |tau|)),
        # so tau<0 is not the same experiment as tau>0. Kept only for reproducibility of
        # earlier scans; pass --delay-pad-fs for a physically consistent scan.
        t_start_pump1 = max(0.0, tau_meep)
        t_start_rest = max(0.0, -tau_meep)
    else:
        pad_meep = float(delay_pad_fs) * fs_to_meep
        t_start_pump1 = pad_meep + tau_meep
        t_start_rest = pad_meep
        if t_start_pump1 < 0.0:
            raise SystemExit(
                "--delay-pad-fs {:.3f} is too small for --pump1-delay-fs {:.3f}: pump1 would "
                "start before t=0. Use a pad >= the largest |negative delay| in the scan, and "
                "keep it FIXED across the scan.".format(float(delay_pad_fs), pump1_delay_fs)
            )

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
    pad_sub_um = float(spec.get("pads", {}).get("substrate_um", 0.0))
    z_right_edge_no_pml = 0.5 * cell_z - run.dpml_z
    if pad_sub_um > 1e-9:
        z_tr = z_right_edge_no_pml - 0.5 * pad_sub_um
    else:
        z_tr = z_right_edge_no_pml - 0.2
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
    fixed_labels = ["pump1", "pump2", "probe", "sb_minus", "sb_plus"]
    nfreq_probe = 15
    probe_freqs = np.linspace(
        freq_probe - 0.5 * df_probe, freq_probe + 0.5 * df_probe, nfreq_probe
    )
    k_probe_center = nfreq_probe // 2
    diagnostics_enabled = bool(getattr(args, "enable_nonlinear_diagnostics", False))
    diagnostic_scan_points = max(5, int(getattr(args, "diagnostic_scan_points", 41)))
    diagnostic_scan_span_factor = max(
        0.05, float(getattr(args, "diagnostic_scan_span_factor", 0.75))
    )
    diagnostic_cavity_span_fraction = float(
        np.clip(getattr(args, "diagnostic_cavity_span_fraction", 0.9), 0.05, 1.0)
    )

    def build_pump1_sources(amp: float) -> List[mp.Source]:
        return circular_sources(
            freq_p1, df_pump1, run.pump_cutoff, amp, "plus", src_center, src_span,
            include_ey=track_ey, start_time=t_start_pump1,
        )

    def build_pump2_sources(amp: float) -> List[mp.Source]:
        return circular_sources(
            freq_p2, df_pump2, run.pump_cutoff, amp, "minus", src_center, src_span,
            include_ey=track_ey, start_time=t_start_rest,
        )

    def build_probe_sources(amp: float) -> List[mp.Source]:
        return linear_sources_45deg(
            freq_probe, df_probe, run.pump_cutoff, amp, src_center, src_span,
            include_ey=track_ey, start_time=t_start_rest,
            azimuth_deg=init_pol_deg, ellipticity_deg=probe_ellipticity_deg,
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

    # Deliberate pump imbalance (intensity ratio P2/P1). 1.0 keeps the balanced sigma+/sigma-
    # configuration that nulls the direct chi3 carrier term.
    pump_amp2 *= float(np.sqrt(max(pump_imbalance, 0.0)))

    sources: List[mp.Source] = []
    sources += build_pump1_sources(pump_amp1)
    sources += build_pump2_sources(pump_amp2)
    sources += build_probe_sources(probe_amp)

    courant_val = float(getattr(args, "courant", None) or 0.5)
    simulation = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=boundary_layers,
        resolution=run.resolution,
        dimensions=simulation_dimensions,
        default_material=mp.air,
        force_complex_fields=True,
        Courant=courant_val,
    )

    monitor_components = [mp.Ex, mp.Ey]
    dft_fields = simulation.add_dft_fields(
        monitor_components, dft_freqs, where=dft_plane_xy
    )
    probe_monitor_components = [mp.Ex, mp.Ey, mp.Hx, mp.Hy]
    trans_monitor = simulation.add_dft_fields(
        probe_monitor_components, freq_probe, df_probe, nfreq_probe, where=dft_plane_xy
    )
    dft_fields_xz = (
        simulation.add_dft_fields(monitor_components, dft_freqs, where=dft_plane_xz)
        if capture_spatial_fields and dft_plane_xz is not None
        else None
    )
    cavity_dft_fields = None
    cavity_monitor_volume = None
    cavity_scan_freqs: Dict[str, np.ndarray] = {}
    cavity_scan_monitors: Dict[str, Any] = {}
    if diagnostics_enabled:
        cavity_length_um = max(float(spec.get("cavity", {}).get("L_um", 0.0)), 1e-6)
        if is_quasi_1d:
            cavity_monitor_volume = mp.Volume(
                center=mp.Vector3(0, 0, cavity_center),
                size=mp.Vector3(0, 0, cavity_length_um),
            )
        else:
            cavity_span = diagnostic_cavity_span_fraction * max(float(run.span_xy), 1e-6)
            cavity_monitor_volume = mp.Volume(
                center=mp.Vector3(0, 0, cavity_center),
                size=mp.Vector3(cavity_span, cavity_span, cavity_length_um),
            )
        cavity_dft_fields = simulation.add_dft_fields(
            monitor_components, dft_freqs, where=cavity_monitor_volume
        )

        def build_hot_scan(center_freq: float, bandwidth: float) -> np.ndarray:
            half_span = max(0.5 * diagnostic_scan_span_factor * float(bandwidth), 0.0025 * center_freq)
            f_lo = max(center_freq - half_span, 1e-6)
            f_hi = center_freq + half_span
            return np.linspace(f_lo, f_hi, diagnostic_scan_points, dtype=float)

        cavity_scan_freqs = {
            "pump1": build_hot_scan(freq_p1, df_pump1),
            "pump2": build_hot_scan(freq_p2, df_pump2),
            "probe": build_hot_scan(freq_probe, df_probe),
        }
        for label, scan_freqs in cavity_scan_freqs.items():
            cavity_scan_monitors[label] = simulation.add_dft_fields(
                monitor_components, scan_freqs.tolist(), where=cavity_monitor_volume
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
        "cavity_fixed": {
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
            "Ex_fwd_mean": [],
            "Ey_fwd_mean": [],
            "Ix": [],
            "Iy": [],
            "S0": [],
            "S0_backward": [],
            "forward_fraction": [],
            "S1": [],
            "S2": [],
            "S3": [],
            "chi_deg": [],
            "dolp": [],
            "docp": [],
            # Total-field (no forward/backward split) Stokes diagnostics, for comparison
            # against the forward-isolated angle above.
            "theta_total_deg": [],
            "S0_total": [],
            "dolp_total": [],
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

    def total_field_rms(arr_ex: np.ndarray, arr_ey: np.ndarray) -> float:
        mag_sq = np.abs(arr_ex) ** 2 + np.abs(arr_ey) ** 2
        return float(np.sqrt(np.mean(mag_sq)))

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

    def forward_transverse_fields(
        ex_arr: np.ndarray | complex,
        ey_arr: np.ndarray | complex,
        hx_arr: np.ndarray | complex,
        hy_arr: np.ndarray | complex,
        n_medium: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Decompose total transverse fields into forward/backward waves for z-propagation.
        For +z in nonmagnetic medium: Hy=n*Ex and Hx=-n*Ey.
        """
        ex = np.asarray(ex_arr)
        ey = np.asarray(ey_arr)
        hx = np.asarray(hx_arr)
        hy = np.asarray(hy_arr)
        n_use = max(float(n_medium), 1e-9)
        ex_fwd = 0.5 * (ex + hy / n_use)
        ex_bwd = 0.5 * (ex - hy / n_use)
        ey_fwd = 0.5 * (ey - hx / n_use)
        ey_bwd = 0.5 * (ey + hx / n_use)
        return ex_fwd, ey_fwd, ex_bwd, ey_bwd

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

        if cavity_dft_fields is not None:
            cavity_ex_vals, cavity_ey_vals, cavity_abs_vals = [], [], []
            cavity_eplus_rms_vals, cavity_eminus_rms_vals = [], []
            for idx in range(len(fixed_freqs)):
                ex_arr = np.asarray(sim.get_dft_array(cavity_dft_fields, mp.Ex, idx))
                ey_arr = np.asarray(sim.get_dft_array(cavity_dft_fields, mp.Ey, idx))
                cavity_ex_vals.append(complex(np.mean(ex_arr)))
                cavity_ey_vals.append(complex(np.mean(ey_arr)))
                cavity_abs_vals.append(total_field_rms(ex_arr, ey_arr))
                eplus_rms, eminus_rms = circular_rms(ex_arr, ey_arr)
                cavity_eplus_rms_vals.append(eplus_rms)
                cavity_eminus_rms_vals.append(eminus_rms)
            time_trace["cavity_fixed"]["Ex"].append(np.array(cavity_ex_vals))
            time_trace["cavity_fixed"]["Ey"].append(np.array(cavity_ey_vals))
            time_trace["cavity_fixed"]["absE"].append(np.array(cavity_abs_vals))
            time_trace["cavity_fixed"]["Eplus_rms"].append(np.array(cavity_eplus_rms_vals))
            time_trace["cavity_fixed"]["Eminus_rms"].append(np.array(cavity_eminus_rms_vals))

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
        hx_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Hx, k_probe_center))
        hy_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Hy, k_probe_center))
        ex_fwd_c, ey_fwd_c, ex_bwd_c, ey_bwd_c = forward_transverse_fields(
            ex_c, ey_c, hx_c, hy_c, n_medium=n_monitor_medium
        )
        stokes_c = stokes_metrics(ex_fwd_c, ey_fwd_c)
        theta_deg = stokes_c["theta_deg"]
        # Same Stokes angle on the raw total field (forward + backward, no split):
        # the "naive detector" reading, for comparison against the forward-isolated one.
        stokes_total_c = stokes_metrics(ex_c, ey_c)
        ix_c = float(np.mean(np.abs(ex_fwd_c) ** 2))
        iy_c = float(np.mean(np.abs(ey_fwd_c) ** 2))
        s0_bwd = float(np.mean(np.abs(ex_bwd_c) ** 2 + np.abs(ey_bwd_c) ** 2))
        s0_fwd = float(stokes_c["S0"])
        s0_tot = max(s0_fwd + s0_bwd, 1e-30)
        forward_fraction = float(s0_fwd / s0_tot)
        time_trace["probe_pol"]["theta_deg"].append(theta_deg)
        time_trace["probe_pol"]["Ex_fwd_mean"].append(complex(np.mean(ex_fwd_c)))
        time_trace["probe_pol"]["Ey_fwd_mean"].append(complex(np.mean(ey_fwd_c)))
        time_trace["probe_pol"]["Ix"].append(ix_c)
        time_trace["probe_pol"]["Iy"].append(iy_c)
        time_trace["probe_pol"]["S0"].append(stokes_c["S0"])
        time_trace["probe_pol"]["S0_backward"].append(s0_bwd)
        time_trace["probe_pol"]["forward_fraction"].append(forward_fraction)
        time_trace["probe_pol"]["S1"].append(stokes_c["S1"])
        time_trace["probe_pol"]["S2"].append(stokes_c["S2"])
        time_trace["probe_pol"]["S3"].append(stokes_c["S3"])
        time_trace["probe_pol"]["chi_deg"].append(stokes_c["chi_deg"])
        time_trace["probe_pol"]["dolp"].append(stokes_c["dolp"])
        time_trace["probe_pol"]["docp"].append(stokes_c["docp"])
        time_trace["probe_pol"]["theta_total_deg"].append(stokes_total_c["theta_deg"])
        time_trace["probe_pol"]["S0_total"].append(stokes_total_c["S0"])
        time_trace["probe_pol"]["dolp_total"].append(stokes_total_c["dolp"])

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
    # Pulse-energy-integrated probe Stokes vector.
    #
    # The run-accumulated DFT over the probe band is, by Parseval, the time integral of the
    # transmitted probe over the whole pulse -- i.e. what a balanced detector that integrates
    # the pulse actually reports (S_V - S_H = -S1). This is the correct observable for a
    # pump-probe delay scan; the tail/final-window estimators below are settled-state
    # measures and are not comparable across delays.
    # ------------------------------------------------------------------ #
    def pulse_integrated_probe_stokes() -> Dict[str, float]:
        ex_parts: List[np.ndarray] = []
        ey_parts: List[np.ndarray] = []
        for k in range(nfreq_probe):
            ex_k = np.asarray(simulation.get_dft_array(trans_monitor, mp.Ex, k))
            ey_k = np.asarray(simulation.get_dft_array(trans_monitor, mp.Ey, k))
            hx_k = np.asarray(simulation.get_dft_array(trans_monitor, mp.Hx, k))
            hy_k = np.asarray(simulation.get_dft_array(trans_monitor, mp.Hy, k))
            ex_f, ey_f, _, _ = forward_transverse_fields(
                ex_k, ey_k, hx_k, hy_k, n_monitor_medium
            )
            ex_parts.append(np.ravel(np.atleast_1d(ex_f)))
            ey_parts.append(np.ravel(np.atleast_1d(ey_f)))
        st = stokes_metrics(np.concatenate(ex_parts), np.concatenate(ey_parts))
        s0 = max(float(st["S0"]), 1e-30)
        st["rotation_deg"] = float(
            wrap_linear_polarization_deg(float(st["theta_deg"]) - init_pol_deg)
        )
        # Balanced-detector signal for a probe launched along the +45 deg diagonal.
        st["balanced_V_minus_H"] = -float(st["S1"])
        st["balanced_V_minus_H_norm"] = -float(st["S1"]) / s0
        st["ellipticity_change_deg"] = float(st["chi_deg"]) - probe_ellipticity_deg
        return {k: float(v) for k, v in st.items()}

    probe_pulse_integrated = pulse_integrated_probe_stokes()

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #
    t_arr = np.array(time_trace["t"])
    fixed_ex = np.vstack(time_trace["fixed"]["Ex"])
    fixed_ey = np.vstack(time_trace["fixed"]["Ey"])
    fixed_abs = np.vstack(time_trace["fixed"]["absE"])
    fixed_eplus_rms = np.vstack(time_trace["fixed"]["Eplus_rms"])
    fixed_eminus_rms = np.vstack(time_trace["fixed"]["Eminus_rms"])
    if diagnostics_enabled and time_trace["cavity_fixed"]["Ex"]:
        cavity_ex = np.vstack(time_trace["cavity_fixed"]["Ex"])
        cavity_ey = np.vstack(time_trace["cavity_fixed"]["Ey"])
        cavity_abs = np.vstack(time_trace["cavity_fixed"]["absE"])
        cavity_eplus_rms = np.vstack(time_trace["cavity_fixed"]["Eplus_rms"])
        cavity_eminus_rms = np.vstack(time_trace["cavity_fixed"]["Eminus_rms"])
    else:
        cavity_ex = None
        cavity_ey = None
        cavity_abs = None
        cavity_eplus_rms = None
        cavity_eminus_rms = None

    probe_ex = np.vstack(time_trace["probe_band"]["Ex"])
    probe_ey = np.vstack(time_trace["probe_band"]["Ey"])
    probe_abs = np.vstack(time_trace["probe_band"]["absE"])
    probe_eplus_rms = np.vstack(time_trace["probe_band"]["Eplus_rms"])
    probe_eminus_rms = np.vstack(time_trace["probe_band"]["Eminus_rms"])
    theta_deg_wrapped = np.array(time_trace["probe_pol"]["theta_deg"])
    theta_deg_unwrapped = unwrap_linear_polarization_deg(theta_deg_wrapped)
    theta_deg_rel_unwrapped = theta_deg_unwrapped - init_pol_deg
    theta_deg_rel = wrap_linear_polarization_deg(
        theta_deg_wrapped - init_pol_deg
    )
    probe_ex_fwd_mean = np.array(time_trace["probe_pol"]["Ex_fwd_mean"], dtype=complex)
    probe_ey_fwd_mean = np.array(time_trace["probe_pol"]["Ey_fwd_mean"], dtype=complex)
    probe_ix = np.array(time_trace["probe_pol"]["Ix"])
    probe_iy = np.array(time_trace["probe_pol"]["Iy"])
    probe_s0 = np.array(time_trace["probe_pol"]["S0"])
    probe_s0_backward = np.array(time_trace["probe_pol"]["S0_backward"])
    probe_forward_fraction = np.array(time_trace["probe_pol"]["forward_fraction"])
    probe_s1 = np.array(time_trace["probe_pol"]["S1"])
    probe_s2 = np.array(time_trace["probe_pol"]["S2"])
    probe_s3 = np.array(time_trace["probe_pol"]["S3"])
    probe_chi_deg = np.array(time_trace["probe_pol"]["chi_deg"])
    probe_dolp = np.array(time_trace["probe_pol"]["dolp"])
    probe_docp = np.array(time_trace["probe_pol"]["docp"])
    # Total-field (no forward/backward split) angle trace, relative to the input.
    theta_total_deg_wrapped = np.array(time_trace["probe_pol"]["theta_total_deg"])
    theta_total_deg_rel = wrap_linear_polarization_deg(
        theta_total_deg_wrapped - init_pol_deg
    )
    probe_s0_total = np.array(time_trace["probe_pol"]["S0_total"])
    probe_dolp_total = np.array(time_trace["probe_pol"]["dolp_total"])

    t_td = np.array(td_env["t"])
    epl_td = np.vstack(td_env["Eplus"])
    emi_td = np.vstack(td_env["Eminus"])
    theta_deg_t_wrapped = np.array(td_env["theta_deg_t"])
    theta_deg_t_unwrapped = unwrap_linear_polarization_deg(theta_deg_t_wrapped)
    theta_deg_t_rel_unwrapped = theta_deg_t_unwrapped - init_pol_deg
    theta_deg_t_rel = wrap_linear_polarization_deg(
        theta_deg_t_wrapped - init_pol_deg
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
    # Final value is mean over [end-M, end] valid points so the last computed sample
    # is always included.
    probe_rotation_tail_points_requested = max(
        1, int(getattr(args, "probe_rotation_tail_points", 64))
    )
    probe_rotation_window_fs = getattr(args, "probe_rotation_window_fs", None)
    if probe_rotation_window_fs is not None:
        probe_rotation_window_fs = float(probe_rotation_window_fs)
        if not np.isfinite(probe_rotation_window_fs) or probe_rotation_window_fs <= 0:
            probe_rotation_window_fs = None
    disable_strength_validity = bool(getattr(args, "disable_strength_validity", False))
    strength_threshold_rel = 0.01
    validity_policy = (
        "finite_only" if disable_strength_validity else "strength_threshold_and_finite"
    )

    def select_final_window_indices(
        values: np.ndarray, valid_mask: np.ndarray, points: int
    ) -> np.ndarray:
        vv = np.asarray(values, dtype=float)
        if vv.size == 0:
            return np.array([], dtype=int)
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.size != vv.size:
            vm = np.ones(vv.size, dtype=bool)
        idx = np.where(vm & np.isfinite(vv))[0]
        if idx.size == 0:
            idx = np.where(np.isfinite(vv))[0]
        if idx.size == 0:
            return np.array([], dtype=int)
        count = max(1, min(int(points), int(idx.size)))
        return idx[-count:]

    def weighted_arithmetic_mean(values: np.ndarray, weights: np.ndarray) -> float:
        vv = np.asarray(values, dtype=float)
        ww = np.asarray(weights, dtype=float)
        valid = np.isfinite(vv) & np.isfinite(ww) & (ww > 0)
        if np.any(valid):
            return float(np.average(vv[valid], weights=ww[valid]))
        vv = vv[np.isfinite(vv)]
        return float(np.mean(vv)) if vv.size else float("nan")

    def weighted_complex_mean(values: np.ndarray, weights: np.ndarray) -> complex:
        vv = np.asarray(values, dtype=complex)
        ww = np.asarray(weights, dtype=float)
        valid = (
            np.isfinite(vv.real)
            & np.isfinite(vv.imag)
            & np.isfinite(ww)
            & (ww > 0)
        )
        if np.any(valid):
            return complex(np.average(vv[valid], weights=ww[valid]))
        vv = vv[np.isfinite(vv.real) & np.isfinite(vv.imag)]
        return complex(np.mean(vv)) if vv.size else complex(np.nan, np.nan)

    def weighted_circular_std_linear_pol_deg(
        values_deg: np.ndarray, weights: np.ndarray
    ) -> float:
        vv = np.asarray(values_deg, dtype=float)
        ww = np.asarray(weights, dtype=float)
        valid = np.isfinite(vv) & np.isfinite(ww) & (ww > 0)
        if not np.any(valid):
            return float("nan")
        angles = np.radians(2.0 * vv[valid])
        w_use = ww[valid]
        c = np.average(np.cos(angles), weights=w_use)
        s = np.average(np.sin(angles), weights=w_use)
        r = float(np.hypot(c, s))
        r = max(min(r, 1.0), 1e-12)
        sigma_rad = np.sqrt(max(-2.0 * np.log(r), 0.0))
        return float(np.degrees(0.5 * sigma_rad))

    def resolve_window_points_from_fs(
        t_fs: np.ndarray, points_requested: int, window_fs: float | None
    ) -> int:
        if window_fs is None:
            return int(points_requested)
        tt = np.asarray(t_fs, dtype=float)
        if tt.size < 2:
            return int(points_requested)
        dt = np.diff(tt)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            return int(points_requested)
        points_from_fs = int(np.ceil(float(window_fs) / float(np.median(dt))))
        return max(1, points_from_fs)

    def complex_parts(value: complex) -> Dict[str, float]:
        vv = complex(value)
        return {
            "real": float(np.real(vv)),
            "imag": float(np.imag(vv)),
            "abs": float(np.abs(vv)),
            "phase_deg": float(np.degrees(np.angle(vv))),
        }

    def summarize_fixed_trace(
        ex_trace: np.ndarray,
        ey_trace: np.ndarray,
        eplus_rms_trace: np.ndarray,
        eminus_rms_trace: np.ndarray,
        abs_trace: np.ndarray | None,
        idx: int,
        label: str,
        weights: np.ndarray,
        tail_idx: np.ndarray,
    ) -> Dict[str, Any]:
        ex_tail = np.asarray(ex_trace[tail_idx, idx], dtype=complex)
        ey_tail = np.asarray(ey_trace[tail_idx, idx], dtype=complex)
        coherent_ex_loc = weighted_complex_mean(ex_tail, weights)
        coherent_ey_loc = weighted_complex_mean(ey_tail, weights)
        coherent_eplus, coherent_eminus = circular_components(
            coherent_ex_loc, coherent_ey_loc
        )
        payload: Dict[str, Any] = {
            "frequency_inv_um": float(fixed_freqs[idx]),
            "wavelength_um": float(1.0 / fixed_freqs[idx]) if fixed_freqs[idx] > 0 else float("nan"),
            "coherent_ex": complex_parts(coherent_ex_loc),
            "coherent_ey": complex_parts(coherent_ey_loc),
            "coherent_eplus": complex_parts(complex(np.asarray(coherent_eplus).item())),
            "coherent_eminus": complex_parts(complex(np.asarray(coherent_eminus).item())),
            "eplus_rms_tail": float(
                weighted_arithmetic_mean(
                    np.asarray(eplus_rms_trace[tail_idx, idx], dtype=float), weights
                )
            ),
            "eminus_rms_tail": float(
                weighted_arithmetic_mean(
                    np.asarray(eminus_rms_trace[tail_idx, idx], dtype=float), weights
                )
            ),
        }
        if abs_trace is not None:
            payload["field_rms_tail"] = float(
                weighted_arithmetic_mean(
                    np.asarray(abs_trace[tail_idx, idx], dtype=float), weights
                )
            )
        if label == "probe":
            e_parallel = (coherent_ex_loc + coherent_ey_loc) / np.sqrt(2.0)
            e_orth = (coherent_ex_loc - coherent_ey_loc) / np.sqrt(2.0)
            ratio = e_orth / e_parallel if np.abs(e_parallel) > 1e-30 else complex(np.nan, np.nan)
            payload["linear_basis_projection"] = {
                "parallel_45deg": complex_parts(e_parallel),
                "orthogonal_minus45deg": complex_parts(e_orth),
                "orth_over_parallel": complex_parts(ratio),
                "power_ratio_orth_over_parallel": (
                    float((np.abs(e_orth) ** 2) / max(np.abs(e_parallel) ** 2, 1e-30))
                    if np.isfinite(np.abs(e_parallel))
                    else float("nan")
                ),
            }
        return payload

    def extract_hot_scan_summary(label: str, center_freq: float) -> Dict[str, Any]:
        scan_freqs = cavity_scan_freqs[label]
        monitor = cavity_scan_monitors[label]
        amplitudes = []
        for scan_idx in range(scan_freqs.size):
            ex_arr = np.asarray(simulation.get_dft_array(monitor, mp.Ex, int(scan_idx)))
            ey_arr = np.asarray(simulation.get_dft_array(monitor, mp.Ey, int(scan_idx)))
            amplitudes.append(total_field_rms(ex_arr, ey_arr))
        scan_abs = np.asarray(amplitudes, dtype=float)
        peak_idx = int(np.argmax(scan_abs))
        peak_freq = float(scan_freqs[peak_idx])
        target_lambda = float(1.0 / center_freq) if center_freq > 0 else float("nan")
        peak_lambda = float(1.0 / peak_freq) if peak_freq > 0 else float("nan")
        return {
            "target_frequency_inv_um": float(center_freq),
            "target_wavelength_um": target_lambda,
            "scan_frequency_inv_um": scan_freqs.tolist(),
            "scan_wavelength_um": (1.0 / scan_freqs).tolist(),
            "scan_field_rms": scan_abs.tolist(),
            "peak_index": peak_idx,
            "peak_frequency_inv_um": peak_freq,
            "peak_wavelength_um": peak_lambda,
            "peak_field_rms": float(scan_abs[peak_idx]),
            "detuning_from_target_inv_um": float(peak_freq - center_freq),
            "detuning_from_target_nm": float((peak_lambda - target_lambda) * 1e3),
            "peak_at_scan_boundary": bool(peak_idx == 0 or peak_idx == (scan_abs.size - 1)),
        }

    probe_rotation_tail_points = resolve_window_points_from_fs(
        t_arr_fs, probe_rotation_tail_points_requested, probe_rotation_window_fs
    )
    probe_rotation_tail_points_td = resolve_window_points_from_fs(
        t_td_fs, probe_rotation_tail_points_requested, probe_rotation_window_fs
    )

    probe_s0_dft = probe_s0
    intensity_threshold_dft = (
        strength_threshold_rel * float(np.max(probe_s0_dft)) if probe_s0_dft.size else 0.0
    )
    if disable_strength_validity:
        valid_probe_dft = np.isfinite(probe_s0_dft)
    else:
        valid_probe_dft = probe_s0_dft > intensity_threshold_dft
    tail_idx_dft = select_final_window_indices(theta_deg_rel, valid_probe_dft, probe_rotation_tail_points)
    if np.any(valid_probe_dft):
        theta_probe_valid_dft = theta_deg_rel[valid_probe_dft]
        probe_rotation_min_rel = float(np.min(theta_probe_valid_dft))
        probe_rotation_max_rel = float(np.max(theta_probe_valid_dft))
        probe_rotation_mean_rel = weighted_linear_mean_deg(
            theta_probe_valid_dft, probe_s0_dft[valid_probe_dft]
        )
    else:
        probe_rotation_min_rel = (
            float(np.min(theta_deg_rel)) if theta_deg_rel.size else float("nan")
        )
        probe_rotation_max_rel = (
            float(np.max(theta_deg_rel)) if theta_deg_rel.size else float("nan")
        )
        probe_rotation_mean_rel = weighted_linear_mean_deg(theta_deg_rel)
    if tail_idx_dft.size:
        dft_tail_weights = (
            np.asarray(probe_s0_dft[tail_idx_dft], dtype=float)
            if probe_s0_dft.size == theta_deg_rel.size
            else np.ones(tail_idx_dft.size, dtype=float)
        )
        probe_rotation_final_rel_window_mean = weighted_linear_mean_deg(
            np.asarray(theta_deg_rel[tail_idx_dft], dtype=float), dft_tail_weights
        )
        probe_rotation_final_rel_unwrapped_window_mean = weighted_arithmetic_mean(
            np.asarray(theta_deg_rel_unwrapped[tail_idx_dft], dtype=float), dft_tail_weights
        )
        probe_rotation_tail_window_fs = (
            float(t_arr_fs[int(tail_idx_dft[0])]),
            float(t_arr_fs[int(tail_idx_dft[-1])]),
        )
    else:
        probe_rotation_final_rel_window_mean = (
            float(theta_deg_rel[-1]) if theta_deg_rel.size else float("nan")
        )
        probe_rotation_final_rel_unwrapped_window_mean = (
            float(theta_deg_rel_unwrapped[-1])
            if theta_deg_rel_unwrapped.size
            else float("nan")
        )
        probe_rotation_tail_window_fs = (float("nan"), float("nan"))

    # Coherent final-window estimator: average Jones components over the selected
    # end window and compute Stokes angle from the averaged complex field.
    dft_window_weights = (
        np.asarray(probe_s0_dft[tail_idx_dft], dtype=float)
        if tail_idx_dft.size and probe_s0_dft.size == theta_deg_rel.size
        else np.ones(tail_idx_dft.size, dtype=float)
    )
    coherent_ex = complex(np.nan, np.nan)
    coherent_ey = complex(np.nan, np.nan)
    coherent_theta_rel = float("nan")
    coherent_chi_deg = float("nan")
    coherent_dolp = float("nan")
    coherent_docp = float("nan")
    coherent_s0 = float("nan")
    coherent_theta_std = float("nan")
    coherent_signal_power = float("nan")
    coherent_noise_power = float("nan")
    coherent_snr_linear = float("nan")
    coherent_snr_db = float("nan")
    coherent_coherence = float("nan")
    if tail_idx_dft.size:
        ex_tail = np.asarray(probe_ex_fwd_mean[tail_idx_dft], dtype=complex)
        ey_tail = np.asarray(probe_ey_fwd_mean[tail_idx_dft], dtype=complex)
        coherent_ex = weighted_complex_mean(ex_tail, dft_window_weights)
        coherent_ey = weighted_complex_mean(ey_tail, dft_window_weights)
        if np.isfinite(coherent_ex.real) and np.isfinite(coherent_ex.imag) and np.isfinite(coherent_ey.real) and np.isfinite(coherent_ey.imag):
            stokes_coherent = stokes_metrics(
                np.array([coherent_ex], dtype=complex),
                np.array([coherent_ey], dtype=complex),
            )
            coherent_theta_rel = float(
                wrap_linear_polarization_deg(
                    stokes_coherent["theta_deg"] - init_pol_deg
                )
            )
            coherent_chi_deg = float(stokes_coherent["chi_deg"])
            coherent_dolp = float(stokes_coherent["dolp"])
            coherent_docp = float(stokes_coherent["docp"])
            coherent_s0 = float(stokes_coherent["S0"])
            coherent_theta_std = weighted_circular_std_linear_pol_deg(
                np.asarray(theta_deg_rel[tail_idx_dft], dtype=float), dft_window_weights
            )
            residual_tail = np.abs(ex_tail - coherent_ex) ** 2 + np.abs(ey_tail - coherent_ey) ** 2
            coherent_noise_power = weighted_arithmetic_mean(
                np.asarray(residual_tail, dtype=float), dft_window_weights
            )
            coherent_signal_power = float(np.abs(coherent_ex) ** 2 + np.abs(coherent_ey) ** 2)
            coherent_snr_linear = float(
                coherent_signal_power / max(coherent_noise_power, 1e-30)
            )
            coherent_snr_db = float(10.0 * np.log10(max(coherent_snr_linear, 1e-30)))
            window_power = weighted_arithmetic_mean(
                np.asarray(np.abs(ex_tail) ** 2 + np.abs(ey_tail) ** 2, dtype=float),
                dft_window_weights,
            )
            coherent_coherence = float(
                coherent_signal_power / max(window_power, 1e-30)
            )

    nonlinear_diagnostics: Dict[str, Any] = {"enabled": bool(diagnostics_enabled)}
    if diagnostics_enabled and t_arr.size:
        diagnostic_tail_idx = (
            np.asarray(tail_idx_dft, dtype=int)
            if tail_idx_dft.size
            else np.array([int(t_arr.size - 1)], dtype=int)
        )
        diagnostic_weights = (
            np.asarray(dft_window_weights, dtype=float)
            if tail_idx_dft.size
            else np.ones(diagnostic_tail_idx.size, dtype=float)
        )
        output_fixed_summary = {
            label: summarize_fixed_trace(
                fixed_ex,
                fixed_ey,
                fixed_eplus_rms,
                fixed_eminus_rms,
                fixed_abs,
                idx,
                label,
                diagnostic_weights,
                diagnostic_tail_idx,
            )
            for idx, label in enumerate(fixed_labels)
        }
        cavity_fixed_summary = {}
        if cavity_ex is not None and cavity_ey is not None:
            cavity_fixed_summary = {
                label: summarize_fixed_trace(
                    cavity_ex,
                    cavity_ey,
                    cavity_eplus_rms,
                    cavity_eminus_rms,
                    cavity_abs,
                    idx,
                    label,
                    diagnostic_weights,
                    diagnostic_tail_idx,
                )
                for idx, label in enumerate(fixed_labels)
            }

        hot_scan_summary = (
            {
                "pump1": extract_hot_scan_summary("pump1", freq_p1),
                "pump2": extract_hot_scan_summary("pump2", freq_p2),
                "probe": extract_hot_scan_summary("probe", freq_probe),
            }
            if cavity_scan_monitors
            else {}
        )
        cavity_p1_dom = (
            float(cavity_fixed_summary["pump1"]["eminus_rms_tail"])
            if cavity_fixed_summary
            else float("nan")
        )
        cavity_p2_dom = (
            float(cavity_fixed_summary["pump2"]["eplus_rms_tail"])
            if cavity_fixed_summary
            else float("nan")
        )
        cavity_p1_orth = (
            float(cavity_fixed_summary["pump1"]["eplus_rms_tail"])
            if cavity_fixed_summary
            else float("nan")
        )
        cavity_p2_orth = (
            float(cavity_fixed_summary["pump2"]["eminus_rms_tail"])
            if cavity_fixed_summary
            else float("nan")
        )
        probe_projection = output_fixed_summary["probe"].get("linear_basis_projection", {})
        nonlinear_diagnostics = {
            "enabled": True,
            "tail_window_fs": [
                float(t_arr_fs[int(diagnostic_tail_idx[0])]),
                float(t_arr_fs[int(diagnostic_tail_idx[-1])]),
            ],
            "cavity_monitor": {
                "center_z_um": float(cavity_center),
                "volume_size_um": (
                    {
                        "x": float(cavity_monitor_volume.size.x),
                        "y": float(cavity_monitor_volume.size.y),
                        "z": float(cavity_monitor_volume.size.z),
                    }
                    if cavity_monitor_volume is not None
                    else None
                ),
                "scan_points": int(diagnostic_scan_points),
                "scan_span_factor_times_source_fwidth": float(diagnostic_scan_span_factor),
            },
            "intracavity_fixed_freqs": cavity_fixed_summary,
            "output_fixed_freqs": output_fixed_summary,
            "hot_frequency_scans": hot_scan_summary,
            "intracavity_pump_drive_metrics": {
                "definition": {
                    "pump1_dominant_component": "|e-| at f_p1",
                    "pump2_dominant_component": "|e+| at f_p2",
                    "pump1_orthogonal_component": "|e+| at f_p1",
                    "pump2_orthogonal_component": "|e-| at f_p2",
                    "dominant_product_metric": "pump1_dominant * pump2_dominant",
                },
                "tail_weighted_rms": {
                    "pump1_dominant": cavity_p1_dom,
                    "pump2_dominant": cavity_p2_dom,
                    "pump1_orthogonal": cavity_p1_orth,
                    "pump2_orthogonal": cavity_p2_orth,
                    "dominant_product": float(cavity_p1_dom * cavity_p2_dom),
                    "purity_pump1": float(cavity_p1_dom / max(cavity_p1_dom + cavity_p1_orth, 1e-30)),
                    "purity_pump2": float(cavity_p2_dom / max(cavity_p2_dom + cavity_p2_orth, 1e-30)),
                },
            },
            "probe_output_projection": probe_projection,
            "sideband_generation": {
                "cavity_sb_minus_field_rms_tail": (
                    float(cavity_fixed_summary.get("sb_minus", {}).get("field_rms_tail", float("nan")))
                    if cavity_fixed_summary
                    else float("nan")
                ),
                "cavity_sb_plus_field_rms_tail": (
                    float(cavity_fixed_summary.get("sb_plus", {}).get("field_rms_tail", float("nan")))
                    if cavity_fixed_summary
                    else float("nan")
                ),
                "output_sb_minus_field_rms_tail": float(output_fixed_summary["sb_minus"]["field_rms_tail"]),
                "output_sb_plus_field_rms_tail": float(output_fixed_summary["sb_plus"]["field_rms_tail"]),
                "output_sb_minus_eplus_rms_tail": float(output_fixed_summary["sb_minus"]["eplus_rms_tail"]),
                "output_sb_minus_eminus_rms_tail": float(output_fixed_summary["sb_minus"]["eminus_rms_tail"]),
                "output_sb_plus_eplus_rms_tail": float(output_fixed_summary["sb_plus"]["eplus_rms_tail"]),
                "output_sb_plus_eminus_rms_tail": float(output_fixed_summary["sb_plus"]["eminus_rms_tail"]),
            },
        }

    probe_rotation_final_rel = (
        coherent_theta_rel
        if np.isfinite(coherent_theta_rel)
        else probe_rotation_final_rel_window_mean
    )
    probe_rotation_final_rel_unwrapped = probe_rotation_final_rel_unwrapped_window_mean
    probe_rotation_final_method = (
        "dft_probe_center_forward_jones_coherent_final_window"
        if np.isfinite(coherent_theta_rel)
        else "dft_monitor_center_frequency_forward_component_principal_linear_angle_final_window_mean"
    )

    # Keep the TD envelope estimate for diagnostics.
    probe_eplus_td = epl_td[:, i_probe]
    probe_eminus_td = emi_td[:, i_probe]
    probe_ex_td = (probe_eplus_td + probe_eminus_td) / np.sqrt(2.0)
    probe_ey_td = 1j * (-probe_eplus_td + probe_eminus_td) / np.sqrt(2.0)
    probe_s0_td = np.abs(probe_ex_td)**2 + np.abs(probe_ey_td)**2
    intensity_threshold_td = (
        strength_threshold_rel * float(np.max(probe_s0_td)) if probe_s0_td.size else 0.0
    )
    if disable_strength_validity:
        valid_probe_td = np.isfinite(probe_s0_td)
    else:
        valid_probe_td = probe_s0_td > intensity_threshold_td
    tail_idx_td = select_final_window_indices(theta_deg_t_rel, valid_probe_td, probe_rotation_tail_points_td)
    if np.any(valid_probe_td):
        theta_probe_valid_td = theta_deg_t_rel[valid_probe_td]
        probe_rotation_min_rel_td = float(np.min(theta_probe_valid_td))
        probe_rotation_max_rel_td = float(np.max(theta_probe_valid_td))
        probe_rotation_mean_rel_td = weighted_linear_mean_deg(
            theta_probe_valid_td, probe_s0_td[valid_probe_td]
        )
    else:
        probe_rotation_min_rel_td = (
            float(np.min(theta_deg_t_rel)) if theta_deg_t_rel.size else float("nan")
        )
        probe_rotation_max_rel_td = (
            float(np.max(theta_deg_t_rel)) if theta_deg_t_rel.size else float("nan")
        )
        probe_rotation_mean_rel_td = weighted_linear_mean_deg(theta_deg_t_rel)
    if tail_idx_td.size:
        td_tail_weights = (
            np.asarray(probe_s0_td[tail_idx_td], dtype=float)
            if probe_s0_td.size == theta_deg_t_rel.size
            else np.ones(tail_idx_td.size, dtype=float)
        )
        probe_rotation_final_rel_td = weighted_linear_mean_deg(
            np.asarray(theta_deg_t_rel[tail_idx_td], dtype=float), td_tail_weights
        )
        probe_rotation_final_rel_td_unwrapped = weighted_arithmetic_mean(
            np.asarray(theta_deg_t_rel_unwrapped[tail_idx_td], dtype=float), td_tail_weights
        )
        probe_rotation_td_tail_window_fs = (
            float(t_td_fs[int(tail_idx_td[0])]),
            float(t_td_fs[int(tail_idx_td[-1])]),
        )
    else:
        probe_rotation_final_rel_td = (
            float(theta_deg_t_rel[-1]) if theta_deg_t_rel.size else float("nan")
        )
        probe_rotation_final_rel_td_unwrapped = (
            float(theta_deg_t_rel_unwrapped[-1])
            if theta_deg_t_rel_unwrapped.size
            else float("nan")
        )
        probe_rotation_td_tail_window_fs = (float("nan"), float("nan"))

    def safe_ratio(num: float, den: float) -> float:
        return float(num / den) if abs(den) > 1e-30 else float("nan")

    def tail_start_index(
        n_points: int, tail_fraction: float = 0.2, tail_points: int | None = None
    ) -> int:
        if n_points <= 0:
            return 0
        if tail_points is not None:
            count = max(1, min(int(tail_points), int(n_points)))
            return int(n_points - count)
        return max(0, int((1.0 - tail_fraction) * n_points))

    def weighted_tail_mean(
        values: np.ndarray,
        weights: np.ndarray,
        tail_fraction: float = 0.2,
        tail_points: int | None = None,
    ) -> float:
        if values.size == 0:
            return float("nan")
        i0 = tail_start_index(values.size, tail_fraction=tail_fraction, tail_points=tail_points)
        v = np.asarray(values[i0:], dtype=float)
        w = np.asarray(weights[i0:], dtype=float)
        valid = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(valid):
            return float(np.mean(v)) if v.size else float("nan")
        return float(np.average(v[valid], weights=w[valid]))

    def weighted_tail_std(
        values: np.ndarray,
        weights: np.ndarray,
        tail_fraction: float = 0.2,
        tail_points: int | None = None,
    ) -> float:
        if values.size == 0:
            return float("nan")
        i0 = tail_start_index(values.size, tail_fraction=tail_fraction, tail_points=tail_points)
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
        values: np.ndarray,
        weights: np.ndarray,
        tail_fraction: float = 0.2,
        tail_points: int | None = None,
    ) -> float:
        if values.size == 0:
            return float("nan")
        i0 = tail_start_index(values.size, tail_fraction=tail_fraction, tail_points=tail_points)
        v = np.asarray(values[i0:], dtype=float)
        w = np.asarray(weights[i0:], dtype=float)
        return weighted_linear_mean_deg(v, w)

    def weighted_tail_linear_std(
        values: np.ndarray,
        weights: np.ndarray,
        tail_fraction: float = 0.2,
        tail_points: int | None = None,
    ) -> float:
        if values.size == 0:
            return float("nan")
        i0 = tail_start_index(values.size, tail_fraction=tail_fraction, tail_points=tail_points)
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

    def stabilized_zoom_window(
        t_fs: np.ndarray,
        theta_rel_deg: np.ndarray,
        weights: np.ndarray,
        valid_mask: np.ndarray,
        tail_points: int = 64,
    ) -> Dict[str, float] | None:
        tt = np.asarray(t_fs, dtype=float)
        th = np.asarray(theta_rel_deg, dtype=float)
        ww = np.asarray(weights, dtype=float)
        vm = np.asarray(valid_mask, dtype=bool)
        if tt.size == 0 or th.size == 0:
            return None
        if vm.size != tt.size:
            vm = np.ones_like(tt, dtype=bool)

        tail_idx = select_final_window_indices(th, vm, tail_points)
        if tail_idx.size < 2:
            return None

        t_tail = tt[tail_idx]
        th_tail = th[tail_idx]
        w_tail = ww[tail_idx] if ww.size == tt.size else np.ones_like(t_tail)
        valid_tail = np.isfinite(t_tail) & np.isfinite(th_tail)
        if not np.any(valid_tail):
            return None

        t_tail = t_tail[valid_tail]
        th_tail = th_tail[valid_tail]
        w_tail = np.asarray(w_tail[valid_tail], dtype=float)

        theta_mean = weighted_linear_mean_deg(th_tail, w_tail)
        theta_std = weighted_tail_linear_std(th_tail, w_tail, tail_fraction=1.0)
        y_min = float(np.min(th_tail))
        y_max = float(np.max(th_tail))
        y_span = max(y_max - y_min, 1e-6)
        y_pad = max(0.02, 0.25 * y_span, 2.0 * theta_std if np.isfinite(theta_std) else 0.0)

        x0 = float(np.min(t_tail))
        x1 = float(np.max(t_tail))
        x_span = max(x1 - x0, 1e-6)
        x_pad = max(0.05 * x_span, 2.0)

        return {
            "x0": float(x0 - x_pad),
            "x1": float(x1 + x_pad),
            "y0": float(y_min - y_pad),
            "y1": float(y_max + y_pad),
            "theta_mean": float(theta_mean),
            "tail_start": float(x0),
            "tail_end": float(x1),
            "tail_point_count": int(tail_idx.size),
        }

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

    probe_theta_tail_rel = weighted_tail_linear_mean(
        theta_deg_rel, probe_s0_dft, tail_points=probe_rotation_tail_points
    )
    probe_theta_tail_std_rel = weighted_tail_linear_std(
        theta_deg_rel, probe_s0_dft, tail_points=probe_rotation_tail_points
    )
    probe_chi_tail = weighted_tail_mean(
        probe_chi_deg, probe_s0_dft, tail_points=probe_rotation_tail_points
    )
    probe_docp_tail = weighted_tail_mean(
        probe_docp, probe_s0_dft, tail_points=probe_rotation_tail_points
    )
    probe_dolp_tail = weighted_tail_mean(
        probe_dolp, probe_s0_dft, tail_points=probe_rotation_tail_points
    )
    probe_s0_tail = weighted_tail_mean(
        probe_s0_dft,
        np.ones_like(probe_s0_dft),
        tail_points=probe_rotation_tail_points,
    )
    probe_tail_points_effective = int(
        min(int(probe_rotation_tail_points), int(probe_s0_dft.size))
    ) if probe_s0_dft.size else 0
    probe_s0_max = float(np.max(probe_s0_dft)) if probe_s0_dft.size else float("nan")
    probe_s0_tail_rel_max = safe_ratio(probe_s0_tail, probe_s0_max)

    # Total-field (no forward/backward split) tail/final aggregates, S0-weighted like
    # the forward-isolated ones, for the three-way comparison.
    probe_s0_total_w = (
        probe_s0_total
        if probe_s0_total.size == theta_total_deg_rel.size
        else np.ones_like(theta_total_deg_rel)
    )
    probe_theta_total_tail_rel = weighted_tail_linear_mean(
        theta_total_deg_rel, probe_s0_total_w, tail_points=probe_rotation_tail_points
    )
    probe_theta_total_tail_std_rel = weighted_tail_linear_std(
        theta_total_deg_rel, probe_s0_total_w, tail_points=probe_rotation_tail_points
    )
    probe_dolp_total_tail = weighted_tail_mean(
        probe_dolp_total, probe_s0_total_w, tail_points=probe_rotation_tail_points
    )
    probe_theta_total_final_rel = (
        float(theta_total_deg_rel[-1]) if theta_total_deg_rel.size else float("nan")
    )
    probe_dolp_total_final = (
        float(probe_dolp_total[-1]) if probe_dolp_total.size else float("nan")
    )
    probe_s0_total_tail = weighted_tail_mean(
        probe_s0_total,
        np.ones_like(probe_s0_total),
        tail_points=probe_rotation_tail_points,
    ) if probe_s0_total.size else float("nan")

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
    ax.plot(
        t_arr_fs,
        theta_total_deg_rel,
        color="0.7",
        lw=1.0,
        label="total-field θ (no fwd/bwd split)",
    )
    ax.plot(t_arr_fs, theta_deg_rel, "k-", lw=1.2, label="forward-isolated θ (incoherent)")
    if np.isfinite(coherent_theta_rel):
        ax.axhline(
            coherent_theta_rel,
            color="C2",
            lw=1.3,
            ls=":",
            label=f"forward coherent (final) = {coherent_theta_rel:.3f}°",
        )
    dft_zoom = stabilized_zoom_window(
        t_arr_fs,
        theta_deg_rel,
        probe_s0_dft,
        valid_probe_dft,
        tail_points=probe_rotation_tail_points,
    )
    if dft_zoom is not None:
        ax.axvspan(
            dft_zoom["tail_start"],
            dft_zoom["tail_end"],
            color="C0",
            alpha=0.12,
            lw=0.0,
            label=f"final {int(probe_rotation_tail_points)}-point window",
        )
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("polarization rotation (deg)")
    ax.set_title("Probe polarization angle vs time (relative to input)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=7)
    plot_paths["probe_rotation"] = str(
        save_figure(fig, "probe_polarization.png", output_dir)
    )
    if dft_zoom is not None:
        fig = plt.figure(figsize=(7.0, 3.6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t_arr_fs, theta_total_deg_rel, color="0.8", lw=1.0, label="total-field θ")
        ax.plot(t_arr_fs, theta_deg_rel, color="0.4", lw=1.0, label="forward θ (incoherent)")
        ax.axhline(
            dft_zoom["theta_mean"],
            color="C3",
            lw=1.2,
            ls="--",
            label="forward incoherent window mean",
        )
        if np.isfinite(coherent_theta_rel):
            ax.axhline(
                coherent_theta_rel,
                color="C2",
                lw=1.3,
                ls=":",
                label=f"forward coherent = {coherent_theta_rel:.3f}°",
            )
        ax.set_xlim(dft_zoom["x0"], dft_zoom["x1"])
        ax.set_ylim(dft_zoom["y0"], dft_zoom["y1"])
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("polarization rotation (deg)")
        ax.set_title("Probe polarization angle (final-window zoom, DFT)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
        plot_paths["probe_rotation_zoom"] = str(
            save_figure(fig, "probe_polarization_zoom.png", output_dir)
        )

    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t_td_fs, theta_deg_t_rel, "k-")
    td_zoom = stabilized_zoom_window(
        t_td_fs,
        theta_deg_t_rel,
        probe_s0_td,
        valid_probe_td,
        tail_points=probe_rotation_tail_points_td,
    )
    if td_zoom is not None:
        ax.axvspan(
            td_zoom["tail_start"],
            td_zoom["tail_end"],
            color="C0",
            alpha=0.12,
            lw=0.0,
            label=f"final {int(probe_rotation_tail_points_td)}-point window",
        )
    ax.set_xlabel("time (fs)")
    ax.set_ylabel("polarization rotation (deg)")
    ax.set_title("Probe polarization angle vs time (relative to input) in time-domain")
    ax.grid(True, alpha=0.3)
    plot_paths["probe_rotation_td"] = str(
        save_figure(fig, "probe_polarization_td.png", output_dir)
    )
    if td_zoom is not None:
        fig = plt.figure(figsize=(7.0, 3.6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(t_td_fs, theta_deg_t_rel, color="0.75", lw=1.0, label="full trace")
        ax.axhline(
            td_zoom["theta_mean"],
            color="C3",
            lw=1.2,
            ls="--",
            label="window mean",
        )
        ax.set_xlim(td_zoom["x0"], td_zoom["x1"])
        ax.set_ylim(td_zoom["y0"], td_zoom["y1"])
        ax.set_xlabel("time (fs)")
        ax.set_ylabel("polarization rotation (deg)")
        ax.set_title("Probe polarization angle (final-window zoom, TD)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        plot_paths["probe_rotation_td_zoom"] = str(
            save_figure(fig, "probe_polarization_td_zoom.png", output_dir)
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
    probe_s0_backward_final = (
        float(probe_s0_backward[-1]) if probe_s0_backward.size else float("nan")
    )
    probe_forward_fraction_final = (
        float(probe_forward_fraction[-1])
        if probe_forward_fraction.size
        else float("nan")
    )
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
        theta_total_deg_rel=theta_total_deg_rel.copy(),
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
        "modeling_dimensions": {
            "requested_dimension": int(dimension),
            "meep_solver_dimensions": int(simulation_dimensions),
            "quasi_1d_collapsed_transverse_cell": bool(is_quasi_1d),
        },
        "run_params": run_params_dict,
        "geometry_file": str(spec_path),
        "materials_model": args.materials,
        "high_index_material": {
            "key": str(high_preset.key),
            "display_name": str(high_preset.display_name),
            "source": str(high_preset.source),
            "slot_in_geometry": str(high_slot),
            "n_constant_used": float(n_high if args.nH is not None else default_high),
            "k_constant_used": float(k_high),
            "kappa_ref_lambda_um": float(args.kappa_ref_lambda),
            "n2_m2_per_w_used": float(n2_high),
            "n_linear_probe": float(n_linear_probe),
            "chi3_si": float(chi3_si),
            "E_chi3_diag_meep": float(e_chi3_meep),
        },
        "cavity_material": {
            "key": str(cavity_material),
            "slot_in_geometry": str(cav_slot),
            "distinct_from_mirrors": bool(mat_cav is not mat_sin),
            "n2_m2_per_w_used": float(n2_cav),
            "n_linear_probe": float(n_linear_cav),
            "chi3_si": float(chi3_si_cav),
            "E_chi3_diag_meep": float(e_chi3_cav),
        },
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
        "monitor_medium_index_probe": float(n_monitor_medium),
        "plot_time_units": {
            "unit": "fs",
            "fs_per_meep_time": float(FS_PER_MEEP),
        },
        "probe_source_state": {
            "azimuth_deg": float(init_pol_deg),
            "ellipticity_deg": float(probe_ellipticity_deg),
            "pump1_delay_fs": float(pump1_delay_fs),
            "pump1_start_time_meep": float(t_start_pump1),
            "pump2_probe_start_time_meep": float(t_start_rest),
            "delay_pad_fs": (float(delay_pad_fs) if delay_pad_fs is not None else None),
            "delay_convention": ("common_pad" if delay_pad_fs is not None else "legacy_split"),
            "pump_imbalance_intensity_ratio": float(pump_imbalance),
        },
        "probe_pulse_integrated": probe_pulse_integrated,
        "probe_rotation_deg": {
            "initial_deg": init_pol_deg,
            "final_relative_deg": probe_rotation_final_rel,
            "final_relative_deg_window_mean": probe_rotation_final_rel_window_mean,
            "max_relative_deg": probe_rotation_max_rel,
            "min_relative_deg": probe_rotation_min_rel,
            "mean_relative_deg": probe_rotation_mean_rel,
            "final_relative_unwrapped_deg": probe_rotation_final_rel_unwrapped,
            "final_relative_unwrapped_deg_window_mean": (
                probe_rotation_final_rel_unwrapped_window_mean
            ),
            "final_window_policy": "mean_over_last_m_valid_points",
            "final_window_points_requested": int(probe_rotation_tail_points_requested),
            "final_window_points_effective": int(probe_rotation_tail_points),
            "final_window_points_used": int(tail_idx_dft.size),
            "final_window_fs_requested": (
                float(probe_rotation_window_fs)
                if probe_rotation_window_fs is not None
                else None
            ),
            "final_window_time_fs": [
                float(probe_rotation_tail_window_fs[0]),
                float(probe_rotation_tail_window_fs[1]),
            ],
            "wrapped_final_relative_deg": (
                float(theta_deg_rel[-1])
                if theta_deg_rel.size
                else float("nan")
            ),
            "raw_final_relative_deg": (
                float(theta_deg_wrapped[-1] - init_pol_deg)
                if theta_deg_wrapped.size
                else float("nan")
            ),
            "method": probe_rotation_final_method,
            "forward_decomposition": {
                "using_fields": ["Ex", "Ey", "Hx", "Hy"],
                "n_medium": float(n_monitor_medium),
                "forward_relation": "Hy=n*Ex and Hx=-n*Ey",
            },
            "coherent_window_estimate": {
                "theta_relative_deg": float(coherent_theta_rel),
                "theta_relative_std_deg": float(coherent_theta_std),
                "chi_deg": float(coherent_chi_deg),
                "dolp": float(coherent_dolp),
                "docp": float(coherent_docp),
                "S0": float(coherent_s0),
                "Ex_forward_mean_real": float(np.real(coherent_ex)),
                "Ex_forward_mean_imag": float(np.imag(coherent_ex)),
                "Ey_forward_mean_real": float(np.real(coherent_ey)),
                "Ey_forward_mean_imag": float(np.imag(coherent_ey)),
                "signal_power": float(coherent_signal_power),
                "noise_power": float(coherent_noise_power),
                "snr_linear": float(coherent_snr_linear),
                "snr_db": float(coherent_snr_db),
                "coherence_factor": float(coherent_coherence),
                "weights": "S0",
            },
            "validity_policy": validity_policy,
            "strength_validity_enabled": bool(not disable_strength_validity),
            "intensity_threshold_rel": (
                float(strength_threshold_rel) if not disable_strength_validity else 0.0
            ),
            "intensity_threshold_abs": float(intensity_threshold_dft),
            "time_domain_reference": {
                "final_relative_deg": probe_rotation_final_rel_td,
                "max_relative_deg": probe_rotation_max_rel_td,
                "min_relative_deg": probe_rotation_min_rel_td,
                "mean_relative_deg": probe_rotation_mean_rel_td,
                "final_relative_unwrapped_deg": probe_rotation_final_rel_td_unwrapped,
                "final_window_policy": "mean_over_last_m_valid_points",
                "final_window_points_requested": int(probe_rotation_tail_points_requested),
                "final_window_points_effective": int(probe_rotation_tail_points_td),
                "final_window_points_used": int(tail_idx_td.size),
                "final_window_fs_requested": (
                    float(probe_rotation_window_fs)
                    if probe_rotation_window_fs is not None
                    else None
                ),
                "final_window_time_fs": [
                    float(probe_rotation_td_tail_window_fs[0]),
                    float(probe_rotation_td_tail_window_fs[1]),
                ],
                "wrapped_final_relative_deg": (
                    float(theta_deg_t_rel[-1])
                    if theta_deg_t_rel.size
                    else float("nan")
                ),
                "raw_final_relative_deg": (
                    float(theta_deg_t_wrapped[-1] - init_pol_deg)
                    if theta_deg_t_wrapped.size
                    else float("nan")
                ),
                "method": "time_domain_probe_envelope_at_output_principal_linear_angle_final_window_mean",
                "validity_policy": validity_policy,
                "strength_validity_enabled": bool(not disable_strength_validity),
                "intensity_threshold_rel": (
                    float(strength_threshold_rel) if not disable_strength_validity else 0.0
                ),
                "intensity_threshold_abs": float(intensity_threshold_td),
            },
        },
        "probe_stokes_dft": {
            "final": {
                "theta_relative_deg": probe_theta_final_rel,
                "chi_deg": probe_chi_final,
                "dolp": probe_dolp_final,
                "docp": probe_docp_final,
                "S0": probe_s0_final,
                "S0_backward": probe_s0_backward_final,
                "forward_fraction": probe_forward_fraction_final,
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
                "window_policy": "mean_over_last_m_points",
                "window_points_requested": int(probe_rotation_tail_points_requested),
                "window_points_effective": int(probe_rotation_tail_points),
                "window_points_used": int(probe_tail_points_effective),
                "window_fs_requested": (
                    float(probe_rotation_window_fs)
                    if probe_rotation_window_fs is not None
                    else None
                ),
                "weights": "S0",
            },
        },
        "probe_stokes_total": {
            "description": (
                "Total-field Stokes angle at the probe monitor (raw Ex,Ey, no "
                "forward/backward split). Naive-detector reading; compare against "
                "probe_stokes_dft (forward, incoherent) and "
                "probe_rotation_deg.final_relative_deg (forward, coherent)."
            ),
            "final": {
                "theta_relative_deg": probe_theta_total_final_rel,
                "dolp": probe_dolp_total_final,
            },
            "tail_weighted": {
                "theta_relative_deg": float(probe_theta_total_tail_rel),
                "theta_relative_std_deg": float(probe_theta_total_tail_std_rel),
                "dolp": float(probe_dolp_total_tail),
                "S0": float(probe_s0_total_tail),
                "window_points_requested": int(probe_rotation_tail_points_requested),
                "weights": "S0",
            },
        },
        "nonlinear_diagnostics": nonlinear_diagnostics,
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
        "plot_paths": plot_paths,
        "theta_deg_rel_I": {
            "pump_intensity_w_cm2": float(run.pump_intensity_w_cm2),
            # Primary metric (unchanged): forward-isolated, coherent window estimate.
            "final_relative_deg": probe_rotation_final_rel,
            # Companion readings for comparison (do not feed the optimizer objective).
            "forward_incoherent_final_relative_deg": float(probe_theta_tail_rel),
            "total_field_final_relative_deg": float(probe_theta_total_tail_rel),
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
        "--high-index-material",
        choices=high_index_material_choices(),
        default="sin",
        help="High-index cavity material preset (sin or tio2).",
    )
    # --- distinct CAVITY material (3-material stacks) -------------------------------- #
    # By default the cavity is made of the same medium as the mirror high-index layers, which
    # is what every geometry in this repo assumed. These flags let the cavity be a DIFFERENT
    # material (e.g. a SiC cavity inside SiN/SiO2 mirrors, as fabricated 2026-08). All default
    # to None => behaviour is bit-for-bit unchanged.
    parser.add_argument(
        "--cavity-material",
        dest="cavity_material",
        choices=high_index_material_choices(),
        default=None,
        help="Material preset for the CAVITY layer when it differs from the mirror high-index "
             "material. Default: cavity uses the same medium as the mirrors.",
    )
    parser.add_argument(
        "--cavity-fit",
        dest="cavity_fit",
        type=str,
        default=None,
        help="CSV with wavelength,n,k for the cavity material when --materials fit "
             "(defaults to --sin-fit if omitted).",
    )
    parser.add_argument(
        "--n-cav",
        dest="n_cav",
        type=float,
        default=None,
        help="Override cavity refractive index for constant/library fallback.",
    )
    parser.add_argument(
        "--k-cav",
        dest="k_cav",
        type=float,
        default=None,
        help="Override cavity extinction coefficient k for constant/library fallback.",
    )
    parser.add_argument(
        "--cavity-n2",
        dest="cavity_n2",
        type=float,
        default=None,
        help="Override the cavity Kerr n2 (m^2/W). Default: the cavity preset's n2. The mirror "
             "high-index layers keep their own n2 and remain nonlinear.",
    )
    parser.add_argument(
        "--nH",
        type=float,
        default=None,
        help="Override high-index refractive index for constant/library fallback.",
    )
    parser.add_argument(
        "--kH",
        type=float,
        default=None,
        help="Override high-index extinction coefficient k for constant/library fallback.",
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
        help="CSV with wavelength_nm,n,k for selected high-index material when --materials fit.",
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
        "--kappa-ref-lambda",
        type=float,
        default=1.55,
        help="Reference wavelength (um) used to map constant k to Meep conductivity.",
    )
    parser.add_argument(
        "--high-index-n2",
        type=float,
        default=None,
        help="Override Kerr nonlinear index n2 (m^2/W) for the selected high-index material.",
    )
    parser.add_argument(
        "--pump-intensity",
        type=float,
        default=None,
        help="Pump intensity in W/cm^2.",
    )
    parser.add_argument(
        "--probe-intensity",
        type=float,
        default=None,
        help="Probe intensity in W/cm^2.",
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
        default=1e-4,
        help=(
            "Field-decay stopping threshold used when --until-time is not provided. "
            "Default 1e-4 (trustworthy 1D everyday value); use 1e-3 for a fast check "
            "or 1e-6 for a converged measurement."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Override default resolution (px/um) for the selected mode.",
    )
    parser.add_argument(
        "--pulse-duration-fs",
        dest="pulse_duration_fs",
        type=float,
        default=None,
        help=(
            "Override the Gaussian pulse-duration parameter (fs) for BOTH pumps and the "
            "probe. NOTE this is the label fed to df_from_pulse_duration, which sets "
            "width = T/(2 ln 2); the resulting INTENSITY FWHM is 1.2011 x T. So the "
            "default 100.0 is a 120.1 fs intensity-FWHM pulse, and a true 100 fs "
            "intensity FWHM needs T = 83.2555. This also sets the probe DFT readout "
            "band (freq_probe +- df/2), so widening the pulse narrows that band."
        ),
    )
    parser.add_argument(
        "--courant",
        type=float,
        default=None,
        help=(
            "Courant factor for the FDTD time step (Meep default 0.5). Lower it "
            "(e.g. 0.25) to keep dispersive (Lorentzian) media stable at coarse "
            "resolution, e.g. res-30 3D runs with high-frequency fit poles."
        ),
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
        "--disable-strength-validity",
        action="store_true",
        help=(
            "Disable S0-based validity threshold when selecting rotation window; "
            "use finite-only validity."
        ),
    )
    parser.add_argument(
        "--enable-nonlinear-diagnostics",
        action="store_true",
        help=(
            "Add cavity DFT monitors and narrow hot-frequency scans around pump/probe "
            "to quantify nonlinear resonance alignment and output-mode projection."
        ),
    )
    parser.add_argument(
        "--diagnostic-scan-points",
        type=int,
        default=41,
        help="Number of frequency samples in each cavity hot-frequency scan.",
    )
    parser.add_argument(
        "--diagnostic-scan-span-factor",
        type=float,
        default=0.75,
        help=(
            "Half-width of each hot-frequency scan, in units of the corresponding "
            "source Gaussian fwidth."
        ),
    )
    parser.add_argument(
        "--diagnostic-cavity-span-fraction",
        type=float,
        default=0.9,
        help=(
            "For 3D runs, fraction of the transverse simulation span included in the "
            "cavity diagnostic monitor."
        ),
    )
    parser.add_argument(
        "--pump1-delay-fs",
        type=float,
        default=0.0,
        help=(
            "Delay of pump1 relative to pump2 and the probe, in fs (pump2 and probe stay "
            "locked together). Positive = pump1 arrives later. 0 reproduces the coincident "
            "three-pulse setup."
        ),
    )
    parser.add_argument(
        "--delay-pad-fs",
        type=float,
        default=None,
        help=(
            "Common start-time offset (fs) applied to ALL sources, so that pump1 is the only "
            "source whose timing changes with --pump1-delay-fs. Required for negative delays: "
            "pass a pad >= |most negative tau| in the scan, held FIXED across the scan. "
            "Without it the code falls back to shifting pump2+probe for tau<0, which keeps the "
            "relative timing but NOT the relative optical phase, so the two halves of a delay "
            "scan are then different experiments. A common offset is physically harmless: it "
            "shifts every field in time, and the Stokes parameters (built from same-frequency "
            "products) are invariant under a global phase."
        ),
    )
    parser.add_argument(
        "--probe-azimuth-deg",
        type=float,
        default=INIT_PROBE_POLARIZATION_DEG,
        help=(
            "Azimuth of the launched probe polarization ellipse, in degrees. Rotation is "
            "reported relative to this. Default 45 (the historical diagonal probe); offsets "
            "model analyzer/waveplate misalignment in balanced detection."
        ),
    )
    parser.add_argument(
        "--probe-ellipticity-deg",
        type=float,
        default=0.0,
        help=(
            "Ellipticity angle chi of the launched probe, in degrees (0 = pure linear, "
            "45 = circular). Models the residual ellipticity of a real probe beam."
        ),
    )
    parser.add_argument(
        "--pump-imbalance",
        type=float,
        default=1.0,
        help=(
            "Pump intensity ratio P2/P1. 1.0 keeps the balanced sigma+/sigma- configuration "
            "that nulls the direct chi3 carrier term; values off 1 reintroduce it."
        ),
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
