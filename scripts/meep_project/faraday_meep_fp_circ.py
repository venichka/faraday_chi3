#!/usr/bin/env python3
"""
Nonlinear cavity pump–probe simulation with dispersive DBR stack.

Geometry is imported from optimized_geometry.json (produced by the optimizer),
and the resonant wavelengths/frequencies are loaded from cavity_modes.json
(written by fp_cavity_modes_spectrum.py). Two strong pumps (counter-rotating
polarizations) and one weak probe interact with an isotropic χ³ in the SiN
cavity. We record temporal envelopes, polarization rotation, and field maps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from geometry_io import load_params
from mode_targeting import (
    build_stack_from_params,
    get_cavity_materials,
    material_index_at_wavelength,
)

# --------------------------------------------------------------------------- #
# Unit helpers
# --------------------------------------------------------------------------- #
UM = 1.0
EPS0 = 8.854187817e-12
C0 = 299792458.0

# conversion from Meep electric-field units (with μm length units) to SI
SCALE_E = 1.0 / (1e-6 * EPS0 * C0)


def meep_field_to_intensity(E_meep: np.ndarray, n_lin: float) -> np.ndarray:
    """Return intensity in W/cm^2 for a complex field envelope."""
    E_SI = np.abs(E_meep) * SCALE_E
    I_SI = 0.5 * n_lin * EPS0 * C0 * (E_SI**2)
    return I_SI / 1e4


def intensity_to_meep_amplitude(I_SI_per_cm2: float, n_lin: float) -> float:
    """Convert desired intensity (W/cm^2) to a Meep field amplitude."""
    I_SI = I_SI_per_cm2 * 1e4  # → W/m^2
    E_SI = np.sqrt(2.0 * I_SI / (n_lin * EPS0 * C0))
    return E_SI / SCALE_E


def df_from_bandwidth(lam_um: float, dlam_um: float) -> float:
    """Frequency half-width (GaussianSource fwidth parameter) from Δλ."""
    return dlam_um / (lam_um * lam_um)


# --------------------------------------------------------------------------- #
# Data classes / trackers
# --------------------------------------------------------------------------- #
@dataclass
class EnvelopeTracker:
    freq: float
    tau: float
    value: complex = 0.0 + 0.0j

    def update(self, field_sample: complex, time: float, dt: float) -> complex:
        alpha = min(1.0, dt / self.tau) if self.tau > 0 else 1.0
        self.value = (1.0 - alpha) * self.value + alpha * field_sample * np.exp(
            -2j * np.pi * self.freq * time
        )
        return self.value


# --------------------------------------------------------------------------- #
# Load geometry & resonant mode data
# --------------------------------------------------------------------------- #
def load_mode_summary(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Resonance summary {path} not found. Run fp_cavity_modes_spectrum.py first."
        )
    with path.open("r") as f:
        data = json.load(f)
    return data


# --------------------------------------------------------------------------- #
# Simulation setup
# --------------------------------------------------------------------------- #
params = load_params(prefer="report")
mode_summary = load_mode_summary(Path("cavity_modes.json"))

# Extract resonant wavelengths/frequencies
lam_probe = mode_summary["probe"]["lambda_um"]
lam_p1 = mode_summary["pump1"]["lambda_um"]
lam_p2 = mode_summary["pump2"]["lambda_um"]
freq_probe = mode_summary["probe"]["frequency"]
freq_p1 = mode_summary["pump1"]["frequency"]
freq_p2 = mode_summary["pump2"]["frequency"]
delta_omega = abs(freq_p1 - freq_p2)
freq_sb_plus = freq_probe + delta_omega
freq_sb_minus = max(freq_probe - delta_omega, 0.0)

# --------------------------------------------------------------------------- #
# Materials (dispersive linear response + χ³)
# --------------------------------------------------------------------------- #
mat_SiN, mat_SiO2 = get_cavity_materials("library")

# χ³ for Si3N4 (n₂ ≈ 2.5e-19 m²/W)
n2_sin_SI = 2.5e-19
n_sin_probe = material_index_at_wavelength(mat_SiN, lam_probe)
chi3_SI = (4.0 / 3.0) * n2_sin_SI * (n_sin_probe**2) * EPS0 * C0
E_chi3_meep = chi3_SI * (SCALE_E**3)
mat_SiN.E_chi3_diag = mp.Vector3(E_chi3_meep, E_chi3_meep, E_chi3_meep)

# indices at specific wavelengths (for intensity conversion)
n_probe = material_index_at_wavelength(mat_SiN, lam_probe)
n_pump1 = material_index_at_wavelength(mat_SiN, lam_p1)
n_pump2 = material_index_at_wavelength(mat_SiN, lam_p2)

# --------------------------------------------------------------------------- #
# Geometry (1D stack along z, finite cross-section in x/y)
# --------------------------------------------------------------------------- #
N_per = params["N_per"]
t_SiN = params["t_SiN"]
t_SiO2 = params["t_SiO2"]
L_cav = params["t_cav"]
pad_air = params["pad_air"]
pad_sub = params["pad_sub"]
dpml = params["dpml"]
cell_margin = params.get("cell_margin", 0.4)

stack_len = pad_air + N_per * (t_SiN + t_SiO2) + L_cav + N_per * (t_SiN + t_SiO2) + pad_sub
cell_z = stack_len + 2 * dpml + cell_margin
span_xy = 3.0  # μm; finite cross-section to allow polarization textures
cell = mp.Vector3(span_xy, span_xy, cell_z)

geometry = build_stack_from_params(
    N_per=N_per,
    t_SiN=t_SiN,
    t_SiO2=t_SiO2,
    L_cav=L_cav,
    dpml=dpml,
    pad_air=pad_air,
    pad_sub=pad_sub,
    mat_SiN=mat_SiN,
    mat_SiO2=mat_SiO2,
    cell_z=cell_z,
)

z_pad_air = dpml + 0.5 * cell_margin
z_left_interface = (
    -0.5 * cell_z + dpml + pad_air + N_per * (t_SiN + t_SiO2)
)
z_cavity_center = z_left_interface + 0.5 * L_cav
z_monitor = 0.5 * cell_z - dpml - 0.4

pml_layers = [mp.PML(dpml, direction=mp.Z)]

# --------------------------------------------------------------------------- #
# Temporal and spectral properties of the sources
# --------------------------------------------------------------------------- #
band_probe_nm = 10.0
band_pump_nm = 30.0
df_probe = df_from_bandwidth(lam_probe, band_probe_nm * 1e-3)
df_pump1 = df_from_bandwidth(lam_p1, band_pump_nm * 1e-3)
df_pump2 = df_from_bandwidth(lam_p2, band_pump_nm * 1e-3)

pulse_duration_fs = 100.0
pulse_duration_meep = pulse_duration_fs * 1e9 / C0  # convert fs → Meep time units
pump_cutoff = 4.0

# Intensities
I_pump_W_cm2 = 1.0e12  # 1 TW/cm^2
I_probe_W_cm2 = 1.0e7  # 10 MW/cm^2

E0_pump1 = intensity_to_meep_amplitude(I_pump_W_cm2, n_pump1)
E0_pump2 = intensity_to_meep_amplitude(I_pump_W_cm2, n_pump2)
E0_probe = intensity_to_meep_amplitude(I_probe_W_cm2, n_probe)


def gaussian_source(frequency: float, fwidth: float, amplitude: complex) -> mp.Source:
    return mp.Source(
        src=mp.GaussianSource(
            frequency=frequency,
            fwidth=fwidth,
            cutoff=pump_cutoff,
            start_time=0.0,
        ),
        component=mp.Ey,
        center=mp.Vector3(0, 0, -0.5 * cell_z + dpml + 0.3),
        size=mp.Vector3(span_xy, span_xy, 0),
        amplitude=0.0,  # overwritten per polarization
    )


def circular_sources(
    frequency: float,
    fwidth: float,
    amplitude: float,
    handedness: str,
) -> List[mp.Source]:
    """Return Ey/Ez sources for circular polarization (handedness 'plus' or 'minus')."""
    phase = 1.0j if handedness == "plus" else -1.0j
    amp = amplitude / np.sqrt(2.0)
    base = mp.GaussianSource(
        frequency=frequency,
        fwidth=fwidth,
        cutoff=pump_cutoff,
        start_time=0.0,
    )
    center = mp.Vector3(0, 0, -0.5 * cell_z + dpml + 0.3)
    size = mp.Vector3(span_xy, span_xy, 0)
    return [
        mp.Source(src=base, component=mp.Ey, center=center, size=size, amplitude=amp),
        mp.Source(src=base, component=mp.Ez, center=center, size=size, amplitude=amp * phase),
    ]


def linear_source_45deg(frequency: float, fwidth: float, amplitude: float) -> List[mp.Source]:
    amp = amplitude / np.sqrt(2.0)
    base = mp.GaussianSource(
        frequency=frequency,
        fwidth=fwidth,
        cutoff=pump_cutoff,
        start_time=0.0,
    )
    center = mp.Vector3(0, 0, -0.5 * cell_z + dpml + 0.3)
    size = mp.Vector3(span_xy, span_xy, 0)
    return [
        mp.Source(src=base, component=mp.Ey, center=center, size=size, amplitude=amp),
        mp.Source(src=base, component=mp.Ez, center=center, size=size, amplitude=amp),
    ]


sources: List[mp.Source] = []
sources += circular_sources(freq_p1, df_pump1, E0_pump1, handedness="plus")
sources += circular_sources(freq_p2, df_pump2, E0_pump2, handedness="minus")
sources += linear_source_45deg(freq_probe, df_probe, E0_probe)

# --------------------------------------------------------------------------- #
# Simulation object
# --------------------------------------------------------------------------- #
resolution = int(params["resolution"])
simulation = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
    force_complex_fields=True,
)

# --------------------------------------------------------------------------- #
# Time-domain diagnostics
# --------------------------------------------------------------------------- #
sample_point = mp.Vector3(0, 0, z_cavity_center)
sample_dt = 0.05  # Meep time units (~0.17 fs)
lp_tau = 0.8  # low-pass constant in Meep units (~2.7 fs)

trackers: Dict[str, EnvelopeTracker] = {
    "probe_plus": EnvelopeTracker(freq_probe, lp_tau),
    "probe_minus": EnvelopeTracker(freq_probe, lp_tau),
    "probe_Ey": EnvelopeTracker(freq_probe, lp_tau),
    "probe_Ez": EnvelopeTracker(freq_probe, lp_tau),
    "pump1_plus": EnvelopeTracker(freq_p1, lp_tau),
    "pump2_minus": EnvelopeTracker(freq_p2, lp_tau),
    "sb_plus_plus": EnvelopeTracker(freq_sb_plus, lp_tau),
    "sb_minus_minus": EnvelopeTracker(freq_sb_minus, lp_tau),
}

histories: Dict[str, List[complex]] = {key: [] for key in trackers}
time_samples: List[float] = []

# Snapshot storage
snapshot_time = 3.0 * pulse_duration_meep
field_planes: Dict[str, np.ndarray] = {}


def sample_callback(sim: mp.Simulation) -> None:
    t = sim.meep_time()
    Ey = sim.get_field_point(mp.Ey, sample_point)
    Ez = sim.get_field_point(mp.Ez, sample_point)
    E_plus = (Ey + 1j * Ez) / np.sqrt(2.0)
    E_minus = (Ey - 1j * Ez) / np.sqrt(2.0)

    time_samples.append(t)

    trackers["probe_plus"].update(E_plus, t, sample_dt)
    trackers["probe_minus"].update(E_minus, t, sample_dt)
    trackers["probe_Ey"].update(Ey, t, sample_dt)
    trackers["probe_Ez"].update(Ez, t, sample_dt)
    trackers["pump1_plus"].update(E_plus, t, sample_dt)
    trackers["pump2_minus"].update(E_minus, t, sample_dt)
    trackers["sb_plus_plus"].update(E_plus, t, sample_dt)
    trackers["sb_minus_minus"].update(E_minus, t, sample_dt)

    for key, tracker in trackers.items():
        histories[key].append(tracker.value)

    if "xy" not in field_planes and t >= snapshot_time:
        xy_plane = sim.get_array(
            center=mp.Vector3(0, 0, z_cavity_center),
            size=mp.Vector3(span_xy, span_xy, 0),
            component=mp.Ey,
        )
        xz_plane = sim.get_array(
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(span_xy, 0, cell_z - 2 * dpml),
            component=mp.Ey,
        )
        field_planes["xy"] = xy_plane
        field_planes["xz"] = xz_plane


simulation.run(
    mp.at_every(sample_dt, sample_callback),
    until=6.0 * pulse_duration_meep,
)

# --------------------------------------------------------------------------- #
# Post-processing
# --------------------------------------------------------------------------- #
times = np.array(time_samples)
times_fs = times * 1e9 / C0

I_probe_plus = meep_field_to_intensity(np.array(histories["probe_plus"]), n_probe)
I_probe_minus = meep_field_to_intensity(np.array(histories["probe_minus"]), n_probe)
I_pump1 = meep_field_to_intensity(np.array(histories["pump1_plus"]), n_pump1)
I_pump2 = meep_field_to_intensity(np.array(histories["pump2_minus"]), n_pump2)
I_sb_plus = meep_field_to_intensity(np.array(histories["sb_plus_plus"]), n_probe)
I_sb_minus = meep_field_to_intensity(np.array(histories["sb_minus_minus"]), n_probe)

Ey_env = np.array(histories["probe_Ey"])
Ez_env = np.array(histories["probe_Ez"])
Q = np.abs(Ey_env) ** 2 - np.abs(Ez_env) ** 2
U = 2.0 * np.real(Ey_env * np.conjugate(Ez_env))
theta_deg = 0.5 * np.degrees(np.arctan2(U, Q))

# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
fig1, ax1 = plt.subplots(figsize=(9, 4))
ax1.plot(times_fs, I_pump1, label=r"$\omega_1$ (e$^+$)")
ax1.plot(times_fs, I_pump2, label=r"$\omega_2$ (e$^-$)")
ax1.plot(times_fs, I_probe_plus, label=r"$\omega_s$ (e$^+$)")
ax1.plot(times_fs, I_probe_minus, label=r"$\omega_s$ (e$^-)$")
ax1.plot(times_fs, I_sb_plus, label=r"$\omega_s + (\omega_1-\omega_2)$")
ax1.plot(times_fs, I_sb_minus, label=r"$\omega_s - (\omega_1-\omega_2)$")
ax1.set_xlabel("time (fs)")
ax1.set_ylabel("Intensity (W/cm²)")
ax1.set_title("Field intensities vs time at cavity center")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)

fig2, ax2 = plt.subplots(figsize=(9, 3))
ax2.plot(times_fs, theta_deg)
ax2.set_xlabel("time (fs)")
ax2.set_ylabel("Probe polarization rotation θ (deg)")
ax2.set_title("Probe polarization rotation in time")
ax2.grid(True, alpha=0.3)

if "xy" in field_planes:
    xy = field_planes["xy"]
    nx, ny = xy.shape
    extent_xy = (
        -0.5 * span_xy,
        0.5 * span_xy,
        -0.5 * span_xy,
        0.5 * span_xy,
    )
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    im1 = ax3.imshow(
        np.abs(xy.T),
        origin="lower",
        extent=extent_xy,
        cmap="inferno",
        aspect="equal",
    )
    ax3.set_xlabel("x (μm)")
    ax3.set_ylabel("y (μm)")
    ax3.set_title("Field magnitude |E_y| (x–y plane)")
    fig3.colorbar(im1, ax=ax3)

if "xz" in field_planes:
    xz = field_planes["xz"]
    nx, nz = xz.shape
    extent_xz = (
        -0.5 * span_xy,
        0.5 * span_xy,
        -0.5 * (cell_z - 2 * dpml),
        0.5 * (cell_z - 2 * dpml),
    )
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    im2 = ax4.imshow(
        np.abs(xz.T),
        origin="lower",
        extent=extent_xz,
        cmap="inferno",
        aspect="auto",
    )
    ax4.set_xlabel("x (μm)")
    ax4.set_ylabel("z (μm)")
    ax4.set_title("Field magnitude |E_y| (x–z plane)")
    fig4.colorbar(im2, ax=ax4)

plt.tight_layout()
plt.show()
