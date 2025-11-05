#%%
# Faraday Rotation Tutorial Simulation (3D, step-by-step)
#
# This script demonstrates how to set up a simple 3D pump–probe simulation in Meep
# in a cell with layered dielectric stack (DBR cavity) along +z.  Two circularly
# polarized pumps (opposite handedness) and a linearly polarized probe are launched
# from z = z_src, propagate along +z, and interact with a χ³ (Kerr) SiN cavity.
#
# Every section is deliberately verbose, using #%% to mimic a notebook cell.
# Adjust parameters in the first cells if you want different wavelengths or intensities.

#%%
# Imports and helper utilities

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Set Matplotlib style for clarity
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

#%%
# Units, conversion helpers, and global constants

EPS0 = 8.854187817e-12  # vacuum permittivity (F/m)
C0 = 299792458.0        # speed of light (m/s)
UM_SCALE = 1.0          # Meep uses μm as the base length unit
SCALE_E = 1.0 / (1e-6 * EPS0 * C0)  # convert Meep E-field → SI (V/m)


def intensity_to_meep_amplitude(intensity_w_cm2: float, n_lin: float) -> float:
    """Convert plane-wave intensity (W/cm²) to a Meep electric-field amplitude."""
    intensity_si = intensity_w_cm2 * 1e4  # cm² → m²
    E_SI = np.sqrt(2.0 * intensity_si / (n_lin * EPS0 * C0))
    return float(E_SI / SCALE_E)


def meep_field_to_intensity(E_meep: np.ndarray, n_lin: float) -> np.ndarray:
    """Convert complex Meep E-field envelope to intensity in W/cm²."""
    E_SI = np.abs(E_meep) * SCALE_E
    intensity_si = 0.5 * n_lin * EPS0 * C0 * (E_SI**2)
    return intensity_si / 1e4


def df_from_bandwidth(lam_um: float, dlam_nm: float) -> float:
    """Gaussian fwidth parameter (Meep) from bandwidth in nm."""
    return (dlam_nm * 1e-3) / (lam_um * lam_um)


#%%
# Simulation parameters (explicit values — tweak here)

@dataclass
class RunParams:
    resolution: int
    span_xy: float
    dpml_z: float
    dpml_xy: float
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


# Choose either QUICK or FULL (uncomment the block you prefer)
QUICK = RunParams(
    resolution=10,
    span_xy=0.8,
    dpml_z=1.0,
    dpml_xy=0.6,
    src_buffer=0.25,
    runtime_factor=10,
    pulse_duration_fs=30.0,
    pump_band_nm=25.0,
    probe_band_nm=8.0,
    pump_intensity_w_cm2=5.0e5,
    probe_intensity_w_cm2=1.0e4,
    nonlinear_scale=0.05,
    sample_dt=0.015,
    lp_tau=0.3,
    capture_fields=True,
)

FULL = RunParams(
    resolution=96,
    span_xy=3.0,
    dpml_z=1.0,
    dpml_xy=1.0,
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
)

RUN = QUICK  # change to FULL for the realistic simulation

#%%
# Optical frequencies and polarizations

lam_probe = 0.8      # μm (probe wavelength)
lam_pump1 = 1.48     # μm (pump 1 wavelength)
lam_pump2 = 1.65     # μm (pump 2 wavelength)

freq_probe = 1.0 / lam_probe
freq_pump1 = 1.0 / lam_pump1
freq_pump2 = 1.0 / lam_pump2
delta_omega = abs(freq_pump1 - freq_pump2)
freq_sb_plus = freq_probe + delta_omega
freq_sb_minus = max(freq_probe - delta_omega, 0.0)

# Frequency range
WAVELENGTH_MIN, WAVELENGTH_MAX = 0.6, 2.0

polarization_note = """
Polarization conventions:
  e⁺  = (Ex + i Ey)/√2  (right-handed circular in x–y)
  e⁻  = (Ex - i Ey)/√2  (left-handed circular in x–y)
  Probe (45°) = (Ex + Ey)/√2
"""
print(polarization_note)

#%%
# Materials (SiO2 and SiN) with optional χ³ in SiN

index_sio2 = 1.45
index_sin = 2.0

n2_sin = 2.5e-19  # m²/W

mat_sio2 = mp.Medium(index=index_sio2)

chi3_si = (4.0 / 3.0) * n2_sin * (index_sin**2) * EPS0 * C0
E_chi3_meep = chi3_si * (SCALE_E**3) * RUN.nonlinear_scale
mat_sin = mp.Medium(index=index_sin, E_chi3_diag=mp.Vector3(E_chi3_meep, E_chi3_meep, E_chi3_meep))

mat_air = mp.Medium(index=1.0)

#%%
# Geometry definition (DBR cavity along z)

pad_air = 0.8
pad_sub = 3.0
dpml = RUN.dpml_z
cell_margin = 0.4

left_mirror = [
    ("SiN", 0.158),
    ("SiO2", 0.318),
    ("SiN", 0.158),
    ("SiO2", 0.318),
    ("SiN", 0.158),
    ("SiO2", 0.318),
]
right_mirror = [
    ("SiO2", 0.318),
    ("SiN", 0.158),
    ("SiO2", 0.318),
    ("SiN", 0.158),
    ("SiO2", 0.318),
    ("SiN", 0.158),
]
spacer_left = 0.0
spacer_right = 0.0
cavity_thickness = 2.1
cavity_material = mat_sin

material_map: Dict[str, mp.Medium] = {
    "SiN": mat_sin,
    "SiO2": mat_sio2,
    "air": mat_air,
}


def build_stack(core_span: float) -> Tuple[List[mp.Block], float, float]:
    total_length = (
        pad_air
        + sum(thk for _, thk in left_mirror)
        + spacer_left
        + cavity_thickness
        + spacer_right
        + sum(thk for _, thk in right_mirror)
        + pad_sub
    )
    cell_z = total_length + 2 * dpml + cell_margin
    geometry: List[mp.Block] = []

    def add_layer(z_start: float, thickness: float, mat: mp.Medium) -> float:
        center = z_start + 0.5 * thickness
        geometry.append(
            mp.Block(
                center=mp.Vector3(0, 0, center),
                size=mp.Vector3(core_span, core_span, thickness),
                material=mat,
            )
        )
        return z_start + thickness

    z = -0.5 * cell_z + dpml
    z += pad_air
    for mat_name, thk in left_mirror:
        z = add_layer(z, thk, material_map[mat_name])
    if spacer_left > 0:
        z = add_layer(z, spacer_left, mat_sio2)
    cavity_start = z
    z = add_layer(z, cavity_thickness, cavity_material)
    cavity_center = cavity_start + 0.5 * cavity_thickness
    if spacer_right > 0:
        z = add_layer(z, spacer_right, mat_sio2)
    for mat_name, thk in right_mirror:
        z = add_layer(z, thk, material_map[mat_name])
    z = add_layer(z, pad_sub, mat_sio2)
    return geometry, cell_z, cavity_center


GEOMETRY, CELL_Z, CAVITY_CENTER = build_stack(RUN.span_xy)

print(f"Total cell height: {CELL_Z:.3f} μm, cavity center at z = {CAVITY_CENTER:.3f} μm")

#%%
# Visualize ε(x,z) cross-section (no sources)

sim_eps = mp.Simulation(
    cell_size=mp.Vector3(RUN.span_xy + 2 * RUN.dpml_xy, RUN.span_xy + 2 * RUN.dpml_xy, CELL_Z),
    geometry=GEOMETRY,
    boundary_layers=[mp.PML(RUN.dpml_z, direction=mp.Z)],
    default_material=mp.air,
    resolution=RUN.resolution,
)

sim_eps.init_sim()
eps_data = sim_eps.get_array(
    center=mp.Vector3(),
    size=mp.Vector3(RUN.span_xy, 0, CELL_Z - 2 * RUN.dpml_z),
    component=mp.Dielectric,
)

extent_x = (-0.5 * RUN.span_xy, 0.5 * RUN.span_xy)
extent_z = (-0.5 * (CELL_Z - 2 * RUN.dpml_z), 0.5 * (CELL_Z - 2 * RUN.dpml_z))

plt.figure(figsize=(6, 4))
plt.imshow(
    eps_data.T,
    origin="lower",
    aspect="auto",
    extent=(extent_x[0], extent_x[1], extent_z[0], extent_z[1]),
    cmap="viridis",
)
plt.colorbar(label="ε")
plt.xlabel("x (μm)")
plt.ylabel("z (μm)")
plt.title("Dielectric profile (x–z plane)")
plt.tight_layout()
plt.show()

#%%
# Gaussian sources (pumps and probe)

df_probe = df_from_bandwidth(lam_probe, RUN.probe_band_nm)
df_pump1 = df_from_bandwidth(lam_pump1, RUN.pump_band_nm)
df_pump2 = df_from_bandwidth(lam_pump2, RUN.pump_band_nm)

amp_probe = intensity_to_meep_amplitude(RUN.probe_intensity_w_cm2, index_sin)
amp_pump1 = intensity_to_meep_amplitude(RUN.pump_intensity_w_cm2, index_sin)
amp_pump2 = intensity_to_meep_amplitude(RUN.pump_intensity_w_cm2, index_sin)

cell = mp.Vector3(RUN.span_xy + 2 * RUN.dpml_xy, RUN.span_xy + 2 * RUN.dpml_xy, CELL_Z)

src_z = -0.5 * CELL_Z + RUN.dpml_z + RUN.src_buffer
src_center = mp.Vector3(0, 0, src_z)
src_span = RUN.span_xy + 2*RUN.dpml_xy  # max(RUN.span_xy - 2 * RUN.src_buffer, 0.2)
src_size = mp.Vector3(src_span, src_span, 0)

sources: List[mp.Source] = []

def add_circular_source(frequency: float, fwidth: float, amplitude: float, handedness: str):
    phase = 1.0j if handedness == "plus" else -1.0j
    amp = amplitude / np.sqrt(2.0)
    pulse = mp.GaussianSource(frequency=frequency, fwidth=fwidth, cutoff=4.0)
    sources.append(mp.Source(pulse, component=mp.Ex, center=src_center, size=src_size, amplitude=amp))
    sources.append(mp.Source(pulse, component=mp.Ey, center=src_center, size=src_size, amplitude=amp * phase))


def add_linear_source(frequency: float, fwidth: float, amplitude: float):
    amp = amplitude / np.sqrt(2.0)
    pulse = mp.GaussianSource(frequency=frequency, fwidth=fwidth, cutoff=4.0)
    sources.append(mp.Source(pulse, component=mp.Ex, center=src_center, size=src_size, amplitude=amp))
    sources.append(mp.Source(pulse, component=mp.Ey, center=src_center, size=src_size, amplitude=amp))


add_circular_source(freq_pump1, df_pump1, amp_pump1, "plus")
add_circular_source(freq_pump2, df_pump2, amp_pump2, "minus")
add_linear_source(freq_probe, df_probe, amp_probe)

#%%
# Set up simulation with PML

boundary_layers = [mp.PML(RUN.dpml_z, direction=mp.Z)]
if RUN.dpml_xy > 0:
    boundary_layers.extend([
        mp.PML(RUN.dpml_xy, direction=mp.X),
        mp.PML(RUN.dpml_xy, direction=mp.Y),
    ])

simulation = mp.Simulation(
    cell_size=cell,
    geometry=GEOMETRY,
    sources=sources,
    boundary_layers=boundary_layers,
    default_material=mp.air,
    resolution=RUN.resolution,
    force_complex_fields=True,
)

# DFT monitors (transmitted fields)

dft_span_xy = mp.Volume(center=mp.Vector3(0, 0, 0.4 * CELL_Z), size=mp.Vector3(src_span, src_span, 0))

# # add monitors
wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, 41) # wavelengths for monitors
frequencies = 1.0 / wavelengths

# # Domain DFT monitor
dft_freqs = [freq_pump1, freq_pump2, freq_probe, freq_sb_minus, freq_sb_plus]

# --- frequency grids and storage for time traces ---
# For the 5 fixed frequencies monitor:
fixed_freqs = np.array(dft_freqs, dtype=float)  # [pump1, pump2, probe, sb-, sb+]

# For the probe-band monitor (fcen/df/nfreq): construct the explicit frequency grid
nfreq_probe = 15
probe_freqs = np.linspace(freq_probe - 0.5*df_probe, freq_probe + 0.5*df_probe, nfreq_probe)  # docs: fcen±df/2
k_probe_center = nfreq_probe // 2 + 1 # center index is the true probe frequency


dft_fields = simulation.add_dft_fields([mp.Ex, mp.Ey],
                                dft_freqs, 
                                where=dft_span_xy)
trans_monitor = simulation.add_dft_fields(
    [mp.Ex, mp.Ey],
    freq_probe,
    df_probe,
    15,
    where=dft_span_xy,
)

# XZ DFT monitor plane at y = 0 (for spatial slices)
dft_span_xz = mp.Volume(
    center=mp.Vector3(0, 0, 0),
    size=mp.Vector3(src_span, 0, CELL_Z - 2 * RUN.dpml_z),
)
dft_fields_xz = simulation.add_dft_fields([mp.Ex, mp.Ey], dft_freqs, where=dft_span_xz)

#%%
simulation.plot2D(output_plane = mp.Volume(center = mp.Vector3(), size = mp.Vector3(3*RUN.span_xy,0, CELL_Z)))

#%%
# Time-domain sampling setup

pulse_duration_meep = RUN.pulse_duration_fs * 1e9 / C0
stop_time = RUN.runtime_factor * pulse_duration_meep

sample_point = mp.Vector3(0, 0, CAVITY_CENTER + 0.05)


snapshot_time = 0.6 * stop_time

# Allocate time-series containers (plane-averaged magnitudes)
time_trace = {
    "t": [],
    "fixed": {  # 5 arbitrary freqs
        "Ex": [], "Ey": [], "absE": []  # each entry is shape (5,)
    },
    "probe_band": {  # 15 freqs around probe
        "Ex": [], "Ey": [], "absE": []  # each entry is shape (15,)
    },
    "probe_pol": {  # polarization angle from center frequency
        "theta_deg": [],  # Faraday rotation angle vs time
        "Ix": [], "Iy": [],  # optional diagnostics
    },
}

# For an XZ snapshot at a specific time:
SNAP_T = snapshot_time  # already defined in your script
xz_snapshot = {
    "taken": False,
    "t": None,
    "freqs": fixed_freqs,
    "Ex_maps": {},  # maps freq -> 2D array Ex(x,z)
    "Ey_maps": {},  # maps freq -> 2D array Ey(x,z)
}

def _plane_avg_mag(arr_ex, arr_ey):
    """Return plane-averaged |E| for complex arrays Ex, Ey of same shape."""
    mag = np.sqrt(np.abs(arr_ex)**2 + np.abs(arr_ey)**2)
    return float(np.mean(mag))

def _stokes_theta_deg(Ex, Ey):
    """
    Linear polarization angle (deg) from complex Ex,Ey (Jones) on a plane:
    S1=|Ex|^2-|Ey|^2, S2=2 Re(Ex Ey*), theta=0.5*atan2(S2,S1).
    """
    S1 = np.mean(np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = 2.0 * np.mean(np.real(Ex * np.conj(Ey)))
    theta = 0.5 * np.arctan2(S2, S1)  # radians
    return np.degrees(theta)


def sample_callback(sim: mp.Simulation):
    # current Meep time (arbitrary units)
    t = sim.meep_time()
    time_trace["t"].append(t)

    # --- 4a) Fixed-frequency DFT monitor (5 freqs) ---
    ex_vals = []
    ey_vals = []
    abs_vals = []
    for idx in range(len(fixed_freqs)):
        Ex = np.asarray(sim.get_dft_array(dft_fields, mp.Ex, idx))
        Ey = np.asarray(sim.get_dft_array(dft_fields, mp.Ey, idx))
        ex_vals.append(np.mean(Ex))  # complex average (optional)
        ey_vals.append(np.mean(Ey))
        abs_vals.append(_plane_avg_mag(Ex, Ey))
    time_trace["fixed"]["Ex"].append(np.array(ex_vals))
    time_trace["fixed"]["Ey"].append(np.array(ey_vals))
    time_trace["fixed"]["absE"].append(np.array(abs_vals))

    # --- 4b) Probe-band DFT monitor (fcen/df/nfreq) ---
    ex_pb = []
    ey_pb = []
    abs_pb = []
    for k in range(nfreq_probe):
        Exk = np.asarray(sim.get_dft_array(trans_monitor, mp.Ex, k))
        Eyk = np.asarray(sim.get_dft_array(trans_monitor, mp.Ey, k))
        ex_pb.append(np.mean(Exk))
        ey_pb.append(np.mean(Eyk))
        abs_pb.append(_plane_avg_mag(Exk, Eyk))
    ex_pb = np.array(ex_pb); ey_pb = np.array(ey_pb); abs_pb = np.array(abs_pb)
    time_trace["probe_band"]["Ex"].append(ex_pb)
    time_trace["probe_band"]["Ey"].append(ey_pb)
    time_trace["probe_band"]["absE"].append(abs_pb)

    # --- 4c) Polarization angle of the probe (center frequency only) ---
    Ex_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Ex, k_probe_center))
    Ey_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Ey, k_probe_center))
    theta_deg = _stokes_theta_deg(Ex_c, Ey_c)  # average over the plane
    time_trace["probe_pol"]["theta_deg"].append(theta_deg)
    time_trace["probe_pol"]["Ix"].append(float(np.mean(np.abs(Ex_c)**2)))
    time_trace["probe_pol"]["Iy"].append(float(np.mean(np.abs(Ey_c)**2)))

    # --- 4d) One-time XZ snapshot of spatial maps at SNAP_T (optional) ---
    if (not xz_snapshot["taken"]) and (t >= SNAP_T):
        for i, f in enumerate(fixed_freqs):
            Ex_map = np.asarray(sim.get_dft_array(dft_fields_xz, mp.Ex, i))
            Ey_map = np.asarray(sim.get_dft_array(dft_fields_xz, mp.Ey, i))
            xz_snapshot["Ex_maps"][f] = Ex_map
            xz_snapshot["Ey_maps"][f] = Ey_map
        xz_snapshot["taken"] = True
        xz_snapshot["t"] = t



#%%
# Run simulation

simulation.run(mp.at_every(RUN.sample_dt, sample_callback),
            #    until=stop_time
               until_after_sources=mp.stop_when_fields_decayed(
    50, mp.Ex, mp.Vector3(0,0,0.4*CELL_Z), 1e-7)
               )

#%%
# Convert lists → arrays
t_arr = np.array(time_trace["t"])
fixed_Ex  = np.vstack(time_trace["fixed"]["Ex"])      # shape (Nt, 5)
fixed_Ey  = np.vstack(time_trace["fixed"]["Ey"])
fixed_abs = np.vstack(time_trace["fixed"]["absE"])

probe_Ex  = np.vstack(time_trace["probe_band"]["Ex"])  # shape (Nt, 15)
probe_Ey  = np.vstack(time_trace["probe_band"]["Ey"])
probe_abs = np.vstack(time_trace["probe_band"]["absE"])
theta_deg = np.array(time_trace["probe_pol"]["theta_deg"])

#%%

# --- (A) |E| vs time for the 5 fixed frequencies ---
plt.figure(figsize=(7.4, 4.2))
for i, f in enumerate(fixed_freqs):
    plt.plot(t_arr, fixed_abs[:, i], label=f"f={f:.3f}")
plt.xlabel("time (Meep units)"); plt.ylabel(r"$\langle |E| \rangle$ (plane-avg)")
plt.title("|E| at five DFT frequencies vs time")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

#%%

# --- (B) Probe-band: |E| vs frequency and time (heatmap) ---
plt.figure(figsize=(7.4, 4.2))
plt.imshow(np.abs(probe_abs.T), aspect="auto", origin="lower",
           extent=[t_arr.min(), t_arr.max(), probe_freqs.min(), probe_freqs.max()])
plt.colorbar(label=r"$\langle |E| \rangle$")
plt.xlabel("time (Meep units)"); plt.ylabel("frequency (1/μm)")
plt.title("Probe-band plane-avg |E| vs (f, t)")
plt.tight_layout(); plt.show()

#%%

# --- (C) Faraday rotation angle of probe vs time ---
plt.figure(figsize=(7.0, 3.6))
plt.plot(t_arr, theta_deg, "k-")
plt.xlabel("time (Meep units)"); plt.ylabel("polarization angle (deg)")
plt.title("Probe polarization angle vs time (center frequency)")
plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

#%%

# --- (D) X–Z spatial maps at SNAP_T (optional) ---
if xz_snapshot["taken"]:
    f_show = fixed_freqs  # plot all; or pick a subset
    ncols = min(3, len(f_show)); nrows = int(np.ceil(len(f_show)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, f_show):
        Em = np.sqrt(np.abs(xz_snapshot["Ex_maps"][f])**2 + np.abs(xz_snapshot["Ey_maps"][f])**2)
        im = ax.imshow(np.abs(Em).T, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"|E|(x,z) at f={f:.3f}, t≈{xz_snapshot['t']:.2f}")
        ax.set_xlabel("x index"); ax.set_ylabel("z index")
    for ax in axes[len(f_show):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.9, label="|E|")
    fig.tight_layout(); plt.show()

# %%
