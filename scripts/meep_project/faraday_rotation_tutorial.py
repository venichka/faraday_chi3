#!/usr/bin/env python3
# %%
# Faraday pump–probe (3D) using your helper APIs:
#   - geometry_io.material_factory, geometry_io.read_json
#   - mode_targeting.get_cavity_materials, mode_targeting.material_index_at_wavelength
#
# Features:
#   • Loads geometry/materials from optimized_geometry.json via your loaders
#   • Optionally replaces SiN/SiO2 with fitted dispersive media via get_cavity_materials
#   • Enforces a “no dispersive medium in PML” clearance
#   • DFT monitors at fixed freqs (pumps + probe + sidebands) and a probe-band monitor
#   • Time-domain demodulated envelopes at the same fixed freqs
#   • e+/e− decomposition for pumps, probe, sidebands
#   • Probe polarization angle from center DFT bin (and subtracts initial polarization)
#
# NOTE: The script assumes your project provides:
#   from geometry_io     import material_factory, read_json as load_geometry_json
#   from mode_targeting  import get_cavity_materials, material_index_at_wavelength

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

import numpy as np
import matplotlib.pyplot as plt
import meep as mp

from geometry_io import material_factory, read_json as load_geometry_json
from mode_targeting import get_cavity_materials, material_index_at_wavelength

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

#%%
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

# Toggle whether to use your fitted dispersive materials for SiN/SiO2
USE_FITTED_MATERIALS = True   # set True to pull CSV fits via get_cavity_materials

# JSON inputs produced by your pipeline
PATH_GEOM  = Path("optimized_geometry_experiment.json")   # stack + material entries
PATH_MODES = Path("cavity_modes_experiment.json")         # probe/pump wavelengths

# Safety margins / placement
MARGIN_Z_CELL = 0.40   # extra padding around PMLs
MARGIN_ZTR    = 0.20   # location of transmission DFT plane before +z PML [um]
CLEARANCE_PML = 0.10   # min gap between any solid/dispersive region and PML [um]

# Manual fallbacks if files are absent (only used when missing)
WL_MANUAL = dict(probe=0.80, pump1=1.48, pump2=1.65)  # μm

# Nonlinearity (SI)
N2_SIN = 2*2.5e-19  # m^2/W

#%%
# ------------------------------------------------------------------------------
# Run profile
# ------------------------------------------------------------------------------

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
    lp_tau: float       # low-pass time constant for TD demod (meep time units)
    capture_fields: bool

QUICK = RunParams(
    resolution=30,
    span_xy=0.8,
    dpml_z=2.0,
    dpml_xy=1.6,
    src_buffer=0.25,
    runtime_factor=0.01,
    pulse_duration_fs=100.0,
    pump_band_nm=10.0,
    probe_band_nm=30.0,
    pump_intensity_w_cm2=1.0e6,
    probe_intensity_w_cm2=1.0e3,
    nonlinear_scale=1.0,
    sample_dt=0.015,
    lp_tau=0.8,
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

RUN = QUICK  # change to FULL for the larger run

#%%
# ------------------------------------------------------------------------------
# Units & helpers
# ------------------------------------------------------------------------------

EPS0 = 8.854187817e-12
C0   = 299792458.0
SCALE_E = 1.0 / (1e-6 * EPS0 * C0)  # Meep E -> SI (V/m)

def intensity_to_meep_amplitude(I_w_cm2: float, n_lin: float) -> float:
    I_SI = I_w_cm2 * 1e4
    E_SI = np.sqrt(2.0 * I_SI / (n_lin * EPS0 * C0))
    return float(E_SI / SCALE_E)

def df_from_bandwidth(lam_um: float, dlam_nm: float) -> float:
    # Convert Δλ (nm) around λ to Meep fwidth (Δf, in 1/μm)
    return (dlam_nm * 1e-3) / (lam_um * lam_um)

# ------------------------------------------------------------------------------
# File I/O via your helpers
# ------------------------------------------------------------------------------

def load_modes(path: Path) -> Dict[str, float]:
    if path.exists():
        d = load_geometry_json(str(path))
        return dict(
            probe=float(d["probe"]["lambda_um"]),
            pump1=float(d["pump1"]["lambda_um"]),
            pump2=float(d["pump2"]["lambda_um"]),
        )
    return WL_MANUAL.copy()

def build_geometry_from_spec(
    spec: Dict,
    mats: Dict[str, mp.Medium],
    span_xy: float,
    dpml_z: float,
    margin_z: float
) -> Tuple[List[mp.Block], float, float, float, float]:
    """
    Build Meep geometry from your optimized_geometry.json spec.
    Returns (geometry, cell_z, cavity_center, zmin_solid, zmax_solid).
    """
    pad_air = float(spec["pads"]["air_um"])
    pad_sub = float(spec["pads"]["substrate_um"])
    dpml_spec = float(spec["pads"]["pml_um"])
    dpml_z = max(dpml_z, dpml_spec)  # abide by spec’s pml thickness if larger

    left_layers  = spec["mirrors"]["left"]
    right_layers = spec["mirrors"]["right"]
    cavity_thk   = float(spec["cavity"]["L_um"])
    cavity_mat   = mats[spec["cavity"]["mat"]]
    spacer_left  = float(spec.get("spacers", {}).get("left_um", 0.0))
    spacer_right = float(spec.get("spacers", {}).get("right_um", 0.0))

    def lsum(layers: Sequence[Dict]) -> float:
        return sum(float(L["thk_um"]) for L in layers)

    stack_len = pad_air + lsum(left_layers) + spacer_left + cavity_thk + spacer_right + lsum(right_layers) + pad_sub
    cell_z = stack_len + 2*dpml_z + margin_z

    geometry: List[mp.Block] = []
    def add(z0: float, thk: float, med: mp.Medium) -> float:
        cz = z0 + 0.5*thk
        geometry.append(
            mp.Block(center=mp.Vector3(0,0,cz), size=mp.Vector3(span_xy, span_xy, thk), material=med)
        )
        return z0 + thk

    # build stack
    z = -0.5*cell_z + dpml_z
    z += pad_air
    zmin_solid = z

    for L in left_layers:
        z = add(z, float(L["thk_um"]), mats[L["mat"]])
    if spacer_left > 0:
        z = add(z, spacer_left, mats["SiO2"])

    cav_start = z
    z = add(z, cavity_thk, cavity_mat)
    cavity_center = cav_start + 0.5*cavity_thk

    if spacer_right > 0:
        z = add(z, spacer_right, mats["SiO2"])
    for L in right_layers:
        z = add(z, float(L["thk_um"]), mats[L["mat"]])

    z = add(z, pad_sub, mats["SiO2"])
    zmax_solid = z

    return geometry, cell_z, cavity_center, zmin_solid, zmax_solid

def assemble_geometry(core_span_xy: float, dpml_z: float, margin_z: float):
    """
    Uses your spec + material_factory to build the DBR cavity.
    Optionally swap in fitted dispersive SiN/SiO2 via get_cavity_materials.
    """
    spec = load_geometry_json(str(PATH_GEOM))
    # Make base materials from the JSON spec
    mats = {
        name: material_factory(name, entry, mp)
        for name, entry in spec["materials"].items()
    }

    # Optionally overwrite SiN/SiO2 with your fitted, frequency-dependent media
    if USE_FITTED_MATERIALS:
        # Edit the arguments below to your CSVs and fit settings
        mat_sin, mat_sio2 = get_cavity_materials(
            model="fit",
            index_high=2.0,           # fallback index used inside your helper when needed
            index_low=1.45,
            sin_csv="si3n4.csv",      # your CSV paths
            sio2_csv="sio2.csv",
            lam_min=600, lam_max=2000,  # nm
            fit_poles=2
        )
        mats["SiN"]  = mat_sin
        mats["SiO2"] = mat_sio2

    geometry, cell_z, cav_ctr, zmin, zmax = build_geometry_from_spec(
        spec, mats, core_span_xy, dpml_z, margin_z
    )
    return geometry, cell_z, cav_ctr, zmin, zmax, mats, spec


#%%
# ------------------------------------------------------------------------------
# Build problem (modes, geometry, materials)
# ------------------------------------------------------------------------------

modes = load_modes(PATH_MODES)
lam_probe = float(modes["probe"])
lam_pump1 = float(modes["pump1"])
lam_pump2 = float(modes["pump2"])

freq_probe  = 1.0 / lam_probe
freq_pump1  = 1.0 / lam_pump1
freq_pump2  = 1.0 / lam_pump2
delta_omega = abs(freq_pump1 - freq_pump2)
freq_sb_plus  = freq_probe + delta_omega
freq_sb_minus = max(freq_probe - delta_omega, 0.0)

geometry, CELL_Z, CAVITY_CENTER, zmin_solid, zmax_solid, materials, spec = assemble_geometry(
    RUN.span_xy, RUN.dpml_z, MARGIN_Z_CELL
)

# Nonlinear SiN (scaled)
sin_med  = materials["SiN"]
sio2_med = materials["SiO2"]

# Use your helper to get n(λ) even for dispersive Medium objects
n_probe_lin = material_index_at_wavelength(sin_med, lam_probe)
EPS0 = 8.854187817e-12; C0 = 299792458.0
SCALE_E = 1.0 / (1e-6 * EPS0 * C0)

chi3_si = (4.0/3.0) * N2_SIN * (n_probe_lin**2) * EPS0 * C0
E_chi3_meep = chi3_si * (SCALE_E**2) * RUN.nonlinear_scale
sin_med.E_chi3_diag = mp.Vector3(E_chi3_meep, E_chi3_meep, E_chi3_meep)

# Clearance check: keep solids away from PMLs
def assert_pml_clearance(cell_z, dpml_z, zmin_solid, zmax_solid, gap):
    zlo_req = -0.5*cell_z + dpml_z + gap
    zhi_req =  0.5*cell_z - dpml_z - gap
    if (zmin_solid < zlo_req) or (zmax_solid > zhi_req):
        raise RuntimeError(
            f"PML–solid clearance violated. Solid z-range [{zmin_solid:.3f},{zmax_solid:.3f}] "
            f"must lie within [{zlo_req:.3f},{zhi_req:.3f}]."
        )
assert_pml_clearance(CELL_Z, RUN.dpml_z, zmin_solid, zmax_solid, CLEARANCE_PML)

print(f"Cell height: {CELL_Z:.3f} μm; cavity center z={CAVITY_CENTER:.3f} μm")

#%%

# Quick dielectric cross-section plot
sim_eps = mp.Simulation(
    cell_size=mp.Vector3(RUN.span_xy + 2*RUN.dpml_xy, RUN.span_xy + 2*RUN.dpml_xy, CELL_Z),
    geometry=geometry,
    boundary_layers=[mp.PML(RUN.dpml_z, direction=mp.Z),
                     mp.PML(RUN.dpml_xy, direction=mp.X),
                     mp.PML(RUN.dpml_xy, direction=mp.Y)],
    default_material=mp.air,
    resolution=RUN.resolution,
)
sim_eps.init_sim()
eps_xz = sim_eps.get_array(
    center=mp.Vector3(), size=mp.Vector3(RUN.span_xy+2*RUN.dpml_xy, 0, CELL_Z - 0*RUN.dpml_z), component=mp.Dielectric
)
plt.figure(figsize=(6,4))
extent_x = (-0.5*RUN.span_xy-RUN.dpml_xy, 0.5*RUN.span_xy+RUN.dpml_xy)
extent_z = (-0.5*(CELL_Z - 0*RUN.dpml_z), 0.5*(CELL_Z - 0*RUN.dpml_z))
plt.imshow(eps_xz.T, origin="lower", aspect="auto", extent=(extent_x[0],extent_x[1],extent_z[0],extent_z[1]), cmap="viridis")
plt.colorbar(label="ε"); plt.xlabel("x (μm)"); plt.ylabel("z (μm)"); plt.title("ε(x,z)")
plt.tight_layout(); plt.show()

#%%
# ------------------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------------------

def add_circular_source(sources, center, size, f, fwidth, amp, handedness):
    phase = 1.0j if handedness == "plus" else -1.0j
    a = amp / np.sqrt(2.0)
    pulse = mp.GaussianSource(frequency=f, fwidth=fwidth, cutoff=4.0)
    sources.append(mp.Source(pulse, component=mp.Ex, center=center, size=size, amplitude=a))
    sources.append(mp.Source(pulse, component=mp.Ey, center=center, size=size, amplitude=a*phase))

def add_linear_source(sources, center, size, f, fwidth, amp):
    a = amp / np.sqrt(2.0)
    pulse = mp.GaussianSource(frequency=f, fwidth=fwidth, cutoff=4.0)
    sources.append(mp.Source(pulse, component=mp.Ex, center=center, size=size, amplitude=a))
    sources.append(mp.Source(pulse, component=mp.Ey, center=center, size=size, amplitude=a))

df_probe = 0.01 #df_from_bandwidth(lam_probe, RUN.probe_band_nm)
df_p1    = 0.01 #df_from_bandwidth(lam_pump1, RUN.pump_band_nm)
df_p2    = 0.01 #df_from_bandwidth(lam_pump2, RUN.pump_band_nm)

# for amplitude, use linear index at probe (any consistent choice is fine)
amp_probe = intensity_to_meep_amplitude(RUN.probe_intensity_w_cm2, n_probe_lin)
amp_p1    = intensity_to_meep_amplitude(RUN.pump_intensity_w_cm2, n_probe_lin)
amp_p2    = intensity_to_meep_amplitude(RUN.pump_intensity_w_cm2, n_probe_lin)

cell_vec  = mp.Vector3(RUN.span_xy + 2*RUN.dpml_xy, RUN.span_xy + 2*RUN.dpml_xy, CELL_Z)
src_z     = -0.5*CELL_Z + RUN.dpml_z + RUN.src_buffer
src_center= mp.Vector3(0,0,src_z)
src_size  = mp.Vector3(RUN.span_xy + 2*RUN.dpml_xy, RUN.span_xy + 2*RUN.dpml_xy, 0)

sources: List[mp.Source] = []
add_circular_source(sources, src_center, src_size, freq_pump1, df_p1, amp_p1, "plus")
add_circular_source(sources, src_center, src_size, freq_pump2, df_p2, amp_p2, "minus")
add_linear_source  (sources, src_center, src_size, freq_probe, df_probe, amp_probe)

#%%
# ------------------------------------------------------------------------------
# Simulation + monitors
# ------------------------------------------------------------------------------

boundary_layers = [mp.PML(RUN.dpml_z, direction=mp.Z)]
if RUN.dpml_xy > 0:
    boundary_layers += [mp.PML(RUN.dpml_xy, direction=mp.X), mp.PML(RUN.dpml_xy, direction=mp.Y)]

simulation = mp.Simulation(
    cell_size=cell_vec,
    geometry=geometry,
    sources=sources,
    boundary_layers=boundary_layers,
    default_material=mp.air,
    resolution=RUN.resolution,
    force_complex_fields=True,
)

# Transmission plane: just before +z PML in uniform region
z_tr = +0.5*CELL_Z - RUN.dpml_z - MARGIN_ZTR
z_tr_cav = -3.6 #CAVITY_CENTER
dft_span_xy = mp.Volume(center=mp.Vector3(0,0,z_tr),
                        size=mp.Vector3(RUN.span_xy, RUN.span_xy, 0))
dft_span_xy_cavity = mp.Volume(center=mp.Vector3(0,0,z_tr_cav),
                               size=mp.Vector3(RUN.span_xy, RUN.span_xy, 0))
dft_span_xz = mp.Volume(center=mp.Vector3(0,0,0), size=mp.Vector3(RUN.span_xy, 0, CELL_Z - 2*RUN.dpml_z))

dft_freqs = [freq_pump1, freq_pump2, freq_probe, freq_sb_minus, freq_sb_plus]
fixed_freqs = np.array(dft_freqs, dtype=float)

nfreq_probe = 15
probe_freqs = np.linspace(freq_probe - 0.5*df_probe, freq_probe + 0.5*df_probe, nfreq_probe)
k_probe_center = nfreq_probe // 2

dft_fields    = simulation.add_dft_fields([mp.Ex, mp.Ey],
                                          dft_freqs,
                                          where=dft_span_xy)
dft_fields_cav = simulation.add_dft_fields([mp.Ex, mp.Ey],
                                          dft_freqs,
                                          where=dft_span_xy_cavity)
trans_monitor = simulation.add_dft_fields([mp.Ex, mp.Ey], freq_probe,
                                          df_probe, nfreq_probe,
                                          where=dft_span_xy)
dft_fields_xz = simulation.add_dft_fields([mp.Ex, mp.Ey],
                                          dft_freqs,
                                          where=dft_span_xz)

#%%
simulation.plot2D(output_plane=mp.Volume(center=mp.Vector3(), size=mp.Vector3(3*RUN.span_xy, 0, CELL_Z)))

#%%

# ------------------------------------------------------------------------------
# Timing
# ------------------------------------------------------------------------------

pulse_duration_meep = RUN.pulse_duration_fs * 1e9 / C0
stop_time = RUN.runtime_factor * pulse_duration_meep
snapshot_time = 2.0 * pulse_duration_meep

# ------------------------------------------------------------------------------
# Sampling and storage
# ------------------------------------------------------------------------------

time_trace = {"t": [],
              "fixed": {"Ex": [], "Ey": [], "absE": []},
              "probe_band": {"Ex": [], "Ey": [], "absE": []},
              "probe_pol": {"theta_deg": [], "Ix": [], "Iy": []}}

time_trace_cav = {"t": [],
                  "fixed": {"Ex": [], "Ey": [], "absE": []},
                  "probe_band": {"Ex": [], "Ey": [], "absE": []},
                  "probe_pol": {"theta_deg": [], "Ix": [], "Iy": []}}

xz_snapshot = {"taken": False, "t": None, "freqs": fixed_freqs, "Ex_maps": {}, "Ey_maps": {}}

xz_td_snapshot = {"taken": False, "t": None, "Ex": None, "Ey": None}

# time-domain demodulated envelopes at fixed freqs
td_env = {"t": [], "Eplus": [], "Eminus": [], "theta_deg_t": []}
_env_plus  = np.zeros(len(fixed_freqs), dtype=complex)
_env_minus = np.zeros(len(fixed_freqs), dtype=complex)

def _plane_avg_mag(ex, ey):
    return float(np.mean(np.sqrt(np.abs(ex)**2 + np.abs(ey)**2)))

def _stokes_theta_deg(Ex, Ey):
    S1 = np.mean(np.abs(Ex)**2 - np.abs(Ey)**2)
    S2 = 2.0 * np.mean(np.real(Ex * np.conj(Ey)))
    return np.degrees(0.5 * np.arctan2(S2, S1))

def sample_callback(sim: mp.Simulation):
    t = sim.meep_time()
    time_trace["t"].append(t)

    # --- Fixed-freq DFT plane averages ---
    ex_vals, ey_vals, abs_vals = [], [], []
    ex_vals_cav, ey_vals_cav, abs_vals_cav = [], [], []
    for i in range(len(fixed_freqs)):
        Ex = np.asarray(sim.get_dft_array(dft_fields, mp.Ex, i))
        Ey = np.asarray(sim.get_dft_array(dft_fields, mp.Ey, i))
        ex_vals.append(np.mean(Ex)); ey_vals.append(np.mean(Ey))
        abs_vals.append(_plane_avg_mag(Ex, Ey))
        Ex_cav = np.asarray(sim.get_dft_array(dft_fields_cav, mp.Ex, i))
        Ey_cav = np.asarray(sim.get_dft_array(dft_fields_cav, mp.Ey, i))
        ex_vals_cav.append(np.mean(Ex_cav))
        ey_vals_cav.append(np.mean(Ey_cav))
        abs_vals_cav.append(_plane_avg_mag(Ex_cav, Ey_cav))
    time_trace["fixed"]["Ex"].append(np.array(ex_vals))
    time_trace["fixed"]["Ey"].append(np.array(ey_vals))
    time_trace["fixed"]["absE"].append(np.array(abs_vals))
    time_trace_cav["fixed"]["Ex"].append(np.array(ex_vals_cav))
    time_trace_cav["fixed"]["Ey"].append(np.array(ey_vals_cav))
    time_trace_cav["fixed"]["absE"].append(np.array(abs_vals_cav))

    # --- Probe-band DFT ---
    ex_pb, ey_pb, abs_pb = [], [], []
    for k in range(nfreq_probe):
        Exk = np.asarray(sim.get_dft_array(trans_monitor, mp.Ex, k))
        Eyk = np.asarray(sim.get_dft_array(trans_monitor, mp.Ey, k))
        ex_pb.append(np.mean(Exk)); ey_pb.append(np.mean(Eyk))
        abs_pb.append(_plane_avg_mag(Exk, Eyk))
    time_trace["probe_band"]["Ex"].append(np.array(ex_pb))
    time_trace["probe_band"]["Ey"].append(np.array(ey_pb))
    time_trace["probe_band"]["absE"].append(np.array(abs_pb))

    # --- Probe polarization at center bin ---
    Ex_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Ex, k_probe_center))
    Ey_c = np.asarray(sim.get_dft_array(trans_monitor, mp.Ey, k_probe_center))
    time_trace["probe_pol"]["theta_deg"].append(_stokes_theta_deg(Ex_c, Ey_c))
    time_trace["probe_pol"]["Ix"].append(float(np.mean(np.abs(Ex_c)**2)))
    time_trace["probe_pol"]["Iy"].append(float(np.mean(np.abs(Ey_c)**2)))

    # --- Time-domain demod at z_tr (plane-avg) ---
    Ex_td = np.asarray(sim.get_array(center=mp.Vector3(0,0,z_tr), size=mp.Vector3(src_size.x, src_size.y, 0), component=mp.Ex))
    Ey_td = np.asarray(sim.get_array(center=mp.Vector3(0,0,z_tr), size=mp.Vector3(src_size.x, src_size.y, 0), component=mp.Ey))
    Ex_mean, Ey_mean = complex(np.mean(Ex_td)), complex(np.mean(Ey_td))
    Eplus_td, Eminus_td = (Ex_mean + 1j*Ey_mean)/np.sqrt(2.0), (Ex_mean - 1j*Ey_mean)/np.sqrt(2.0)

    alpha = RUN.sample_dt / max(RUN.lp_tau, 1e-9)
    eplus_env  = np.zeros(len(fixed_freqs), complex)
    eminus_env = np.zeros(len(fixed_freqs), complex)
    for i, f in enumerate(fixed_freqs):
        rot = np.exp(-2j*np.pi*f*t)
        _env_plus[i]  = (1-alpha)*_env_plus[i]  + alpha*(Eplus_td*rot)
        _env_minus[i] = (1-alpha)*_env_minus[i] + alpha*(Eminus_td*rot)
        eplus_env[i], eminus_env[i] = _env_plus[i], _env_minus[i]
    td_env["t"].append(t)
    td_env["Eplus"].append(eplus_env)
    td_env["Eminus"].append(eminus_env)
    td_env["theta_deg_t"].append(np.degrees(0.5*np.angle(eplus_env[2]/eminus_env[2])))

    # --- One-time XZ snapshot ---
    if (not xz_snapshot["taken"]) and (t >= snapshot_time):
        for i, f in enumerate(fixed_freqs):
            Ex_map = np.asarray(sim.get_dft_array(dft_fields_xz, mp.Ex, i))
            Ey_map = np.asarray(sim.get_dft_array(dft_fields_xz, mp.Ey, i))
            xz_snapshot["Ex_maps"][f] = Ex_map
            xz_snapshot["Ey_maps"][f] = Ey_map
        xz_snapshot["taken"] = True; xz_snapshot["t"] = t

        # ---- One-time time-domain XZ snapshot at snapshot_time ----
    if (not xz_td_snapshot["taken"]) and (t >= snapshot_time):
        # time-domain instantaneous fields on the same XZ plane
        Ex_map_td = np.asarray(sim.get_array(
            center=mp.Vector3(0,0,0),
            size=mp.Vector3(RUN.span_xy, 0, CELL_Z - 2*RUN.dpml_z),
            component=mp.Ex,
        ))
        Ey_map_td = np.asarray(sim.get_array(
            center=mp.Vector3(0,0,0),
            size=mp.Vector3(RUN.span_xy, 0, CELL_Z - 2*RUN.dpml_z),
            component=mp.Ey,
        ))
        xz_td_snapshot["Ex"] = Ex_map_td
        xz_td_snapshot["Ey"] = Ey_map_td
        xz_td_snapshot["taken"] = True
        xz_td_snapshot["t"] = t


#%%
# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------

pt_probe = mp.Vector3(0, 0, z_tr + 0.1)
simulation.run(
    mp.at_every(RUN.sample_dt, sample_callback),
    until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, pt_probe, 1e-6)
)

#%%
# ------------------------------------------------------------------------------
# Post-processing & plots
# ------------------------------------------------------------------------------

init_polarization_deg = 45.0  # subtract input polarization to get rotation
t_arr = np.array(time_trace["t"])

fixed_Ex  = np.vstack(time_trace["fixed"]["Ex"])
fixed_Ey  = np.vstack(time_trace["fixed"]["Ey"])
fixed_abs = np.vstack(time_trace["fixed"]["absE"])
Epl_dft   = (fixed_Ex + 1j*fixed_Ey)/np.sqrt(2.0)
Emi_dft   = (fixed_Ex - 1j*fixed_Ey)/np.sqrt(2.0)

probe_abs = np.vstack(time_trace["probe_band"]["absE"])
theta_deg = np.array(time_trace["probe_pol"]["theta_deg"]) - init_polarization_deg

# DFT inside cavity
fixed_Ex_cav  = np.vstack(time_trace_cav["fixed"]["Ex"])
fixed_Ey_cav  = np.vstack(time_trace_cav["fixed"]["Ey"])
fixed_abs_cav = np.vstack(time_trace_cav["fixed"]["absE"])
Epl_dft_cav   = (fixed_Ex_cav + 1j*fixed_Ey_cav)/np.sqrt(2.0)
Emi_dft_cav   = (fixed_Ex_cav - 1j*fixed_Ey_cav)/np.sqrt(2.0)

t_td   = np.array(td_env["t"])
Epl_td = np.vstack(td_env["Eplus"])
Emi_td = np.vstack(td_env["Eminus"])
theta_deg_t = np.array(td_env["theta_deg_t"]) - init_polarization_deg

i_p1, i_p2, i_probe, i_sb_minus, i_sb_plus = range(5)

# Pumps (DFT)
plt.figure(figsize=(7.2,4))
plt.plot(t_arr, np.abs(Emi_dft[:, i_p1]), label=f"pump1 DFT |e-| f={fixed_freqs[i_p1]:.3f}")
plt.plot(t_arr, np.abs(Epl_dft[:, i_p2]), label=f"pump2 DFT |e+| f={fixed_freqs[i_p2]:.3f}")
plt.xlabel("time (Meep)"); plt.ylabel("|E| (DFT plane-avg)"); plt.title("Pumps (DFT)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# Pumps (TD demod)
plt.figure(figsize=(7.2,4))
plt.plot(t_td, np.abs(Emi_td[:, i_p1]), label="pump1 TD |e-|")
plt.plot(t_td, np.abs(Epl_td[:, i_p2]), label="pump2 TD |e+|")
plt.xlabel("time (Meep)"); plt.ylabel("|E| (TD demod)"); plt.title("Pumps (time-domain)")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# Probe & sidebands (DFT)
def plot_pair(label, idx):
    plt.plot(t_arr, np.abs(Epl_dft[:, idx]), '-',  label=f"{label} DFT |e+|")
    plt.plot(t_arr, np.abs(Emi_dft[:, idx]), '--', label=f"{label} DFT |e-|")
plt.figure(figsize=(7.6,4.6))
plot_pair("probe", i_probe); plot_pair("sb−", i_sb_minus); plot_pair("sb+", i_sb_plus)
plt.xlabel("time (Meep)"); plt.ylabel("|E| (DFT plane-avg)")
plt.title("Probe & sidebands (DFT)"); plt.grid(True, alpha=0.3); plt.legend(ncol=2); plt.tight_layout(); plt.show()

# Probe & sidebands (TD demod)
def plot_pair_td(label, idx):
    plt.plot(t_td, np.abs(Epl_td[:, idx]), '-',  label=f"{label} TD |e+|")
    plt.plot(t_td, np.abs(Emi_td[:, idx]), '--', label=f"{label} TD |e-|")
plt.figure(figsize=(7.6,4.6))
plot_pair_td("probe", i_probe); plot_pair_td("sb−", i_sb_minus); plot_pair_td("sb+", i_sb_plus)
plt.xlabel("time (Meep)"); plt.ylabel("|E| (TD demod)")
plt.title("Probe & sidebands (time-domain)"); plt.grid(True, alpha=0.3); plt.legend(ncol=2); plt.tight_layout(); plt.show()

# Pumps (DFT) inside cavity
plt.figure(figsize=(7.2,4))
plt.plot(t_arr, np.abs(Emi_dft_cav[:, i_p1]), label=f"pump1 DFT |e-| f={fixed_freqs[i_p1]:.3f}")
plt.plot(t_arr, np.abs(Epl_dft_cav[:, i_p2]), label=f"pump2 DFT |e+| f={fixed_freqs[i_p2]:.3f}")
plt.xlabel("time (Meep)"); plt.ylabel("|E| (DFT plane-avg)"); plt.title("Pumps (DFT) inside cavity")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# Probe & sidebands (DFT) inside cavity
def plot_pair_cav(label, idx):
    plt.plot(t_arr, np.abs(Epl_dft_cav[:, idx]), '-',  label=f"{label} DFT |e+|")
    plt.plot(t_arr, np.abs(Emi_dft_cav[:, idx]), '--', label=f"{label} DFT |e-|")
plt.figure(figsize=(7.6,4.6))
plot_pair_cav("probe", i_probe); plot_pair_cav("sb−", i_sb_minus); plot_pair_cav("sb+", i_sb_plus)
plt.xlabel("time (Meep)"); plt.ylabel("|E| (DFT plane-avg)")
plt.title("Probe & sidebands (DFT) inside cavity"); plt.grid(True, alpha=0.3); plt.legend(ncol=2); plt.tight_layout(); plt.show()

# Probe band heatmap
plt.figure(figsize=(7.4,4.2))
plt.imshow(np.abs(probe_abs.T), aspect="auto", origin="lower",
           extent=[t_arr.min(), t_arr.max(), probe_freqs.min(), probe_freqs.max()])
plt.colorbar(label=r"$\langle |E| \rangle$")
plt.xlabel("time (Meep)"); plt.ylabel("frequency (1/μm)")
plt.title("Probe-band plane-avg |E| vs (f,t)"); plt.tight_layout(); plt.show()

# Polarization rotation
plt.figure(figsize=(7.0,3.6))
plt.plot(t_arr, theta_deg, "k-")
plt.xlabel("time (Meep)"); plt.ylabel("polarization rotation (deg)")
plt.title("Probe rotation vs time (center DFT bin)"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(7.0,3.6))
plt.plot(t_arr, theta_deg_t, "k-")
plt.xlabel("time (Meep)"); plt.ylabel("polarization rotation (deg)")
plt.title("Probe rotation vs time (from time-domain)"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

# --- (E1) X–Z spatial maps from DFT monitor (|E| at each fixed frequency) ---
if xz_snapshot["taken"]:
    # extents (μm) matching the XZ monitor volume
    x_half = 0.5 * RUN.span_xy
    z_half = 0.5 * (CELL_Z - 2*RUN.dpml_z)
    extent_xz = (-x_half, x_half, -z_half, z_half)

    f_show = fixed_freqs  # plot all fixed freqs
    ncols = min(3, len(f_show))
    nrows = int(np.ceil(len(f_show)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, f in zip(axes, f_show):
        Ex_map = np.asarray(xz_snapshot["Ex_maps"][f])
        Ey_map = np.asarray(xz_snapshot["Ey_maps"][f])
        Emag = np.sqrt(np.abs(Ex_map)**2 + np.abs(Ey_map)**2)
        im = ax.imshow(Emag.T, origin="lower", aspect="auto", cmap="magma",
                       extent=extent_xz)
        ax.set_title(f"|E|(x,z) at f={f:.3f}, t≈{xz_snapshot['t']:.2f}")
        ax.set_xlabel("x (μm)"); ax.set_ylabel("z (μm)")

    for ax in axes[len(f_show):]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.9, label="|E|")
    fig.tight_layout(); plt.show()

# --- (E2) X–Z spatial map from **time-domain** snapshot (instantaneous |E|) ---
if xz_td_snapshot["taken"]:
    x_half = 0.5 * RUN.span_xy
    z_half = 0.5 * (CELL_Z - 2*RUN.dpml_z)
    extent_xz = (-x_half, x_half, -z_half, z_half)

    Ex_td = np.asarray(xz_td_snapshot["Ex"])
    Ey_td = np.asarray(xz_td_snapshot["Ey"])
    Emag_td = np.sqrt(np.abs(Ex_td)**2 + np.abs(Ey_td)**2)

    plt.figure(figsize=(4.4, 4.4))
    plt.imshow(Emag_td.T, origin="lower", aspect="auto", cmap="viridis",
               extent=extent_xz)
    plt.colorbar(label="|E| (instantaneous)")
    plt.xlabel("x (μm)"); plt.ylabel("z (μm)")
    plt.title(f"Instantaneous |E|(x,z) at t≈{xz_td_snapshot['t']:.2f}")
    plt.tight_layout(); plt.show()

# %%
