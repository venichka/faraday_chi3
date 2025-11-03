# fp_cavity_modes_spectrum.py
#
# Air | [SiO2/Si3N4]×3 | Si3N4 cavity | [SiO2/Si3N4]×3 | SiO2 substrate
# Thicknesses (nm): Si3N4=260, SiO2=160, cavity=1543  → μm units below.

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
# import json

from geometry_io import load_params, export_params_json, export_geometry_json

# choose which source to prefer:
#   prefer="report"  -> read optimize_report.json first, then fall back to optimized_geometry.json
#   prefer="geom"    -> read optimized_geometry.json first, then fall back
params = load_params(prefer="report")

# Meep + geometry params
t_SiN = params["t_SiN"]
t_SiO2 = params["t_SiO2"]
t_cav = params["t_cav"]
N_per = params["N_per"]
pad_air = params["pad_air"]
pad_sub = params["pad_sub"]
dpml = params["dpml"]
resolution = params["resolution"]
cell_margin = params["cell_margin"]

# legacy spacers (optimizer now sets these to 0.0)
sL_opt = 0.0
sR_opt = 0.0

# with open("optimize_report.json") as f:   # or whatever filename you used
#     rep = json.load(f)
#
# L_cav_opt = rep["best_theta"]["L_cav_um"]
# sL_opt = rep["best_theta"]["sL_um"]
# sR_opt = rep["best_theta"]["sR_um"]
# cell_margin = rep["best_theta"]["cell_margin_um"]
#
#
# print("Using meep version:", mp.__version__)
#
# # ---------- User parameters ----------
# # geometry (μm)
# t_SiN = 0.260         # leave your layer values as-is
# t_SiO2 = 0.160
# t_cav = L_cav_opt #+ sL_opt + sR_opt   # << use optimized cavity length
# # sL_opt = 0.0
# # sR_opt = 0.0
# N_per = 3
# pad_air = 0.8
# pad_sub = 3.0
# dpml = 1.0

# materials (indices)
n_SiN, n_SiO2 = 2.00, 1.45
chi3_SiN = 0.0
mat_SiN = mp.Medium(index=n_SiN,  chi3=chi3_SiN)
mat_SiO2 = mp.Medium(index=n_SiO2)

# spectrum + resolution (unchanged)
wl_min, wl_max = 0.6, 2.0
fmin, fmax = 1/wl_max, 1/wl_min
fcen, df = 0.5*(fmin+fmax), (fmax-fmin)
nfreq = 800
resolution = 100


# ---------- Build 1D stack along z ----------
def build_stack():
    geometry = []
    z_left = -0.5*cell_z + dpml  # start at left edge inside PML

    def advance(t):
        nonlocal z_left
        z_left += t

    def add_block(thk, mat):
        nonlocal z_left
        center = z_left + 0.5*thk
        geometry.append(
            mp.Block(center=mp.Vector3(0, 0, center),
                     size=mp.Vector3(mp.inf, mp.inf, thk),
                     material=mat)
        )
        z_left += thk

    # 1) left air spacer
    advance(pad_air)

    # 2) LEFT DBR: (SiN, SiO2) × N_per  → ends with SiO2 near cavity
    for _ in range(N_per):
        add_block(t_SiN,  mat_SiN)
        add_block(t_SiO2, mat_SiO2)

    # 2.5) optimized LEFT SiO2 spacer next to cavity
    if sL_opt > 0:
        add_block(sL_opt, mat_SiO2)

    # 3) CAVITY (SiN) with optimized length
    add_block(t_cav, mat_SiN)

    # 3.5) optimized RIGHT SiO2 spacer next to cavity
    if sR_opt > 0:
        add_block(sR_opt, mat_SiO2)

    # 4) RIGHT DBR: (SiO2, SiN) × N_per  → starts with SiO2 near cavity
    for _ in range(N_per):
        add_block(t_SiO2, mat_SiO2)
        add_block(t_SiN,  mat_SiN)

    # 5) SiO2 substrate spacer
    add_block(pad_sub, mat_SiO2)

    return geometry


# --- update the total lengths to include spacers and margin ---
stack_z = pad_air + N_per*(t_SiO2+t_SiN) + sL_opt + \
    t_cav + sR_opt + N_per*(t_SiO2+t_SiN) + pad_sub
cell_z = stack_z + 2*dpml + cell_margin   # << use the margin from JSON
cell = mp.Vector3(0, 0, cell_z)
# ---------- Helpers: plotting ----------


def plot_epsilon(sim, title="ε(z)"):
    # one-dimensional ε profile sampled along z
    eps_1d = sim.get_array(center=mp.Vector3(), size=mp.Vector3(
        0, 0, cell_z), component=mp.Dielectric)
    zgrid = np.linspace(-0.5*cell_z, 0.5*cell_z, eps_1d.size)
    plt.figure(figsize=(8, 3))
    plt.plot(zgrid, eps_1d)
    plt.axvspan(-0.5*cell_z, -0.5*cell_z+dpml,
                color='k', alpha=0.05, label='PML')
    plt.axvspan(0.5*cell_z-dpml,  0.5*cell_z, color='k', alpha=0.05)
    plt.xlabel("z (μm)")
    plt.ylabel("ε")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_mode_profile(sim, freqs, dft, label="|Ex|(z)"):
    zgrid = np.linspace(-0.5*cell_z, 0.5*cell_z,
                        sim.get_dft_array(dft, mp.Ex, 0).size)
    plt.figure(figsize=(8, 3))
    for i, f in enumerate(freqs):
        field = sim.get_dft_array(dft, mp.Ex, i)
        plt.plot(zgrid, np.abs(field),
                 label=f"λ={1/f:.1f} nm" if 1/f < 0.01 else f"λ={1e3/f:.0f} nm")
    plt.legend()
    plt.xlabel("z (μm)")
    plt.ylabel(label)
    plt.tight_layout()
    plt.show()


# ---------- 1) Build cavity & plot epsilon ----------
geometry = build_stack()
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    boundary_layers=[mp.PML(dpml)],
                    default_material=mp.air,
                    resolution=resolution,
                    dimensions=1)

# initialize once to access eps
sim.init_sim()
# uses get_array(mp.Dielectric)
plot_epsilon(sim, "Dielectric profile of the cavity")

# ---------- 2) Find resonant modes around 0.8 μm and 1.6 μm with Harminv ---


def find_modes(wavelength, bandwidth_factor=0.2, run_time=400):
    f0 = 1/wavelength
    df_local = bandwidth_factor * f0
    src = [mp.Source(mp.GaussianSource(f0, fwidth=df_local),
                     component=mp.Ex,
                     center=mp.Vector3(0, 0, -0.5*cell_z+dpml+0.2),
                     amplitude=1.0)]
    sim.reset_meep()
    sim.change_sources(src)
    mon_pt = mp.Vector3()  # center of cavity
    h = mp.Harminv(mp.Ex, mon_pt, f0, df_local)
    sim.run(mp.after_sources(h), until=run_time)
    return h.modes


modes_probe = find_modes(0.8)   # around 800 nm
modes_pump = find_modes(1.6, bandwidth_factor=0.4)   # around 1600 nm

print("\n=== Harminv modes near 800 nm ===")
for md in modes_probe:
    print(md)
print("\n=== Harminv modes near 1600 nm ===")
for md in modes_pump:
    print(md)

# ---------- 3) Mode field profiles via DFT monitors ----------
# take up to the first few modes near each band and plot |Ex|(z)


def modes_to_freqs(modes, maxn=3):
    freqs = []
    for m in modes[:maxn]:
        if m.decay > 0:  # skip growing artifacts
            continue
        freqs.append(m.freq)
    return freqs


freqs = modes_to_freqs(modes_probe, 3) + modes_to_freqs(modes_pump, 3)
if len(freqs) > 0:
    # narrowband source just to excite steady state near these freqs
    src = [mp.Source(mp.ContinuousSource(fcen, width=0),
                     component=mp.Ex,
                     center=mp.Vector3(0, 0, -0.5*cell_z+dpml+0.2),
                     amplitude=1.0)]
    sim.reset_meep()
    sim.change_sources(src)
    vol = mp.Volume(center=mp.Vector3(),
                    size=mp.Vector3(0, 0, cell_z-2*dpml-0.02))
    dft = sim.add_dft_fields([mp.Ex], freqs, where=vol)
    sim.run(until=800)  # long enough to settle
    plot_mode_profile(sim, freqs, dft, label="|Ex|(z) (DFT)")
else:
    print("No stable modes found to plot.")


# ---------- 3) Mode field profiles for ALL Harminv modes (robust to grid size)

def _dedup_mode_freqs(hmodes, rel_tol=1e-4):
    fs = sorted([m.freq for m in hmodes if np.isfinite(m.freq) and m.freq > 0])
    uniq = []
    for f in fs:
        if not uniq or abs(f - uniq[-1]) / uniq[-1] > rel_tol:
            uniq.append(f)
    return uniq


def gather_and_plot_modes(all_modes, use_cw_solver=False, tol=1e-9,
                          max_iters=10000, bicgL=10):
    """
    For every Harminv-found mode, run an independent CW/DFT solve at that frequency
    and plot |Ex|(z). z-grid length is taken from the returned data to avoid mismatch.
    """
    freqs = _dedup_mode_freqs(all_modes)
    if not freqs:
        print("[modes] No valid Harminv frequencies found.")
        return

    # monitor region (exclude PMLs by a small margin)
    field_len = cell_z - 2*dpml - 0.02
    vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 0, field_len))

    fig, ax = plt.subplots(figsize=(9, 3))

    for f in freqs:
        lam = 1.0 / f
        # fresh sim per mode
        src = [mp.Source(mp.ContinuousSource(f, width=0 if not use_cw_solver else 0.02*f),
                         component=mp.Ex,
                         center=mp.Vector3(0, 0, -0.5*cell_z + dpml + 0.2),
                         amplitude=1.0)]
        sim_mode = mp.Simulation(cell_size=cell,
                                 geometry=geometry,
                                 boundary_layers=[mp.PML(dpml)],
                                 default_material=mp.air,
                                 resolution=resolution,
                                 dimensions=1,
                                 sources=src,
                                 force_complex_fields=use_cw_solver)

        if use_cw_solver:
            # Frequency-domain solver (steady-state at f)
            sim_mode.init_sim()
            sim_mode.solve_cw(tol, max_iters, bicgL)  # see Meep freq-domain tutorial
            ex = sim_mode.get_array(vol=vol, component=mp.Ex)
        else:
            # Time-domain DFT at single frequency
            dft = sim_mode.add_dft_fields([mp.Ex], f, 0, 1, where=vol)
            Nperiods = 60
            sim_mode.run(until=Nperiods / f)
            ex = sim_mode.get_dft_array(dft, mp.Ex, 0)

        # ensure 1D vector
        ex = np.ravel(ex)
        # build z-grid using the actual number of samples (avoid off-by-few mismatch)
        N = ex.size
        zgrid = np.linspace(-0.5*field_len, 0.5*field_len, N)

        ax.plot(zgrid, np.abs(ex), label=f"λ={1e3*lam:.0f} nm")

    ax.set_xlabel("z (μm)")
    ax.set_ylabel("|Ex|(z)")
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("Cavity mode profiles (all Harminv-found modes)")
    fig.tight_layout()
    plt.show()


# Use all Harminv modes (probe + pump windows)
all_harminv_modes = list(modes_probe) + list(modes_pump)
gather_and_plot_modes(all_harminv_modes, use_cw_solver=False, tol=1e-9,
                      max_iters=15000, bicgL=10)

# ---------- 4) Reflectance spectrum ----------
# Standard two-run normalization:
#   (a) run without structure → save incident-field DFT data
#   (b) run with structure and load_minus_flux_data → reflected power / incident power


def reflectance_spectrum():
    # placement of source and monitors
    src_z = -0.5*cell_z + dpml + 0.2
    # just to the right of the source (left of structure)
    refl_pt = src_z + 0.1
    # --- Normalization run (no geometry) ---
    sim_norm = mp.Simulation(cell_size=cell,
                             boundary_layers=[mp.PML(dpml)],
                             default_material=mp.air,
                             resolution=resolution,
                             dimensions=1,
                             sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df),
                                                component=mp.Ex,
                                                center=mp.Vector3(0, 0, src_z))])
    refl = sim_norm.add_flux(fcen, df, nfreq, mp.FluxRegion(
        center=mp.Vector3(0, 0, refl_pt)))
    sim_norm.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(0, 0, refl_pt), 1e-8))
    # cache Fourier-transformed incident fields
    incident_data = sim_norm.get_flux_data(refl)
    freqs = mp.get_flux_freqs(refl)
    Inc = np.array(mp.get_fluxes(refl))

    # --- Structure run ---
    sim_struct = mp.Simulation(cell_size=cell,
                               geometry=geometry,
                               boundary_layers=[mp.PML(dpml)],
                               default_material=mp.air,
                               resolution=resolution,
                               dimensions=1,
                               sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df),
                                                  component=mp.Ex,
                                                  center=mp.Vector3(0, 0, src_z))])
    # same refl monitor
    refl2 = sim_struct.add_flux(
        fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(0, 0, refl_pt)))
    # subtract incident fields (equivalent to load_minus_flux+scale -1 per API docs)
    sim_struct.load_minus_flux_data(refl2, incident_data)
    sim_struct.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, mp.Vector3(0, 0, refl_pt), 1e-8))
    R = - np.array(mp.get_fluxes(refl2)) / Inc
    return freqs, R


freqs, R = reflectance_spectrum()

# plot R(λ)
wl = 1/np.array(freqs)
plt.figure(figsize=(7, 4))
plt.plot(1e3*wl, R, lw=1.5)
plt.gca().invert_xaxis()
plt.axvline(800,  color='k', ls='--', lw=0.7)  # probe
plt.axvline(1600, color='k', ls=':',  lw=0.7)  # pumps
plt.xlabel("wavelength (nm)")
plt.ylabel("Reflectance")
plt.title("Cavity reflectance spectrum")
plt.tight_layout()
plt.show()
