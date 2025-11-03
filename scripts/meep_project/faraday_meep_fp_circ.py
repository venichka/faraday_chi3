#!/usr/bin/env mp
# -*- coding: utf-8 -*-
# FP-DBR cavity (Si3N4/SiO2), three pulses:
# Pumps: circular (e+, e-) at 1650/1550 nm; Probe: linear 45° at 800 nm.
# Kerr χ(3) isotropic; plots: geometry ε slice, field snapshot, intensities vs time, θ(t).

import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# ---------- Units & constants ----------
um = 1.0
c0 = 299792458.0
eps0 = 8.854187817e-12

# ---------- Materials ----------
n_air = 1.0
n_sio2 = 1.45
n_sin = 2.00

# Nonlinearity (isotropic Kerr) for Si3N4
n2_sin_SI = 2.5e-19  # m^2/W
scale_E = 1.0 / (1e-6 * eps0 * c0)


def I_to_Eamp_meep(I_SI, n_lin):
    E_SI = np.sqrt(2.0 * I_SI / (n_lin * eps0 * c0))
    return E_SI / scale_E


# Intensities
I_pump_SI = 1e12 * 1e4  # 1 TW/cm^2 -> W/m^2
I_probe_SI = 1e3 * 1e4  # 1 kW/cm^2 -> W/m^2
E0_pump = I_to_Eamp_meep(I_pump_SI,  n_sin)
E0_probe = I_to_Eamp_meep(I_probe_SI, n_sin)

# Convert n2 -> chi3 (SI), then to Meep units for E_chi3_diag
chi3_SI = (4.0/3.0) * n2_sin_SI * (n_sin**2) * eps0 * c0
E_chi3_MP = chi3_SI * (scale_E**3)

mat_air = mp.Medium(index=n_air)
mat_sio2 = mp.Medium(index=n_sio2)
mat_sin = mp.Medium(index=n_sin, E_chi3_diag=mp.Vector3(
    E_chi3_MP, E_chi3_MP, E_chi3_MP))

# ---------- Wavelengths/BWs ----------
lam_s = 0.800
dlam_s = 0.010  # 10 nm
lam1 = 1.650
dlam_p = 0.030  # 30 nm (pumps)
lam2 = 1.550


def f_from_lam(lam_um): return 1.0/lam_um
def df_from_lam(lam_um, dlam_um): return dlam_um/(lam_um**2)


fs, dfs = f_from_lam(lam_s), df_from_lam(lam_s, dlam_s)  # probe freq
f1, df1 = f_from_lam(lam1),  df_from_lam(lam1, dlam_p)  # pump 1 freq
f2, df2 = f_from_lam(lam2),  df_from_lam(lam2, dlam_p)  # pump 2 freq

# --- Sidebands around the probe from pump beating ---
Delta = abs(f1 - f2)
fsb_plus = fs + Delta
fsb_minus = fs - Delta

# a reasonable narrowband width for sidebands (match probe BW)
df_sb = 6*dfs      # bandwidth for sideband DFT boxes (same style as probe)
nf_sb = 11         # number of DFT bins for the sidebands

# frequency
nfreq = 15  # number of freq bins
df = 6*dfs  # span ~±3*dfs about fs


# ---------- Cavity ----------
t_SiN, t_SiO2, t_cav = 0.260, 0.160, 1.543
N_per = 3

dpml, pad = 0.8, 2.0
sx = dpml + pad + N_per*(t_SiN+t_SiO2) + t_cav + \
    N_per*(t_SiN+t_SiO2) + pad + dpml
sy = 2.0
sz = 2.0

cell = mp.Vector3(sx, sy, sz)
pml_layers = [mp.PML(dpml)]

geometry = []
x0 = -0.5*sx + dpml + pad    # cursor at start of free space (air)

# ---- Left DBR: (Si3N4, SiO2)^3 ----
for _ in range(N_per):
    # Si3N4
    geometry.append(mp.Block(material=mat_sin,
                             size=mp.Vector3(t_SiN, mp.inf, mp.inf),
                             center=mp.Vector3(x0 + 0.5*t_SiN, 0, 0)))
    x0 += t_SiN
    # SiO2
    geometry.append(mp.Block(material=mat_sio2,
                             size=mp.Vector3(t_SiO2, mp.inf, mp.inf),
                             center=mp.Vector3(x0 + 0.5*t_SiO2, 0, 0)))
    x0 += t_SiO2

# ---- Cavity (Si3N4) ----
geometry.append(mp.Block(material=mat_sin,
                         size=mp.Vector3(t_cav, mp.inf, mp.inf),
                         center=mp.Vector3(x0 + 0.5*t_cav, 0, 0)))
x0 += t_cav

# ---- Right DBR: (SiO2, Si3N4)^3 ----
for _ in range(N_per):
    # SiO2
    geometry.append(mp.Block(material=mat_sio2,
                             size=mp.Vector3(t_SiO2, mp.inf, mp.inf),
                             center=mp.Vector3(x0 + 0.5*t_SiO2, 0, 0)))
    x0 += t_SiO2
    # Si3N4
    geometry.append(mp.Block(material=mat_sin,
                             size=mp.Vector3(t_SiN, mp.inf, mp.inf),
                             center=mp.Vector3(x0 + 0.5*t_SiN, 0, 0)))
    x0 += t_SiN

# ---- Substrate fill (SiO2) : only from end of stack to right PML ----
x_right_stack = x0
x_right_edge = 0.5*sx - dpml       # start of right PML
w_sub = max(0.0, x_right_edge - x_right_stack)

if w_sub > 0:
    geometry.append(mp.Block(material=mat_sio2,
                             size=mp.Vector3(w_sub, mp.inf, mp.inf),
                             center=mp.Vector3(x_right_stack + 0.5*w_sub, 0, 0)))

# ---------- Sources ----------
# Normal incidence along +x -> E in (y,z). Circular via ±pi/2 phase shift between Ey and Ez.
src_center = mp.Vector3(-0.5*sx + dpml + 0.5*pad, 0, 0)


def gsrc(fcen, fwidth, amp, comp, amp_comp):
    return mp.Source(src=mp.GaussianSource(frequency=fcen, fwidth=fwidth),
                     component=comp, center=src_center,
                     size=mp.Vector3(0, sy, sz),
                     amplitude=amp*amp_comp)


sources = []
# Pump 1: e+  => Ey + i Ez  (phase +π/2 on Ez)
sources += [gsrc(f1, df1, E0_pump/np.sqrt(2), mp.Ey, amp_comp=1.0),
            gsrc(f1, df1, E0_pump/np.sqrt(2), mp.Ez, amp_comp=1.0j)]
# Pump 2: e-  => Ey - i Ez  (phase -π/2 on Ez)
sources += [gsrc(f2, df2, E0_pump/np.sqrt(2), mp.Ey, amp_comp=1.0),
            gsrc(f2, df2, E0_pump/np.sqrt(2), mp.Ez, amp_comp=-1.0j)]
# Probe: linear 45° in y–z (Ey=Ez in phase)
sources += [gsrc(fs, dfs, E0_probe/np.sqrt(2), mp.Ey, amp_comp=1.0),
            gsrc(fs, dfs, E0_probe/np.sqrt(2), mp.Ez, amp_comp=1.0)]

# ---------- Simulation ----------
resolution = int(np.ceil(10.0/lam_s))  # ≥30 px per 0.8 µm
symms = [mp.Mirror(mp.Y, phase=+1), mp.Mirror(mp.Z, phase=+1)]
sim = mp.Simulation(cell_size=cell, boundary_layers=pml_layers,
                    geometry=geometry, sources=sources,
                    symmetries=symms,
                    resolution=resolution,
                    force_complex_fields=True)

# Transmission monitor (DFT) for ~800 nm Stokes
mon_x = -0.5*sx + dpml + pad + N_per * \
    (t_SiN+t_SiO2) + t_cav + N_per*(t_SiN+t_SiO2) + 0.3

# --- Separate DFT monitors per band (probe, pumps, sidebands) ---
dft_probe = sim.add_dft_fields(
    [mp.Ey, mp.Ez], fs, 6*dfs, 15,
    where=mp.Volume(center=mp.Vector3(mon_x, 0, 0),
                    size=mp.Vector3(0, 4.0, 4.0))
)

dft_p1 = sim.add_dft_fields(
    [mp.Ey, mp.Ez], f1, 6*df1, 11,
    where=mp.Volume(center=mp.Vector3(mon_x, 0, 0),
                    size=mp.Vector3(0, 4.0, 4.0))
)

dft_p2 = sim.add_dft_fields(
    [mp.Ey, mp.Ez], f2, 6*df2, 11,
    where=mp.Volume(center=mp.Vector3(mon_x, 0, 0),
                    size=mp.Vector3(0, 4.0, 4.0))
)

dft_sb_plus = sim.add_dft_fields(
    [mp.Ey, mp.Ez], fsb_plus, df_sb, nf_sb,
    where=mp.Volume(center=mp.Vector3(mon_x, 0, 0),
                    size=mp.Vector3(0, 4.0, 4.0))
)

dft_sb_minus = sim.add_dft_fields(
    [mp.Ey, mp.Ez], fsb_minus, df_sb, nf_sb,
    where=mp.Volume(center=mp.Vector3(mon_x, 0, 0),
                    size=mp.Vector3(0, 4.0, 4.0))
)


# Time sampling at a point (on axis)
probe_pt = mp.Vector3(mon_x, 0, 0)
Ey_t, Ez_t, t_meep = [], [], []


def collect(sim):
    Ey_t.append(sim.get_field_point(mp.Ey, probe_pt))
    Ez_t.append(sim.get_field_point(mp.Ez, probe_pt))
    t_meep.append(sim.meep_time())


# total time (~3 ps); sample every ~2 fs
# fs_per_meep = 1.0/(c0*1e-6)  # fs / meep_time
# dt_meep = 2.0 / fs_per_meep
# correct fs per Meep-time unit (fs per a/c)
fs_per_meep = (1e-6 / c0) * 1e15   # ≈ 3.3356409519815204 fs / Meep-unit
dt_meep = 2.0 / fs_per_meep    # 2 fs sampling -> ≈ 0.599584916 Meep-units


T_total_ps = 1.0
sim_time = T_total_ps*1e3 / fs_per_meep

# ---------- Pre-run: epsilon slice for geometry plot ----------
# Grab a 2D epsilon slice (x–y plane at z=0)
# Note: field/epsilon array extents differ by 1 sometimes; we plot separately.  :contentReference[oaicite:3]{index=3}
# Initialize structure/fields without running time stepping
sim.init_sim()  # <-- correct initializer

# 2D epsilon slice (x–y at z=0). Center/size form is valid.
eps_slice = sim.get_array(center=mp.Vector3(),
                          size=mp.Vector3(sx, sy, 0),
                          component=mp.Dielectric)

# ---------- Run ----------
# pick a field component & point near your transmission monitor
# Time sampling at a point (off the symmetry planes)
probe_pt = mp.Vector3(mon_x, 0.2, 0.2)   # was (mon_x, 0, 0)

# plt.figure(figsize=(6, 3.5))
# plt.imshow(eps_slice.T, origin='lower',
#            extent=[-sx/2, sx/2, -sy/2, sy/2], aspect='auto')
# plt.colorbar(label='ε')
# plt.title('Geometry: ε slice (z=0)')
# plt.xlabel('x (µm)')
# plt.ylabel('y (µm)')
# plt.tight_layout()
# plt.show()


# Fast termination: also check decay off the symmetry planes
sim.run(
    mp.at_every(dt_meep, collect),
    until_after_sources=mp.stop_when_fields_decayed(
        10,                  # check more frequently than 50
        mp.Ey,
        mp.Vector3(mon_x, 0.2, 0.2),
        1e-7
    )
)


# ---------- Field snapshot in cavity (x–y at cavity center) ----------
xcav = -0.5*sx + dpml + pad + N_per*(t_SiN+t_SiO2) + 0.5*t_cav
Ey_xy = sim.get_array(center=mp.Vector3(xcav, 0, 0),
                      size=mp.Vector3(0, sy, sz), component=mp.Ey)
Ez_xy = sim.get_array(center=mp.Vector3(xcav, 0, 0),
                      size=mp.Vector3(0, sy, sz), component=mp.Ez)

# ---------- Time-domain Stokes & θ(t) ----------
Ey_t = np.array(Ey_t)
Ez_t = np.array(Ez_t)
t_meep = np.array(t_meep)
tt_ps = t_meep * fs_per_meep * 1e-3

# Narrowband around fs to isolate probe


def narrowband(u, t_m, f0, bw):
    """
    Gaussian bandpass around center frequency f0 with width bw.
    Works for complex field samples u(t). Frequencies are in Meep units (1/µm),
    which are consistent with the source/DFT 'frequency' in Meep.  # c = 1 units
    """
    u = np.asarray(
        u, dtype=np.complex128)   # ensure complex array, no object dtype
    t_m = np.asarray(t_m, dtype=np.float64)

    if u.size < 8:
        return u  # too few samples to filter

    dt = (t_m[1] - t_m[0])                   # Meep time units
    U = np.fft.fft(u)
    # same units as Meep 'frequency' (1/µm)
    freqs = np.fft.fftfreq(u.size, d=dt)
    win = np.exp(-0.5 * ((freqs - f0) / bw)**2)
    u_nb = np.fft.ifft(U * win)
    return u_nb


Ey_nb = narrowband(Ey_t, t_meep, fs, dfs)
Ez_nb = narrowband(Ez_t, t_meep, fs, dfs)

# --- Narrowband envelopes for pumps & sidebands (time-domain) ---
Ey1_nb = narrowband(Ey_t, t_meep, f1, df1)
Ez1_nb = narrowband(Ez_t, t_meep, f1, df1)
I_pump1_t = np.abs(Ey1_nb)**2 + np.abs(Ez1_nb)**2

Ey2_nb = narrowband(Ey_t, t_meep, f2, df2)
Ez2_nb = narrowband(Ez_t, t_meep, f2, df2)
I_pump2_t = np.abs(Ey2_nb)**2 + np.abs(Ez2_nb)**2

# back to single-BW (dfs) in time
Eysbp_nb = narrowband(Ey_t, t_meep, fsb_plus,  df_sb/6)
Ezsbp_nb = narrowband(Ez_t, t_meep, fsb_plus,  df_sb/6)
I_sb_plus_t = np.abs(Eysbp_nb)**2 + np.abs(Ezsbp_nb)**2

Eysbm_nb = narrowband(Ey_t, t_meep, fsb_minus, df_sb/6)
Ezsbm_nb = narrowband(Ez_t, t_meep, fsb_minus, df_sb/6)
I_sb_minus_t = np.abs(Eysbm_nb)**2 + np.abs(Ezsbm_nb)**2

# --- Probe in circular basis: E+ = (Ey + i Ez)/sqrt(2), E- = (Ey - i Ez)/sqrt(2)
Eprobe_plus = (Ey_nb + 1j*Ez_nb)/np.sqrt(2.0)
Eprobe_minus = (Ey_nb - 1j*Ez_nb)/np.sqrt(2.0)
I_probe_plus_t = np.abs(Eprobe_plus)**2
I_probe_minus_t = np.abs(Eprobe_minus)**2


S0 = np.abs(Ey_nb)**2 + np.abs(Ez_nb)**2
S1 = np.abs(Ey_nb)**2 - np.abs(Ez_nb)**2
S2 = 2.0*np.real(Ey_nb*np.conj(Ez_nb))
S3 = -2.0*np.imag(Ey_nb*np.conj(Ez_nb))
theta_t = 0.5*np.arctan2(S2, S1)  # radians

# ---------- Frequency-domain θ at ~800 nm ----------
# choose the DFT bin closest to fs
# frequencies of the bins (for picking the one closest to fs)

f_bins = np.linspace(fs-df/2, fs+df/2, nfreq)
k = int(np.argmin(np.abs(f_bins - fs)))
Ey_d = sim.get_dft_array(dft_probe, mp.Ey, k).mean()
Ez_d = sim.get_dft_array(dft_probe, mp.Ez, k).mean()
S1f = (np.abs(Ey_d)**2 - np.abs(Ez_d)**2)
S2f = 2.0*np.real(Ey_d*np.conj(Ez_d))
theta_freq = 0.5*np.arctan2(S2f, S1f)

print(f"[RESULT] θ(freq @ ~800 nm) = {theta_freq*180/np.pi:.3f} deg")
print(f"[RESULT] θ(t) length = {len(theta_t)} samples.")

# ---------- Plots ----------
plt.figure(figsize=(6, 3.5))
plt.imshow(eps_slice.T, origin='lower',
           extent=[-sx/2, sx/2, -sy/2, sy/2], aspect='auto')
plt.colorbar(label='ε')
plt.title('Geometry: ε slice (z=0)')
plt.xlabel('x (µm)')
plt.ylabel('y (µm)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3.5))
plt.imshow((np.abs(Ey_xy.T)**2 + np.abs(Ez_xy.T)**2), origin='lower',
           extent=[-sy/2, sy/2, -sz/2, sz/2], aspect='equal')
plt.colorbar(label='|E|^2 (arb)')
plt.title('Field intensity in cavity plane (x=xcav)')
plt.xlabel('y (µm)')
plt.ylabel('z (µm)')
plt.tight_layout()
plt.show()

# Intensities vs time (point monitor): plot pump+probe envelopes (arb units)
plt.figure(figsize=(6, 3.0))
plt.plot(tt_ps, np.abs(Ey_t + 1j*Ez_t)**2/2, lw=1.0)
plt.plot(tt_ps, np.abs(Ey_t - 1j*Ez_t)**2/2, lw=1.0)
plt.xlabel('time (ps)')
plt.ylabel('Intensity ∝ |E|^2 (arb)')
plt.title('Total intensity at transmission probe point vs time')
plt.tight_layout()
plt.show()


# --- Three stacked intensity plots (shared time axis) ---
fig, axes = plt.subplots(3, 1, figsize=(7, 7.8), sharex=True)

# Top: probe E+ and E- (narrowband @ fs)
axes[0].plot(tt_ps, I_probe_plus_t,  lw=1.1, label='probe  E⁺')
axes[0].plot(tt_ps, I_probe_minus_t, lw=1.1, label='probe  E⁻')
axes[0].set_ylabel('|E|$^2$ (arb)')
axes[0].set_title('Narrowband intensities at transmission point')
axes[0].legend(loc='upper right', frameon=False)

# Middle: pumps
axes[1].plot(tt_ps, I_pump1_t, lw=1.1, label='pump 1 (1650 nm, e⁺)')
axes[1].plot(tt_ps, I_pump2_t, lw=1.1, label='pump 2 (1550 nm, e⁻)')
axes[1].set_ylabel('|E|$^2$ (arb)')
axes[1].legend(loc='upper right', frameon=False)

# Bottom: sidebands
axes[2].plot(tt_ps, I_sb_plus_t,  lw=1.1, label='sideband + (fs+Δ)')
axes[2].plot(tt_ps, I_sb_minus_t, lw=1.1, label='sideband − (fs−Δ)')
axes[2].set_ylabel('|E|$^2$ (arb)')
axes[2].set_xlabel('time (ps)')
axes[2].legend(loc='upper right', frameon=False)

fig.tight_layout()
plt.show()
# (Headless? use: plt.savefig("intensities_probe_pumps_sidebands.png", dpi=220); plt.close())

plt.figure(figsize=(6, 3.0))
plt.plot(tt_ps, theta_t*180/np.pi, lw=1.0)
plt.xlabel('time (ps)')
plt.ylabel(r'$\Theta(t)$ (deg)')
plt.title('Probe rotation angle θ(t) (narrowband ~800 nm)')
plt.tight_layout()
plt.show()
