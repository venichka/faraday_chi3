#!/usr/bin/env python3
"""
Supplementary mode-analysis plots for the SiC L=3.2um study.

Reuses the 1D stack/sim helpers from fp_cavity_modes_spectrum.py and the dispersive
material fit from mode_targeting.get_cavity_materials, then produces:

  mode_analysis/epsilon_profile.png   - eps(z) of the SiC/SiO2 stack at the probe wavelength
  mode_analysis/mode_profiles.png     - |Ex|(z) at probe / pump1 / pump2 (DFT), eps(z) shaded
  mode_analysis/fwm_overlap.png       - the |E_probe * E_pump1 * E_pump2|(z) product profile
  mode_analysis/overlaps.json         - normalized spatial-overlap integrals

Run from the meep_project/ directory (so sic.csv / sio2.csv resolve):
  MPLBACKEND=Agg python SiC_optimizations/sic_L3p2um/mode_plots.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import meep as mp

# Allow imports of the project modules regardless of where this is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fp_cavity_modes_spectrum import (
    CavityConfig, build_geometry, make_simulation, epsilon_profile, FCEN, DF,
)
from mode_targeting import get_cavity_materials
from geometry_io import load_params

HERE = Path("SiC_optimizations/sic_L3p2um")
OUT = HERE / "mode_analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Material model for this study (SiC high-index, SiO2 low-index), 3-pole stable fit.
FIT = dict(sin_csv="sic.csv", sio2_csv="sio2.csv", lam_min=600, lam_max=2000, fit_poles=3)

# numpy>=2.0 renamed trapz -> trapezoid
_trap = getattr(np, "trapezoid", getattr(np, "trapz", None))


def load_config() -> CavityConfig:
    p = load_params(report_json="__no_report__.json",
                    geom_json=str(HERE / "geometry.json"), prefer="geom")
    return CavityConfig(t_SiN=p["t_SiN"], t_SiO2=p["t_SiO2"], t_cav=p["t_cav"],
                        N_per=p["N_per"], pad_air=p["pad_air"], pad_sub=p["pad_sub"],
                        dpml=p["dpml"], resolution=p["resolution"],
                        cell_margin=p["cell_margin"])


def field_profiles(cfg, geometry, cell_z, freqs):
    """Steady-state |Ex|(z) at each frequency from a broadband DFT run."""
    src_z = -0.5 * cell_z + cfg.dpml + 0.2
    src = [mp.Source(mp.GaussianSource(FCEN, fwidth=DF), component=mp.Ex,
                     center=mp.Vector3(0, 0, src_z), amplitude=1.0)]
    sim = make_simulation(cfg, geometry, cell_z, sources=src, force_complex_fields=True)
    monitor_len = cell_z - 2 * cfg.dpml - 0.02
    vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 0, monitor_len))
    dft = sim.add_dft_fields([mp.Ex], freqs, where=vol)
    settle = max(1500, int(np.ceil(60 / min(freqs))))
    sim.run(until=settle)
    n = sim.get_dft_array(dft, mp.Ex, 0).size
    z = np.linspace(-0.5 * monitor_len, 0.5 * monitor_len, n)
    fields = [np.abs(sim.get_dft_array(dft, mp.Ex, i)) for i in range(len(freqs))]
    return z, fields, monitor_len


def main():
    cfg = load_config()
    mat_sic, mat_sio2 = get_cavity_materials(
        model="fit", high_index_material="sic",
        sin_csv=FIT["sin_csv"], sio2_csv=FIT["sio2_csv"],
        lam_min=FIT["lam_min"], lam_max=FIT["lam_max"], fit_poles=FIT["fit_poles"])
    geometry, cell_z = build_geometry(cfg, mat_sic, mat_sio2)

    modes = json.loads((HERE / "cavity_modes.json").read_text())
    roles = ["probe", "pump1", "pump2"]
    freqs = [float(modes[r]["frequency"]) for r in roles]
    lams_nm = [1e3 / f for f in freqs]
    lam_probe = 1.0 / freqs[0]

    # --- eps(z) profile (at probe wavelength) ---
    zeps, eps = epsilon_profile(geometry, cell_z, lam_probe)
    plt.figure(figsize=(9, 3))
    plt.plot(zeps, eps, lw=1.2)
    plt.axvspan(-0.5 * cell_z, -0.5 * cell_z + cfg.dpml, color="k", alpha=0.06)
    plt.axvspan(0.5 * cell_z - cfg.dpml, 0.5 * cell_z, color="k", alpha=0.06)
    plt.xlabel("z (µm)"); plt.ylabel("ε  (at probe λ)")
    plt.title(f"SiC/SiO₂ stack ε(z),  L_cav={cfg.t_cav:.2f} µm")
    plt.tight_layout(); plt.savefig(OUT / "epsilon_profile.png", dpi=140); plt.close()

    # --- field profiles |Ex|(z) ---
    z, fields, mlen = field_profiles(cfg, geometry, cell_z, freqs)
    # eps resampled onto field grid for shading
    eps_on_z = np.interp(z, zeps, eps)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax2 = ax.twinx()
    ax2.fill_between(z, eps_on_z, color="0.85", step=None, alpha=0.6, zorder=0)
    ax2.set_ylabel("ε(z)  (shaded)")
    for role, f, lam, prof in zip(roles, freqs, lams_nm, fields):
        ax.plot(z, prof / prof.max(), lw=1.3, label=f"{role}: {lam:.0f} nm (Q={modes[role].get('Q')})", zorder=3)
    ax.set_xlabel("z (µm)"); ax.set_ylabel("|Ex|(z), normalized")
    ax.set_title("Mode field profiles (DFT steady state)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "mode_profiles.png", dpi=140); plt.close(fig)

    # --- overlaps ---
    def norm(a):
        return a / np.sqrt(_trap(a * a, z))
    up, u1, u2 = (norm(f) for f in fields)
    # pairwise intensity overlap and triple FWM overlap (all normalized)
    def iov(a, b):
        return float(_trap((a**2) * (b**2), z))
    triple = float(_trap(up * u1 * u2 * np.abs(up), z))  # ~ probe-weighted FWM density
    fwm_density = up * u1 * u2
    overlaps = {
        "intensity_overlap_probe_pump1": iov(up, u1),
        "intensity_overlap_probe_pump2": iov(up, u2),
        "intensity_overlap_pump1_pump2": iov(u1, u2),
        "fwm_triple_overlap_norm": float(_trap(np.abs(fwm_density), z)),
        "field_peak_in_cavity": {
            role: float(prof[np.abs(z) <= 0.5 * cfg.t_cav].max() / prof.max())
            for role, prof in zip(roles, fields)
        },
    }
    (OUT / "overlaps.json").write_text(json.dumps(overlaps, indent=2))

    plt.figure(figsize=(9, 3))
    plt.plot(z, np.abs(fwm_density) / np.abs(fwm_density).max(), color="C3", lw=1.3)
    plt.axvspan(-0.5 * cfg.t_cav, 0.5 * cfg.t_cav, color="C0", alpha=0.08, label="cavity")
    plt.xlabel("z (µm)"); plt.ylabel("|E_pr·E_p1·E_p2| (norm)")
    plt.title("FWM spatial overlap density")
    plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / "fwm_overlap.png", dpi=140); plt.close()

    print("[mode_plots] wrote epsilon_profile.png, mode_profiles.png, fwm_overlap.png, overlaps.json")
    print("[overlaps]", json.dumps(overlaps, indent=2))


if __name__ == "__main__":
    main()
