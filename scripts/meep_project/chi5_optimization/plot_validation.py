#!/usr/bin/env python
"""Visual control for the TMM engine:
  (1) TMM-vs-Meep reflectance overlay (the headline validation), modes marked;
  (2) cavity-mode |E(z)|^2 field profiles over the eps(z) stack (TMM).
Writes PNGs to chi5_optimization/validation/. Needs only numpy + matplotlib (+ tmm.py).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tmm  # same directory

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
OUT = HERE / "validation"
MODE_COLORS = {"probe": "tab:blue", "pump1": "tab:green", "pump2": "tab:purple"}


def load_meep(csv):
    p = Path(csv)
    if not p.exists():
        return None
    d = np.loadtxt(p, delimiter=",", skiprows=1)
    return d[:, 0] / 1000.0, d[:, 1]   # lam_um, R


def reflectance_plot(name, geom, modes, meep, wl=(0.6, 2.0), npts=1400):
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    lam = np.linspace(wl[0], wl[1], npts)
    Rtmm = np.array([tmm.rt_at(layers, idx, L)[0] for L in lam])
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    m = load_meep(meep)
    if m is not None:
        ax.plot(m[0] * 1000, m[1], color="0.45", lw=1.6, label="Meep FDTD", zorder=2)
    ax.plot(lam * 1000, Rtmm, color="tab:red", lw=1.1, label="TMM (analytic)", zorder=3)
    for k, c in MODE_COLORS.items():
        lam_nm = 1000.0 / modes[k]["frequency"]
        ax.axvline(lam_nm, color=c, ls=":", lw=1.1, label=f"{k} {lam_nm:.0f}nm")
    if m is not None:
        rms = np.sqrt(np.mean((np.interp(m[0], lam, Rtmm) - m[1]) ** 2))
        ax.text(0.02, 0.04, f"TMM-Meep RMS = {rms:.3f}", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("Reflectance")
    ax.set_title(f"{name}: TMM vs Meep reflectance"); ax.set_ylim(-0.02, 1.03)
    ax.legend(fontsize=7, ncol=2, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"reflectance_{name}.png", dpi=140); plt.close(fig)


def fields_plot(name, geom, modes):
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    zc0 = sum(float(l["thk_um"]) for l in geom["mirrors"]["left"])
    zc1 = zc0 + float(geom["cavity"]["L_um"])
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 7.4), sharex=True)
    for ax, (k, c) in zip(axs, MODE_COLORS.items()):
        mo = tmm.find_mode(layers, idx, modes[k]["frequency"])
        z, E, eps = tmm.field_profile(layers, idx, mo["freq"])
        I = np.abs(E) ** 2; I = I / I.max()
        ax2 = ax.twinx()
        ax2.fill_between(z, eps, color="0.85", lw=0, zorder=0)
        ax2.set_ylabel("eps(z)", color="0.55"); ax2.set_ylim(0, eps.max() * 1.05)
        ax.axvspan(zc0, zc1, color="0.97", zorder=0)
        ax.plot(z, I, color=c, lw=1.2, zorder=3)
        ax.set_ylabel("|E|^2 (norm)"); ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        V = tmm.mode_volume(z, E, eps)
        ax.set_title(f"{k}: lam={1000*mo['lambda_um']:.1f}nm  Q={mo['Q']:.0f}  Veff={V:.2f}um",
                     fontsize=9, loc="left")
    axs[-1].set_xlabel("z (um)   [shaded band = cavity defect]")
    fig.suptitle(f"{name}: TMM cavity-mode field profiles")
    fig.tight_layout(); fig.savefig(OUT / f"fields_{name}.png", dpi=140); plt.close(fig)


def run(name, geom_path, modes_path, meep_csv):
    geom = json.load(open(geom_path)); modes = json.load(open(modes_path))
    reflectance_plot(name, geom, modes, meep_csv)
    fields_plot(name, geom, modes)
    print(f"{name}: wrote reflectance_{name}.png + fields_{name}.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    run("sin_best_absolute",
        MEEP / "SiN_optimizations/best_absolute/geometry.json",
        MEEP / "SiN_optimizations/best_absolute/cavity_modes.json",
        OUT / "meep_refl_sin_best_absolute.csv")
    run("sic_L3p2um",
        MEEP / "SiC_optimizations/sic_L3p2um/geometry.json",
        MEEP / "SiC_optimizations/sic_L3p2um/cavity_modes.json",
        OUT / "meep_refl_sic_L3p2um.csv")
