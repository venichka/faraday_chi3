#!/usr/bin/env python
"""Estimate the AS-BUILT layer thicknesses of SIN51 from where the lab finds its resonances.

The fabricated sample differs slightly from the best_absolute design, which is why the pumps
had to move from 1521.5 / 1574.0 nm to 1560 / 1620 nm to sit on resonances (user, 2026-09-24),
with the probe at 800 nm. Deposition errors are modelled per material -- one thickness scale
for every SiN layer (mirror layers AND the SiN cavity), one for every SiO2 layer -- which is how
a PECVD rate miscalibration acts. The TMM mode finder (validated to <0.7% against FDTD) then
locates the pump-band and probe-band resonances of each candidate stack.

Targets: a pump-band resonance at 1560 nm, one at 1620 nm, and a probe resonance at 800 nm.
Three targets, two parameters: the residual is a consistency check on the model.

The misfit has a long diagonal valley (SiN and SiO2 errors trade off against each other), so
the split between the two materials is poorly determined; the uniform-scale member of the
valley is saved too, as the other end of the range for the simulations.

  python asbuilt_geometry.py      -> asbuilt/geometry.json (best), asbuilt/geometry_uniform.json,
                                     asbuilt/asbuilt_result.json, figs/asbuilt_geometry.png
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

import common as K  # this folder's common (plot style)

HERE = Path(__file__).resolve().parent
C = K.dbr_harness()  # chi5_dbr_design/common.py, loaded under its own name (both are "common")

TARGETS = {"pump_a": 1560.0, "pump_b": 1620.0, "probe": 800.0}
T = C._tmm()
IDX = T.index_map()


def scaled(g, s_sin, s_sio2):
    g = copy.deepcopy(g)
    sc = {"SiN": s_sin, "SiO2": s_sio2}
    for side in ("left", "right"):
        for L in g["mirrors"][side]:
            L["thk_um"] *= sc[L["mat"]]
    g["cavity"]["L_um"] *= sc[g["cavity"]["mat"]]
    return g


def modes(g, lo_nm, hi_nm):
    layers = T.build_layers(g)
    ms = T.find_modes_in_band(layers, IDX, 1e3 / hi_nm, 1e3 / lo_nm, "SiO2")
    return sorted(({"nm": 1e3 / m["freq"], "freq": m["freq"], "Q": m["Q"]} for m in ms),
                  key=lambda m: m["nm"])


def misfit(g):
    p, q = modes(g, 1450, 1720), modes(g, 770, 830)
    if not p or not q:
        return None
    near = lambda ms, x: min(ms, key=lambda m: abs(m["nm"] - x))  # noqa: E731
    hits = {"pump_a": near(p, TARGETS["pump_a"]), "pump_b": near(p, TARGETS["pump_b"]),
            "probe": near(q, TARGETS["probe"])}
    if hits["pump_a"] is hits["pump_b"]:
        return None
    d = np.array([hits[k]["nm"] - TARGETS[k] for k in TARGETS])
    return float(np.sqrt(np.mean(d ** 2))), hits, p, q


def main():
    base = C.load_base_geometry()
    s = np.arange(0.97, 1.0601, 0.0025)
    grid = np.full((len(s), len(s)), np.nan)
    for i, a in enumerate(s):
        for j, b in enumerate(s):
            m = misfit(scaled(base, a, b))
            if m:
                grid[i, j] = m[0]
    i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
    # refine
    fs = np.arange(s[i] - 0.004, s[i] + 0.0041, 0.0005)
    fo = np.arange(s[j] - 0.004, s[j] + 0.0041, 0.0005)
    best = None
    for a in fs:
        for b in fo:
            m = misfit(scaled(base, a, b))
            if m and (best is None or m[0] < best[0]):
                best = (m[0], a, b, m)
    rms, a, b, (_, hits, p, q) = best
    uni = min(((misfit(scaled(base, x, x)) or (1e9,))[0], x) for x in np.arange(0.98, 1.05, 0.0005))
    g = scaled(base, a, b)
    (HERE / "asbuilt").mkdir(exist_ok=True)
    json.dump(g, open(HERE / "asbuilt" / "geometry.json", "w"), indent=2)
    # the other end of the degenerate valley, kept for a sensitivity check in the simulations
    json.dump(scaled(base, uni[1], uni[1]), open(HERE / "asbuilt" / "geometry_uniform.json", "w"),
              indent=2)
    nominal = {"pump": modes(base, 1450, 1720), "probe": modes(base, 760, 850)}
    res = {"model": "one thickness scale per material (SiN incl. cavity, SiO2)",
           "s_SiN": float(a), "s_SiO2": float(b), "rms_nm": rms,
           "uniform_scale": {"s": float(uni[1]), "rms_nm": float(uni[0])},
           "targets_nm": TARGETS,
           "matched": {k: {"nm": v["nm"], "freq": v["freq"], "Q": v["Q"]} for k, v in hits.items()},
           "asbuilt_pump_modes": p, "asbuilt_probe_modes": modes(g, 760, 850),
           "nominal_pump_modes": nominal["pump"], "nominal_probe_modes": nominal["probe"],
           "layer_thicknesses_um": {"SiN": base["mirrors"]["left"][0]["thk_um"] * a,
                                    "SiO2": base["mirrors"]["left"][1]["thk_um"] * b,
                                    "cavity": base["cavity"]["L_um"] * a}}
    json.dump(res, open(HERE / "asbuilt" / "asbuilt_result.json", "w"), indent=2)
    print("per-material fit: s_SiN = {:.4f}, s_SiO2 = {:.4f}, rms {:.2f} nm".format(a, b, rms))
    for k, v in hits.items():
        print("  {:6s} target {:.0f} -> resonance {:.1f} nm (Q {:.0f})".format(k, TARGETS[k], v["nm"], v["Q"]))
    print("uniform-scale fit: s = {:.4f}, rms {:.2f} nm".format(uni[1], uni[0]))
    print("as-built probe modes:", [round(m["nm"], 1) for m in res["asbuilt_probe_modes"]])
    print("as-built pump modes :", [(round(m["nm"], 1), round(m["Q"])) for m in p])

    plt = K.style()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axs[0]
    im = ax.pcolormesh(s, s, grid.T, cmap="Blues_r", shading="auto", vmax=20)
    ax.plot(a, b, marker="o", ms=9, mfc="none", mec=K.ORANGE, mew=2, label="best (per material)")
    ax.plot([0.97, 1.06], [0.97, 1.06], color=K.INK2, lw=0.8, ls=":", label="uniform scale")
    ax.plot(1, 1, marker="x", color=K.INK, ms=8, label="design")
    ax.set_xlabel("SiN thickness scale")
    ax.set_ylabel("SiO₂ thickness scale")
    ax.set_title("rms miss of the 1560 / 1620 / 800 nm resonances")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="rms (nm)")
    ax = axs[1]
    lam = np.linspace(760, 1720, 6000)
    for gg, col, lab in ((base, K.INK2, "design (best_absolute)"), (g, K.BLUE, "as-built estimate")):
        layers = T.build_layers(gg)
        R, Tt = T.spectrum(layers, IDX, 1e3 / lam, "SiO2")
        ax.plot(lam, Tt, color=col, lw=1.0, label=lab)
    for v in TARGETS.values():
        ax.axvline(v, color=K.ORANGE, lw=1.2)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmission (TMM)")
    ax.set_title("resonances: orange = where the lab pumps / probes")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    K.save(fig, "asbuilt_geometry.png")


if __name__ == "__main__":
    main()
