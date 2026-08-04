#!/usr/bin/env python
"""Detailed side-by-side of the FABRICATED cavity and the optimized designs.

Produces
  docs/cmp_spectra.png    transmittance/reflectance of every design, over the pump band and the
                          probe band, with the operating wavelengths and the cavity modes marked
  docs/cmp_structure.png  refractive-index (stack) profiles drawn to scale, and the mode combs
  docs/comparison.md      the numbers: geometry, modes+Q, operating point, rotation, nonlinearity

Everything spectral is TMM (analytic, no FDTD) -- validated against the committed FDTD modes to
<0.3%. The rotation and nonlinear numbers are read from the FDTD stages.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import s3_validate as S3  # noqa: E402

sys.path.insert(0, str(C.MEEP / "chi5_optimization"))
import tmm  # noqa: E402

DOCS = HERE / "docs"
RUNS = HERE / "runs"
S2 = RUNS / "s2_fdtd" / "s2_result.json"
COLORS = {"baseline": "C3", "cand13": "C0", "cand16": "C2", "cand15": "C4", "cand07": "C1"}


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def load(name):
    p = RUNS / name
    return json.load(open(p)) if p.exists() else {}


def designs():
    """[(label, geom, op)] with the fabricated cavity first."""
    fin = S3.load_finalists(S2)
    out = [x for x in fin if x[0] == "baseline"] + [x for x in fin if x[0] != "baseline"]
    return out


def spectrum(geom, lam_lo, lam_hi, n=3000):
    """(lambda_nm, R, T) on a uniform FREQUENCY grid, returned sorted by INCREASING wavelength.
    The sort matters: lambda = 1/f runs backwards relative to f, and np.interp silently returns
    nonsense for a decreasing x-array (it put every operating-point marker at T = 0)."""
    idx, layers = tmm.index_map(), tmm.build_layers(geom)
    f = np.linspace(1.0 / lam_hi, 1.0 / lam_lo, n)
    R, T = tmm.spectrum(layers, idx, f)
    lam = 1000.0 / f
    o = np.argsort(lam)
    return lam[o], np.asarray(R)[o], np.asarray(T)[o]


def index_profile(geom, npts=4000):
    """n(z) of the deposited stack at 800 nm, from the incident face."""
    idx = tmm.index_map()
    layers = tmm.build_layers(geom)
    z, nz = [], []
    pos = 0.0
    for d, mat in layers:
        nval = float(np.real(idx[mat](0.8)))
        z += [pos, pos + d]
        nz += [nval, nval]
        pos += d
    return np.array(z) * 1000.0, np.array(nz)      # nm


# ------------------------------------------------------------------------ figures --- #
def fig_spectra(ds, path):
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.6))
    for ax, (lo, hi, title) in zip(axes, [(0.760, 0.980, "probe band"),
                                          (1.350, 1.900, "pump band")]):
        for label, geom, op in ds:
            lam, R, T = spectrum(geom, lo, hi)
            c = COLORS.get(label, "C7")
            lw = 2.2 if label == "baseline" else 1.4
            ax.plot(lam, T, color=c, lw=lw, alpha=0.9,
                    label="{}{}".format(label, "  (fabricated)" if label == "baseline" else ""))
            # operating wavelengths of this design
            marks = ([op["probe_nm"]] if title.startswith("probe")
                     else [1000.0 / op["pump1"], 1000.0 / op["pump2"]])
            for m in marks:
                if lo * 1000 <= m <= hi * 1000:
                    ax.plot([m], [np.interp(m, lam, T)], "v", color=c, ms=9,
                            markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        ax.set_xlabel("wavelength (nm)")
        ax.set_ylabel("transmittance")
        ax.set_title("{} — cavity transmittance; ▼ marks each design's operating wavelength(s)"
                     .format(title), fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, ncol=3)
        style(ax)
    fig.suptitle("Cavity spectra — fabricated vs optimized (TMM, validated to <0.3% vs FDTD)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


def fig_structure(ds, path):
    n = len(ds)
    fig, axes = plt.subplots(n, 1, figsize=(12.5, 1.85 * n + 1.2), squeeze=False, sharex=True)
    for ax, (label, geom, op) in zip(axes[:, 0], ds):
        z, nz = index_profile(geom)
        c = COLORS.get(label, "C7")
        ax.fill_between(z, 1.0, nz, step="pre", color=c, alpha=0.35, lw=0)
        ax.plot(z, nz, color=c, lw=1.2)
        p = C.geometry_params(geom)
        ax.set_ylabel("n", fontsize=9)
        ax.set_ylim(1.3, 2.25)
        ax.annotate("{}{}   {}+{} pairs | SiN {:.0f} nm / SiO$_2$ {:.0f} nm "
                    "($t_{{lo}}/t_{{hi}}$={:.2f}) | cavity {:.0f} nm | stack {:.2f} $\\mu$m"
                    .format(label, "  (FABRICATED)" if label == "baseline" else "",
                            p["n_left"], p["n_right"], p["t_hi"] * 1000, p["t_lo"] * 1000,
                            p["t_lo"] / p["t_hi"], p["L_cav"] * 1000, p["stack_um"]),
                    (0.005, 0.93), xycoords="axes fraction", va="top", fontsize=8, color=c,
                    weight="bold" if label == "baseline" else "normal")
        style(ax)
    axes[-1, 0].set_xlabel(r"depth from the incident face (nm)")
    fig.suptitle("Deposition stacks to scale — refractive index at 800 nm", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


# ------------------------------------------------------------------------- report --- #
def mode_table(geom, lo, hi):
    idx, layers = tmm.index_map(), tmm.build_layers(geom)
    ms = tmm.find_modes_in_band(layers, idx, 1.0 / hi, 1.0 / lo)
    ms.sort(key=lambda m: m["lambda_um"])
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    DOCS.mkdir(parents=True, exist_ok=True)
    ds = designs()
    s2 = json.load(open(S2))
    by = {r["label"]: r for r in s2["ranking"]}
    inten = load("s3_validate/intensity_result.json")
    tol = load("s3_validate/tolerance_result.json")
    d3 = load("s3_validate/3d_result.json")
    s4 = (load("s4_delay/s4_result.json") or {}).get("designs", {})

    fig_spectra(ds, DOCS / "cmp_spectra.png")
    fig_structure(ds, DOCS / "cmp_structure.png")

    base_t = abs(by["baseline"]["best"]["theta_chi5_deg"])
    out = ["# Fabricated cavity vs optimized designs — detailed comparison", "",
           "All rotations are the **carrier-averaged, pulse-integrated** χ⁽⁵⁾ rotation at",
           "I_pump = 1e12 W/cm², 100 fs intensity FWHM, balanced σ⁺σ⁻ counter-rotating pumps.",
           "Spectra and mode tables are TMM (validated <0.3% against the committed FDTD modes).",
           "", "![stacks](cmp_structure.png)", "", "![spectra](cmp_spectra.png)", ""]

    # ---- geometry ---- #
    out += ["## 1. Geometry", "",
            "| design | pairs L/R | SiN (nm) | SiO₂ (nm) | t_lo/t_hi | cavity (nm) | stack (µm) |",
            "|---|---|---|---|---|---|---|"]
    for label, geom, op in ds:
        p = C.geometry_params(geom)
        out.append("| {}{} | {} / {} | {:.1f} | {:.1f} | **{:.2f}** | {:.0f} | {:.2f} |".format(
            label, " *(fabricated)*" if label == "baseline" else "",
            p["n_left"], p["n_right"], p["t_hi"] * 1000, p["t_lo"] * 1000,
            p["t_lo"] / p["t_hi"], p["L_cav"] * 1000, p["stack_um"]))
    out += ["", "The **mirror detuning t_lo/t_hi is the single strongest design lever** "
                "(Spearman −0.598 against rotation) and the fabricated cavity sits on the wrong "
                "side of it: 1.45 (thick SiO₂) against 0.26–0.81 for the optimized designs.", ""]

    # ---- operating point ---- #
    out += ["## 2. Operating point", "",
            "| design | probe (nm) | pump1 (nm) | pump2 (nm) | separation (nm) | Δ (1/µm) |",
            "|---|---|---|---|---|---|"]
    for label, geom, op in ds:
        out.append("| {} | {:.1f} | {:.1f} | {:.1f} | {:.1f} | {:.4f} |".format(
            label, op["probe_nm"], 1000.0 / op["pump1"], 1000.0 / op["pump2"],
            abs(1000.0 / op["pump2"] - 1000.0 / op["pump1"]), op["delta"]))
    out.append("")

    # ---- modes ---- #
    out += ["## 3. Cavity modes (TMM)", ""]
    for label, geom, op in ds:
        pr = mode_table(geom, 0.760, 0.980)
        pu = mode_table(geom, 1.350, 1.900)
        out += ["**{}** — probe band: ".format(label)
                + ", ".join("{:.1f} nm (Q {:.0f}){}".format(
                    m["lambda_um"] * 1000, m["Q"],
                    " ←probe" if abs(m["lambda_um"] * 1000 - op["probe_nm"]) < 3 else "")
                    for m in pr), "",
                "&nbsp;&nbsp;&nbsp;&nbsp;pump band: "
                + ", ".join("{:.0f} nm (Q {:.0f})".format(m["lambda_um"] * 1000, m["Q"])
                            for m in pu), ""]
    out += ["Every pump-band mode has Q ≈ 40–130 while a 100 fs pump only resolves "
            "Q_cap = f/fwidth ≈ 12, so **all of them are unresolved by the pulse and the "
            "intracavity buildup is saturated** — which is why pump placement is not about "
            "hitting a mode centre.", "",
            "⭐ **The clearest illustration is the fabricated cavity itself.** Its pump-band Q "
            "climbs steeply with wavelength — 1493 nm (Q 41), 1525 nm (Q 41) … 1752 nm (Q 103), "
            "1865 nm (Q 130) — because 1700–1870 nm is its mirror stopband (see the deep "
            "transmittance minimum in the lower spectra panel). Yet **its best operating point "
            "is the pump pair at 1493 / 1525 nm, i.e. on its two LOWEST-Q pump modes**, and the "
            "high-Q modes inside the stopband are useless. With buildup saturated, Q carries no "
            "information; what matters is the four-wave-mixing and sideband placement. Ranking "
            "pump centres by Q — the obvious thing to do — actively selects the wrong ones.", ""]

    # ---- rotation ---- #
    out += ["## 4. Rotation", "",
            "| design | θ_χ5 1D | vs fabricated | θ_χ5 3D | 3D/1D | fringe 1D | "
            "**contrast** 1D → 3D | DoLP 1D / 3D |", "|---|---|---|---|---|---|---|---|"]
    for label, geom, op in ds:
        r = by[label]
        t = abs(r["best"]["theta_chi5_deg"])
        f = r["best"]["theta_fringe_amp_deg"]
        v3 = d3.get(label) or {}
        t3 = v3.get("theta_3d")
        c3 = (t3 / v3["fringe_3d"]) if (t3 and v3.get("fringe_3d")) else None
        out.append("| {} | {:.5f}° | {:.2f}× | {} | {} | {:.5f}° | {:.2f} → {} | {:.3f} / {} |"
                   .format(label, t, t / base_t,
                           "{:.4f}°".format(t3) if t3 else "—",
                           "{:.1f}×".format(v3["ratio"]) if v3.get("ratio") else "—",
                           f, t / f, "**{:.2f}**".format(c3) if c3 else "—",
                           r["best"]["dolp"],
                           "{:.3f}".format(v3["dolp_3d"]) if v3.get("dolp_3d") else "—"))
    out += ["", "`contrast = θ_χ5 / carrier-fringe amplitude` — above 1 the effect is the "
                "dominant feature of a delay trace; below 1 it hides under the fringe.", ""]

    # ---- nonlinearity ---- #
    out += ["## 5. Nonlinear behaviour", "",
            "| design | local log-log slopes (2.5e11→4e12) | global | θ at 4e12 (1D) | "
            "DoLP at 4e12 | σ=3 nm worst | σ=5 nm worst |", "|---|---|---|---|---|---|---|"]
    for label, geom, op in ds:
        d = inten.get(label)
        tl = (tol.get(label) or {}).get("sigmas", {})
        sl = ", ".join("{:.2f}".format(s) for _a, _b, s in d["local_slopes"]) if d else "—"
        hi = d["rows"][-1] if d else None
        out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            label, sl, "{:.2f}".format(d["global_slope"]) if d else "—",
            "{:.3f}°".format(hi["theta"]) if hi else "—",
            "{:.3f}".format(hi["dolp"]) if hi else "—",
            "{:.0f}%".format(100 * tl["3.0"]["min_rel"]) if "3.0" in tl else "—",
            "{:.0f}%".format(100 * tl["5.0"]["min_rel"]) if "5.0" in tl else "—"))
    out += ["", "A χ⁽⁵⁾ cascade gives slope 2; χ⁽³⁾ gives 1. The tolerance columns are the worst "
                "of 12 independent Gaussian layer-error draws, with the operating point held "
                "fixed (no post-fabrication re-tuning).", ""]

    # ---- delay trace ---- #
    if s4:
        out += ["## 6. Predicted delay trace", "",
                "| design | true effect at τ=0 | phase-stable line reads | sign | "
                "effect/fringe at overlap |", "|---|---|---|---|---|"]
        for label, geom, op in ds:
            d = s4.get(label)
            if not d:
                continue
            rows = sorted(d["rows"], key=lambda r: r["tau_fs"])
            tau = np.array([r["tau_fs"] for r in rows])
            env = np.array([r["theta_avg_deg"] for r in rows])
            si = np.array([r["theta_single_deg"] for r in rows])
            fr = np.array([r["fringe_amp_deg"] for r in rows])
            i0 = int(np.argmin(abs(tau)))
            core = abs(tau) <= 50
            same = np.sign(env[i0]) == np.sign(si[i0])
            out.append("| {} | {:+.5f}° | {:+.5f}° | {} | {:.2f} |".format(
                label, env[i0], si[i0], "same" if same else "**OPPOSITE**",
                float(np.median(np.abs(env[core]) / np.maximum(fr[core], 1e-12)))))
        out += ["", "![trace](s4_trace.png)", ""]

    path = DOCS / "comparison.md"
    path.write_text("\n".join(out))
    print("->", path)


if __name__ == "__main__":
    main()
