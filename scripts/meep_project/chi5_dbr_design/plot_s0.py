#!/usr/bin/env python
"""Figures for Stage 0 (s0_harness.py).

  s0_carrier.png       the carrier fringe vs the rectified effect, on the baseline cavity --
                       the whole principle of the campaign in one panel
  s0_pulse.png         what the 100 fs (intensity FWHM) correction does
  s0_estimators.png    every cheap estimator against the objective + the L trends

Convention: `theta_chi5` is the carrier-averaged, pulse-integrated rotation (the objective =
the balanced-detector observable); `legacy` is the tail-window azimuth that produced the
published 0.137 deg; `fringe` is the coherent artifact's fundamental amplitude.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "s0_harness"
DOCS = HERE / "docs"

C_EFFECT, C_FRINGE, C_LEGACY, C_SINGLE = "C0", "C3", "C1", "C2"


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_carrier(res, path):
    """The four carrier sub-samples on the baseline: a large oscillation whose mean is the
    small rectified effect. Shows why a single-phase measurement is not the effect."""
    pa = res.get("part_a", {})
    if not pa:
        return
    fig, axes = plt.subplots(1, len(pa), figsize=(5.4 * len(pa), 4.2), squeeze=False)
    for ax, (tag, r) in zip(axes[0], pa.items()):
        sub = np.array(r["theta_sub_deg"], dtype=float)
        n = len(sub)
        phase = np.arange(n) / n
        T1 = 1.0 / (res["config"]["base_freqs"]["pump1"] * 0.299792458)
        # EXACT band-limited interpolation of the 4 uniform samples: DC + fundamental +
        # Nyquist (2nd harmonic) is 4 degrees of freedom for 4 points, so this passes through
        # every sample rather than being an eyeballed guide.  The 2nd-harmonic term is why
        # N = 4 is the minimum defensible sub-sample count -- N = 2 could not resolve it, and
        # a fringe with a strong 2nd harmonic would survive a 2-point "average".
        ph = np.linspace(0, 1, 400)
        dc = sub.mean()
        c1 = np.sum(sub * np.exp(-2j * np.pi * phase)) / n
        c2 = float(np.real(np.sum(sub * np.exp(-4j * np.pi * phase)) / n))
        recon = (dc + 2 * np.abs(c1) * np.cos(2 * np.pi * ph - np.angle(c1))
                 + c2 * np.cos(4 * np.pi * ph))
        ax.plot(ph * T1, recon, color=C_FRINGE, lw=1.3, alpha=0.5, zorder=1,
                label="exact DFT interpolation (2nd harm. {:.0f}%)".format(
                    100 * abs(c2) / max(2 * abs(c1), 1e-30)))
        ax.plot(phase * T1, sub, "o", color=C_FRINGE, ms=9, zorder=3,
                label="single-phase samples (what one run gives)")
        # NB: format() must NOT be applied to the LaTeX literal -- adjacent string literals
        # concatenate first, and "$\theta_{\chi^5}$" would then be read as a format field.
        ax.axhline(r["theta_chi5_deg"], color=C_EFFECT, lw=2.2, zorder=2,
                   label=r"carrier-averaged $\theta_{\chi^5}$ = "
                         + "{:+.5f}".format(r["theta_chi5_deg"]) + r"$\degree$")
        ax.axhline(0.0, color="0.6", lw=0.8, ls=":", zorder=0)
        ratio = r["theta_fringe_amp_deg"] / max(abs(r["theta_chi5_deg"]), 1e-12)
        ax.set_title("{}  ({:.0f} fs intensity FWHM)\nfringe amplitude "
                     "{:.4f}$\\degree$ = {:.0f}$\\times$ the effect".format(
                         tag, r["intensity_fwhm_fs"], r["theta_fringe_amp_deg"], ratio),
                     fontsize=10)
        ax.set_xlabel(r"pump1 delay within one optical period $T_1$ (fs)")
        ax.set_ylabel(r"probe rotation $\theta$ (deg)")
        ax.legend(fontsize=8, loc="best")
        style(ax)
    fig.suptitle("The carrier fringe is not the effect — SiN best_absolute, "
                 r"$\tau=0$, $I=10^{12}$ W/cm$^2$", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


def fig_pulse(res, path):
    """Part A: the estimators side by side at both pulse settings (grouped bars, log scale --
    the three quantities span two decades, which is the point)."""
    pa = res.get("part_a", {})
    if len(pa) < 1:
        return
    keys = ["theta_chi5_deg", "theta_fringe_amp_deg", "theta_legacy_deg"]
    names = [r"$\theta_{\chi^5}$ (objective)", "fringe amplitude", "legacy (tail-window)"]
    cols = [C_EFFECT, C_FRINGE, C_LEGACY]
    tags = list(pa.keys())
    x = np.arange(len(keys))
    w = 0.8 / len(tags)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for i, tag in enumerate(tags):
        vals = [abs(pa[tag][k]) for k in keys]
        pos = x + (i - (len(tags) - 1) / 2) * w
        bars = ax.bar(pos, vals, w * 0.88, label="{} ({:.0f} fs FWHM)".format(
            tag, pa[tag]["intensity_fwhm_fs"]),
            color=[cols[j] for j in range(len(keys))],
            alpha=0.55 if i == 0 else 1.0,
            edgecolor="white", linewidth=1.2,
            hatch="//" if i == 0 else None)
        for b, v in zip(bars, vals):
            ax.annotate("{:.4f}".format(v), (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel(r"$|\theta|$ (deg, log scale)")
    ax.set_title("Pulse-duration correction and the estimator hierarchy\n"
                 "(hatched = legacy 120.1 fs label, solid = true 100 fs intensity FWHM)",
                 fontsize=10)
    ax.legend(fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


def fig_estimators(res, path):
    """Part B: does any 1-run estimator rank geometries like the 4-run objective?"""
    rows = res.get("part_b", [])
    if len(rows) < 3:
        print("  (part B incomplete: {} rows)".format(len(rows)))
        return
    obj = np.array([abs(r["theta_chi5_deg"]) for r in rows])
    L = np.array([r["params"]["L_cav"] for r in rows])
    dlt = np.array([r["op"]["delta"] for r in rows])
    npair = np.array([r["params"]["n_left"] for r in rows])
    comps = [("theta_single_phase_deg", "single phase (1 sim)", C_SINGLE, True),
             ("theta_legacy_deg", "legacy tail-window (old objective)", C_LEGACY, True),
             ("theta_fringe_amp_deg", "carrier-fringe amplitude", C_FRINGE, False)]
    corr = res.get("correlations", {})

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.4))
    # --- top row: cheap estimator vs the objective -------------------------------------- #
    for ax, (key, name, col, take_abs) in zip(axes[0], comps):
        y = np.array([abs(r[key]) if take_abs else r[key] for r in rows])
        for d, mk in zip(sorted(set(dlt)), ["o", "s"]):
            m = dlt == d
            ax.scatter(obj[m], y[m], s=np.where(npair[m] == 4, 95, 60), c=col, marker=mk,
                       edgecolor="white", linewidth=0.8,
                       label=r"$\Delta$ = {:.3f}".format(d), zorder=3)
        rho = corr.get(key.replace("theta_", "").replace("_deg", ""), np.nan)
        ax.set_xlabel(r"objective  $|\theta_{\chi^5}|$  (deg)")
        ax.set_ylabel("{} (deg)".format(name))
        ax.set_title("{}\nSpearman $\\rho$ = {:+.3f}".format(name, rho), fontsize=10)
        ax.legend(fontsize=8, title="marker size = pairs/side", title_fontsize=7)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(8)
        style(ax)
    # --- bottom row: the L trend for each estimator, one axis each (never dual-axis) ----- #
    for ax, (key, name, col, take_abs) in zip(axes[1], comps):
        for d, mk in zip(sorted(set(dlt)), ["o", "s"]):
            for np_ in sorted(set(npair)):
                m = (dlt == d) & (npair == np_)
                if not m.any():
                    continue
                o = np.argsort(L[m])
                y = np.array([abs(r[key]) if take_abs else r[key] for r in rows])[m][o]
                ax.plot(L[m][o], y, mk + "-", color=col, ms=6, lw=1.3,
                        alpha=1.0 if np_ == 3 else 0.5,
                        label=r"$\Delta$={:.3f}, {} pairs".format(d, np_))
        ax.set_xlabel(r"cavity length $L$ ($\mu$m)")
        ax.set_ylabel("{} (deg)".format(name))
        ax.set_title("{} vs cavity length".format(name), fontsize=10)
        ax.legend(fontsize=7)
        style(ax)
    # objective's own L trend, overlaid as a reference line on the first bottom panel
    ax = axes[1][0]
    for d, mk in zip(sorted(set(dlt)), ["o", "s"]):
        m = (dlt == d) & (npair == 3)
        if m.any():
            o = np.argsort(L[m])
            ax.plot(L[m][o], obj[m][o], mk + "--", color=C_EFFECT, ms=5, lw=1.6,
                    label=r"OBJECTIVE $\Delta$={:.3f}".format(d))
    ax.legend(fontsize=7)

    axes[1][0].annotate(
        "the objective (blue, dashed) is FLAT and ~15$\\times$ smaller;\n"
        "the $L$-growth visible in a 1-sim estimator is the FRINGE",
        (0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=7.5, color="0.25",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
    axes[1][0].legend(fontsize=6.5, loc="upper left", ncol=2)
    fig.suptitle("Stage 0 Part B — can a 1-run estimator screen geometries?  "
                 r"No: $\rho\approx0$ for all three."
                 "\n(each case at its TMM-proposed operating point, so the objective range is "
                 "compressed — Stage 2 gives every geometry its own FDTD-optimised point)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(OUT / "s0_result.json"))
    args = ap.parse_args()
    res = json.load(open(args.result))
    DOCS.mkdir(parents=True, exist_ok=True)
    fig_carrier(res, DOCS / "s0_carrier.png")
    fig_pulse(res, DOCS / "s0_pulse.png")
    fig_estimators(res, DOCS / "s0_estimators.png")


if __name__ == "__main__":
    main()
