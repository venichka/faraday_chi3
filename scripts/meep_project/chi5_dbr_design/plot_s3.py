#!/usr/bin/env python
"""Figures for Stage 3 (s3_validate.py) -- do the finalists survive validation?

  s3_intensity.png   |theta_chi5| vs pump intensity, with LOCAL log-log slopes.  chi5 => 2,
                     chi3 => 1.  The response is a crossover, so the local slope is the
                     meaningful quantity; a global fit averages across it.
  s3_tolerance.png   signal retained under independent Gaussian layer-thickness error
                     (sigma = 3 and 5 nm), at a FIXED operating point.
  s3_3d.png          1D vs 3D on the chi5 channel, and what happens to the contrast.
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
OUT = HERE / "runs" / "s3_validate"
DOCS = HERE / "docs"
COLORS = ["C0", "C2", "C4", "C5", "C3"]


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_intensity(path):
    p = OUT / "intensity_result.json"
    if not p.exists():
        return
    res = json.load(open(p))
    if not res:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for i, (label, d) in enumerate(res.items()):
        I = np.array([r["I"] for r in d["rows"]])
        th = np.array([r["theta"] for r in d["rows"]])
        c = COLORS[i % len(COLORS)]
        ax.plot(I, th, "o-", color=c, ms=7, lw=1.6,
                label="{}  (global {:.2f})".format(label, d["global_slope"]))
    # slope-2 reference anchored on the first series' first point
    first = next(iter(res.values()))
    I0, t0 = first["rows"][0]["I"], first["rows"][0]["theta"]
    Iref = np.array([r["I"] for r in first["rows"]])
    ax.plot(Iref, t0 * (Iref / I0) ** 2, ls="--", color="0.4", lw=1.3,
            label=r"$\propto I^2$ ($\chi^{(5)}$)")
    ax.plot(Iref, t0 * (Iref / I0) ** 1, ls=":", color="0.6", lw=1.3,
            label=r"$\propto I^1$ ($\chi^{(3)}$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"pump intensity (W/cm$^2$)")
    ax.set_ylabel(r"$|\theta_{\chi^5}|$ (deg)")
    ax.set_title("intensity scaling", fontsize=10)
    ax.legend(fontsize=7.5)
    style(ax)

    ax = axes[1]
    for i, (label, d) in enumerate(res.items()):
        mid = [np.sqrt(a * b) for a, b, _ in d["local_slopes"]]
        sl = [s for _, _, s in d["local_slopes"]]
        ax.plot(mid, sl, "o-", color=COLORS[i % len(COLORS)], ms=7, lw=1.6, label=label)
    ax.axhline(2.0, color="0.4", ls="--", lw=1.3)
    ax.annotate(r"$\chi^{(5)}$", (0.02, 2.02), xycoords=("axes fraction", "data"), fontsize=9,
                color="0.4")
    ax.axhline(1.0, color="0.6", ls=":", lw=1.3)
    ax.annotate(r"$\chi^{(3)}$", (0.02, 1.02), xycoords=("axes fraction", "data"), fontsize=9,
                color="0.6")
    ax.set_xscale("log")
    ax.set_xlabel(r"pump intensity (W/cm$^2$, geometric midpoint)")
    ax.set_ylabel("LOCAL log-log slope")
    ax.set_title(r"local slope — read this, not the global fit"
                 "\n(the response is a $\\chi^{(3)}\\rightarrow\\chi^{(5)}$ crossover)",
                 fontsize=10)
    ax.legend(fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


def fig_tolerance(path):
    p = OUT / "tolerance_result.json"
    if not p.exists():
        return
    res = json.load(open(p))
    labels = [k for k, v in res.items() if v.get("sigmas")]
    if not labels:
        return
    sigmas = sorted({s for v in res.values() for s in v.get("sigmas", {})}, key=float)
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(labels) + 3), 5))
    w = 0.8 / len(sigmas)
    for j, sig in enumerate(sigmas):
        data, pos = [], []
        for i, lab in enumerate(labels):
            s = res[lab]["sigmas"].get(sig)
            if not s:
                continue
            nom = res[lab].get("nominal") or np.median(s["values"])
            data.append(np.array(s["values"]) / nom * 100.0)
            pos.append(i + (j - (len(sigmas) - 1) / 2) * w)
        if not data:
            continue
        bp = ax.boxplot(data, positions=pos, widths=w * 0.8, patch_artist=True,
                        medianprops=dict(color="black", lw=1.4), showfliers=True,
                        flierprops=dict(marker=".", ms=4, mfc="0.5", mec="0.5"))
        for b in bp["boxes"]:
            b.set_facecolor("C0" if j == 0 else "C1")
            b.set_alpha(0.75)
            b.set_edgecolor("white")
    ax.axhline(100, color="0.4", ls="--", lw=1.2)
    ax.annotate("nominal", (0.005, 100), xycoords=("axes fraction", "data"), fontsize=8,
                color="0.4", va="bottom")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$|\theta_{\chi^5}|$ retained (% of nominal)")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.75) for c in ("C0", "C1")]
    ax.legend(handles, [r"$\sigma$ = {} nm".format(s) for s in sigmas], fontsize=8)
    ax.set_title("Fabrication tolerance — independent Gaussian error on every layer,\n"
                 "operating point held at the nominal design (no post-fab re-tuning)",
                 fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


def fig_3d(path):
    p = OUT / "3d_result.json"
    if not p.exists():
        return
    res = json.load(open(p))
    labels = [k for k in res if res[k].get("theta_3d") is not None]
    if not labels:
        return
    t3 = np.array([res[k]["theta_3d"] for k in labels])
    t1 = np.array([res[k].get("theta_1d", np.nan) for k in labels])
    f3 = np.array([res[k].get("fringe_3d", np.nan) for k in labels])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.bar(x - 0.2, t1, 0.38, label="1D", color="C0", edgecolor="white", lw=1.2)
    ax.bar(x + 0.2, t3, 0.38, label="3D", color="C2", edgecolor="white", lw=1.2)
    for xi, (a, b) in enumerate(zip(t1, t3)):
        if np.isfinite(a) and a > 0:
            ax.annotate("{:.1f}$\\times$".format(b / a), (xi + 0.2, b), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"$|\theta_{\chi^5}|$ (deg)")
    ax.set_title("1D vs 3D on the $\\chi^{(5)}$ channel", fontsize=10)
    ax.legend(fontsize=8)
    style(ax)

    ax = axes[1]
    contrast = t3 / np.maximum(f3, 1e-12)
    ax.bar(x, contrast, 0.6, color=np.where(contrast >= 1, "C2", "C1"),
           edgecolor="white", lw=1.2)
    ax.axhline(1.0, color="0.35", ls="--", lw=1.4)
    for xi, c in enumerate(contrast):
        ax.annotate("{:.2f}".format(c), (xi, c), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(r"contrast in 3D = $|\theta_{\chi^5}|$ / fringe")
    ax.set_title("does the effect still stand above the fringe in 3D?", fontsize=10)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    DOCS.mkdir(parents=True, exist_ok=True)
    fig_intensity(DOCS / "s3_intensity.png")
    fig_tolerance(DOCS / "s3_tolerance.png")
    fig_3d(DOCS / "s3_3d.png")


if __name__ == "__main__":
    main()
