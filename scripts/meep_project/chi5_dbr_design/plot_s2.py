#!/usr/bin/env python
"""Figures for Stage 2 (s2_fdtd.py) -- the FDTD ranking on the physical objective.

  s2_ranking.png    every geometry's best carrier-averaged |theta_chi5|, vs the fabricated
                    baseline, with the fringe shown alongside so the two channels stay distinct
  s2_trends.png     what FDTD actually rewards (cavity length, pair count, mirror detuning,
                    Delta) + whether the analytic pre-rank had any skill
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

DOCS = HERE / "docs"
POOL_COLOR = {"proxy": "C0", "diverse": "C2", "control": "C3"}


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_ranking(res, path):
    rank = res["ranking"]
    labels = [r["label"] for r in rank]
    th = np.array([abs(r["best"]["theta_chi5_deg"]) for r in rank])
    fr = np.array([r["best"]["theta_fringe_amp_deg"] for r in rank])
    dolp = np.array([r["best"]["dolp"] for r in rank])
    pools = [r["pool"] for r in rank]
    base = next((abs(r["best"]["theta_chi5_deg"]) for r in rank if r["label"] == "baseline"), None)

    y = np.arange(len(rank))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(4.5, 0.42 * len(rank) + 2.2)),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    ax = axes[0]
    ax.barh(y, th, color=[POOL_COLOR.get(p, "C7") for p in pools],
            edgecolor="white", linewidth=1.2, height=0.72, zorder=3)
    if base:
        ax.axvline(base, color="C3", ls="--", lw=1.5, zorder=4)
        ax.annotate("fabricated baseline\n(best_absolute)", (base, y[0] + 0.6),
                    fontsize=8, color="C3", ha="left", va="bottom")
    for yi, (t, d, lab) in enumerate(zip(th, dolp, labels)):
        rel = "  {:.2f}x".format(t / base) if base else ""
        warn = "  DoLP {:.2f} !".format(d) if d < res["config"]["dolp_min"] else ""
        ax.annotate("{:.5f}{}{}".format(t, rel, warn), (t, y[yi]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5,
                    color="C3" if warn else "0.25")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"carrier-averaged, pulse-integrated  $|\theta_{\chi^5}|$  (deg)")
    ax.set_xlim(0, th.max() * 1.35 if len(th) else 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in
               (POOL_COLOR["proxy"], POOL_COLOR["diverse"], POOL_COLOR["control"])]
    ax.legend(handles, ["proxy-ranked pool", "diversity pool", "fabricated control"], fontsize=8)
    ax.set_title("THE OBJECTIVE — the physical, measurable rotation", fontsize=10)
    style(ax)

    # Contrast: is the effect above or below the coherent fringe it has to be extracted from?
    # This is the axis that decides whether the lab can SEE the effect, and it ranks the
    # designs almost oppositely to raw |theta|.
    ax = axes[1]
    contrast = th / np.maximum(fr, 1e-12)
    ax.barh(y, contrast, color=np.where(contrast >= 1.0, "C2", "C1"),
            edgecolor="white", linewidth=1.2, height=0.72, zorder=3)
    ax.axvline(1.0, color="0.35", ls="--", lw=1.4, zorder=4)
    ax.annotate("effect = fringe", (1.0, y[0] + 0.6), rotation=90, fontsize=7.5,
                color="0.35", ha="right", va="top")
    for yi, (c, f) in enumerate(zip(contrast, fr)):
        ax.annotate("  {:.2f}   (fringe {:.3f}$\\degree$)".format(c, f), (c, y[yi]),
                    xytext=(3, 0), textcoords="offset points", va="center", fontsize=7,
                    color="0.25")
    ax.set_yticks(y)
    ax.set_yticklabels([])
    ax.set_xscale("log")
    ax.set_xlabel(r"contrast = $|\theta_{\chi^5}|$ / carrier-fringe amplitude")
    ax.set_xlim(min(contrast.min() * 0.6, 0.05), max(contrast.max() * 6, 3))
    ax.set_title("CAN IT BE SEEN?\ngreen = effect exceeds the fringe it hides under",
                 fontsize=10)
    style(ax)

    fig.suptitle("Stage 2 — 1D FDTD ranking (res {}, decay {}, "
                 "$I=10^{{12}}$ W/cm$^2$, 100 fs)".format(
                     res["config"]["res"], res["config"]["decay"]), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


def fig_trends(res, path, s1_path=None):
    rank = res["ranking"]
    th = np.array([abs(r["best"]["theta_chi5_deg"]) for r in rank])
    L = np.array([r["params"]["L_cav"] for r in rank])
    npair = np.array([r["params"]["n_left"] + r["params"]["n_right"] for r in rank])
    ratio = np.array([r["params"]["t_lo"] / r["params"]["t_hi"] for r in rank])
    delta = np.array([r["best"]["op"]["delta"] for r in rank])
    pools = [r["pool"] for r in rank]
    cols = [POOL_COLOR.get(p, "C7") for p in pools]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    for ax, x, xl, ttl in [
            (axes[0][0], L, r"cavity length $L$ ($\mu$m)",
             r"(a) cavity length — 1D FDTD previously gave max$|\theta|\propto L^{+1.2}$"),
            (axes[0][1], npair, "total mirror pairs (left + right)",
             "(b) mirror pairs"),
            (axes[1][0], ratio, r"mirror detuning  $t_{lo}/t_{hi}$",
             r"(c) mirror detuning (best_absolute $\approx$ 1.45)"),
            (axes[1][1], delta, r"$\Delta$ at the best operating point (1/$\mu$m)",
             r"(d) pump splitting $\Delta$")]:
        ax.scatter(x, th, c=cols, s=90, edgecolor="white", linewidth=1.0, zorder=3)
        for xi, ti, r in zip(x, th, rank):
            if r["label"] == "baseline":
                ax.annotate("baseline", (xi, ti), xytext=(6, 4), textcoords="offset points",
                            fontsize=7.5, color="C3")
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$|\theta_{\chi^5}|$ (deg)")
        ax.set_title(ttl, fontsize=9.5)
        style(ax)
    axes[1][1].axvline(C.DELTA_MAX_INBAND, color="0.5", ls=":", lw=1.2)
    axes[1][1].annotate("readout band limit", (C.DELTA_MAX_INBAND, axes[1][1].get_ylim()[1]),
                        rotation=90, fontsize=7, ha="right", va="top", color="0.4")

    fig.suptitle("Stage 2 — what the FDTD objective actually rewards", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)

    # proxy-skill figure, only if Stage 1 is available
    if not s1_path or not Path(s1_path).exists():
        return
    s1 = json.load(open(s1_path))
    fom = {"cand{:02d}".format(i): r["fom"] for i, r in enumerate(s1["top"])}
    pairs = [(fom[r["label"]], abs(r["best"]["theta_chi5_deg"]), r["pool"])
             for r in rank if r["label"] in fom]
    if len(pairs) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for tag in ("proxy", "diverse"):
        m = [(a, b) for a, b, t in pairs if t == tag]
        if m:
            ax.scatter([a for a, _ in m], [b for _, b in m], s=95, color=POOL_COLOR[tag],
                       edgecolor="white", linewidth=1.0, label="{} pool".format(tag), zorder=3)
    rho = C.spearman([a for a, _, _ in pairs], [b for _, b, _ in pairs])
    ax.set_xscale("log")
    ax.set_xlabel("analytic v3 FoM (Stage 1 pre-rank)")
    ax.set_ylabel(r"1D FDTD $|\theta_{\chi^5}|$ (deg)")
    ax.set_title("Does the analytic proxy have any skill?\n"
                 r"Spearman $\rho$ = {:+.3f}  (n = {})".format(rho, len(pairs)), fontsize=10)
    ax.legend(fontsize=8)
    style(ax)
    fig.tight_layout()
    p2 = Path(path).parent / "s2_proxy_skill.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print("->", p2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(HERE / "runs" / "s2_fdtd" / "s2_result.json"))
    ap.add_argument("--s1", default=str(HERE / "runs" / "s1_screen" / "s1_result.json"))
    args = ap.parse_args()
    res = json.load(open(args.result))
    DOCS.mkdir(parents=True, exist_ok=True)
    fig_ranking(res, DOCS / "s2_ranking.png")
    fig_trends(res, DOCS / "s2_trends.png", args.s1)


if __name__ == "__main__":
    main()
