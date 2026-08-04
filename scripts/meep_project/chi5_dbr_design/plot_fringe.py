#!/usr/bin/env python
"""Figure for docs/fringe_vs_effect.md -- the four measurements that establish the
fringe/effect distinction.

  fringe_vs_effect.png
    (a) the carrier fringe itself: the 4 sub-samples of the FABRICATED sample at fixed
        envelope delay, the exact fundamental reconstructed from them, and the mean that
        survives averaging.  This is the whole method in one panel.
    (b) Stage 4 delay traces: effect and fringe share the same ~100 fs envelope and are
        separable only by the carrier phase.
    (c) Stage 3 intensity sweep: the effect is I^2 (homodyne, |A_sb|^2), the fringe is
        ~I^1.2-1.7 (heterodyne, A_sb.A_s*), so contrast grows with intensity.
    (d) Stage 2, 21 designs: contrast is an axis ORTHOGONAL to raw rotation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DOCS = HERE / "docs"


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fundamental(y):
    """Exact discrete projection onto the k=1 harmonic: 2|sum y_j exp(-i phi_j)|/N."""
    y = np.asarray(y, dtype=float)
    n = y.size
    phi = 2.0 * np.pi * np.arange(n) / n
    z = np.sum(y * np.exp(-1j * phi))
    return 2.0 * np.abs(z) / n, np.angle(z)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))

    # -- (a) the fringe, resolved -------------------------------------------------------- #
    ax = axes[0][0]
    s0 = json.load(open(RUNS / "s0_harness" / "s0_result.json"))
    sub = np.array(s0["part_a"]["true100"]["theta_sub_deg"], dtype=float)
    mean = float(np.mean(sub))
    amp, psi = fundamental(sub)
    n = sub.size
    phi = 2.0 * np.pi * np.arange(n) / n
    T1 = 1.0 / (s0["config"]["base_freqs"]["pump1"] * 0.299792458)   # fs

    # exact discrete harmonics: with N=4 real samples we determine c0, c1 (complex) and c2
    # (real) exactly -- 1 + 2 + 1 = 4 degrees of freedom.
    c1 = np.sum(sub * np.exp(-1j * phi)) / n
    c2 = float(np.real(np.sum(sub * np.exp(-2j * phi)) / n))
    pg = np.linspace(0, 2 * np.pi, 400)
    ax.plot(pg / (2 * np.pi) * T1, mean + amp * np.cos(pg + psi), "-", color="C3", lw=2.0,
            label=r"$k{=}1$ fundamental, $A_1=%.4f°$" % amp)
    ax.plot(pg / (2 * np.pi) * T1,
            mean + amp * np.cos(pg + psi) + c2 * np.cos(2 * pg), "-", color="0.45", lw=1.1,
            label=r"$+\,k{=}2$ ($%.0f\%%$ of $A_1$) — exact through all 4 points"
                  % (100 * abs(c2) / amp))
    ax.plot(phi / (2 * np.pi) * T1, sub, "o", ms=11, color="C0", mec="white", mew=1.4,
            zorder=5, label="the 4 sub-samples actually simulated")
    ax.axhline(mean, color="C2", lw=2.2, ls="--",
               label=r"mean ($k{=}0$) = $\theta_{\chi^5}$ = %.5f°" % mean)
    ax.axhline(0, color="0.75", lw=0.8)
    ax.annotate("", xy=(T1 * 0.5, mean), xytext=(T1 * 0.5, sub[0]),
                arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.2))
    ax.annotate("  a single-phase\n  measurement is\n  {:.0f}× the effect".format(abs(sub[0] / mean)),
                (T1 * 0.5, (mean + sub[0]) / 2), fontsize=8.5, color="0.3", va="center")
    ax.set_xlabel(r"pump-1 delay within one optical period  $\tau$ (fs),  $T_1$ = %.3f fs" % T1)
    ax.set_ylabel(r"$\theta$ (deg)")
    ax.set_title("(a) the carrier fringe, resolved\n"
                 "fabricated sample, $I=10^{12}$ W/cm$^2$, envelope delay fixed at 0",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    style(ax)

    # -- (b) Stage 4 delay traces -------------------------------------------------------- #
    ax = axes[0][1]
    s4 = json.load(open(RUNS / "s4_delay" / "s4_result.json"))
    for name, col in (("cand16", "C2"), ("cand13", "C0"), ("baseline", "C3")):
        rows = s4["designs"][name]["rows"]
        tau = np.array([r["tau_fs"] for r in rows])
        o = np.argsort(tau)
        tau = tau[o]
        th = np.abs([r["theta_avg_deg"] for r in rows])[o]
        fr = np.abs([r["fringe_amp_deg"] for r in rows])[o]
        ax.plot(tau, th, "o-", color=col, ms=4.5, lw=1.8, label="%s  effect" % name)
        ax.plot(tau, fr, "^:", color=col, ms=4.5, lw=1.2, alpha=0.65,
                label="%s  fringe" % name)
    ax.set_yscale("log")
    ax.set_ylim(4e-4, 4e-1)
    ax.axhspan(4e-4, 5e-3, color="0.85", alpha=0.5, zorder=0)
    ax.annotate("shaded: $N{=}4$ residual ($k{=}4$ leakage) + numerics.\n"
                "The wings are floor, not signal — which is why the\n"
                "effect is read from an ENVELOPE FIT, not one point.",
                (-300, 3.4e-1), fontsize=7.5, color="0.35", va="top")
    ax.set_xlabel(r"envelope delay $\tau$ (fs)")
    ax.set_ylabel("amplitude (deg)")
    ax.set_title("(b) both share the same ~100 fs envelope\n"
                 "only the carrier phase separates them", fontsize=10)
    ax.legend(fontsize=7.2, ncol=3, loc="lower center")
    style(ax)

    # -- (c) intensity scaling ----------------------------------------------------------- #
    ax = axes[1][0]
    s3 = json.load(open(RUNS / "s3_validate" / "intensity_result.json"))
    for name, col in (("cand16", "C2"), ("cand13", "C0"), ("baseline", "C3")):
        r = s3[name]["rows"]
        I = np.array([x["I"] for x in r])
        th = np.abs([x["theta"] for x in r])
        fr = np.abs([x["fringe"] for x in r])
        pe = np.polyfit(np.log(I), np.log(th), 1)[0]
        pf = np.polyfit(np.log(I), np.log(fr), 1)[0]
        ax.loglog(I, th, "o-", color=col, ms=6, lw=1.9,
                  label=r"%s effect   $p=%.2f$" % (name, pe))
        ax.loglog(I, fr, "^:", color=col, ms=6, lw=1.2, alpha=0.65,
                  label=r"%s fringe   $p=%.2f$" % (name, pf))
    I = np.array([2.5e11, 4e12])
    ax.loglog(I, 3e-3 * (I / 2.5e11) ** 2, "-", color="0.45", lw=1.0)
    ax.annotate(r"$\propto I^2$", (1.2e12, 4e-2), color="0.45", fontsize=9)
    ax.loglog(I, 8e-3 * (I / 2.5e11) ** 1, "-", color="0.65", lw=1.0)
    ax.annotate(r"$\propto I^1$", (1.6e12, 3.5e-2), color="0.65", fontsize=9)
    ax.set_xlabel(r"pump intensity (W/cm$^2$)")
    ax.set_ylabel("amplitude (deg)")
    ax.set_title(r"(c) effect $\propto I^2$ (homodyne $|A_{sb}|^2$),"
                 "\n" r"fringe $\propto I^{1.2-1.7}$ (heterodyne $A_{sb}A_s^*$)", fontsize=10)
    ax.legend(fontsize=7.2, ncol=1, loc="upper left")
    style(ax)

    # -- (d) contrast is an orthogonal axis ---------------------------------------------- #
    ax = axes[1][1]
    s2 = json.load(open(RUNS / "s2_fdtd" / "s2_result.json"))
    R = s2["ranking"]
    th = np.array([abs(r["best"]["theta_chi5_deg"]) for r in R])
    ct = np.array([abs(r["best"]["theta_chi5_deg"])
                   / max(r["best"]["theta_fringe_amp_deg"], 1e-12) for r in R])
    lab = [r["label"] for r in R]
    hi = {"cand13": "C0", "cand16": "C2", "baseline": "C3", "cand07": "C1"}
    for i, L in enumerate(lab):
        c = hi.get(L, "0.65")
        ax.loglog(th[i], ct[i], "o", ms=12 if L in hi else 7, color=c,
                  mec="white", mew=1.2, zorder=4 if L in hi else 2)
        if L in hi:
            ax.annotate("  " + L, (th[i], ct[i]), fontsize=9, color=c, va="center")
    ax.axhline(1.0, color="crimson", ls="--", lw=1.5)
    ax.annotate("contrast = 1: the effect exceeds the fringe", (th.min() * 1.05, 1.09),
                fontsize=8.5, color="crimson")

    def spearman(x, y):
        def rank(v):
            o = np.argsort(v)
            r = np.empty_like(o, dtype=float)
            r[o] = np.arange(len(v))
            return r
        a, b = rank(x), rank(y)
        return float(np.corrcoef(a, b)[0, 1])

    ax.set_xlabel(r"$|\theta_{\chi^5}|$ (deg)   — what the old objective maximized")
    ax.set_ylabel(r"contrast  $|\theta_{\chi^5}|$ / fringe amplitude")
    ax.set_title("(d) 21 designs: contrast is a SEPARATE axis\n"
                 r"Spearman$(\theta$, contrast$) = %.3f$" % spearman(th, ct), fontsize=10)
    style(ax)

    fig.suptitle("The carrier fringe vs the χ⁵ effect — "
                 "1D FDTD, SiN/SiO₂ DBR, 100 fs pulses", fontsize=13)
    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    p = DOCS / "fringe_vs_effect.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("->", p)


if __name__ == "__main__":
    main()
