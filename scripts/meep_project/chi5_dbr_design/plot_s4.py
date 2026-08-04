#!/usr/bin/env python
"""Figure for Stage 4 -- the trace the lab would actually record.

  s4_trace.png   V-H vs pump1 delay for each design, in the two readouts:
                   * carrier-averaged  = a delay line that is NOT phase stable, or is
                     deliberately dithered by >= 5.1 fs. This is the rectified chi5 envelope.
                   * single phase      = a phase-stable line, one run per delay. This is
                     fringe + envelope, and is what the current experiment appears to record.
                 The shaded band is the peak-to-peak swing the fringe would produce over one
                 pump1 optical period at that delay -- i.e. how much the trace moves if the
                 delay drifts by a fraction of a wavelength.
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
DOCS = HERE / "docs"


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(HERE / "runs" / "s4_delay" / "s4_result.json"))
    args = ap.parse_args()
    res = json.load(open(args.result))
    designs = res["designs"]
    if not designs:
        raise SystemExit("no completed designs in the result file")

    order = [k for k in ("cand16", "cand13", "baseline") if k in designs]
    order += [k for k in designs if k not in order]
    fig, axes = plt.subplots(1, len(order), figsize=(6.2 * len(order), 5.0), squeeze=False)

    for ax, label in zip(axes[0], order):
        d = designs[label]
        rows = sorted(d["rows"], key=lambda r: r["tau_fs"])
        tau = np.array([r["tau_fs"] for r in rows])
        env = np.array([r["theta_avg_deg"] for r in rows])
        single = np.array([r["theta_single_deg"] for r in rows])
        fr = np.array([r["fringe_amp_deg"] for r in rows])

        ax.fill_between(tau, env - fr / 2, env + fr / 2, color="C1", alpha=0.22, lw=0,
                        label="fringe swing over one $T_1$")
        ax.plot(tau, single, "o-", color="C1", ms=4, lw=1.2, alpha=0.9,
                label="phase-stable line (1 run/delay)")
        ax.plot(tau, env, "o-", color="C0", ms=5, lw=2.0,
                label=r"carrier-averaged = the $\chi^{(5)}$ effect")
        ax.axhline(0, color="0.6", lw=0.8, ls=":")

        # Contrast must be read where the effect EXISTS. Averaging the ratio over all delays is
        # meaningless: in the wings the pulses no longer overlap, the envelope goes to zero by
        # construction, and effect/fringe -> 0 there regardless of the design. Quote the core
        # (|tau| <= 50 fs, i.e. within the pulse overlap).
        core = np.abs(tau) <= 50.0
        peak = np.max(np.abs(env))
        contrast = float(np.median(np.abs(env[core]) / np.maximum(fr[core], 1e-12)))
        # envelope width, for comparison with the ~100 fs pulse
        a = np.abs(env)
        idx = np.where(a >= a.max() / 2)[0]
        fwhm = tau[idx[-1]] - tau[idx[0]] if len(idx) > 1 else float("nan")
        verdict = ("effect DOMINATES the trace" if contrast > 1 else
                   "effect hidden under the fringe")
        ax.set_title("{} — {}\npeak envelope {:.4f}$\\degree$, FWHM {:.0f} fs | "
                     "effect/fringe at overlap = {:.2f}".format(label, verdict, peak, fwhm,
                                                                contrast),
                     fontsize=9.5,
                     color=("C2" if contrast > 1 else "0.15"))
        ax.set_xlabel(r"pump1 delay $\tau$ (fs)")
        ax.set_ylabel(r"probe rotation $\theta$ (deg)")
        ax.legend(fontsize=7.5, loc="best")
        style(ax)

    fig.suptitle("Stage 4 — what the balanced detector would record vs pump1 delay "
                 r"($I=10^{12}$ W/cm$^2$, 100 fs)", fontsize=12)
    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "s4_trace.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


if __name__ == "__main__":
    main()
