#!/usr/bin/env python
"""Figures for Stage 5 -- what the EXISTING fabricated sample can give without refabrication.

  s5_existing.png   (a) best rotation achievable at each probe cavity mode -- the dominant knob
                    (b) the (pump centre x Delta) map at the winning probe -- the optimum and
                        how sharply it is peaked, i.e. how precisely the lab must tune
                    (c) rotation vs FWM energy mismatch |2 f_pump - f_probe| -- why the winning
                        point is where it is
                    (d) sensitivity cuts through the optimum in centre and in Delta
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
AS_FAB = 0.003519          # as-fabricated operating point, true 100 fs pulse (Stage 0 Part A)
AS_FAB_PROBE = 800.1


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(HERE / "runs" / "s5_existing" / "s5_result.json"))
    args = ap.parse_args()
    res = json.load(open(args.result))
    rows = res["rows"]
    th = np.array([abs(r["theta_chi5_deg"]) for r in rows])
    probe = np.array([r["op"]["probe_nm"] for r in rows])
    cen = np.array([1000.0 / r["op"]["center"] for r in rows])
    dlt = np.array([r["op"]["delta"] for r in rows])
    mis = np.array([abs(2 * r["op"]["center"] - r["op"]["probe"]) for r in rows])
    best = int(np.argmax(th))
    bp = probe[best]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) best achievable per probe mode ------------------------------------------------- #
    ax = axes[0][0]
    ps = sorted(set(np.round(probe, 1)), reverse=True)
    bestper = [th[np.isclose(probe, p, atol=0.05)].max() for p in ps]
    cols = ["C2" if abs(p - bp) < 0.05 else ("C3" if abs(p - AS_FAB_PROBE) < 0.5 else "C0")
            for p in ps]
    ax.barh(range(len(ps)), bestper, color=cols, edgecolor="white", lw=1.2)
    ax.set_yticks(range(len(ps)))
    ax.set_yticklabels(["{:.1f}".format(p) for p in ps], fontsize=8)
    ax.axvline(AS_FAB, color="C3", ls="--", lw=1.4)
    ax.annotate("as-fabricated\noperating point", (AS_FAB, len(ps) - 0.5), fontsize=8,
                color="C3", ha="left", va="top")
    for i, v in enumerate(bestper):
        ax.annotate("  {:.4f}°  ({:.1f}×)".format(v, v / AS_FAB), (v, i), va="center",
                    fontsize=7.5, color="0.25")
    ax.set_xlabel(r"best achievable $|\theta_{\chi^5}|$ (deg)")
    ax.set_ylabel("probe cavity mode (nm)")
    ax.set_xlim(0, max(bestper) * 1.45)
    ax.set_title("(a) the probe mode is the dominant knob\n"
                 "green = optimum, red = currently used", fontsize=10)
    style(ax)

    # (b) centre x Delta map at the winning probe ---------------------------------------- #
    ax = axes[0][1]
    m = np.isclose(probe, bp, atol=0.05)
    sc = ax.scatter(cen[m], dlt[m], c=th[m], s=210, cmap="viridis", edgecolor="white", lw=0.8)
    ax.plot([cen[best]], [dlt[best]], "*", ms=26, mfc="none", mec="crimson", mew=2.2, zorder=5)
    ax.annotate("  optimum {:.4f}°".format(th[best]), (cen[best], dlt[best]), fontsize=9,
                color="crimson", va="center")
    ax.set_xlabel("pump centre wavelength (nm)")
    ax.set_ylabel(r"pump splitting $\Delta$ (1/$\mu$m)")
    ax.set_title(r"(b) operating map at probe {:.1f} nm".format(bp), fontsize=10)
    fig.colorbar(sc, ax=ax, label=r"$|\theta_{\chi^5}|$ (deg)")
    style(ax)

    # (c) FWM energy matching ------------------------------------------------------------ #
    ax = axes[1][0]
    for p, col in ((bp, "C2"), (AS_FAB_PROBE, "C3")):
        k = np.isclose(probe, p, atol=0.05)
        ax.scatter(mis[k], th[k], s=70, color=col, edgecolor="white", lw=0.7,
                   label="probe {:.1f} nm".format(p), zorder=3)
    other = ~(np.isclose(probe, bp, atol=0.05) | np.isclose(probe, AS_FAB_PROBE, atol=0.5))
    ax.scatter(mis[other], th[other], s=26, color="0.7", lw=0, label="other probe modes",
               zorder=2)
    mis_fab = abs(2 * (1 / 1.5215 + 1 / 1.5740) / 2 - 1 / 0.8001)
    ax.axvline(mis_fab, color="C3", ls="--", lw=1.3)
    ax.annotate("as-fabricated\nmismatch", (mis_fab, ax.get_ylim()[1] * 0.92), fontsize=8,
                color="C3", ha="left", va="top")
    ax.set_xlabel(r"FWM energy mismatch  $|2f_{pump} - f_{probe}|$  (1/$\mu$m)")
    ax.set_ylabel(r"$|\theta_{\chi^5}|$ (deg)")
    ax.set_title("(c) the winning point is near-perfectly octave-matched\n"
                 r"(at fixed probe, $\rho$ = −0.70 at the optimum mode)", fontsize=10)
    ax.legend(fontsize=8)
    style(ax)

    # (d) sensitivity cuts through the optimum ------------------------------------------- #
    # Both knobs on ONE axis, as fractional detuning from their optimum values -- a dual x-axis
    # would make the two curves' crossings meaningless.
    ax = axes[1][1]
    cut_d = m & np.isclose(dlt, dlt[best])
    o = np.argsort(cen[cut_d])
    ax.plot((cen[cut_d][o] / cen[best] - 1) * 100, th[cut_d][o] / th[best] * 100,
            "o-", color="C0", ms=6,
            label="pump centre  (optimum {:.1f} nm)".format(cen[best]))
    cut_c = m & np.isclose(cen, cen[best])
    o2 = np.argsort(dlt[cut_c])
    ax.plot((dlt[cut_c][o2] / dlt[best] - 1) * 100, th[cut_c][o2] / th[best] * 100,
            "s--", color="C1", ms=6,
            label=r"pump splitting $\Delta$  (optimum {:.4f})".format(dlt[best]))
    # the probe, for scale: best per probe mode vs its detuning from the winning mode
    pv = np.array(sorted(set(np.round(probe, 1))))
    bv = np.array([th[np.isclose(probe, p, atol=0.05)].max() for p in pv])
    ax.plot((pv / bp - 1) * 100, bv / th[best] * 100, "^:", color="C2", ms=6,
            label="probe mode  (optimum {:.1f} nm)".format(bp))
    ax.axhline(100, color="0.6", lw=0.8, ls=":")
    ax.axhline(50, color="0.8", lw=0.8, ls=":")
    ax.axvline(0, color="0.8", lw=0.8, ls=":")
    ax.set_xlabel("detuning from the optimum (%)")
    ax.set_ylabel("% of the optimum rotation")
    ax.set_xlim(-8, 8)
    ax.set_title("(d) how precisely must the lab tune?\n"
                 "the pump knobs are forgiving; the probe is not", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower center")
    style(ax)

    fig.suptitle("Stage 5 — the EXISTING fabricated sample, retuned only "
                 r"(geometry frozen, $I=10^{12}$ W/cm$^2$, 100 fs)", fontsize=12)
    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "s5_existing.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


if __name__ == "__main__":
    main()
