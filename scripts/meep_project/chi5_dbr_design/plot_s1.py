#!/usr/bin/env python
"""Figures for Stage 1 (s1_screen.py) -- what the fabricable design space looks like.

  s1_space.png    where feasible cavities live, the fab boundary, and where the Stage-2 pool sits

The proxy FoM is shown as a colour only, never as a ranking claim: it selects the pool, and
1D FDTD in Stage 2 does the ranking. See s1_screen.py's header for why.
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
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parent
DOCS = HERE / "docs"


def style(ax):
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=str(HERE / "runs" / "s1_screen" / "s1_result.json"))
    args = ap.parse_args()
    res = json.load(open(args.result))
    # the full feasible set lives in a compressed side file (too big to track as JSON)
    if "all_feasible" in res:
        feas = res["all_feasible"]
    else:
        sys.path.insert(0, str(HERE))
        import s1_screen
        feas = s1_screen.load_feasible(Path(args.result).parent / "s1_feasible.npz")
    top = res["top"]
    base = res.get("baseline") or {}
    cfg = res["config"]
    DOCS.mkdir(parents=True, exist_ok=True)

    t_hi = np.array([r["params"]["t_hi"] for r in feas])
    t_lo = np.array([r["params"]["t_lo"] for r in feas])
    L = np.array([r["params"]["L_cav"] for r in feas])
    stack = np.array([r["params"]["stack_um"] for r in feas])
    fom = np.array([max(r["fom"], 1e-12) for r in feas])
    probe = np.array([r["probe_nm"] for r in feas])

    def pool_xy(kx, ky):
        gx = {"t_hi": lambda r: r["params"]["t_hi"], "t_lo": lambda r: r["params"]["t_lo"],
              "L_cav": lambda r: r["params"]["L_cav"],
              "stack_um": lambda r: r["params"]["stack_um"],
              "probe_nm": lambda r: r["probe_nm"]}
        return ([gx[kx](r) for r in top], [gx[ky](r) for r in top],
                [r.get("pool", "proxy") for r in top])

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))

    # (a) mirror-layer plane -------------------------------------------------------------- #
    ax = axes[0][0]
    sc = ax.scatter(t_hi * 1000, t_lo * 1000, c=fom, s=7, norm=LogNorm(), cmap="viridis",
                    alpha=0.55, linewidths=0)
    px, py, pt = pool_xy("t_hi", "t_lo")
    for tag, mk, lab in (("proxy", "o", "pool: proxy-ranked"), ("diverse", "^", "pool: diversity")):
        m = [i for i, t in enumerate(pt) if t == tag]
        if m:
            ax.scatter(np.array(px)[m] * 1000, np.array(py)[m] * 1000, marker=mk, s=110,
                       facecolor="none", edgecolor="crimson", linewidth=1.6, label=lab, zorder=4)
    if base.get("params"):
        ax.scatter([base["params"]["t_hi"] * 1000], [base["params"]["t_lo"] * 1000], marker="*",
                   s=320, color="white", edgecolor="black", linewidth=1.2, zorder=5,
                   label="best_absolute (fabricated)")
    lim = [1000 * min(t_hi.min(), t_lo.min()), 1000 * max(t_hi.max(), t_lo.max())]
    ax.plot(lim, lim, ls=":", color="0.5", lw=1, zorder=1)
    ax.annotate("equal thickness", (lim[1] * 0.93, lim[1] * 0.93), fontsize=7, color="0.45",
                rotation=45, ha="center", va="bottom")
    ax.annotate("diagonal bands = loci where a cavity mode\nlands in the probe window",
                (0.03, 0.06), xycoords="axes fraction", fontsize=7.5, color="0.3",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.85))
    ax.set_xlabel("SiN layer $t_{hi}$ (nm)")
    ax.set_ylabel("SiO$_2$ layer $t_{lo}$ (nm)")
    ax.set_title("(a) mirror layer thicknesses of feasible cavities", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    fig.colorbar(sc, ax=ax, label="analytic v3 FoM (pool selector, NOT a ranking)")
    style(ax)

    # (b) cavity length vs total stack, with the fab cap ---------------------------------- #
    ax = axes[0][1]
    sc = ax.scatter(L, stack, c=fom, s=7, norm=LogNorm(), cmap="viridis", alpha=0.55,
                    linewidths=0)
    px, py, pt = pool_xy("L_cav", "stack_um")
    for tag, mk in (("proxy", "o"), ("diverse", "^")):
        m = [i for i, t in enumerate(pt) if t == tag]
        if m:
            ax.scatter(np.array(px)[m], np.array(py)[m], marker=mk, s=110, facecolor="none",
                       edgecolor="crimson", linewidth=1.6, zorder=4)
    if base.get("params"):
        ax.scatter([base["params"]["L_cav"]], [base["params"]["stack_um"]], marker="*", s=320,
                   color="white", edgecolor="black", linewidth=1.2, zorder=5)
    cap = cfg["fab"]["stack_max_um"]
    ax.axhline(cap, color="crimson", ls="--", lw=1.4)
    ax.annotate("fab cap {:.0f} $\\mu$m — BINDING:\nmax|$\\theta$| $\\propto L^{{+1.2}}$ wants a "
                "long cavity,\nmore mirror pairs want thickness".format(cap),
                (0.03, 0.06), xycoords="axes fraction", va="bottom", fontsize=7.5,
                color="crimson",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="crimson", alpha=0.85))
    ax.set_xlabel(r"cavity length $L$ ($\mu$m)")
    ax.set_ylabel(r"total deposited stack ($\mu$m)")
    ax.set_title("(b) the fabrication budget is the real constraint", fontsize=10)
    fig.colorbar(sc, ax=ax, label="analytic v3 FoM")
    style(ax)

    # (c) feasibility funnel -------------------------------------------------------------- #
    ax = axes[1][0]
    counts = res["counts"]
    order = sorted(counts.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in order]
    vals = [v for _, v in order]
    cols = ["C2" if n == "ok" else "C3" if n == "fab" else "C1" for n in names]
    bars = ax.barh(range(len(names))[::-1], vals, color=cols, edgecolor="white", linewidth=1.2)
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names, fontsize=9)
    tot = sum(vals)
    for b, v in zip(bars, vals):
        ax.annotate(" {} ({:.1f}%)".format(v, 100 * v / tot),
                    (v, b.get_y() + b.get_height() / 2), va="center", fontsize=8)
    ax.set_xlabel("candidates")
    ax.set_title("(c) screen outcome, {} Sobol candidates".format(cfg["n_samples"]), fontsize=10)
    ax.set_xlim(0, max(vals) * 1.25)
    style(ax)

    # (d) probe wavelength achieved vs the allowed windows -------------------------------- #
    ax = axes[1][1]
    ax.hist(probe, bins=60, color="C0", alpha=0.75, edgecolor="white", linewidth=0.5)
    for lo, hi in [(790, 810), (850, 950)]:
        ax.axvspan(lo, hi, color="C2", alpha=0.13, zorder=0)
    ax.annotate("allowed probe windows\n(lab-tunable)", (0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="C2")
    px, _, pt = pool_xy("probe_nm", "probe_nm")
    for x in px:
        ax.axvline(x, color="crimson", lw=0.9, alpha=0.8, zorder=3)
    ax.set_xlabel("probe cavity-mode wavelength (nm)")
    ax.set_ylabel("feasible cavities")
    ax.set_title("(d) where the probe mode lands (red = Stage-2 pool)", fontsize=10)
    style(ax)

    fig.suptitle("Stage 1 — fabricable design space (SiN/SiO$_2$, $\\leq$6 pairs/side, "
                 "stack $\\leq$ {:.0f} $\\mu$m, layers $\\geq$ {:.0f} nm)".format(
                     cap, cfg["fab"]["t_layer_min"] * 1000), fontsize=12)
    fig.tight_layout()
    path = DOCS / "s1_space.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("->", path)


if __name__ == "__main__":
    main()
