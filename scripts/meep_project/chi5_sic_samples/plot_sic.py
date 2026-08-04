#!/usr/bin/env python
"""Figures for the fabricated SiC-cavity samples.

  s0_intensity.png  (a) rotation and carrier fringe vs pump intensity, with the chi5 I^2 law
                    (b) DoLP and the local log-log slope -- the two things that decide where
                        the response is still a meaningful chi5 azimuth, and hence where the
                        reference intensity must sit
  s1_scan.png       (a) best achievable rotation at each probe cavity mode, lab reach marked
                    (b) contrast (effect / fringe) at each probe mode -- the second axis
                    (c) the (pump centre x Delta) map at the winning probe
                    (d) rotation vs FWM energy mismatch

Palette: Okabe-Ito. matplotlib's default tab10 orange/green pair is DeltaE 0.7 under
protanopia -- indistinguishable -- so it is not used for categorical series here.
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
import common_sic as S  # noqa: E402

RUNS = HERE / "runs"
DOCS = HERE / "docs"

# Validated categorical palette (all 6 checks pass, light surface).
CLR = {"L3p2": "#0072B2", "L4p8": "#D55E00", "ref": "#009E73"}
LBL = {"L3p2": "SiC L = 3.2 $\\mu$m", "L4p8": "SiC L = 4.8 $\\mu$m"}
SEQ = "viridis"          # sequential: single-hue-ish, perceptually uniform
I_REF = 1e11             # chosen by s0_intensity.py
SIN_REF_DEG = 0.003519   # fabricated SiN sample, as built, at 1e12 W/cm^2


def style(ax):
    ax.grid(True, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ figure 1 --- #
def fig_intensity():
    p = RUNS / "s0_intensity" / "s0_result.json"
    if not p.exists():
        print("skip s0_intensity: no data")
        return
    d = json.load(open(p))
    # Three panels, one quantity each. DoLP and the log-log slope are different measures on
    # different scales -- putting them on one axis (or worse, two y-axes) would make their
    # crossings meaningless.
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))

    ax = axes[0]
    for k, s in d["samples"].items():
        rows = s["rows"]
        if not rows:
            continue
        I = np.array([r["I"] for r in rows])
        th = np.array([r["theta"] for r in rows])
        fr = np.array([r["fringe"] for r in rows])
        ax.loglog(I, th, "o-", color=CLR[k], ms=6, lw=2.0, label=LBL[k] + "  effect")
        ax.loglog(I, fr, "^:", color=CLR[k], ms=5, lw=1.2, alpha=0.6,
                  label=LBL[k] + "  fringe")
    lo = np.array([2.5e10, 1e12])
    ax.loglog(lo, 3.4e-4 * (lo / 1e11) ** 2, "-", color="0.5", lw=1.0, zorder=1)
    ax.annotate(r"$\propto I^2$  (pure $\chi^{(5)}$)", (3.0e11, 1.7e-3), color="0.4",
                fontsize=9, rotation=0)
    ax.axvline(I_REF, color="0.25", ls="--", lw=1.4)
    ax.annotate("reference\nintensity", (I_REF * 1.12, ax.get_ylim()[0] * 3), fontsize=8.5,
                color="0.25")
    ax.set_xlabel(r"pump intensity (W/cm$^2$)")
    ax.set_ylabel("carrier-averaged amplitude (deg)")
    ax.set_title("(a) both cavities are clean $I^2$ at low intensity\n"
                 "the fringe stays above the effect at this operating point", fontsize=10)
    ax.legend(fontsize=7.6, loc="upper left")
    style(ax)

    # (b) DoLP alone -- is the transmitted probe still linearly polarized?
    ax = axes[1]
    for k, s in d["samples"].items():
        rows = s["rows"]
        if not rows:
            continue
        ax.semilogx([r["I"] for r in rows], [r["dolp"] for r in rows], "o-",
                    color=CLR[k], ms=6, lw=2.0, label=LBL[k])
    ax.axhline(0.99, color="#B22222", lw=1.3, ls="--")
    ax.annotate("DoLP 0.99 — below this the probe is\npartly depolarized and its azimuth\n"
                "is not a clean measurement", (1.1e10, 0.977), fontsize=8, color="#B22222",
                va="top")
    ax.axvline(I_REF, color="0.25", ls="--", lw=1.4)
    ax.annotate("reference\nintensity", (I_REF * 1.15, 0.83), fontsize=8.5, color="0.25")
    ax.set_xlabel(r"pump intensity (W/cm$^2$)")
    ax.set_ylabel("degree of linear polarization")
    ax.set_ylim(0.78, 1.008)
    ax.set_title("(b) the polarization survives only below ~$2\\times10^{11}$\n"
                 "at $10^{12}$ the L=4.8 probe is at DoLP 0.81", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    style(ax)

    # (c) local log-log slope alone -- is it still chi5?
    ax = axes[2]
    for k, s in d["samples"].items():
        rows = s["rows"]
        if len(rows) < 3:
            continue
        Im, sl = [], []
        for i in range(1, len(rows) - 1):
            Im.append(rows[i]["I"])
            sl.append(float(np.log(rows[i + 1]["theta"] / rows[i - 1]["theta"]) /
                            np.log(rows[i + 1]["I"] / rows[i - 1]["I"])))
        ax.semilogx(Im, sl, "o-", color=CLR[k], ms=6, lw=2.0, label=LBL[k])
    ax.axhline(2.0, color="0.45", lw=1.2, ls=":")
    ax.axhspan(1.9, 2.1, color="0.85", alpha=0.6, zorder=0)
    ax.annotate(r"shaded: within 5% of $I^2$ — pure $\chi^{(5)}$", (0.03, 0.97),
                xycoords="axes fraction", fontsize=8.5, color="0.35", va="top")
    ax.axvline(I_REF, color="0.25", ls="--", lw=1.4)
    ax.set_xlabel(r"pump intensity (W/cm$^2$)")
    ax.set_ylabel(r"local slope   $d\ln|\theta| / d\ln I$")
    ax.set_title(r"(c) $\chi^{(5)}$ rolls over into saturation above $\sim2\times10^{11}$",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left")
    style(ax)

    fig.suptitle("Fabricated SiC-cavity samples — choosing the pump intensity "
                 "(1D FDTD, 100 fs, carrier-averaged)", fontsize=12)
    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "s0_intensity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("->", out)


# ------------------------------------------------------------------ figure 2 --- #
def fig_scan():
    data = {}
    for k in S.CAVITY_LENGTHS_UM:
        p = RUNS / "s1_scan" / k / "result.json"
        if p.exists():
            d = json.load(open(p))
            if d.get("rows"):
                data[k] = d
    if not data:
        print("skip s1_scan: no data yet")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))

    def per_probe(d):
        by = {}
        for r in d["rows"]:
            key = round(r["op"]["probe_nm"], 1)
            if key not in by or abs(r["theta_chi5_deg"]) > abs(by[key]["theta_chi5_deg"]):
                by[key] = r
        return by

    # (a) best rotation per probe mode
    ax = axes[0][0]
    for k, d in data.items():
        by = per_probe(d)
        xs = sorted(by)
        ys = [abs(by[x]["theta_chi5_deg"]) for x in xs]
        ax.plot(xs, ys, "o-", color=CLR[k], ms=6, lw=1.8, label=LBL[k])
        reach = [x for x in xs if S.in_windows(x / 1000.0, S.PROBE_WINDOWS_NOW)]
        ax.plot(reach, [abs(by[x]["theta_chi5_deg"]) for x in reach], "o", ms=12,
                mfc="none", mec=CLR[k], mew=2.0)
    for lo, hi in S.PROBE_WINDOWS_NOW:
        ax.axvspan(lo * 1000, hi * 1000, color="0.85", alpha=0.55, zorder=0)
    ax.annotate("shaded = probe reachable today (ringed markers)", (0.02, 0.03),
                xycoords="axes fraction", fontsize=8, color="0.35", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("probe cavity mode (nm)")
    ax.set_ylabel(r"best $|\theta_{\chi^5}|$ (deg)")
    ax.set_title("(a) the probe mode is the dominant knob", fontsize=10)
    ax.legend(fontsize=8)
    style(ax)

    # (b) contrast per probe mode
    ax = axes[0][1]
    for k, d in data.items():
        by = {}
        for r in d["rows"]:
            key = round(r["op"]["probe_nm"], 1)
            c = abs(r["theta_chi5_deg"]) / max(r["theta_fringe_amp_deg"], 1e-12)
            if key not in by or c > by[key]:
                by[key] = c
        xs = sorted(by)
        ax.plot(xs, [by[x] for x in xs], "o-", color=CLR[k], ms=6, lw=1.8, label=LBL[k])
    ax.axhline(1.0, color="#B22222", ls="--", lw=1.5)
    ax.annotate("contrast = 1: the effect exceeds the fringe", (0.03, 0.06),
                xycoords="axes fraction", fontsize=8.5, color="#B22222")
    for lo, hi in S.PROBE_WINDOWS_NOW:
        ax.axvspan(lo * 1000, hi * 1000, color="0.85", alpha=0.55, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("probe cavity mode (nm)")
    ax.set_ylabel(r"best contrast  $|\theta_{\chi^5}|$ / fringe")
    ax.set_title("(b) can the effect be seen without delay dithering?", fontsize=10)
    ax.legend(fontsize=8)
    style(ax)

    # (c) centre x Delta map at the best WELL-SAMPLED probe mode.
    # Not simply the best probe overall: modes near the edge of the 1400-2000 nm pump range
    # lose most of their (centre, Delta) grid to that constraint, so the global-max mode can
    # carry only a couple of points and its "map" would be meaningless.
    ax = axes[1][0]

    def counts(d):
        c = {}
        for r in d["rows"]:
            c[round(r["op"]["probe_nm"], 1)] = c.get(round(r["op"]["probe_nm"], 1), 0) + 1
        return c

    # Prefer the RECOMMENDED operating point -- the frozen "legible" finalist (max signal
    # subject to contrast >= 1). That is the map the lab actually needs to tune against.
    k = bestr = None
    fp = RUNS / "s3_finalists" / "finalists.json"
    if fp.exists():
        fin = [f for f in json.load(open(fp))
               if f["kind"] == "legible" and f["sample"] in data]
        if fin:
            f = max(fin, key=lambda f: abs(f["theta_1d_deg"]))
            k = f["sample"]
            bestr = max((r for r in data[k]["rows"]
                         if abs(r["op"]["probe_nm"] - f["op"]["probe_nm"]) < 0.05),
                        key=lambda r: abs(r["theta_chi5_deg"]), default=None)
    if bestr is None:
        cand = []
        for kk, d in data.items():
            c = counts(d)
            nmax = max(c.values())
            for r in d["rows"]:
                if c[round(r["op"]["probe_nm"], 1)] >= max(6, 0.8 * nmax):
                    cand.append((abs(r["theta_chi5_deg"]), kk, r))
        if not cand:
            cand = [(abs(r["theta_chi5_deg"]), kk, r)
                    for kk, d in data.items() for r in d["rows"]]
        _, k, bestr = max(cand, key=lambda t: t[0])
    d = data[k]
    bp = bestr["op"]["probe_nm"]
    sel = [r for r in d["rows"] if abs(r["op"]["probe_nm"] - bp) < 0.05]
    cen = np.array([r["op"]["center_nm"] for r in sel])
    dlt = np.array([r["op"]["delta"] for r in sel])
    th = np.array([abs(r["theta_chi5_deg"]) for r in sel])
    ct_all = np.array([abs(r["theta_chi5_deg"]) / max(r["theta_fringe_amp_deg"], 1e-12)
                       for r in sel])
    sc = ax.scatter(cen, dlt, c=th, s=190, cmap=SEQ, edgecolor="white", lw=0.8)
    # ring the points where the effect actually beats the fringe -- the usable region
    leg = ct_all >= 1.0
    if leg.any():
        ax.scatter(cen[leg], dlt[leg], s=330, facecolor="none", edgecolor="#B22222", lw=1.8)
    ax.plot([bestr["op"]["center_nm"]], [bestr["op"]["delta"]], "*", ms=24, mfc="none",
            mec="#B22222", mew=2.2, zorder=5)
    ax.annotate("  max $|\\theta|$ here: {:.5f}°".format(abs(bestr["theta_chi5_deg"])),
                (bestr["op"]["center_nm"], bestr["op"]["delta"]), fontsize=8.5, color="#B22222",
                va="center")
    ax.annotate("red rings: contrast $\\geq$ 1 ({} of {} points)".format(int(leg.sum()),
                                                                        len(sel)),
                (0.02, 0.03), xycoords="axes fraction", fontsize=8, color="#B22222")
    fig.colorbar(sc, ax=ax, label=r"$|\theta_{\chi^5}|$ (deg)")
    ax.set_xlabel("pump centre wavelength (nm)")
    ax.set_ylabel(r"pump splitting $\Delta$ (1/$\mu$m)")
    ax.set_title("(c) operating map, {} at probe {:.1f} nm".format(LBL[k], bp), fontsize=10)
    style(ax)

    # (d) the two axes are not the same axis
    ax = axes[1][1]
    for k, d in data.items():
        th = np.array([abs(r["theta_chi5_deg"]) for r in d["rows"]])
        ct = np.array([abs(r["theta_chi5_deg"]) / max(r["theta_fringe_amp_deg"], 1e-12)
                       for r in d["rows"]])
        ax.scatter(th, ct, s=26, color=CLR[k], alpha=0.5, lw=0, label=LBL[k])
        acc = [i for i, r in enumerate(d["rows"])
               if S.in_windows(r["op"]["probe_nm"] / 1000.0, S.PROBE_WINDOWS_NOW)]
        ax.scatter(th[acc], ct[acc], s=64, facecolor="none", edgecolor=CLR[k], lw=1.3)
    ax.axhline(1.0, color="#B22222", ls="--", lw=1.5)
    ax.annotate("contrast = 1", (0.02, 0.955), xycoords="axes fraction", fontsize=8.5,
                color="#B22222")
    ax.annotate("ringed = probe reachable today", (0.02, 0.03), xycoords="axes fraction",
                fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$|\theta_{\chi^5}|$ (deg)   — signal")
    ax.set_ylabel(r"contrast — legibility")
    ax.set_title("(d) signal and legibility are different axes:\n"
                 "the largest rotations sit lowest in contrast", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    style(ax)

    fig.suptitle("Fabricated SiC-cavity samples — operating-point map "
                 r"($I = 10^{11}$ W/cm$^2$, 100 fs, carrier-averaged)", fontsize=12)
    fig.tight_layout()
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "s1_scan.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="both", choices=["intensity", "scan", "both"])
    a = ap.parse_args()
    if a.which in ("intensity", "both"):
        fig_intensity()
    if a.which in ("scan", "both"):
        fig_scan()


if __name__ == "__main__":
    main()
