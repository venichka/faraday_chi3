#!/usr/bin/env python
"""Figures for the physically-consistent delay study (delay_physics.py).

Produces, from delay_physics_result.json:
  delay_effect.png            rotation vs pump1 delay -- absolute and relative to tau=0,
                              for BOTH estimators (legacy settled + pulse-integrated)
  ellipticity_effect.png      the same for probe ellipticity chi0, absolute and relative
  delay_and_ellipticity.png   both effects on one plot

All rotations are in degrees. `legacy` is the tail/final-window azimuth (the estimator behind
the published 0.137 deg); `pulse` is the pulse-energy-integrated Stokes readout (the balanced-
detector observable). Every point is carrier-averaged over one pump1 optical period; the
shaded bands show the peak-to-peak spread across that period, i.e. how much a phase-stable
(fringe-resolving) measurement would swing at that delay.
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
EL_COLORS = {"el0": "C0", "el5": "C1", "el10": "C2", "el20": "C3"}
EL_LABEL = {"el0": r"$\chi_0=0^\circ$ (linear)", "el5": r"$\chi_0=5^\circ$",
            "el10": r"$\chi_0=10^\circ$", "el20": r"$\chi_0=20^\circ$"}
# pump-band cavity mode spacings of this geometry (fs) -- the expected delay periodicities
MODE_BEATS = [152.8, 234.6, 92.6]


def arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def at_zero(rows, key):
    t = arr(rows, "tau_fs")
    return float(arr(rows, key)[int(np.argmin(np.abs(t)))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=None)
    args = ap.parse_args()
    root = Path(args.result).parent if args.result else HERE / "delay_physics"
    res = json.load(open(args.result or (root / "delay_physics_result.json")))
    fams = res["families"]
    clean = fams.get("el0") or []
    if not clean:
        raise SystemExit("no chi0=0 data in the result file")

    tau = arr(clean, "tau_fs")
    th_p = arr(clean, "theta_pulse_deg")
    th_l = arr(clean, "theta_legacy_deg")
    # Unbiased fundamental amplitude of the carrier fringe (exact DFT projection over the
    # uniform sub-sample phases). Peak-to-peak of 4 samples is biased by up to 30%.
    ptp_p = arr(clean, "theta_pulse_fringe_amp_deg")
    ptp_l = arr(clean, "theta_legacy_fringe_amp_deg")
    chi = arr(clean, "chi_pulse_deg")
    th_p0, th_l0 = at_zero(clean, "theta_pulse_deg"), at_zero(clean, "theta_legacy_deg")

    print("=== Delay study | {} | I_pump={:.0e} W/cm^2 ===".format(
        res["design"], res["pump_intensity_w_cm2"]))
    print("  pad {:.0f} fs (pump1 is the only source that moves, both signs of tau);"
          " {} carrier sub-samples".format(res["pad_fs"], res["subsamples"]))
    print("\n--- ROTATION vs DELAY (chi0 = 0) ---")
    print("  {:>9s} {:>14s} {:>14s} {:>10s} {:>10s} {:>11s}".format(
        "tau (fs)", "theta_pulse", "theta_legacy", "th_p/th_p0", "th_l/th_l0", "chi (deg)"))
    for i in range(len(tau)):
        print("  {:+9.1f} {:+14.6f} {:+14.6f} {:10.3f} {:10.3f} {:+11.5f}".format(
            tau[i], th_p[i], th_l[i], th_p[i] / th_p0 if th_p0 else np.nan,
            th_l[i] / th_l0 if th_l0 else np.nan, chi[i]))
    print("\n  at tau=0 (no delay):  theta_pulse = {:+.6f} deg   theta_legacy = {:+.6f} deg"
          .format(th_p0, th_l0))
    print("  carrier fringe ptp at tau=0: pulse {:.6f} deg, legacy {:.6f} deg"
          .format(at_zero(clean, "theta_pulse_fringe_ptp_deg"),
                  at_zero(clean, "theta_legacy_fringe_ptp_deg")))
    j = int(np.argmax(np.abs(th_p)))
    print("  |theta_pulse| is largest at tau = {:+.1f} fs: {:+.6f} deg ({:.2f}x the tau=0 value)"
          .format(tau[j], th_p[j], th_p[j] / th_p0 if th_p0 else np.nan))

    print("\n--- CARRIER-FRINGE AMPLITUDE (the channel that actually carries the delay signal) ---")
    far = np.abs(tau) >= 300
    for nm, a_ in (("pulse-integrated", ptp_p), ("legacy settled", ptp_l)):
        pk = a_[int(np.argmin(np.abs(tau)))]
        print("  {:18s} peak {:.5f} deg at tau=0, baseline(|tau|>=300) {:.5f} +/- {:.5f}"
              "  -> contrast {:.1f}x".format(nm, pk, a_[far].mean(), a_[far].std(),
                                             pk / max(a_[far].mean(), 1e-12)))
    print("  carrier-AVERAGED means are FLAT: pulse {:+.5f} +/- {:.5f}, legacy {:+.5f} +/- {:.5f} deg"
          .format(th_p.mean(), th_p.std(), th_l.mean(), th_l.std()))

    # ---------------- figure 1: delay ----------------
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.4))
    a = ax[0, 2]
    a.semilogy(tau, np.maximum(ptp_p, 1e-6), "o-", ms=3.5, lw=1.4, color="C0",
               label="pulse-integrated")
    a.semilogy(tau, np.maximum(ptp_l, 1e-6), "s-", ms=3.5, lw=1.4, color="C1",
               label="legacy settled")
    a.set_ylabel(r"carrier-fringe amplitude in $\theta$ (deg)")
    a.set_title("Fringe amplitude - THIS carries the delay signal")

    a = ax[1, 2]
    for lab, y, col in (("pulse-integrated", ptp_p, "C0"), ("legacy settled", ptp_l, "C1")):
        a.plot(tau, y / y[int(np.argmin(np.abs(tau)))], "o-", ms=3.5, lw=1.3, color=col, label=lab)
    a.plot(tau, np.exp(-(tau / (120.0 / (2 * np.sqrt(np.log(2))))) ** 2), "k--", lw=1.2,
           label="120 fs pulse envelope")
    a.set_ylabel("fringe amplitude / peak")
    a.set_title("Envelope shape vs the 120 fs pulse")

    a = ax[0, 0]
    a.fill_between(tau, th_p - ptp_p, th_p + ptp_p, color="C0", alpha=.18,
                   label="carrier-fringe swing")
    a.plot(tau, th_p, "o-", ms=3.5, lw=1.4, color="C0", label=r"$\theta$ pulse-integrated")
    a.axhline(th_p0, color="k", ls=":", lw=1.2, label=r"no delay ($\tau=0$): {:+.4f}$^\circ$".format(th_p0))
    a.set_ylabel(r"$\theta$ (deg)")
    a.set_title("Pulse-integrated rotation vs delay (absolute)")

    a = ax[0, 1]
    a.fill_between(tau, th_l - ptp_l, th_l + ptp_l, color="C1", alpha=.18,
                   label="carrier-fringe swing")
    a.plot(tau, th_l, "o-", ms=3.5, lw=1.4, color="C1", label=r"$\theta$ legacy (settled)")
    a.axhline(th_l0, color="k", ls=":", lw=1.2, label=r"no delay: {:+.4f}$^\circ$".format(th_l0))
    a.set_ylabel(r"$\theta$ (deg)")
    a.set_title("Legacy settled rotation vs delay (absolute)")

    a = ax[1, 0]
    if th_p0:
        a.plot(tau, th_p / th_p0, "o-", ms=3.5, lw=1.4, color="C0", label="pulse-integrated")
    if th_l0:
        a.plot(tau, th_l / th_l0, "s-", ms=3.5, lw=1.4, color="C1", label="legacy settled")
    a.axhline(1.0, color="k", ls=":", lw=1.2, label=r"no-delay value")
    a.axhline(0.0, color="0.6", lw=.8)
    a.set_ylabel(r"$\theta(\tau)\,/\,\theta(0)$")
    a.set_title("Relative to the non-delayed case")

    a = ax[1, 1]
    a.plot(tau, chi, "o-", ms=3.5, lw=1.4, color="C2", label=r"output ellipticity $\chi$")
    a.axhline(at_zero(clean, "chi_pulse_deg"), color="k", ls=":", lw=1.2,
              label=r"no delay: {:+.4f}$^\circ$".format(at_zero(clean, "chi_pulse_deg")))
    a.set_ylabel(r"$\chi$ (deg)")
    a.set_title("Output ellipticity vs delay")

    for a in ax.ravel():
        for b in MODE_BEATS[:1]:
            for k in (-2, -1, 1, 2):
                a.axvline(k * b, color="0.88", lw=0.8, zorder=0)
        a.axvline(0, color="0.7", lw=0.9, zorder=0)
        a.set_xlabel(r"pump1 delay $\tau$ (fs)")
        a.legend(fontsize=8)
        a.grid(alpha=.3)
    fig.suptitle("Effect of pump1 delay - SiN best_absolute, $I_{pump}=10^{12}$ W/cm$^2$"
                 " (grey lines = 152.8 fs pump-band mode beat)", fontsize=11)
    fig.tight_layout()
    fig.savefig(root / "delay_effect.png", dpi=140)
    print("\n  -> {}".format(root / "delay_effect.png"))

    # ---------------- figure 2: ellipticity ----------------
    print("\n--- ROTATION vs PROBE ELLIPTICITY ---")
    print("  {:>6s} {:>15s} {:>15s} {:>12s} {:>12s}".format(
        "chi0", "theta_pulse(0)", "theta_legacy(0)", "vs chi0=0", "chi_out(0)"))
    keys = [k for k in ("el0", "el5", "el10", "el20") if fams.get(k)]
    for k in keys:
        r = fams[k]
        tp, tl = at_zero(r, "theta_pulse_deg"), at_zero(r, "theta_legacy_deg")
        print("  {:>6s} {:+15.6f} {:+15.6f} {:12.3f} {:+12.5f}".format(
            k.replace("el", "") + " deg", tp, tl, tp / th_p0 if th_p0 else np.nan,
            at_zero(r, "chi_pulse_deg")))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    a = ax[0, 0]
    for k in keys:
        r = fams[k]
        a.plot(arr(r, "tau_fs"), arr(r, "theta_pulse_deg"), "o-", ms=3.5, lw=1.3,
               color=EL_COLORS[k], label=EL_LABEL[k])
    a.set_ylabel(r"$\theta$ (deg)")
    a.set_title("Pulse-integrated rotation, by probe ellipticity (absolute)")

    a = ax[0, 1]
    for k in keys:
        r = fams[k]
        a.plot(arr(r, "tau_fs"), arr(r, "theta_legacy_deg"), "s-", ms=3.5, lw=1.3,
               color=EL_COLORS[k], label=EL_LABEL[k])
    a.set_ylabel(r"$\theta$ (deg)")
    a.set_title("Legacy settled rotation, by probe ellipticity (absolute)")

    a = ax[1, 0]
    t0 = arr(clean, "tau_fs")
    for k in keys:
        if k == "el0":
            continue
        r = fams[k]
        tt = arr(r, "tau_fs")
        ref = np.interp(tt, t0, th_p)
        a.plot(tt, arr(r, "theta_pulse_deg") - ref, "o-", ms=3.5, lw=1.3,
               color=EL_COLORS[k], label=EL_LABEL[k] + r" $-\ \chi_0=0$")
    a.axhline(0, color="k", ls=":", lw=1.2)
    a.set_ylabel(r"$\Delta\theta$ (deg)")
    a.set_title(r"Ellipticity contribution alone (pulse-integrated)")

    a = ax[1, 1]
    c0 = [float(k.replace("el", "")) for k in keys]
    a.plot(c0, [at_zero(fams[k], "theta_pulse_deg") for k in keys], "o-", color="C0",
           label=r"pulse-integrated, $\tau=0$")
    a.plot(c0, [at_zero(fams[k], "theta_legacy_deg") for k in keys], "s-", color="C1",
           label=r"legacy settled, $\tau=0$")
    a.set_xlabel(r"probe ellipticity $\chi_0$ (deg)")
    a.set_ylabel(r"$\theta$ (deg)")
    a.set_title(r"Rotation at zero delay vs probe ellipticity")
    a.legend(fontsize=8)
    a.grid(alpha=.3)

    for a in ax.ravel()[:3]:
        a.axvline(0, color="0.7", lw=0.9, zorder=0)
        a.set_xlabel(r"pump1 delay $\tau$ (fs)")
        a.legend(fontsize=8)
        a.grid(alpha=.3)
    fig.suptitle("Effect of probe ellipticity - SiN best_absolute", fontsize=11)
    fig.tight_layout()
    fig.savefig(root / "ellipticity_effect.png", dpi=140)
    print("  -> {}".format(root / "ellipticity_effect.png"))

    # ---------------- figure 3: both on one plot ----------------
    fig, a = plt.subplots(figsize=(10.5, 6.2))
    for k in keys:
        r = fams[k]
        a.plot(arr(r, "tau_fs"), arr(r, "theta_pulse_deg"), "o-", ms=4, lw=1.6,
               color=EL_COLORS[k], label=EL_LABEL[k] + "  (pulse-integrated)")
        a.plot(arr(r, "tau_fs"), arr(r, "theta_legacy_deg"), "s--", ms=3, lw=1.1,
               color=EL_COLORS[k], alpha=.55, label=EL_LABEL[k] + "  (legacy settled)")
    a.axhline(0, color="0.6", lw=.8)
    a.axvline(0, color="0.7", lw=.9)
    for kk in (-2, -1, 1, 2):
        a.axvline(kk * MODE_BEATS[0], color="0.9", lw=.8, zorder=0)
    a.set_xlabel(r"pump1 delay $\tau$ (fs)")
    a.set_ylabel(r"rotation $\theta$ (deg)")
    a.set_title("Delay and probe ellipticity together - solid = pulse-integrated, "
                "dashed = legacy settled")
    a.legend(fontsize=8, ncol=2)
    a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(root / "delay_and_ellipticity.png", dpi=140)
    print("  -> {}".format(root / "delay_and_ellipticity.png"))


if __name__ == "__main__":
    main()
