#!/usr/bin/env python
"""Plots + mechanism analysis for the pump-probe delay scan (delay_scan.py).

Produces, from delay_scan_result.json:
  delay_trace.png       V-H vs tau (the lab observable) + rotation and ellipticity channels
  delay_systematics.png the ellipticity / azimuth / imbalance families vs the clean reference
  delay_carrier.png     the fine carrier-resolved scan
and prints the numbers that discriminate the candidate mechanisms:
  * oscillation period (FFT of the delay trace) vs the 152 fs pump-beat / round-trip prediction
  * damping: fitted decay of the ringing, converted to an effective pump-band Q
  * pedestal at large |tau| (single-pump chi3) vs peak-minus-pedestal (two-pump chi5)

  python chi5_optimization/plot_delay_scan.py
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
C0_UM_FS = 0.299792458


def oscillation_analysis(tau, sig, pedestal):
    """Dominant period of the delay trace, from a zero-padded FFT of the pedestal-subtracted
    signal with the tau=0 correlation peak retained (the ringing rides on it)."""
    out = {}
    if len(tau) < 8:
        return out
    y = np.asarray(sig, dtype=float) - pedestal
    dt = float(np.mean(np.diff(tau)))
    # Remove the symmetric envelope so the FFT sees the oscillation, not the peak:
    # subtract a smooth Gaussian fit to |y| centred at tau=0.
    env = np.exp(-(tau / max(np.std(tau), 1e-9)) ** 2)
    env *= float(np.max(np.abs(y)) / max(np.max(env), 1e-30))
    resid = y - np.sign(y[np.argmin(np.abs(tau))]) * env
    n = 1 << int(np.ceil(np.log2(len(resid) * 16)))
    spec = np.abs(np.fft.rfft(resid * np.hanning(len(resid)), n=n))
    freqs = np.fft.rfftfreq(n, d=dt)  # 1/fs
    lo = freqs > 1.0 / (4.0 * (tau.max() - tau.min()))  # ignore DC / ultra-slow
    if lo.any() and spec[lo].max() > 0:
        f_peak = float(freqs[lo][np.argmax(spec[lo])])
        if f_peak > 0:
            out["dominant_period_fs"] = 1.0 / f_peak
    return out


def damping_to_Q(period_fs, decay_fs, lam_um):
    """A ringing that decays with time constant tau_d corresponds to intracavity energy
    lifetime tau_d, i.e. Q = omega * tau_d = 2 pi c tau_d / lambda."""
    if not (decay_fs and decay_fs > 0):
        return float("nan")
    return float(2.0 * np.pi * C0_UM_FS * decay_fs / lam_um)


def fit_ring_decay(tau, sig, pedestal):
    """Fit |y| envelope decay on tau>0 to A*exp(-tau/tau_d); returns tau_d in fs."""
    y = np.abs(np.asarray(sig, dtype=float) - pedestal)
    m = (tau > 0) & (y > 0)
    if m.sum() < 4:
        return float("nan")
    try:
        p = np.polyfit(tau[m], np.log(y[m]), 1)
        return float(-1.0 / p[0]) if p[0] < 0 else float("nan")
    except Exception:
        return float("nan")


def panel(ax, tau, y, label, color=None, marker="o"):
    ax.plot(tau, y, marker=marker, ms=3, lw=1.2, label=label, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=None)
    args = ap.parse_args()
    root = Path(args.result).parent if args.result else HERE / "delay_scan"
    res = json.load(open(args.result or (root / "delay_scan_result.json")))

    beat = res.get("predicted_beat_fs", float("nan"))
    print("=== Delay scan analysis | {} ===".format(res.get("design")))
    print("  predicted pump-beat / round-trip period: {:.1f} fs".format(beat))

    s1 = res.get("stage1") or {}
    if s1:
        tau = np.array(s1["tau_fs"], dtype=float)
        sig = np.array(s1["vmh_norm"], dtype=float)
        rot = np.array(s1["rotation_deg"], dtype=float)
        chi = np.array(s1["chi_deg"], dtype=float)
        ped = s1["pedestal_vmh_norm"]

        osc = oscillation_analysis(tau, sig, ped)
        tau_d = fit_ring_decay(tau, sig, ped)
        span = float(tau.max() - tau.min())
        edge = np.abs(tau) >= 0.8 * np.max(np.abs(tau))
        scatter = float(np.std(sig[edge]))
        scatter_rot = float(np.std(rot[edge]))
        ped_rot = s1.get("pedestal_rotation_deg", float(np.mean(rot[edge])))
        print("\n--- Stage 1 (clean balanced sigma+/sigma-) ---")
        print("  {:<24s} {:>14s}   {:>16s}".format("", "V-H (norm)", "theta (deg)"))
        print("  {:<24s} {:+14.5e}   {:+16.5f}".format(
            "peak at tau=0", s1["peak_vmh_norm"], s1["peak_rotation_deg"]))
        print("  {:<24s} {:+14.5e}   {:+16.5f}".format("pedestal at large |tau|", ped, ped_rot))
        print("  {:<24s} {:+14.5e}   {:+16.5f}".format(
            "peak - pedestal", s1["peak_minus_pedestal"],
            s1["peak_rotation_deg"] - ped_rot))
        print("  {:<24s} {:14.5e}   {:16.5f}".format(
            "per-point edge scatter", scatter, scatter_rot))
        n_edge = int(edge.sum())
        err = scatter / np.sqrt(max(n_edge, 1))
        print("  -> pedestal is {:.1f} sigma from zero ({:d} edge points); the peak/pedestal"
              " ratio is\n     only a measurement if this is >> 1.".format(
                  abs(ped) / err if err > 0 else float("nan"), n_edge))
        # An "oscillation" whose period exceeds the scan span is not resolved: the FFT is just
        # picking its lowest retained bin. Reporting a ring-down Q from it is meaningless.
        per = osc.get("dominant_period_fs")
        if per and per < 0.5 * span:
            print("  measured period           = {:.1f} fs   (predicted {:.1f} fs, ratio {:.2f})".format(
                per, beat, per / beat))
            if np.isfinite(tau_d):
                print("  ringing decay time        = {:.1f} fs  => {:.2f} oscillation periods"
                      "  => effective pump Q ~ {:.0f}".format(
                          tau_d, tau_d / beat, damping_to_Q(beat, tau_d, 1.5215)))
                print("  (designed pump Q = 70 => 0.37 periods; 2-3 periods needs Q ~ 190-470)")
        else:
            print("  NO resolved oscillation: best FFT period {} fs vs scan span {:.0f} fs."
                  "\n     A period >= half the span is the FFT's lowest bin, not a ringing -- no"
                  " effective Q is quoted.".format(
                      "{:.0f}".format(per) if per else "n/a", span))
            osc.pop("dominant_period_fs", None)
            tau_d = float("nan")
        s1["oscillation"] = osc
        s1["ring_decay_fs"] = tau_d

        fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
        panel(axes[0], tau, sig, "V - H (pulse-integrated)", "C0")
        axes[0].axhline(ped, color="0.5", ls="--", lw=1, label="large-|tau| pedestal (chi3)")
        axes[0].set_ylabel(r"$(S_V-S_H)/S_0$")
        axes[0].set_title("Balanced detection vs pump1 delay - SiN best_absolute, "
                          r"$I_{pump}=10^{12}$ W/cm$^2$")
        panel(axes[1], tau, rot, "rotation " + r"$\theta$", "C1")
        axes[1].set_ylabel(r"$\theta$ (deg)")
        panel(axes[2], tau, chi, "ellipticity " + r"$\chi$", "C2")
        axes[2].set_ylabel(r"$\chi$ (deg)")
        axes[2].set_xlabel(r"pump1 delay $\tau$ (fs)")
        for a in axes:
            for k in (-2, -1, 1, 2):
                a.axvline(k * beat, color="0.85", lw=0.8, zorder=0)
            a.axvline(0, color="0.7", lw=0.8, zorder=0)
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / "delay_trace.png", dpi=150)
        print("  -> {}".format(root / "delay_trace.png"))

    fine = res.get("fine") or {}
    if fine:
        ftau = np.array(fine["tau_fs"], dtype=float)
        fsig = np.array(fine["vmh_norm"], dtype=float)
        frot = np.array(fine["rotation_deg"], dtype=float)
        fchi = np.array(fine["chi_deg"], dtype=float)
        print("\n--- Fine carrier-resolved scan ---")
        print("  carrier-period modulation ptp = {:.3e}  ({:.2%} of mean level)".format(
            fine.get("carrier_ptp", float("nan")),
            fine.get("carrier_ptp_over_mean", float("nan"))))
        # The fine grid samples whole carrier periods, so the plain mean IS the carrier-averaged
        # (delay-phase-averaged) value -- what a scan without interferometric phase stability
        # measures. Quote it next to the tau=0 value, which sits at the fringe maximum.
        print("  {:<26s} {:>14s}   {:>16s}   {:>12s}".format(
            "", "V-H (norm)", "theta (deg)", "chi (deg)"))
        print("  {:<26s} {:+14.5e}   {:+16.5f}   {:+12.4f}".format(
            "at tau=0 (fringe maximum)", fsig[0], frot[0], fchi[0]))
        print("  {:<26s} {:+14.5e}   {:+16.5f}   {:+12.4f}".format(
            "carrier-AVERAGED", fsig.mean(), frot.mean(), fchi.mean()))
        print("  {:<26s} {:14.5e}   {:16.5f}   {:12.4f}".format(
            "fringe amplitude (ptp/2)", np.ptp(fsig) / 2, np.ptp(frot) / 2, np.ptp(fchi) / 2))
        print("  -> if the pump arm is NOT interferometrically phase-stable, the experiment"
              " measures the\n     carrier-averaged row, not the tau=0 row.")
        fine["carrier_averaged_vmh_norm"] = float(fsig.mean())
        fine["carrier_averaged_rotation_deg"] = float(frot.mean())
        fine["carrier_averaged_chi_deg"] = float(fchi.mean())
        fine["fringe_amplitude_rotation_deg"] = float(np.ptp(frot) / 2)
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
        panel(axes[0], ftau, fsig, "V - H", "C3")
        axes[0].axhline(fsig.mean(), color="k", ls=":", lw=1.2, label="carrier-averaged")
        axes[0].set_ylabel(r"$(S_V-S_H)/S_0$")
        panel(axes[1], ftau, frot, r"rotation $\theta$", "C1")
        axes[1].axhline(frot.mean(), color="k", ls=":", lw=1.2, label="carrier-averaged")
        axes[1].set_ylabel(r"$\theta$ (deg)")
        axes[1].set_xlabel(r"$\tau$ (fs)")
        axes[0].set_title("Carrier-resolved scan ({:.3f} fs carrier period)".format(
            res.get("carrier_period_fs", float("nan"))))
        for a in axes:
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / "delay_carrier.png", dpi=150)
        print("  -> {}".format(root / "delay_carrier.png"))

    s2 = res.get("stage2") or {}
    if s2:
        ref = s2.get("reference") or {}
        print("\n--- Stage 2 systematics (contrast relative to the clean reference) ---")
        print("  {:<10s} {:>12s} {:>12s} {:>12s} {:>12s} {:>12s} {:>10s}".format(
            "family", "peak-ped", "theta p-p", "theta(0)", "vs ref", "cos(2chi0)", "pedestal"))
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex=True)
        if ref:
            for ax in axes.ravel()[:2]:
                panel(ax, np.array(ref["tau_fs"]), np.array(ref["vmh_norm"]),
                      "clean (45 deg linear)", "k")
            for ax in axes.ravel()[2:]:
                panel(ax, np.array(ref["tau_fs"]), np.array(ref["rotation_deg"]),
                      "clean (45 deg linear)", "k")
        for tag, fam in s2.items():
            if tag == "reference" or not fam:
                continue
            print("  {:<10s} {:12.4e} {:12.5f} {:12.5f} {:>12} {:>12} {:10.3e}".format(
                tag, fam.get("peak_minus_pedestal", float("nan")),
                fam.get("peak_minus_pedestal_rotation_deg", float("nan")),
                fam.get("peak_rotation_deg", float("nan")),
                "{:.4f}".format(fam["contrast_vs_reference"]) if "contrast_vs_reference" in fam else "-",
                "{:.4f}".format(fam["cos2chi_prediction"]) if "cos2chi_prediction" in fam else "-",
                fam.get("pedestal_vmh_norm", float("nan"))))
            col = 0 if tag.startswith("ellip") else 1
            panel(axes[0, col], np.array(fam["tau_fs"]), np.array(fam["vmh_norm"]), tag)
            panel(axes[1, col], np.array(fam["tau_fs"]), np.array(fam["rotation_deg"]), tag)
        axes[0, 0].set_title("Probe ellipticity")
        axes[0, 1].set_title("Azimuth misalignment / pump imbalance")
        for a in axes[0, :]:
            a.set_ylabel(r"$(S_V-S_H)/S_0$")
        for a in axes[1, :]:
            a.set_ylabel(r"$\theta$ (deg)")
            a.set_xlabel(r"pump1 delay $\tau$ (fs)")
        for a in axes.ravel():
            a.legend(fontsize=7)
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(root / "delay_systematics.png", dpi=150)
        print("  -> {}".format(root / "delay_systematics.png"))

    json.dump(res, open(root / "delay_scan_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
