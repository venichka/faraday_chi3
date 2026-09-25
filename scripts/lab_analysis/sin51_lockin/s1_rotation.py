#!/usr/bin/env python
"""Stage 1 -- convert the lock-in X into a rotation angle and characterise the pedestal.

theta(tau) = X * ROT_DEG_PER_V (calibration and its assumptions: common.py docstring).

A strong overlap feature would dominate every pedestal statistic, so it is located robustly
first and excluded: points more than 5 MAD from the median of the 3-point running mean mark it,
and +-250 fs around the largest of them is left out. On what remains, three models for the slow
part of the trace are fitted under AR(1) noise:
  M0  constant pedestal
  M1  constant + linear drift (the scan runs in time as well as in delay)
  M2  constant + smoothed step at tau0 (a long-lived pump2-induced response switching on when
      pump2 overtakes the probe)

Per dataset (python s1_rotation.py [146 235 626]; default all):
  figs/<tag>/s1_rotation.png, results/<tag>/s1_result.json
"""
import numpy as np

import common as K

STEP_WIDTH_FS = 150.0     # rise width of the M2 step (~ the pulse cross-correlation)
FEATURE_MAD = 5.0
FEATURE_HALF_FS = 250.0


def step(t, t0, w=STEP_WIDTH_FS):
    return 0.5 * (1.0 + np.tanh((t - t0) / (w / 2.0)))


def feature_window(t, th):
    """(centre, half-width) of the overlap feature, or None if nothing stands out."""
    rm = np.convolve(th, np.ones(3) / 3, mode="same")
    dev = rm - np.median(rm)
    mad = 1.4826 * np.median(np.abs(dev))
    if np.max(np.abs(dev)) < FEATURE_MAD * mad:
        return None
    return float(t[int(np.argmax(np.abs(dev)))]), FEATURE_HALF_FS


def main(tag):
    d = K.load(tag)
    t = d["tau"]
    th = d["X"] * K.ROT_DEG_PER_V
    thy = d["Y"] * K.ROT_DEG_PER_V
    fw = feature_window(t, th)
    keep = np.ones(len(t), bool) if fw is None else np.abs(t - fw[0]) > fw[1]
    tk, yk = t[keep], th[keep]
    n = len(tk)
    one = np.ones(n)

    rho = K.ar1_rho(yk - yk.mean())
    c0, chi0, _ = K.gls(yk, one[:, None], rho)
    s2w = chi0 / (n - 1)                     # white-noise variance after whitening
    sig_pt = np.sqrt(s2w / (1 - rho ** 2))   # marginal per-point std
    c1, chi1, cov1 = K.gls(yk, np.c_[one, (tk - tk.mean()) / 1000.0], rho)
    t0s = np.arange(tk.min() + 100, tk.max() - 100 + 1, 5.0)
    fits = [K.gls(yk, np.c_[one, step(tk, t0)], rho) for t0 in t0s]
    dchi_step = np.array([(chi0 - f[1]) / s2w for f in fits])
    ib = int(np.argmax(dchi_step))
    cs, _, covs = fits[ib]
    dchi_lin = (chi0 - chi1) / s2w
    bic = {"M0": n * np.log(chi0 / n) + 1 * np.log(n),
           "M1": n * np.log(chi1 / n) + 2 * np.log(n),
           "M2": n * np.log((chi0 - dchi_step[ib] * s2w) / n) + 3 * np.log(n)}
    ped = float(c0[0])
    phase_ped = float(np.degrees(np.arctan2(np.median(thy[keep]), np.median(th[keep]))))

    res = {
        "calibration_deg_per_V": K.ROT_DEG_PER_V,
        "feature_window_fs": None if fw is None else [fw[0] - fw[1], fw[0] + fw[1]],
        "feature_peak_fs": None if fw is None else fw[0],
        "pedestal_deg": ped, "pedestal_V": ped / K.ROT_DEG_PER_V,
        "pedestal_err_deg": float(np.sqrt(s2w / np.sum(K.whiten(one, rho) ** 2))),
        "noise_per_point_deg": float(sig_pt), "ar1_rho": rho,
        "quadrature_Y_median_deg": float(np.median(thy[keep])),
        "lockin_phase_of_pedestal_deg": phase_ped,
        "drift_deg_per_ps": float(c1[1]), "drift_err": float(np.sqrt(cov1[1, 1] * s2w)),
        "drift_dchi2": float(dchi_lin),
        "step_best_tau0_fs": float(t0s[ib]), "step_deg": float(cs[1]),
        "step_err_deg": float(np.sqrt(covs[1, 1] * s2w)), "step_dchi2": float(dchi_step[ib]),
        "bic": bic,
    }
    print("[{}] pedestal {:.3f} +- {:.3f} deg, noise {:.3f} deg/pt, rho {:.2f}, lock-in phase "
          "{:.1f} deg; feature {}; drift dchi2 {:.1f}, step dchi2 {:.1f}".format(
              tag, ped, res["pedestal_err_deg"], sig_pt, rho, phase_ped,
              "none" if fw is None else "at {:.0f} fs (excluded +-{:.0f})".format(*fw),
              dchi_lin, dchi_step[ib]))
    K.save_result(tag, "s1", res)

    # ---- figure ---------------------------------------------------------------------------- #
    plt = K.style()
    fig, axs = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True,
                            gridspec_kw={"height_ratios": [2.2, 1.2, 1.2]})
    ax = axs[0]
    ax.axhspan(ped - sig_pt, ped + sig_pt, color=K.GRID, alpha=0.7, lw=0)
    if fw is not None:
        ax.axvspan(fw[0] - fw[1], fw[0] + fw[1], color=K.YELLOW, alpha=0.15, lw=0)
        ax.text(fw[0], ax.get_ylim()[1] if False else th.max(), " overlap feature (excluded)",
                fontsize=8, color=K.INK2, va="top")
    ax.plot(t, th, color=K.BLUE, marker="o", ms=3.5, lw=0.9, label="{} from X".format(K.THETA))
    ax.axhline(ped, color=K.ORANGE, lw=1.4, ls="--", label="M0 constant pedestal")
    ax.plot(tk, c1[0] + c1[1] * (tk - tk.mean()) / 1000.0, color=K.AQUA, lw=1.4, ls="none",
            marker=".", ms=2, label="M1 linear drift")
    ax.plot(tk, cs[0] + cs[1] * step(tk, t0s[ib]), color=K.MAGENTA, lw=0, marker=".", ms=2,
            label="M2 step at τ₀ = {:.0f} fs".format(t0s[ib]))
    ax.set_ylabel("rotation {} (deg)".format(K.THETA))
    ax.set_title("{}\npedestal {:.3f}° ± {:.3f}°, noise {:.3f}° per point".format(
        K.title(tag), ped, res["pedestal_err_deg"], sig_pt))
    ax.legend(loc="upper right", ncol=2, fontsize=8)

    ax = axs[1]
    ax.plot(t, thy, color=K.BLUE, marker="o", ms=3, lw=0.8)
    ax.axhline(0, color=K.INK2, lw=0.8)
    ax.set_ylabel("Y (deg-equiv.)")
    ax.set_title("Out-of-phase channel: pedestal lock-in phase ≈ {:.1f}°".format(phase_ped))

    ax = axs[2]
    ax.plot(t0s, dchi_step, color=K.MAGENTA, lw=1.6)
    ax.axhline(dchi_lin, color=K.AQUA, lw=1.2, ls="--", label="linear drift Δχ² = {:.1f}".format(dchi_lin))
    ax.set_ylabel("Δχ² vs constant")
    ax.set_xlabel("pump₂ delay τ (fs, stage read-back, common axis)")
    ax.set_title("Pedestal models outside the feature: step Δχ² {:.1f} at {:.0f} fs, drift Δχ² {:.1f}".format(
        dchi_step[ib], t0s[ib], dchi_lin))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s1_rotation.png", tag)


if __name__ == "__main__":
    for tag in K.tags_from_argv():
        main(tag)
