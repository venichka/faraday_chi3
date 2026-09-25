#!/usr/bin/env python
"""Stage 0 -- the raw data, as recorded, plus the sanity checks everything later relies on.

Per dataset (python s0_raw.py [146 235 626]; default all):
  figs/<tag>/s0_raw.png     X, Y, R and lock-in phase vs delay
  figs/<tag>/s0_checks.png  stage calibration (double pass?), set-point vs read-back delay,
                            R vs hypot(X, Y), autocorrelation of X
"""
import numpy as np

import common as K


def main(tag):
    d = K.load(tag)
    t, tn, X, Y, R, ph = d["tau"], d["tau_nom"], d["X"], d["Y"], d["R"], d["phase"]
    plt = K.style()

    # ---- numbers ------------------------------------------------------------------------ #
    slope = np.polyfit(d["pos_um"], tn, 1)[0]
    resid = t - tn
    xc = X - np.median(X)
    ac = np.array([np.corrcoef(xc[:-k], xc[k:])[0, 1] for k in range(1, 11)])
    res = {"n": len(t), "step_fs": K.step_fs(d), "stage_fs_per_um": float(slope),
           "double_pass_fs_per_um": 2 / K.C_UM_PER_FS,
           "readback_minus_setpoint_rms_fs": float(resid.std()),
           "readback_minus_setpoint_max_fs": float(np.abs(resid - resid.mean()).max()),
           "X_median_V": float(np.median(X)), "Y_median_V": float(np.median(Y)),
           "lockin_phase_median_deg": float(np.median(ph)), "X_autocorr_lag1": float(ac[0])}
    print("[{}] N = {}, step {:.3f} fs, stage {:.4f} fs/um (2/c = {:.4f}), read-back - set point "
          "rms {:.3f} fs; X median {:.3e} V, lag-1 autocorr {:.2f}".format(
              tag, res["n"], res["step_fs"], slope, 2 / K.C_UM_PER_FS, resid.std(),
              res["X_median_V"], ac[0]))
    K.save_result(tag, "s0", res)

    # ---- figure 1: raw channels -------------------------------------------------------- #
    fig, axs = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for ax, y, lab in zip(axs[:3], (X * 1e6, Y * 1e6, R * 1e6), ("X (µV)", "Y (µV)", "R (µV)")):
        ax.plot(t, y, color=K.BLUE, marker="o", ms=3.5, lw=1.0)
        ax.axhline(0, color=K.INK2, lw=0.8)
        ax.set_ylabel(lab)
    axs[0].axhline(np.median(X) * 1e6, color=K.ORANGE, lw=1.2, ls="--")
    axs[0].text(t.min(), np.median(X) * 1e6, " median", color=K.INK2, va="bottom", fontsize=8)
    axs[3].plot(t, ph, color=K.BLUE, marker="o", ms=3.5, lw=1.0)
    axs[3].set_ylabel("lock-in phase (deg)")
    axs[3].set_xlabel("pump₂ delay τ (fs, stage read-back, common axis)")
    axs[0].set_title("Raw lock-in channels — {}".format(K.title(tag)))
    K.save(fig, "s0_raw.png", tag)

    # ---- figure 2: sanity checks -------------------------------------------------------- #
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    ax = axs[0, 0]
    ax.plot(d["pos_um"], tn, color=K.BLUE, marker="o", ms=3, lw=0)
    ax.plot(d["pos_um"], 2 * d["pos_um"] / K.C_UM_PER_FS + tn[0], color=K.ORANGE, lw=1.2,
            label="double pass, 2x/c")
    ax.set_xlabel("stage travel (µm)")
    ax.set_ylabel("set-point delay (fs)")
    ax.set_title("Stage: {:.4f} fs/µm vs 2/c = {:.4f}".format(slope, 2 / K.C_UM_PER_FS))
    ax.legend(loc="upper left")

    ax = axs[0, 1]
    ax.plot(tn, resid, color=K.BLUE, marker="o", ms=3, lw=0.8)
    ax.set_xlabel("set-point delay (fs)")
    ax.set_ylabel("read-back − set point (fs)")
    ax.set_title("Actual delay deviates {:.2f} fs rms".format(resid.std()))

    ax = axs[1, 0]
    h = np.hypot(X, Y) * 1e6
    ax.plot(h, R * 1e6, color=K.BLUE, marker="o", ms=3.5, lw=0)
    lim = [min(R.min() * 1e6, h.min()) * 0.9, max(R.max() * 1e6, h.max()) * 1.05]
    ax.plot(lim, lim, color=K.INK2, lw=0.8)
    ax.set_xlabel("hypot(X, Y) (µV)")
    ax.set_ylabel("R (µV)")
    ax.set_title("R vs hypot(X, Y): corr {:.2f}".format(np.corrcoef(h, R)[0, 1]))

    ax = axs[1, 1]
    ax.bar(np.arange(1, 11), ac, color=K.BLUE, width=0.6)
    ax.axhline(0, color=K.INK2, lw=0.8)
    band = 2 / np.sqrt(len(X))
    ax.axhspan(-band, band, color=K.GRID, alpha=0.6, lw=0)
    ax.set_xlabel("lag (points)")
    ax.set_ylabel("autocorrelation of X")
    ax.set_title("X autocorrelation (grey: ±2/√N)")
    fig.suptitle(K.title(tag), fontweight="bold")
    fig.tight_layout()
    K.save(fig, "s0_checks.png", tag)


if __name__ == "__main__":
    for tag in K.tags_from_argv():
        main(tag)
