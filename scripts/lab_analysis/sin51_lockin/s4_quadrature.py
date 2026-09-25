#!/usr/bin/env python
"""Stage 4 -- the quadrature control: is the overlap signal a chopper-synchronous optical signal?

The lock-in demodulates the 500 Hz chopping of pump2. Every pump2-induced change in V-H -- the
pedestal, the effect, the fringe -- follows the chopper and so appears at ONE lock-in phase. The
pedestal fixes that phase (~4-6 deg), so the signal is analysed in the frame rotated to it:
  X' = X cos(phi0) + Y sin(phi0)   (in phase with the pedestal: the rotation signal)
  Y' = -X sin(phi0) + Y cos(phi0)  (quadrature: noise, pickup, or a signal with its own timing)

  1  X' and Y' vs delay
  2  inside the overlap window: the phase of the excess signal, atan2(dY', dX'). ~0 means it is
     the same chopper-synchronous optical signal as the pedestal.
  3  sliding-window variance of X' and Y' (7 points), with look-elsewhere-corrected thresholds
  4  noise budget in the wings: sigma_X'^2 = sigma_Y'^2 + (pedestal-proportional part)^2

Per dataset (python s4_quadrature.py [146 235 626]):
  figs/<tag>/s4_quadrature.png, results/<tag>/s4_result.json
"""
import numpy as np

import common as K

WIN = 7
N_SURR = 1000
RNG = np.random.default_rng(7)


def ar1_series(n, m, rho, rng):
    e = rng.standard_normal((n, m))
    x = np.empty_like(e)
    x[0] = e[0] / np.sqrt(1 - rho ** 2)
    for k in range(1, n):
        x[k] = rho * x[k - 1] + e[k]
    return x * np.sqrt(1 - rho ** 2)          # unit marginal variance


def local_var(r, win):
    """Moving mean of r^2 over `win` points (r already standardized)."""
    k = np.ones(win) / win
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, r ** 2)


def main(tag):
    d = K.load(tag)
    t, n = d["tau"], len(d["tau"])
    s1 = K.load_result(tag, "s1")
    s3 = K.load_result(tag, "s3")["candidate"]
    phi0 = np.radians(s1["lockin_phase_of_pedestal_deg"])
    thx = d["X"] * K.ROT_DEG_PER_V
    thy = d["Y"] * K.ROT_DEG_PER_V
    xp = thx * np.cos(phi0) + thy * np.sin(phi0)
    yp = -thx * np.sin(phi0) + thy * np.cos(phi0)
    t_c = s1["feature_peak_fs"] if s1["feature_peak_fs"] is not None else s3["tau0_fs"]
    win = np.abs(t - t_c) <= 150
    far = np.abs(t - t_c) > 250
    ped_x, ped_y = np.median(xp[far]), np.median(yp[far])
    dx, dy = xp[win] - ped_x, yp[win] - ped_y

    # 2: phase of the excess signal (weighted by its size)
    sig_phase = float(np.degrees(np.arctan2(np.sum(dy * dx), np.sum(dx * dx))))
    ratio = float(np.sum(dy * dx) / max(np.sum(dx * dx), 1e-30))
    # 3: sliding variance, each channel standardized by its own wing noise
    sx, sy = xp[far].std(), yp[far].std()
    vx = local_var((xp - ped_x) / sx, WIN)
    vy = local_var((yp - ped_y) / sy, WIN)
    tc = t[WIN // 2: n - WIN // 2]
    rho_x, rho_y = K.ar1_rho(xp[far] - ped_x), K.ar1_rho(yp[far] - ped_y)
    nx = local_var(ar1_series(n, N_SURR, rho_x, RNG), WIN).max(0)
    ny = local_var(ar1_series(n, N_SURR, rho_y, RNG), WIN).max(0)
    jx, jy = int(np.argmax(vx)), int(np.argmax(vy))
    # 4: noise budget
    mult = np.sqrt(max(sx ** 2 - sy ** 2, 0.0))
    res = {"lockin_phase_deg": float(np.degrees(phi0)), "window_centre_fs": float(t_c),
           "excess_phase_deg": sig_phase, "quadrature_to_inphase_ratio": ratio,
           "var_X_max": float(vx[jx]), "var_X_at_fs": float(tc[jx]),
           "var_X_p_global": float(np.mean(nx >= vx[jx])), "var_X_thresh95": float(np.quantile(nx, 0.95)),
           "var_Y_max": float(vy[jy]), "var_Y_at_fs": float(tc[jy]),
           "var_Y_p_global": float(np.mean(ny >= vy[jy])), "var_Y_thresh95": float(np.quantile(ny, 0.95)),
           "noise_budget": {"sigma_X_deg": float(sx), "sigma_Y_deg": float(sy),
                            "pedestal_proportional_deg": float(mult),
                            "pedestal_fluctuation_frac": float(mult / max(ped_x, 1e-12))}}
    print("[{}] excess phase {:+.1f} deg (dY'/dX' = {:+.3f}); local variance X' {:.1f}x (p {:.3f}), "
          "Y' {:.1f}x (p {:.3f}); noise X' {:.3f}, Y' {:.3f} deg, pedestal-proportional {:.0%}".format(
              tag, sig_phase, ratio, vx[jx], res["var_X_p_global"], vy[jy], res["var_Y_p_global"],
              sx, sy, res["noise_budget"]["pedestal_fluctuation_frac"]))
    K.save_result(tag, "s4", res)

    # ---- figure --------------------------------------------------------------------------- #
    plt = K.style()
    fig = plt.figure(figsize=(10, 10.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1], width_ratios=[1.6, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(t, xp - ped_x, color=K.BLUE, marker="o", ms=3, lw=0.9, label="X′ − pedestal (in phase)")
    ax.plot(t, yp - ped_y, color=K.ORANGE, marker="o", ms=3, lw=0.9, label="Y′ − median (quadrature)")
    ax.axvspan(t_c - 150, t_c + 150, color=K.YELLOW, alpha=0.15, lw=0)
    ax.axhline(0, color=K.INK2, lw=0.8)
    ax.set_ylabel("{} (deg)".format(K.THETA))
    ax.set_xlabel("pump₂ delay τ (fs)")
    ax.set_title("{}\n1 · lock-in rotated to the pedestal phase ({:.1f}°); shaded: overlap window".format(
        K.title(tag), np.degrees(phi0)))
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(tc, vx, color=K.BLUE, lw=1.4, label="X′")
    ax.plot(tc, vy, color=K.ORANGE, lw=1.4, label="Y′")
    ax.axhline(res["var_X_thresh95"], color=K.BLUE, lw=0.9, ls="--")
    ax.axhline(res["var_Y_thresh95"], color=K.ORANGE, lw=0.9, ls="--")
    ax.set_yscale("log")
    ax.set_ylabel("local variance / wing noise²")
    ax.set_xlabel("pump₂ delay τ (fs)")
    ax.set_title("3 · {}-point sliding variance (dashed: 95% global)".format(WIN))
    ax.legend(loc="upper right", fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(dx, dy, color=K.BLUE, marker="o", ms=4, lw=0)
    lim = max(np.abs(dx).max(), np.abs(dy).max(), 1e-3) * 1.1
    ax.plot([-lim, lim], [0, 0], color=K.INK2, lw=0.8)
    ax.plot([-lim, lim], [-lim * ratio, lim * ratio], color=K.ORANGE, lw=1.2,
            label="fit: {:+.1f}°".format(sig_phase))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("ΔX′ (deg)")
    ax.set_ylabel("ΔY′ (deg)")
    ax.set_title("2 · excess signal in the window")
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(gs[2, :])
    ax.hist((xp[far] - ped_x), bins=20, color=K.BLUE, alpha=0.7, label="X′ wings, σ = {:.3f}°".format(sx))
    ax.hist((yp[far] - ped_y), bins=20, color=K.ORANGE, alpha=0.6, label="Y′ wings, σ = {:.3f}°".format(sy))
    ax.set_xlabel("{} (deg) about the median".format(K.THETA))
    ax.set_ylabel("points")
    ax.set_title("4 · noise away from the overlap: pedestal-proportional part {:.3f}° ({:.0%} of the pedestal)".format(
        mult, res["noise_budget"]["pedestal_fluctuation_frac"]))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s4_quadrature.png", tag)


if __name__ == "__main__":
    for tag in K.tags_from_argv():
        main(tag)
