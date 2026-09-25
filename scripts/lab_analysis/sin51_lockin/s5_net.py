#!/usr/bin/env python
"""Stage 5 -- the net (fringe-free) rotation: the effect and the fringe, with the fringe treated as
what the data show it to be.

Stage 3's coherent model (one fringe of fixed carrier and phase across the scan) misfits the delay
scans badly (reduced chi2 ~100-190 inside the envelope), and the two scans read completely different
values at the same delay (-1020 fs: 3.53 deg vs 0.17 deg) while agreeing on the envelope. The fine
structure is a fast (~5 fs) fringe whose optical phase is NOT stable from point to point. Each
point then samples that fringe at an effectively random phase, so:

    theta_n = P + E G(tau_n) + noise_n,   Var(noise_n) = sigma^2 + (1/2) A^2 G(tau_n)^2

  P      pedestal;   E G    the phase-averaged (k = 0) rotation = the chi5 observable
  A G    fringe amplitude (k = +-1) under the same envelope G (Gaussian, centre tau0, FWHM w)
  sigma  the per-point noise measured away from the overlap (stage 1)

Fitted by maximum likelihood on a (tau0, w, A) grid with P, E solved by weighted least squares
at every node; errors from the profile likelihood (Delta(-2 ln L) = 1). This removes the fringe
by AVERAGING over its phase, which is valid whatever the phase does, rather than by subtracting a
model of it, which is only valid if the phase is stable.

Per dataset (python s5_net.py [146 235 626]):
  figs/<tag>/s5_net.png, results/<tag>/s5_result.json
"""
import numpy as np

import common as K

TAU0_HALF = 200.0                                   # search +-200 fs around the stage-1 feature
W_GRID = np.arange(60.0, 420.0 + 1e-9, 10.0)
A_GRID = np.arange(0.0, 4.0 + 1e-9, 0.02)
N_AVG = 3


def nll_grid(t, th, sig, t0s, ws, As):
    """-2 ln L on the grid, profiled over P and E (weighted LS). Returns (nll, P, E, varE)."""
    shape = (len(t0s), len(ws), len(As))
    nll = np.full(shape, np.inf)
    P = np.zeros(shape)
    E = np.zeros(shape)
    vE = np.zeros(shape)
    for i, t0 in enumerate(t0s):
        for j, w in enumerate(ws):
            g = K.gauss(t, t0, w)
            v = sig ** 2 + 0.5 * np.outer(As ** 2, g ** 2)           # (nA, n)
            wt = 1.0 / v
            # weighted LS for [1, g] per A, vectorized
            s0, s1, s2 = wt.sum(1), (wt * g).sum(1), (wt * g * g).sum(1)
            b0, b1 = (wt * th).sum(1), (wt * g * th).sum(1)
            det = s0 * s2 - s1 ** 2
            p = (s2 * b0 - s1 * b1) / det
            e = (s0 * b1 - s1 * b0) / det
            r = th[None, :] - p[:, None] - e[:, None] * g[None, :]
            nll[i, j] = (r ** 2 * wt).sum(1) + np.log(v).sum(1)
            P[i, j], E[i, j], vE[i, j] = p, e, s0 / det
    return nll, P, E, vE


def interval(x, prof):
    """Profile-likelihood 68% interval (Delta = 1) of a 1D profile."""
    d = prof - prof.min()
    ok = x[d <= 1.0]
    return float(ok.min()), float(ok.max())


def fit(tag):
    d = K.load(tag)
    t = d["tau"]
    th = d["X"] * K.ROT_DEG_PER_V
    s1 = K.load_result(tag, "s1")
    sig = s1["noise_per_point_deg"]
    c = s1["feature_peak_fs"] if s1["feature_peak_fs"] is not None else \
        K.load_result(tag, "s3")["candidate"]["tau0_fs"]
    t0s = np.arange(c - TAU0_HALF, c + TAU0_HALF + 1e-9, 5.0)
    nll, P, E, vE = nll_grid(t, th, sig, t0s, W_GRID, A_GRID)
    i, j, k = np.unravel_index(np.argmin(nll), nll.shape)
    # profile intervals (E via its own profile: min over grid of nll at fixed E is approximated by
    # the conditional Gaussian at each node; A, tau0, w by direct profiling)
    E_lo = (E - np.sqrt(vE))[i, j, k]
    E_hi = (E + np.sqrt(vE))[i, j, k]
    prof_A = nll.min(axis=(0, 1))
    prof_t0 = nll.min(axis=(1, 2))
    prof_w = nll.min(axis=(0, 2))
    # E profile over all nodes: for each node, nll(E) = nll_node + (E - E_node)^2 / vE_node
    Es = np.linspace(E[i, j, k] - 6 * np.sqrt(vE[i, j, k]), E[i, j, k] + 6 * np.sqrt(vE[i, j, k]), 241)
    prof_E = np.array([np.min(nll + (e - E) ** 2 / vE) for e in Es])
    # the no-fringe (A = 0) and no-effect (E = 0) alternatives, for the likelihood-ratio tests
    nll_A0 = nll[:, :, 0].min()
    nll_E0 = np.min(nll + E ** 2 / vE)
    res = {"dataset": tag, "sigma_deg": sig,
           "tau0_fs": float(t0s[i]), "tau0_68": interval(t0s, prof_t0),
           "w_fs": float(W_GRID[j]), "w_68": interval(W_GRID, prof_w),
           "pedestal_deg": float(P[i, j, k]),
           "effect_deg": float(E[i, j, k]), "effect_68": interval(Es, prof_E),
           "effect_err_conditional_deg": float(np.sqrt(vE[i, j, k])),
           "fringe_amp_deg": float(A_GRID[k]), "fringe_amp_68": interval(A_GRID, prof_A),
           "contrast": float(abs(E[i, j, k]) / max(A_GRID[k], 1e-12)),
           "dnll_no_fringe": float(nll_A0 - nll.min()), "dnll_no_effect": float(nll_E0 - nll.min()),
           "coherent_model_chi2_red": K.load_result(tag, "s3")["candidate"]["chi2_red_window"]}
    return res, dict(t=t, th=th, sig=sig)


def main(tag):
    res, data = fit(tag)
    print("[{}] tau0 {:.0f} fs, FWHM {:.0f} fs; effect {:+.3f} deg [{:+.3f}, {:+.3f}]; fringe {:.2f} deg "
          "[{:.2f}, {:.2f}]; contrast {:.2f}; -2lnL gain: fringe {:.1f}, effect {:.1f}".format(
              tag, res["tau0_fs"], res["w_fs"], res["effect_deg"], *res["effect_68"],
              res["fringe_amp_deg"], *res["fringe_amp_68"], res["contrast"],
              res["dnll_no_fringe"], res["dnll_no_effect"]))
    K.save_result(tag, "s5", res)

    t, th, sig = data["t"], data["th"], data["sig"]
    t0, w, P, E, A = res["tau0_fs"], res["w_fs"], res["pedestal_deg"], res["effect_deg"], res["fringe_amp_deg"]
    plt = K.style()
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 10.5), gridspec_kw={"height_ratios": [1, 1.15, 1.15]})
    ax = axs[0]
    ax.plot(t, th, color=K.BLUE, marker="o", ms=3, lw=0.9, label="measured {} (X)".format(K.THETA))
    ax.axhline(P, color=K.ORANGE, lw=1.2, ls="--", label="pedestal {:.3f}°".format(P))
    ax.axvspan(t0 - w, t0 + w, color=K.YELLOW, alpha=0.15, lw=0)
    ax.set_ylabel("{} (deg)".format(K.THETA))
    ax.set_xlabel("pump₂ delay τ (fs)")
    ax.set_title("{}\n1 · full scan (shaded: overlap, τ₀ ± FWHM)".format(K.title(tag)))
    ax.legend(loc="upper right", fontsize=8)

    zoom = np.abs(t - t0) < 3 * w + 100
    tf = np.linspace(t0 - 3 * w - 100, t0 + 3 * w + 100, 4001)
    gf = K.gauss(tf, t0, w)
    ax = axs[1]
    ax.fill_between(tf, P + E * gf - A * gf, P + E * gf + A * gf, color=K.MAGENTA, alpha=0.18, lw=0,
                    label="fringe band: effect ± A·G, A = {:.2f}°".format(A))
    ax.fill_between(tf, P + E * gf - np.sqrt(sig ** 2 + 0.5 * (A * gf) ** 2),
                    P + E * gf + np.sqrt(sig ** 2 + 0.5 * (A * gf) ** 2), color=K.GRID, alpha=0.9, lw=0,
                    label="±1σ expected scatter (noise + random-phase fringe)")
    ax.plot(t[zoom], th[zoom], color=K.BLUE, marker="o", ms=4, lw=0.8, label="measured")
    ax.plot(tf, P + E * gf, color=K.ORANGE, lw=1.8, label="pedestal + effect")
    ax.set_ylabel("{} (deg)".format(K.THETA))
    ax.set_title("2 · the fringe: a {:.2f}° swing at random phase under the envelope (coherent-fit χ²_red was {:.0f})".format(
        A, res["coherent_model_chi2_red"]))
    ax.legend(loc="upper right", fontsize=7.5)

    ax = axs[2]
    r = th - P
    k = np.ones(N_AVG) / N_AVG
    n = len(t)
    rm = np.convolve(r, k, mode="valid")
    tm = t[N_AVG // 2: n - N_AVG // 2]
    gm = K.gauss(tm, t0, w)
    se = np.sqrt(sig ** 2 + 0.5 * (A * gm) ** 2) / np.sqrt(N_AVG)
    zm = np.abs(tm - t0) < 3 * w + 100
    ax.fill_between(tm[zm], rm[zm] - se[zm], rm[zm] + se[zm], color=K.GRID, alpha=0.9, lw=0,
                    label="±1σ of the {}-point mean".format(N_AVG))
    ax.plot(tm[zm], rm[zm], color=K.BLUE, lw=1.6, marker="o", ms=3,
            label="{}-point mean of {} − pedestal (fringe averaged down)".format(N_AVG, K.THETA))
    ax.plot(tf, E * gf, color=K.ORANGE, lw=2.0,
            label="effect (k = 0): {:+.3f}° [{:+.3f}, {:+.3f}], FWHM {:.0f} fs".format(
                E, *res["effect_68"], w))
    ax.axhline(0, color=K.INK2, lw=0.8)
    ax.set_ylabel("net {} (deg)".format(K.THETA))
    ax.set_xlabel("pump₂ delay τ (fs)")
    ax.set_title("3 · net rotation: fringe removed by averaging over its phase")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s5_net.png", tag)


if __name__ == "__main__":
    for tag in K.tags_from_argv():
        main(tag)
