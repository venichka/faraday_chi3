#!/usr/bin/env python
"""Stage 8 -- the chi5 (k = 0) term without any fringe model: a phase-marginal likelihood.

Every fringe model in stage 7 needs a carrier, and the data do not fix one (the two OPAs are not
phase-locked, so the fringe phase at each point is effectively random). This stage drops the
carrier altogether. Each point is

    y_n = P_k + E G_e(u_n) + A G_f(u_n) cos(phi_n) + eps_n,   phi_n ~ Uniform(0, 2 pi),  eps ~ N(0, sigma_k^2)

and phi_n is integrated out EXACTLY (numerically), instead of being approximated by a Gaussian of
variance A^2 G_f^2 / 2 as in stage 5. The exact marginal is an arcsine law convolved with the
noise: with A G_f >> sigma the points pile up near y = P + E G_e +- A G_f and the k = 0 level is the
MIDPOINT of that band -- much better determined than by the sample mean. The two scans share
E, A and both envelopes; each has its own pedestal P_k and overlap centre tau0_k.

Nonlinear parameters (tau0_k, w_e, w_f, A, E, P_k) are optimised with Nelder-Mead from several
starts; E gets a profile-likelihood interval (Delta(-2lnL) = 1) and a 95% one (= 3.84).

  python s8_phase_marginal.py   -> figs/combined/s8_phase_marginal.png, results/combined/s8_result.json
"""
import numpy as np
from scipy.optimize import minimize

import common as K

PHI = np.linspace(0, 2 * np.pi, 96, endpoint=False)
COS = np.cos(PHI)


def nll_scan(sc, P, tau0, E, w_e, A, w_f):
    u = sc["t"] - tau0
    ge, gf = K.gauss(u, 0.0, w_e), K.gauss(u, 0.0, w_f)
    mu = P + E * ge
    # per point: log mean_phi N(y - mu - A gf cos phi; 0, sigma^2)
    r = (sc["y"][:, None] - mu[:, None] - (A * gf)[:, None] * COS[None, :]) / sc["sig"]
    lp = -0.5 * r ** 2
    m = lp.max(axis=1)
    return -2.0 * np.sum(m + np.log(np.mean(np.exp(lp - m[:, None]), axis=1)) - np.log(sc["sig"] * np.sqrt(2 * np.pi)))


def total(x, scans, E_fixed=None):
    n = len(scans)
    P = x[:n]
    t0 = x[n:2 * n]
    E, w_e, A, w_f = (x[2 * n:] if E_fixed is None else np.r_[E_fixed, x[2 * n:]])
    if not (60 <= w_e <= 400 and 60 <= w_f <= 400 and 0 <= A <= 6):
        return 1e9
    return sum(nll_scan(sc, P[k], t0[k], E, w_e, A, w_f) for k, sc in enumerate(scans))


def fit(scans, E_fixed=None):
    n = len(scans)
    best = None
    for E0 in ((1.0, 1.8, 2.5) if E_fixed is None else (None,)):
        for w0 in (100.0, 150.0):
            x0 = [np.median(sc["y"]) for sc in scans] + [-1060.0] * n
            x0 += ([E0] if E_fixed is None else []) + [w0, 2.0, w0 + 40]
            r = minimize(total, np.array(x0), args=(scans, E_fixed), method="Nelder-Mead",
                         options={"maxiter": 20000, "xatol": 1e-4, "fatol": 1e-6})
            if best is None or r.fun < best.fun:
                best = r
    return best


def main():
    scans = []
    for tag in K.DELAY_TAGS:
        d = K.load(tag)
        scans.append({"tag": tag, "t": d["tau"], "y": d["X"] * K.ROT_DEG_PER_V,
                      "sig": K.load_result(tag, "s1")["noise_per_point_deg"]})
    n = len(scans)
    r = fit(scans)
    P, t0, (E, w_e, A, w_f) = r.x[:n], r.x[n:2 * n], r.x[2 * n:]
    nll0 = r.fun
    Es = np.linspace(-0.5, 4.0, 91)
    prof = np.array([fit(scans, E_fixed=e).fun for e in Es])
    ok68 = Es[prof - nll0 <= 1.0]
    ok95 = Es[prof - nll0 <= 3.84]
    nllE0 = fit(scans, E_fixed=0.0).fun
    res = {"E_deg": float(E), "E_68": [float(ok68.min()), float(ok68.max())],
           "E_95": [float(ok95.min()), float(ok95.max())], "E_dnll_vs_zero": float(nllE0 - nll0),
           "A_deg": float(A), "w_e_fs": float(w_e), "w_f_fs": float(w_f),
           "pedestals_deg": {sc["tag"]: float(p) for sc, p in zip(scans, P)},
           "tau0_fs": {sc["tag"]: float(t) for sc, t in zip(scans, t0)},
           "m2lnL": float(nll0), "n": int(sum(len(sc["t"]) for sc in scans)),
           "profile": {"E": Es.tolist(), "m2lnL": prof.tolist()}}
    K.save_result("combined", "s8", res)
    print("phase-marginal fit: E = {:+.2f} deg, 68% [{:+.2f}, {:+.2f}], 95% [{:+.2f}, {:+.2f}]; "
          "E = 0 disfavoured by dnll {:.1f} ({:.1f} sigma); A = {:.2f} deg; envelopes effect {:.0f} / fringe {:.0f} fs; "
          "tau0 {:.0f} / {:.0f} fs; -2lnL {:.1f} for {} pts".format(
              E, *res["E_68"], *res["E_95"], res["E_dnll_vs_zero"], np.sqrt(max(res["E_dnll_vs_zero"], 0)), A,
              w_e, w_f, *t0, nll0, res["n"]))

    plt = K.style()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axs[0]
    tf = np.linspace(-1400, -700, 2001)
    cols = {"146": K.BLUE, "235": K.ORANGE}
    for k, sc in enumerate(scans):
        m = (sc["t"] > -1400) & (sc["t"] < -700)
        ax.plot(sc["t"][m], sc["y"][m] - P[k], color=cols[sc["tag"]], marker="o", ms=4, lw=0.7,
                label="{} − pedestal".format(sc["tag"]))
    tm = np.mean(t0)
    ge, gf = K.gauss(tf, tm, w_e), K.gauss(tf, tm, w_f)
    ax.fill_between(tf, E * ge - A * gf, E * ge + A * gf, color=K.MAGENTA, alpha=0.15, lw=0,
                    label="phase band: k=0 ± fringe (A = {:.2f}°)".format(A))
    ax.plot(tf, E * ge, color=K.INK, lw=2.0, label="k = 0 term: {:+.2f}° [{:+.2f}, {:+.2f}]".format(E, *res["E_68"]))
    ax.axhline(0, color=K.INK2, lw=0.8)
    ax.set_xlabel("pump₂ delay τ (fs)")
    ax.set_ylabel("{} − pedestal (deg)".format(K.THETA))
    ax.set_title("Phase-marginal fit: no fringe carrier assumed (envelopes {:.0f} / {:.0f} fs)".format(w_e, w_f))
    ax.legend(loc="upper right", fontsize=8)
    ax = axs[1]
    ax.plot(Es, prof - nll0, color=K.BLUE, lw=1.6)
    ax.axhline(1.0, color=K.INK2, lw=0.8, ls="--")
    ax.axhline(3.84, color=K.INK2, lw=0.8, ls=":")
    ax.set_ylim(0, 25)
    ax.set_xlabel("k = 0 amplitude E (deg)")
    ax.set_ylabel("Δ(−2 ln L)")
    ax.set_title("Profile likelihood of E (68% / 95% lines)")
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s8_phase_marginal.png", "combined")


if __name__ == "__main__":
    main()
