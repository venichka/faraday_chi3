#!/usr/bin/env python
"""Stage 3 -- look for the pulse-overlap feature, separate fringe from effect, subtract the fringe.

Model (theta in deg, tau from the stage read-back, G a Gaussian envelope of FWHM w at tau0):

    theta(tau) = P + a G + G [c cos(w2 tau) + s sin(w2 tau)]
                 ped.  effect   fringe (k = +-1), w2 = 2 pi c / lambda2

Effect and fringe share ONE envelope -- they are the same FWM sideband read out at second and
first order (chi5_dbr_design/docs/fringe_vs_effect.md) -- so G is common. tau = 0 is nominal,
so tau0 is always scanned. lambda2 (the delayed pump) is handled two ways:
  * FIXED at the delayed pump, 1620 nm -- the physical search, with the smallest
    look-elsewhere penalty -- and, as a control, at the other pump, 1560 nm. With a
    phase-unstable fringe (stage 4) the control fits about as well: the significance then comes
    from the envelope-shaped bump and scatter, not from coherent fringe detail;
  * FREE over 1450-1650 nm -- a check that the data do not prefer some other wavelength.
For each grid point P, a, c, s are linear and fitted exactly under AR(1) noise. The noise level
and rho come from stage 1, i.e. from OUTSIDE the overlap feature: a strong feature would
otherwise inflate the noise estimate several-fold. Parameter errors are further scaled by
sqrt(reduced chi2) inside the envelope when the model misfits there (chirp, envelope shape).
The envelope width is scanned and the best-fitting one (at the delayed pump's lambda2) is used
for the figures, along with a lambda2 profile at the candidate.

Significance includes the look-elsewhere effect: the grid-maximum Delta-chi2 is compared with
the same maximum on noise-only surrogates, each studentized with its own noise estimate:
  * parametric: AR(1) Gaussian noise with the measured sigma and rho
  * empirical: block bootstrap (blocks of 5, random signs) of the measured residuals with the
    candidate region EXCLUDED -- a plain circular shift would carry the candidate itself into
    every surrogate and make any real feature look insignificant
In the free search, lambda2 values whose aliased fringe period exceeds 4x the envelope width are
masked (near 1499 nm the fringe freezes into a bump, collinear with the effect).

Fringe subtraction is reported at the delayed pump's wavelength whatever its significance -- but
stage 4 shows the fringe is phase-unstable, so this subtraction removes only one random
realisation of it. Stage 5 does the valid (phase-averaged) version.

Per dataset (python s3_search.py [146 235 626]):
  figs/<tag>/s3_search.png         Delta-chi2 vs tau0 (fixed and free lambda2), the (tau0, lambda2)
                                   map, and the lambda2 profile at the candidate
  figs/<tag>/s3_decomposition.png  raw -> fitted fringe -> fringe-subtracted net rotation
"""
import json

import numpy as np

import common as K

W_FWHM = (100.0, 125.0, 150.0, 200.0, 300.0)  # envelope FWHM: pulses 100-150 fs (user)
W_MAIN = 125.0          # default when a dataset has no clear preference (and for stage 4)
TAU0_STEP, LAM_STEP = 5.0, 1.0
N_SURR = 400
RNG = np.random.default_rng(20260923)


def design(t, t0, lam, w):
    g = K.gauss(t, t0, w)
    om = 2 * np.pi / K.fringe_period_fs(lam)
    return np.c_[np.ones_like(t), g, g * np.cos(om * t), g * np.sin(om * t)]


def build_bases(t, rho, t0s, lams, w):
    """Whitened orthonormal bases for every grid point. Column 0 spans the constant (the same
    for all points), so Delta-chi2 vs the pedestal-only model is the norm of the rest."""
    Qf = np.empty((len(t0s), len(lams), len(t), 3))   # effect + fringe directions
    Qe = np.empty((len(t0s), len(t)))                 # effect-only direction
    for i, t0 in enumerate(t0s):
        for j, lam in enumerate(lams):
            A = K.whiten(design(t, t0, lam, w), rho)
            q, _ = np.linalg.qr(A)
            Qf[i, j] = q[:, 1:]
        Ae = K.whiten(design(t, t0, lams[0], w)[:, :2], rho)
        Qe[i] = np.linalg.qr(Ae)[0][:, 1]
    return Qf, Qe


def dchi2_maps(Qf, Qe, Yw, s2w):
    """Yw: whitened data, shape (n,) or (n, m). Returns (full map, effect-only profile)."""
    Yw = Yw.reshape(len(Yw), -1)
    pf = np.einsum("ijnk,nm->ijkm", Qf, Yw)
    full = (pf ** 2).sum(axis=2) / s2w                 # (t0, lam, m)
    eff = (np.einsum("in,nm->im", Qe, Yw) ** 2) / s2w  # (t0, m)
    return full, eff


def white_var(Yw, q0):
    """Per-column noise variance, estimated exactly as for the data: whitened residual variance
    of the constant-only model. Studentizing every surrogate this way keeps data and null on
    the same footing (a fixed sigma from the data would shrink surrogates built from quieter
    residuals and make the test anti-conservative)."""
    Yw = Yw.reshape(len(Yw), -1)
    return ((Yw ** 2).sum(0) - (q0 @ Yw) ** 2) / (len(Yw) - 1)


def null_maxima(Qf, Qe, Yw, q0, lam_ok, cols, chunk=40):
    """Per surrogate: max over the masked grid, max of the effect-only profile, and the max over
    tau0 in each fixed-lambda column `cols`."""
    mf, me, mc = [], [], []
    for k in range(0, Yw.shape[1], chunk):
        blk = Yw[:, k:k + chunk]
        f, e = dchi2_maps(Qf, Qe, blk, white_var(blk, q0))
        mf.append(f[:, lam_ok].max(axis=(0, 1)))
        me.append(e.max(axis=0))
        mc.append(f[:, cols].max(axis=0))           # (len(cols), m)
    return np.concatenate(mf), np.concatenate(me), np.concatenate(mc, axis=1)


def block_bootstrap(r, n, m, block=5, rng=None):
    """m surrogates of length n from residuals r: circular blocks, a random sign per block."""
    out = np.empty((n, m))
    nb = int(np.ceil(n / block))
    for c in range(m):
        starts = rng.integers(0, len(r), nb)
        signs = rng.choice([-1.0, 1.0], nb)
        seq = np.concatenate([sg * np.take(r, np.arange(s0, s0 + block), mode="wrap")
                              for s0, sg in zip(starts, signs)])
        out[:, c] = seq[:n]
    return out


def fit_at(t, th, rho, s2w, t0, lam, w):
    """Linear fit at one grid point. Errors are scaled by sqrt(reduced chi2) over the envelope
    (|tau - t0| <= w) when the model misfits there."""
    A = design(t, t0, lam, w)
    coef, _, cov = K.gls(th, A, rho)
    r = th - A @ coef
    win = np.abs(t - t0) <= w
    chi2_red = float(np.mean(r[win] ** 2) / (s2w / (1 - rho ** 2))) if win.sum() > 4 else 1.0
    err = np.sqrt(np.diag(cov) * s2w * max(chi2_red, 1.0))
    return {"tau0_fs": float(t0), "lambda2_nm": float(lam), "w_fs": float(w),
            "pedestal_deg": float(coef[0]), "effect_deg": float(coef[1]),
            "effect_err_deg": float(err[1]), "fringe_c_deg": float(coef[2]),
            "fringe_s_deg": float(coef[3]), "fringe_amp_deg": float(np.hypot(coef[2], coef[3])),
            "fringe_err_deg": float(max(err[2], err[3])), "chi2_red_window": chi2_red,
            "contrast": abs(float(coef[1])) / max(float(np.hypot(coef[2], coef[3])), 1e-12)}


def main(tag):
    d = K.load(tag)
    t = d["tau"]
    step = K.step_fs(d)
    th = d["X"] * K.ROT_DEG_PER_V
    n = len(t)
    s1 = K.load_result(tag, "s1")
    rho = s1["ar1_rho"]
    ped = s1["pedestal_deg"]
    sig_pt = s1["noise_per_point_deg"]           # from OUTSIDE the feature
    s2w = sig_pt ** 2 * (1 - rho ** 2)
    one = np.ones(n)
    resid0 = th - ped
    q0 = K.whiten(one, rho)
    q0 = q0 / np.linalg.norm(q0)

    t0s = np.arange(t.min() + 60, t.max() - 60 + 1e-9, TAU0_STEP)
    lams = np.arange(K.LAM_SCAN_NM[0], K.LAM_SCAN_NM[1] + 1e-9, LAM_STEP)
    cols = [int(np.argmin(abs(lams - l))) for l in K.LAM_LAB_NM]
    jd = cols[K.LAM_LAB_NM.index(K.LAM_DELAYED_NM)]
    out = {"dataset": tag, "noise_per_point_deg": sig_pt, "rho": rho, "step_fs": step,
           "lab_lambdas_nm": list(K.LAM_LAB_NM), "gain_note": K.GAIN_NOTE, "widths": {}}

    # surrogates (built once, reused for every width)
    e = RNG.standard_normal((n, N_SURR)) * np.sqrt(s2w)
    ar = np.empty_like(e)
    ar[0] = e[0] / np.sqrt(1 - rho ** 2)
    for k in range(1, n):
        ar[k] = rho * ar[k - 1] + e[k]

    maps = {}
    for w in W_FWHM:
        Qf, Qe = build_bases(t, rho, t0s, lams, w)
        full, eff = dchi2_maps(Qf, Qe, K.whiten(th, rho), s2w)
        full, eff = full[..., 0], eff[:, 0]
        lam_ok = K.alias_period_fs(lams, step) < 4 * w
        free = np.where(lam_ok[None, :], full, 0.0)
        i, j = np.unravel_index(np.argmax(free), free.shape)
        mx_par, mxe_par, mc_par = null_maxima(Qf, Qe, K.whiten(ar, rho), q0, lam_ok, cols)
        keep = np.abs(t - t0s[i]) > 250.0
        emp = block_bootstrap(resid0[keep] - resid0[keep].mean(), n, N_SURR, rng=RNG)
        mx_emp, _, mc_emp = null_maxima(Qf, Qe, K.whiten(emp, rho), q0, lam_ok, cols)
        ie = int(np.argmax(eff))
        i0 = int(np.argmin(abs(t0s)))

        res = {"free": {**fit_at(t, th, rho, s2w, t0s[i], lams[j], w),
                        "dchi2": float(free[i, j]),
                        "global_p_parametric": float(np.mean(mx_par >= free[i, j])),
                        "global_p_empirical": float(np.mean(mx_emp >= free[i, j])),
                        "thresh95_parametric": float(np.quantile(mx_par, 0.95)),
                        "masked_lambda2_nm": [float(lams[~lam_ok].min()), float(lams[~lam_ok].max())]
                        if (~lam_ok).any() else None},
               "effect_only": {"tau0_fs": float(t0s[ie]), "dchi2": float(eff[ie]),
                               "global_p_parametric": float(np.mean(mxe_par >= eff[ie])),
                               "thresh95_parametric": float(np.quantile(mxe_par, 0.95))},
               "fixed": {}}
        for c, lam, mp, me_ in zip(cols, K.LAM_LAB_NM, mc_par, mc_emp):
            ic = int(np.argmax(full[:, c]))
            f0 = fit_at(t, th, rho, s2w, 0.0, lam, w)
            res["fixed"][str(int(lam))] = {
                **fit_at(t, th, rho, s2w, t0s[ic], lam, w),
                "dchi2": float(full[ic, c]),
                "global_p_parametric": float(np.mean(mp >= full[ic, c])),
                "global_p_empirical": float(np.mean(me_ >= full[ic, c])),
                "thresh95_parametric": float(np.quantile(mp, 0.95)),
                "thresh99_parametric": float(np.quantile(mp, 0.99)),
                "at_nominal_zero": {"dchi2": float(full[i0, c]),
                                    "effect_deg": f0["effect_deg"], "effect_err_deg": f0["effect_err_deg"],
                                    "effect_95ul_deg": abs(f0["effect_deg"]) + 1.96 * f0["effect_err_deg"],
                                    "fringe_amp_deg": f0["fringe_amp_deg"],
                                    "fringe_95ul_deg": f0["fringe_amp_deg"] + 1.96 * f0["fringe_err_deg"]},
            }
        # lambda2 profile around the delayed-pump candidate (tau0 within +-60 fs of it)
        ic = int(np.argmax(full[:, jd]))
        near = np.abs(t0s - t0s[ic]) <= 60
        prof = full[near].max(axis=0)
        res["lambda2_profile"] = {"lambda_nm": lams.tolist(), "dchi2": prof.tolist()}
        dl = str(int(K.LAM_DELAYED_NM))
        res["candidate"] = {**res["fixed"][dl], "lambda2_nm": K.LAM_DELAYED_NM}
        out["widths"][str(int(w))] = res
        maps[w] = dict(full=full, free=free, eff=eff, res=res, prof=prof)

        fr = res["free"]
        f = res["fixed"][dl]
        print("[{}] w={:3.0f}: at {:.0f} nm tau0 {:.0f} fs, dchi2 {:.1f} (95% {:.1f}), p {:.3f}/{:.3f}; "
              "effect {:+.3f}+-{:.3f}, fringe {:.3f} deg, chi2_red {:.1f} | free: {:.0f} nm, dchi2 {:.1f}".format(
                  tag, w, K.LAM_DELAYED_NM, f["tau0_fs"], f["dchi2"], f["thresh95_parametric"],
                  f["global_p_parametric"], f["global_p_empirical"], f["effect_deg"],
                  f["effect_err_deg"], f["fringe_amp_deg"], f["chi2_red_window"],
                  fr["lambda2_nm"], fr["dchi2"]))

    # the width used for figures and later stages: best fit at the delayed pump's lambda2
    w_best = max(W_FWHM, key=lambda w: maps[w]["res"]["candidate"]["dchi2"])
    out["w_best_fs"] = w_best
    out["candidate"] = out["widths"][str(int(w_best))]["candidate"]
    K.save_result(tag, "s3", out)
    print("[{}] best envelope {:.0f} fs".format(tag, w_best))

    # ---- figure 1: the search ------------------------------------------------------------ #
    plt = K.style()
    m = maps[w_best]
    res = m["res"]
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 11.5), gridspec_kw={"height_ratios": [1.1, 1.2, 0.9]})
    ax = axs[0]
    for c, lam, col in zip(cols, K.LAM_LAB_NM, (K.BLUE, K.AQUA)):
        f = res["fixed"][str(int(lam))]
        role = "delayed pump" if lam == K.LAM_DELAYED_NM else "control"
        ax.plot(t0s, m["full"][:, c], color=col, lw=1.7 if lam == K.LAM_DELAYED_NM else 1.1,
                label="λ₂ = {:.0f} nm, {} (p = {:.3f})".format(lam, role, f["global_p_parametric"]))
        ax.axhline(f["thresh95_parametric"], color=col, lw=0.9, ls="--")
    ax.plot(t0s, m["free"].max(axis=1), color=K.INK2, lw=1.0, ls=":", label="λ₂ free")
    ax.plot(t0s, m["eff"], color=K.ORANGE, lw=1.3, label="effect only, no fringe")
    ax.set_yscale("symlog", linthresh=20)
    ax.set_ylabel("Δχ² vs flat pedestal")
    ax.set_xlabel("envelope centre τ₀ (fs)")
    ax.set_title("{}\nwhere does an overlap feature fit? (envelope {:.0f} fs; dashed: 95% global)".format(
        K.title(tag), w_best))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax = axs[1]
    im = ax.pcolormesh(t0s, lams, m["free"].T, cmap="Blues", shading="auto")
    for lam in K.LAM_LAB_NM:
        ax.axhline(lam, color=K.ORANGE, lw=1.0)
        ax.text(t0s[-1], lam, " {:.0f}".format(lam), color=K.INK2, fontsize=8, va="center")
    ax.set_ylabel("delayed-pump λ₂ (nm)")
    ax.set_xlabel("envelope centre τ₀ (fs)")
    ax.set_title("Δχ² over (τ₀, λ₂) — orange: lab pumps; white: fringe frozen (masked)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=axs[1], shrink=0.8, pad=0.02)
    cb.set_label("Δχ²")
    ax = axs[2]
    ax.plot(lams, m["prof"].max() - m["prof"], color=K.BLUE, lw=1.6)
    for lam in K.LAM_LAB_NM:
        ax.axvline(lam, color=K.ORANGE, lw=1.0)
    ax.set_ylim(0, max(10.0, 1.05 * float(np.max(m["prof"].max() - m["prof"]))))
    ax.set_ylabel("χ² − χ²_min")
    ax.set_xlabel("delayed-pump λ₂ (nm)")
    ax.set_title("λ₂ profile at the candidate (single scan: note the aliasing partners)")
    fig.tight_layout()
    K.save(fig, "s3_search.png", tag)

    # ---- figure 2: decomposition at the delayed pump's wavelength ------------------------- #
    cnd = res["candidate"]
    t0, lam, w = cnd["tau0_fs"], cnd["lambda2_nm"], w_best
    om = 2 * np.pi / K.fringe_period_fs(lam)
    g = K.gauss(t, t0, w)
    fringe_s = g * (cnd["fringe_c_deg"] * np.cos(om * t) + cnd["fringe_s_deg"] * np.sin(om * t))
    tf = np.linspace(t.min(), t.max(), 40001)
    gf = K.gauss(tf, t0, w)
    fringe_f = gf * (cnd["fringe_c_deg"] * np.cos(om * tf) + cnd["fringe_s_deg"] * np.sin(om * tf))
    net = th - fringe_s
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 10), sharex=True)
    ax = axs[0]
    ax.plot(t, th, color=K.BLUE, marker="o", ms=3.5, lw=0.9, label="measured {}".format(K.THETA))
    ax.plot(t, cnd["pedestal_deg"] + cnd["effect_deg"] * g + fringe_s, color=K.ORANGE, lw=1.4,
            label="fit: pedestal + effect + fringe (sampled)")
    ax.set_ylabel("{} (deg)".format(K.THETA))
    ax.set_title("{}\n1 · raw rotation and fit at λ₂ = {:.0f} nm: τ₀ = {:.0f} fs, envelope {:.0f} fs".format(
        K.title(tag), lam, t0, w))
    ax.legend(loc="upper right", fontsize=8)
    ax = axs[1]
    ax.plot(tf, fringe_f, color=K.GRID, lw=0.4, label="fringe, true {:.2f} fs period".format(
        K.fringe_period_fs(lam)))
    ax.plot(t, fringe_s, color=K.MAGENTA, marker="o", ms=3.5, lw=1.0,
            label="fringe as sampled (aliased to ~{:.0f} fs)".format(float(K.alias_period_fs(lam, step))))
    ax.axhline(0, color=K.INK2, lw=0.8)
    ax.set_ylabel("fringe (deg)")
    ax.set_title("2 · calculated fringe: amplitude {:.3f}° ± {:.3f}°".format(
        cnd["fringe_amp_deg"], cnd["fringe_err_deg"]))
    ax.legend(loc="upper right", fontsize=8)
    ax = axs[2]
    ax.axhspan(cnd["pedestal_deg"] - sig_pt, cnd["pedestal_deg"] + sig_pt, color=K.GRID, alpha=0.7, lw=0)
    ax.plot(t, net, color=K.BLUE, marker="o", ms=3.5, lw=0.9, label="{} − fitted fringe".format(K.THETA))
    ax.plot(tf, cnd["pedestal_deg"] + cnd["effect_deg"] * gf, color=K.ORANGE, lw=1.6,
            label="pedestal + effect: {:+.3f}° ± {:.3f}°".format(cnd["effect_deg"], cnd["effect_err_deg"]))
    ax.set_ylabel("net {} (deg)".format(K.THETA))
    ax.set_xlabel("pump₂ delay τ (fs, stage read-back, common axis)")
    ax.set_title("3 · net rotation, fringe fit subtracted (grey: ±1σ per point; fit χ²_red {:.1f} in the envelope)".format(
        cnd["chi2_red_window"]))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s3_decomposition.png", tag)


if __name__ == "__main__":
    for tag in K.tags_from_argv():
        main(tag)
