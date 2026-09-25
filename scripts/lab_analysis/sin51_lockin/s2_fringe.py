#!/usr/bin/env python
"""Stage 2 -- what the carrier fringe should look like in THIS scan, before looking for it.

The fringe oscillates with the DELAYED pump's carrier phase: period T2 = lambda2 / c ~ 5.1-5.3 fs.
Sampled every ~20 fs it aliases. Three things follow and are plotted:

  A  apparent (aliased) period vs lambda2 at the dataset's mean step. It diverges where the
     step is a whole number of carrier periods (the fringe freezes into a smooth bump: 1499 nm
     at 20 fs, 1492 / 1790 nm at 29.85 fs), and a real cosine at frequency f is sampled
     IDENTICALLY to one at k/step - f, so each lambda2 has mirror partners -- different ones for
     different steps, which is why two scans at different steps constrain lambda2 far better
     than either alone (used in s6_compare).
  B  an ideal trace (pedestal + common envelope x [effect + fringe]) at the lab's two pump
     wavelengths (1560 / 1620 nm), drawn continuously and as the lab would sample it.
  C  how precisely the fringe could pin lambda2: correlation between sampled fringe patterns
     at lambda2 and at the design value, inside one envelope.

Per dataset (python s2_fringe.py [146 235 626]):  figs/<tag>/s2_fringe.png
"""
import numpy as np

import common as K

ENV_FWHM_FS = 125.0          # envelope FWHM fitted to the delay scans (~120-130 fs)
FRINGE_OVER_EFFECT = 2.0     # illustrative: the scans suggest a fringe ~1-2x the effect


def partners(lam_nm, step):
    """Wavelengths in the scan range whose fringe is sampled identically at this step."""
    f = K.C_UM_PER_FS / (lam_nm * 1e-3)
    out = []
    for k in range(1, 20):
        fp = k / step - f
        if fp > 0:
            lp = K.C_UM_PER_FS / fp * 1e3
            if K.LAM_SCAN_NM[0] <= lp <= K.LAM_SCAN_NM[1] and abs(lp - lam_nm) > 1:
                out.append(round(lp, 1))
    return out


def main(tag):
    d = K.load(tag)
    t = d["tau"]
    step = K.step_fs(d)
    s1 = K.load_result(tag, "s1")
    tc = s1["feature_peak_fs"] if s1["feature_peak_fs"] is not None else 0.0
    lam = np.linspace(*K.LAM_SCAN_NM, 3001)
    freeze = [step * K.C_UM_PER_FS * 1e3 / m for m in range(1, 30)
              if K.LAM_SCAN_NM[0] <= step * K.C_UM_PER_FS * 1e3 / m <= K.LAM_SCAN_NM[1]]
    res = {"step_fs": step, "freeze_nm": freeze, "lab": {}}
    for l in K.LAM_LAB_NM:
        res["lab"][str(int(l))] = {"period_fs": float(K.fringe_period_fs(l)),
                                   "apparent_period_fs": float(K.alias_period_fs(l, step)),
                                   "identical_trace_nm": partners(l, step)}
        print("[{}] step {:.3f} fs: lambda2 {:.0f} nm -> T2 {:.3f} fs, apparent {:.0f} fs; "
              "identical trace at {} nm".format(tag, step, l, K.fringe_period_fs(l),
                                               float(K.alias_period_fs(l, step)),
                                               partners(l, step)))

    plt = K.style()
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.semilogy(lam, K.alias_period_fs(lam, step), color=K.BLUE, lw=1.8,
                label="step {:.3f} fs".format(step))
    for l in K.LAM_DESIGN_NM:
        ax.axvline(l, color=K.INK2, lw=0.8, ls=":")
        ax.text(l, 3e3, " design {:.1f}".format(l), color=K.INK2, fontsize=8, rotation=90, va="top")
    for l in K.LAM_LAB_NM:
        ax.axvline(l, color=K.ORANGE, lw=1.4)
        ax.text(l, 3e3, " lab {:.0f}{}".format(l, " (delayed)" if l == K.LAM_DELAYED_NM else ""),
                color=K.INK, fontsize=8, rotation=90, va="top")
    for lf in freeze:
        ax.axvline(lf, color=K.MAGENTA, lw=1.0)
        ax.text(lf, 50, " frozen {:.0f}".format(lf), color=K.INK2, fontsize=8)
    ax.axhspan(80, 200, color=K.YELLOW, alpha=0.12, lw=0)
    ax.text(lam[0] + 3, 125, "≈ envelope width: fringe and effect look alike", fontsize=8,
            color=K.INK2, va="center")
    ax.set_ylim(40, 4e3)
    ax.set_xlabel("delayed-pump wavelength λ₂ (nm)")
    ax.set_ylabel("apparent fringe period (fs)")
    ax.set_title("A · {} — aliasing of the λ₂/c ≈ 5.1–5.7 fs fringe".format(K.title(tag)))
    ax.legend(loc="upper right")

    tf = np.linspace(tc - 400, tc + 400, 8001)
    ts = t[(t > tc - 400) & (t < tc + 400)]
    for j, l in enumerate(K.LAM_LAB_NM):
        ax = fig.add_subplot(gs[1, j])
        w = 2 * np.pi / K.fringe_period_fs(l)
        ideal = lambda x: K.gauss(x, tc, ENV_FWHM_FS) * (1 + FRINGE_OVER_EFFECT * np.cos(w * x + 0.4))  # noqa: E731
        ax.plot(tf, ideal(tf), color=K.GRID, lw=0.5, label="true trace (fringe unresolved)")
        ax.plot(tf, K.gauss(tf, tc, ENV_FWHM_FS), color=K.ORANGE, lw=1.6, label="effect alone")
        ax.plot(ts, ideal(ts), color=K.BLUE, marker="o", ms=3.5, lw=1.0, label="as sampled here")
        ax.axhline(0, color=K.INK2, lw=0.8)
        ax.set_xlabel("τ (fs)")
        ax.set_title("B · λ₂ = {:.0f} nm → looks {:.0f}-fs periodic".format(
            l, float(K.alias_period_fs(l, step))))
        if j == 0:
            ax.set_ylabel("θ (units of the effect)")
            ax.legend(loc="lower left", fontsize=7)

    for j, l0 in enumerate(K.LAM_LAB_NM):
        ax = fig.add_subplot(gs[2, j])
        g = K.gauss(ts, tc, ENV_FWHM_FS)
        ref = np.exp(1j * 2 * np.pi * ts / K.fringe_period_fs(l0)) * g
        corr = []
        for l in lam:
            z = np.exp(1j * 2 * np.pi * ts / K.fringe_period_fs(l)) * g
            # phase-free match: a real cosine with free phase cannot tell z from conj(z)
            c = max(abs(np.vdot(ref, z)), abs(np.vdot(ref, np.conj(z))))
            corr.append(c / np.linalg.norm(ref) / np.linalg.norm(z))
        corr = np.array(corr)
        ax.plot(lam, corr, color=K.BLUE, lw=1.4)
        ax.axvline(l0, color=K.INK2, lw=0.8, ls=":")
        ax.set_xlabel("λ₂ (nm)")
        ax.set_title("C · how well a {:.0f} nm fringe pins λ₂ here".format(l0))
        if j == 0:
            ax.set_ylabel("pattern match (0–1)")
    fig.tight_layout()
    K.save(fig, "s2_fringe.png", tag)
    K.save_result(tag, "s2", res)


if __name__ == "__main__":
    for tag in K.tags_from_argv():
        main(tag)
