#!/usr/bin/env python
"""Stage 6 -- all scans together: reproducibility, the joint effect / fringe estimate, the locked
reference, and the comparison with the as-built simulation.

  1  overlay of every dataset on the common delay axis (same stage position <=> same delay)
  2  joint random-phase-fringe fit of the delay scans (the stage-5 model): shared overlap centre,
     envelope width, effect E and fringe amplitude A; a pedestal per scan. Two scans sample the
     fringe at independent phases, so averaging over both pins the effect far better.
  3  pooled profile: both scans, pedestal-subtracted, in 20 fs bins -- the bin MEAN traces the
     effect (k = 0), the bin RMS traces the fringe (k = +-1); each against the joint fit
  4  the locked reference (626) fitted with the same envelope, for its amplitude relative to the
     delay scans
  5  the simulation (sim/sim_result.json, from sim_labpoint.py), if present: as-built 1D trace
     scaled to 3D by the tau = 0 3D/1D ratio, against the measured effect and fringe envelopes

  figs/combined/s6_overlay.png, figs/combined/s6_joint.png, results/combined/s6_result.json
"""
import json

import numpy as np

import common as K
import s5_net as S5

BIN_FS = 20.0


def joint_fit(scans, t0s, ws, As):
    """-2 ln L summed over scans; shared (tau0, w, A, E), one pedestal per scan (weighted LS)."""
    nS = len(scans)
    best = None
    grid = np.full((len(t0s), len(ws), len(As)), np.inf)
    Egrid = np.zeros_like(grid)
    vEgrid = np.zeros_like(grid)
    for i, t0 in enumerate(t0s):
        for j, w in enumerate(ws):
            gs = [K.gauss(s["t"], t0, w) for s in scans]
            for k, A in enumerate(As):
                # normal equations for [P_1..P_nS, E]
                M = np.zeros((nS + 1, nS + 1))
                b = np.zeros(nS + 1)
                logv = 0.0
                ws_ = []
                for m, (s, g) in enumerate(zip(scans, gs)):
                    v = s["sig"] ** 2 + 0.5 * (A * g) ** 2
                    wt = 1.0 / v
                    ws_.append(wt)
                    logv += np.log(v).sum()
                    M[m, m] += wt.sum()
                    M[m, nS] += (wt * g).sum()
                    M[nS, m] += (wt * g).sum()
                    M[nS, nS] += (wt * g * g).sum()
                    b[m] += (wt * s["th"]).sum()
                    b[nS] += (wt * g * s["th"]).sum()
                coef = np.linalg.solve(M, b)
                cov = np.linalg.inv(M)
                chi = sum(((s["th"] - coef[m] - coef[nS] * g) ** 2 * wt).sum()
                          for m, (s, g, wt) in enumerate(zip(scans, gs, ws_)))
                grid[i, j, k] = chi + logv
                Egrid[i, j, k] = coef[nS]
                vEgrid[i, j, k] = cov[nS, nS]
                if best is None or grid[i, j, k] < best[0]:
                    best = (grid[i, j, k], i, j, k, coef)
    return grid, Egrid, vEgrid, best


def main():
    scans = []
    for tag in K.DELAY_TAGS:
        d = K.load(tag)
        s1 = K.load_result(tag, "s1")
        scans.append({"tag": tag, "t": d["tau"], "th": d["X"] * K.ROT_DEG_PER_V,
                      "sig": s1["noise_per_point_deg"], "s5": K.load_result(tag, "s5")})
    c = np.mean([s["s5"]["tau0_fs"] for s in scans])
    t0s = np.arange(c - 120, c + 120 + 1e-9, 5.0)
    ws = np.arange(80.0, 400.0 + 1e-9, 10.0)
    As = np.arange(0.0, 4.0 + 1e-9, 0.02)
    grid, Eg, vEg, (nll0, i, j, k, coef) = joint_fit(scans, t0s, ws, As)
    Es = np.linspace(Eg[i, j, k] - 6 * np.sqrt(vEg[i, j, k]), Eg[i, j, k] + 6 * np.sqrt(vEg[i, j, k]), 241)
    prof_E = np.array([np.min(grid + (e - Eg) ** 2 / vEg) for e in Es])
    t0, w, A, E = float(t0s[i]), float(ws[j]), float(As[k]), float(coef[-1])
    joint = {"tau0_fs": t0, "tau0_68": S5.interval(t0s, grid.min(axis=(1, 2))),
             "w_fs": w, "w_68": S5.interval(ws, grid.min(axis=(0, 2))),
             "effect_deg": E, "effect_68": S5.interval(Es, prof_E),
             "effect_dnll_zero": float(np.min(grid + Eg ** 2 / vEg) - nll0),
             "fringe_amp_deg": A, "fringe_amp_68": S5.interval(As, grid.min(axis=(0, 1))),
             "pedestals_deg": {s["tag"]: float(p) for s, p in zip(scans, coef[:-1])},
             "stage_position_mm": K.X_REF_MM + (t0 - K.X_REF_FS) * K.C_UM_PER_FS / 2 / 1000.0}
    joint["contrast"] = abs(E) / max(A, 1e-12)
    print("joint (146+235): tau0 {:.0f} fs {}, FWHM {:.0f} fs {}, effect {:+.3f} deg {} "
          "(-2lnL vs 0: {:.1f}), fringe {:.2f} deg {}, contrast {:.2f}; stage {:.4f} mm".format(
              t0, joint["tau0_68"], w, joint["w_68"], E, joint["effect_68"], joint["effect_dnll_zero"],
              A, joint["fringe_amp_68"], joint["contrast"], joint["stage_position_mm"]))

    # locked reference with the same envelope
    d6 = K.load("626")
    s16 = K.load_result("626", "s1")
    th6 = d6["X"] * K.ROT_DEG_PER_V
    g6 = K.gauss(d6["tau"], t0, w)
    nll6, P6, E6, vE6 = S5.nll_grid(d6["tau"], th6, s16["noise_per_point_deg"], np.array([t0]),
                                    np.array([w]), As)
    k6 = int(np.argmin(nll6[0, 0]))
    locked = {"effect_deg": float(E6[0, 0, k6]), "effect_err_deg": float(np.sqrt(vE6[0, 0, k6])),
              "fringe_amp_deg": float(As[k6]), "fringe_amp_68": S5.interval(As, nll6[0, 0]),
              "pedestal_deg": float(P6[0, 0, k6]),
              "fringe_ratio_to_delay_scans": float(As[k6] / A) if A else None}
    print("locked 626 at the same envelope: effect {:+.3f} +- {:.3f} deg, fringe {:.2f} deg ({:.0%} of the "
          "delay scans), pedestal {:.3f} deg".format(locked["effect_deg"], locked["effect_err_deg"],
                                                    locked["fringe_amp_deg"],
                                                    locked["fringe_ratio_to_delay_scans"] or 0,
                                                    locked["pedestal_deg"]))

    # pooled binned profile
    edges = np.arange(t0 - 400, t0 + 400 + 1e-9, BIN_FS)
    tb = 0.5 * (edges[1:] + edges[:-1])
    pooled = [np.concatenate([s["th"][(s["t"] >= a) & (s["t"] < b)] - joint["pedestals_deg"][s["tag"]]
                              for s in scans]) for a, b in zip(edges[:-1], edges[1:])]
    nb = np.array([len(p) for p in pooled])
    mean_b = np.array([p.mean() if len(p) else np.nan for p in pooled])
    rms_b = np.array([np.sqrt(np.mean((p - E * K.gauss(tc, t0, w)) ** 2)) if len(p) else np.nan
                      for p, tc in zip(pooled, tb)])

    # simulation
    sim = None
    sp = K.HERE / "sim" / "sim_result.json"
    if sp.exists():
        sim = json.load(open(sp))
        tr = sim.get("trace_1d") or []
        d3 = sim.get("tau0_3d")
        b1 = (sim.get("bracket_1d") or {}).get("best", {}).get("125")
        scale = (abs(d3["effect_deg"]) / abs(b1["effect_deg"]) if d3 and b1 else None,
                 d3["fringe_amp_deg"] / b1["fringe_amp_deg"] if d3 and b1 else None)
        sim_summary = {"tau0_1d": b1, "tau0_3d": d3, "scale_3d_over_1d": scale,
                       "bracket_1d": sim.get("bracket_1d")}
    out = {"joint_delay_scans": joint, "locked_626": locked,
           "per_scan": {s["tag"]: s["s5"] for s in scans},
           "pooled_profile": {"tau_fs": tb.tolist(), "n": nb.tolist(), "mean_deg": mean_b.tolist(),
                              "rms_about_effect_deg": rms_b.tolist()},
           "simulation": sim_summary if sim else None,
           "calibration": K.GAIN_NOTE, "intensity_W_cm2": K.I_PEAK_W_CM2}
    K.save_result("combined", "s6", out)

    # ---- figure 1: overlay --------------------------------------------------------------- #
    plt = K.style()
    cols = {"146": K.BLUE, "235": K.ORANGE, "626": K.AQUA}
    fig, axs = plt.subplots(2, 1, figsize=(10, 8.5))
    for ax, zoom in zip(axs, (False, True)):
        for tag in K.DATASETS:
            d = K.load(tag)
            m = np.abs(d["tau"] - t0) < 400 if zoom else np.ones(len(d["tau"]), bool)
            ax.plot(d["tau"][m], (d["X"] * K.ROT_DEG_PER_V)[m], color=cols[tag], marker="o", ms=3.5 if zoom else 2.5,
                    lw=1.0, label="{} — {}".format(tag, K.DATASETS[tag]["label"]))
        ax.axvline(t0, color=K.INK2, lw=0.8, ls=":")
        ax.set_ylabel("{} (deg)".format(K.THETA))
        ax.set_xlabel("pump₂ delay τ (fs, common stage axis)")
    axs[0].set_title("All scans on one delay axis — overlap at τ₀ = {:.0f} fs (stage {:.4f} mm)".format(
        t0, joint["stage_position_mm"]))
    axs[0].legend(loc="upper right", fontsize=8)
    axs[1].set_title("Zoom: the envelope reproduces, the fine structure does not (fringe phase)")
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s6_overlay.png", "combined")

    # ---- figure 2: joint decomposition + simulation -------------------------------------- #
    tf = np.linspace(t0 - 400, t0 + 400, 4001)
    gf = K.gauss(tf, t0, w)
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 11), sharex=True)
    ax = axs[0]
    for s in scans:
        m = np.abs(s["t"] - t0) < 400
        ax.plot(s["t"][m], s["th"][m] - joint["pedestals_deg"][s["tag"]], color=cols[s["tag"]],
                marker="o", ms=4, lw=0.8, label="{} − pedestal".format(s["tag"]))
    ax.fill_between(tf, E * gf - A * gf, E * gf + A * gf, color=K.MAGENTA, alpha=0.15, lw=0,
                    label="effect ± fringe amplitude (joint fit)")
    ax.plot(tf, E * gf, color=K.INK, lw=1.8, label="effect (joint fit)")
    ax.axhline(0, color=K.INK2, lw=0.8)
    ax.set_ylabel("{} − pedestal (deg)".format(K.THETA))
    ax.set_title("1 · both delay scans and the joint fit: fringe {:.2f}° [{:.2f}, {:.2f}], FWHM {:.0f} fs".format(
        A, *joint["fringe_amp_68"], w))
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[1]
    ok = nb > 0
    se_b = np.array([np.sqrt((np.mean([s["sig"] for s in scans]) ** 2 + 0.5 * (A * K.gauss(tc, t0, w)) ** 2) / max(n, 1))
                     for tc, n in zip(tb, nb)])
    ax.errorbar(tb[ok], mean_b[ok], yerr=se_b[ok], color=K.BLUE, marker="o", ms=4, lw=0, elinewidth=1,
                capsize=2, label="pooled bin mean (k = 0)")
    ax.plot(tf, E * gf, color=K.ORANGE, lw=1.8,
            label="effect: {:+.2f}° [{:+.2f}, {:+.2f}]".format(E, *joint["effect_68"]))
    ax.axhline(0, color=K.INK2, lw=0.8)
    ax.set_ylabel("net {} (deg)".format(K.THETA))
    ax.set_title("2 · net rotation: pooled {:.0f} fs bins, fringe averaged over both scans' phases".format(BIN_FS))
    ax.legend(loc="upper right", fontsize=8)

    ax = axs[2]
    ax.plot(tb[ok], rms_b[ok], color=K.MAGENTA, marker="o", ms=4, lw=0, label="pooled bin rms about the effect")
    ax.plot(tf, np.sqrt(np.mean([s["sig"] for s in scans]) ** 2 + 0.5 * (A * gf) ** 2), color=K.MAGENTA, lw=1.4,
            label="model: √(σ² + ½A²G²)")
    ax.set_ylabel("rms (deg)")
    ax.set_xlabel("pump₂ delay τ (fs, common stage axis)")
    ax.set_title("3 · the fringe: rms of the random-phase fringe under the same envelope")
    if sim and sim.get("trace_1d"):
        # SHAPE comparison: the simulated fringe is scaled to the measured one, and the simulated
        # effect by the SAME factor, so the plot shows the simulated effect-to-fringe ratio
        tr = sim["trace_1d"]
        ts = np.array([r["tau_fs"] for r in tr])
        es = np.array([r["effect_deg"] for r in tr])
        fs = np.array([r["fringe_amp_deg"] for r in tr])
        k = A / fs.max()
        axs[2].plot(t0 + ts, np.sqrt(np.mean([s["sig"] for s in scans]) ** 2 + 0.5 * (k * fs) ** 2),
                    color=K.AQUA, lw=1.4, ls="--", label="simulated fringe (as-built, 1D), scaled ×{:.0f}".format(k))
        axs[1].plot(t0 + ts, k * es, color=K.AQUA, lw=1.4, ls="--",
                    label="simulated effect, same ×{:.0f} (sim effect/fringe = {:.2f})".format(
                        k, np.max(np.abs(es)) / fs.max()))
        axs[1].legend(loc="upper right", fontsize=8)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    K.footnote(fig)
    K.save(fig, "s6_joint.png", "combined")


if __name__ == "__main__":
    main()
