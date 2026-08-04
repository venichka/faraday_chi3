#!/usr/bin/env python
"""Stage 1 -- wide analytic screen of the fabricable design space.

Purpose and, just as importantly, its LIMITS. The 2026-06 campaign established by direct test
that the analytic chi5 figure of merit CANNOT rank geometries against FDTD: the original
energy-normalized score came out anti-correlated (Spearman -0.92; proxy ~ L^-3 while FDTD
~ L^+1.2), and even the corrected v3 -- which does get the sign right (+0.84 on the L family) --
only *selects a good pool*, it does not *order* it (in the hybrid run the FDTD winner was
analytic rank 8/10, and analytic #1 was FDTD #5). It also cannot pick the operating point at
all, because the symmetry-break Re(Sigma) is a delicate difference of two nearly-cancelling arms.

So Stage 1 does exactly two jobs and no more:

  1. FEASIBILITY (hard filter, uses TMM only where it is validated: mode f0 to <0.7%, Q to the
     right order).  A geometry survives iff it is fabricable AND has a probe cavity mode inside
     an allowed probe window AND at least one resonant pump mode in the mid-IR band.  Most of
     the Sobol space fails this -- an arbitrary DBR simply has no mode at 800 nm.

  2. COARSE PRE-RANK (soft, to choose which survivors are worth FDTD).  chi5_score_v3 at the
     corrected 100 fs pulse, maximized over a small operating-point grid.  Treated as a pool
     selector, never as a ranking.

Stage 2 (1D FDTD, carrier-averaged) owns the real ranking and the operating point.

Design space -- 5 parameters, one more than any previous search here:
    n_left, n_right in [2, 6]   mirror pairs per side, ALLOWED TO DIFFER
    t_hi, t_lo      >= 80 nm    SiN / SiO2 layer thicknesses
    L_cav                       cavity length
subject to total stack <= 12 um.  Asymmetric mirrors are deliberate: a cavity whose response is
symmetric about w_s gives Re[Delta chi] = 0, i.e. pure ellipticity and ZERO net rotation, so
net rotation requires breaking the w_s +- Delta symmetry (isotropic_derivation / very_general).

  python chi5_dbr_design/s1_screen.py --n-samples 32768 --workers 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

sys.path.insert(0, str(C.MEEP / "chi5_optimization"))
import objective as OBJ  # noqa: E402
import optimize as OPT   # noqa: E402

OUT = HERE / "runs" / "s1_screen"

# Search bounds.  Deliberately wide: this is a from-scratch search, not a refinement of
# best_absolute (whose t_hi/t_lo = 0.2375/0.3442 sit comfortably inside).
BOUNDS = [(C.N_PAIRS_MIN - 0.49, C.N_PAIRS_MAX + 0.49),   # n_left  (rounded)
          (C.N_PAIRS_MIN - 0.49, C.N_PAIRS_MAX + 0.49),   # n_right (rounded)
          (C.T_LAYER_MIN, C.T_LAYER_MAX),                 # t_hi
          (C.T_LAYER_MIN, C.T_LAYER_MAX),                 # t_lo
          (C.L_CAV_MIN, C.L_CAV_MAX)]                     # L_cav

PRERANK_MAX_CENTERS = 3          # top-Q pump centers considered per geometry
MIN_PROBE_Q = 20.0               # below Q_cap ~ 22 the probe mode is not a mode in any useful sense
MIN_PUMP_Q = 10.0


def make_geometry(base, p):
    return C.build_geometry(base, int(round(p[0])), int(round(p[1])),
                            float(p[2]), float(p[3]), float(p[4]))


def screen_one(args):
    """Feasibility + coarse pre-rank for one candidate. Returns a record or None."""
    p, chi_iso = args
    base = C.load_base_geometry()
    geom = make_geometry(base, p)
    if C.fab_violations(geom):
        return {"status": "fab"}
    try:
        pr = C.probe_modes(geom)
        pu = C.pump_modes(geom)
    except Exception:
        return {"status": "tmm_error"}
    pr = [m for m in pr if m["Q"] >= MIN_PROBE_Q]
    pu = [m for m in pu if m["Q"] >= MIN_PUMP_Q]
    if not pr:
        return {"status": "no_probe_mode"}
    if not pu:
        return {"status": "no_pump_mode"}

    # --- coarse pre-rank over a small operating-point grid ------------------------------- #
    try:
        ctx = OBJ.make_ctx(geom)
    except Exception:
        return {"status": "ctx_error"}
    f_probe = pr[0]["freq"]
    best, best_op = None, None
    for m in pu[:PRERANK_MAX_CENTERS]:
        for d in C.DELTA_GRID:
            f1, f2 = m["freq"] + 0.5 * d, m["freq"] - 0.5 * d
            freqs = {"probe": f_probe, "pump1": f1, "pump2": f2,
                     "sb_plus": f_probe + d, "sb_minus": f_probe - d}
            try:
                s = OBJ.chi5_score_v3(geom, freqs, chi_iso, pulse_fs=C.PULSE_LABEL_FS, ctx=ctx)
            except Exception:
                continue
            if best is None or s["fom_rotation"] > best["fom_rotation"]:
                best = s
                best_op = {"probe": f_probe, "center": float(m["freq"]),
                           "pump1": f1, "pump2": f2, "delta": float(d),
                           "Q_probe": float(pr[0]["Q"]), "Q_pump": float(m["Q"])}
    if best is None:
        return {"status": "no_operating_point"}
    return {"status": "ok",
            "params": C.geometry_params(geom),
            "op": best_op,
            "fom": best["fom_rotation"],
            "fom_ellipticity": best["fom_ellipticity"],
            "ReSigma": best["ReSigma"], "ImSigma": best["ImSigma"],
            "buildup": best["buildup"], "L_interaction": best["L_interaction"],
            "probe_nm": 1000.0 / f_probe,
            "n_probe_modes": len(pr), "n_pump_modes": len(pu)}


FEASIBLE_COLS = ("n_left", "n_right", "t_hi", "t_lo", "L_cav", "stack_um")


def save_feasible(recs, path):
    """Compressed columnar dump of every feasible candidate (plots need it, git should not)."""
    cols = {k: np.array([r["params"][k] for r in recs], dtype=float) for k in FEASIBLE_COLS}
    cols["fom"] = np.array([r["fom"] for r in recs], dtype=float)
    cols["probe_nm"] = np.array([r["probe_nm"] for r in recs], dtype=float)
    cols["delta"] = np.array([r["op"]["delta"] for r in recs], dtype=float)
    np.savez_compressed(path, **cols)


def load_feasible(path):
    """-> list of records shaped like the ones the plots expect.

    NOTE the arrays are materialized ONCE: indexing an NpzFile re-decompresses the whole
    column on every access, so pulling scalars straight out of it inside a loop costs
    n_rows x n_cols decompressions (~200k here, minutes instead of milliseconds)."""
    with np.load(path) as d:
        cols = {k: d[k] for k in list(FEASIBLE_COLS) + ["fom", "probe_nm", "delta"]}
    n = len(cols["fom"])
    return [{"params": {k: float(cols[k][i]) for k in FEASIBLE_COLS},
             "fom": float(cols["fom"][i]), "probe_nm": float(cols["probe_nm"][i]),
             "op": {"delta": float(cols["delta"][i])}} for i in range(n)]


def select_pool(recs, topk):
    """Half the Stage-2 pool by proxy FoM, half by geometric diversity.

    WHY NOT ALL PROXY.  The analytic FoM has a documented history of getting geometry ranking
    exactly backwards on this very design space (Spearman -0.92 before the v2/v3 correction),
    and even v3 is only a coarse pool selector.  If it is wrong again, an all-proxy pool would
    hand FDTD a set of uniformly bad geometries and the campaign would learn nothing.  Filling
    half the pool by farthest-point sampling over the feasible set guarantees Stage 2 sees a
    spread of the actual design space, and -- as a free by-product -- measures whether the
    proxy has ANY skill, by comparing the two halves' FDTD outcomes.

    METHOD: farthest-point sampling on min-max-normalized [n_left, n_right, t_hi, t_lo, L]
    picks spread-out REPRESENTATIVES, then each contributes the best-FoM candidate in its
    Voronoi cell.  Taking the representatives themselves would fill the pool with the corners
    of the space (farthest-point always runs to extremes -- in testing it returned L ~ 1.1-1.5 um
    cavities, which the validated max|theta| ~ L^+1.2 trend says are poor).  Best-in-cell keeps
    the coverage while spending each FDTD slot on a locally sensible design.
    """
    if not recs:
        return []
    n_proxy = max(1, topk // 2)
    n_div = topk - n_proxy
    pool = [dict(r, pool="proxy") for r in recs[:n_proxy]]
    taken = set(range(n_proxy))
    keys = ("n_left", "n_right", "t_hi", "t_lo", "L_cav")
    X = np.array([[r["params"][k] for k in keys] for r in recs], dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    Xn = (X - lo) / np.where(hi - lo > 0, hi - lo, 1.0)

    # spread-out representatives, seeded from the proxy winners already in the pool
    reps = list(range(n_proxy))
    d = np.min(np.linalg.norm(Xn[:, None, :] - Xn[None, reps, :], axis=2), axis=1)
    for _ in range(n_div):
        j = int(np.argmax(d))
        if d[j] <= 0:
            break
        reps.append(j)
        d = np.minimum(d, np.linalg.norm(Xn - Xn[j], axis=1))

    # assign every feasible candidate to its nearest representative, take the best FoM per cell
    new_reps = reps[n_proxy:]
    if new_reps:
        assign = np.argmin(np.linalg.norm(Xn[:, None, :] - Xn[None, new_reps, :], axis=2), axis=1)
        for c in range(len(new_reps)):
            members = [i for i in np.where(assign == c)[0] if i not in taken]
            if not members:
                continue
            best = max(members, key=lambda i: recs[i]["fom"])
            taken.add(best)
            pool.append(dict(recs[best], pool="diverse"))
    return pool[:topk]


def validate_tmm_against_record():
    """Sanity: the TMM mode finder must reproduce the committed FDTD modes of best_absolute."""
    geom, modes = C.load_base_geometry(), C.load_base_modes()
    import tmm
    idx, layers = tmm.index_map(), tmm.build_layers(geom)
    print("  TMM vs committed FDTD modes (SiN best_absolute):")
    ok = True
    for name in ("probe", "pump1", "pump2"):
        f_fdtd = modes[name]["frequency"]
        m = tmm.find_mode(layers, idx, f_fdtd)
        if m is None:
            print("    {:6s} no TMM peak near {:.4f}".format(name, f_fdtd))
            ok = False
            continue
        dl = (m["lambda_um"] - 1.0 / f_fdtd) * 1000.0
        rel = abs(m["freq"] - f_fdtd) / f_fdtd * 100
        print("    {:6s} f_fdtd {:.4f}  f_tmm {:.4f}  ({:+.2f} nm, {:.2f}%)  Q_tmm {:6.1f}"
              .format(name, f_fdtd, m["freq"], dl, rel, m["Q"]))
        ok &= rel < 1.0
    print("    -> {}".format("OK" if ok else "MISMATCH -- investigate before trusting the screen"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT
    out.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1 | analytic feasibility screen + coarse pre-rank ===")
    print("  fab limits: {}-{} pairs/side, layers >= {:.0f} nm, stack <= {:.0f} um".format(
        C.N_PAIRS_MIN, C.N_PAIRS_MAX, C.T_LAYER_MIN * 1000, C.STACK_MAX_UM))
    print("  probe windows {} um, pump band {} um, Delta grid {}".format(
        C.PROBE_WINDOWS, C.PUMP_BAND, C.DELTA_GRID))
    print("  pulse: label {:.4f} fs = {:.0f} fs intensity FWHM, fwidth {:.6f} /um".format(
        C.PULSE_LABEL_FS, C.PULSE_INTENSITY_FWHM_FS, C.FWIDTH))
    validate_tmm_against_record()

    pts = OPT.sobol(BOUNDS, args.n_samples, args.seed)
    print("\n  screening {} Sobol candidates on {} workers...".format(len(pts), args.workers))
    t0 = time.time()
    recs, counts = [], {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(screen_one, (p, 1.0)) for p in pts]
        for k, fu in enumerate(as_completed(futs)):
            r = fu.result()
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "ok":
                recs.append(r)
            if (k + 1) % 2000 == 0:
                print("    {}/{} ({:.0f}s, {} feasible)".format(
                    k + 1, len(pts), time.time() - t0, len(recs)), flush=True)
    print("  screen done in {:.0f}s".format(time.time() - t0))
    print("\n  outcome counts:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print("    {:22s} {:6d}  ({:.1f}%)".format(k, v, 100 * v / len(pts)))

    recs.sort(key=lambda r: r["fom"], reverse=True)
    top = select_pool(recs, args.topk)

    print("\n  Stage-2 pool: {} geometries ({} proxy-ranked + {} diversity-sampled)".format(
        len(top), sum(1 for r in top if r["pool"] == "proxy"),
        sum(1 for r in top if r["pool"] == "diverse")))
    print("  (pool selector only -- FDTD in Stage 2 does the real ranking):")
    print("  {:>4s} {:>8s} {:>3s} {:>3s} {:>8s} {:>8s} {:>8s} {:>7s} {:>9s} {:>7s} {:>6s}".format(
        "#", "pool", "nL", "nR", "t_hi", "t_lo", "L_cav", "stack", "fom", "probe", "Delta"))
    for i, r in enumerate(top):
        p = r["params"]
        print("  {:>4d} {:>8s} {:>3d} {:>3d} {:>8.4f} {:>8.4f} {:>8.3f} {:>7.2f} {:>9.2e} "
              "{:>7.1f} {:>6.3f}".format(
                  i, r["pool"], p["n_left"], p["n_right"], p["t_hi"], p["t_lo"], p["L_cav"],
                  p["stack_um"], r["fom"], r["probe_nm"], r["op"]["delta"]))

    # baseline for reference, scored the same way
    base = C.load_base_geometry()
    bp = C.geometry_params(base)
    base_rec = screen_one((np.array([bp["n_left"], bp["n_right"], bp["t_hi"], bp["t_lo"],
                                     bp["L_cav"]]), 1.0))
    if base_rec.get("status") == "ok":
        print("  {:>4s} {:>8s} {:>3d} {:>3d} {:>8.4f} {:>8.4f} {:>8.3f} {:>7.2f} {:>9.2e} "
              "{:>7.1f} {:>6.3f}".format(
                  "base", "control", bp["n_left"], bp["n_right"], bp["t_hi"], bp["t_lo"],
                  bp["L_cav"], bp["stack_um"], base_rec["fom"], base_rec["probe_nm"],
                  base_rec["op"]["delta"]))
        print("  NOTE the pool's proxy FoM runs ~{:.0f}x the baseline's. The 2026-06 campaign saw "
              "the same\n       pattern and 1D FDTD then INVERTED it. Treat as unranked until "
              "Stage 2.".format(top[0]["fom"] / max(base_rec["fom"], 1e-30)))

    res = {"config": {"n_samples": args.n_samples, "seed": args.seed, "bounds": BOUNDS,
                      "pulse_label_fs": C.PULSE_LABEL_FS,
                      "delta_grid": C.DELTA_GRID,
                      "fab": {"n_pairs": [C.N_PAIRS_MIN, C.N_PAIRS_MAX],
                              "t_layer_min": C.T_LAYER_MIN, "stack_max_um": C.STACK_MAX_UM}},
           "counts": counts, "n_feasible": len(recs),
           "baseline": base_rec, "top": top}
    path = out / "s1_result.json"
    json.dump(res, open(path, "w"), indent=2)

    # The full feasible set is ~22k records (~18 MB as JSON) -- far too big to track, but the
    # plots need it. Keep the tracked result small and put the bulk in a compressed side file.
    save_feasible(recs, out / "s1_feasible.npz")
    print("\n-> {}  ({} feasible records in s1_feasible.npz)".format(path, len(recs)))


if __name__ == "__main__":
    main()
