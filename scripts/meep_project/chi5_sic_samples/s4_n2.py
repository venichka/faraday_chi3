#!/usr/bin/env python
"""Stage 4 -- how the effect, the fringe and their contrast depend on the SiC n2.  (1D, carrier-averaged)

The report originally claimed contrast is "much less sensitive" to n2 "since effect and fringe
both grow with n2".  The fringe-vs-effect mechanism says otherwise: the fringe 2Re[A_sb A_s*] is
FIRST order in chi3 (~n2), the effect |A_sb|^2 is SECOND order (~n2^2), so contrast ~n2^1 at
small signal.  This stage measures it.

Sweeps the cavity n2 over the measured a-SiC film range (Opt. Lett. 49, 4389) at three frozen
finalists (I = 1e11), plus an equivalence control: at L4p8_nowbest, n2 = 5e-18 with I scaled by
n2/5e-18 -- tests whether n2 and I enter only as the product (the SiN mirrors' n2 is NOT scaled
in the n2 arm, so they need not be exactly equivalent).

  python chi5_sic_samples/s4_n2.py --part 0/2 --workers 30      # one slice per node
  python chi5_sic_samples/s4_n2.py --aggregate-only
"""
import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_sic as S  # noqa: E402
import common as C      # noqa: E402

OUT = HERE / "runs" / "s4_n2"
I_REF = 1e11
N2_REF = S.SIC_N2_M2_PER_W
N2S = [3.04e-18, 4.40e-18, 5.00e-18, 6.70e-18]    # PECVD a-SiC films (OL 49, 4389) + baseline
FINALISTS = ["L4p8_nowbest", "L4p8_legible_fallback", "L3p2_legible"]
EQUIV = "L4p8_nowbest"


def finalists():
    fin = json.load(open(HERE / "runs" / "s3_finalists" / "finalists.json"))
    return {f["label"]: f for f in fin}


def cases():
    fin = finalists()
    cs = []
    for lab in FINALISTS:
        for n2 in N2S:
            cs.append(("n2", lab, n2, I_REF))
    for n2 in N2S:
        if n2 != N2_REF:
            cs.append(("Ieq", EQUIV, N2_REF, I_REF * n2 / N2_REF))
    jobs = []
    for kind, lab, n2, I in cs:
        f = fin[lab]
        geom = S.sic_geometry(S.CAVITY_LENGTHS_UM[f["sample"]])
        for sub, tau in enumerate(C.subsample_taus(f["op"]["pump1"])):
            d = OUT / kind / lab / "n2_{:.2e}_I{:.3e}".format(n2, I) / "s{}".format(sub)
            jobs.append(dict(kind=kind, label=lab, n2=n2, I=I, sub=sub, tau=tau,
                             dir=d, geom=geom, op=f["op"]))
    return cs, jobs


def run(job):
    return S.run_case(job["dir"], job["geom"], job["op"], job["sub"], job["tau"],
                      C.RES_1D, C.DECAY_1D, C.PAD_FS, 1, job["I"],
                      extra=["--cavity-n2", repr(job["n2"])])   # argparse: last wins


def aggregate(cs):
    fin = finalists()
    rows = []
    for kind, lab, n2, I in cs:
        base = OUT / kind / lab / "n2_{:.2e}_I{:.3e}".format(n2, I)
        recs = [C.read_case(base / "s{}".format(s)) for s in range(C.SUBSAMPLES)]
        if any(r is None for r in recs):
            continue
        a = C.carrier_average(recs)
        th, fr = abs(a["theta_chi5_deg"]), a["theta_fringe_amp_deg"]
        rows.append(dict(kind=kind, label=lab, n2=n2, I=I, theta_deg=a["theta_chi5_deg"],
                         fringe_deg=fr, contrast=th / max(fr, 1e-12), dolp=a["dolp"],
                         eff_n2I=n2 * I / (N2_REF * I_REF)))
    print("{:5s} {:22s} {:>9s} {:>9s} {:>11s} {:>10s} {:>8s} {:>7s}".format(
        "kind", "finalist", "n2", "I", "|theta|", "fringe", "contr", "DoLP"))
    for r in rows:
        print("{kind:5s} {label:22s} {n2:9.2e} {I:9.2e} {t:11.6f} {fringe_deg:10.6f} "
              "{contrast:8.4f} {dolp:7.4f}".format(t=abs(r["theta_deg"]), **r))
    # power-law exponents vs n2 (n2 arm only), per finalist
    print("\nlog-log exponents vs n2 (least squares over the 4 films), I = 1e11:")
    fits = {}
    for lab in FINALISTS:
        rs = [r for r in rows if r["kind"] == "n2" and r["label"] == lab]
        if len(rs) < 3:
            continue
        x = [math.log(r["n2"]) for r in rs]
        out = {}
        for key in ("theta_deg", "fringe_deg", "contrast"):
            y = [math.log(abs(r[key])) for r in rs]
            mx, my = sum(x) / len(x), sum(y) / len(y)
            out[key] = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
        by_n2 = {r["n2"]: r for r in rs}
        if not all(n in by_n2 for n in (N2_REF, min(N2S), max(N2S))):
            continue                     # another slice is still running
        ref, lo, hi = by_n2[N2_REF], by_n2[min(N2S)], by_n2[max(N2S)]
        out["contrast_ratio_lo"] = lo["contrast"] / ref["contrast"]
        out["contrast_ratio_hi"] = hi["contrast"] / ref["contrast"]
        out["theta_ratio_lo"] = abs(lo["theta_deg"] / ref["theta_deg"])
        out["theta_ratio_hi"] = abs(hi["theta_deg"] / ref["theta_deg"])
        out["theta_1d_frozen"] = fin[lab]["theta_1d_deg"]
        out["contrast_1d_frozen"] = fin[lab]["contrast_1d"]
        fits[lab] = out
        print("  {:22s} p_theta {:.2f}  p_fringe {:.2f}  p_contrast {:.2f}   "
              "contrast x{:.3f} (3.04) .. x{:.3f} (6.70)   theta x{:.3f} .. x{:.3f}".format(
                  lab, out["theta_deg"], out["fringe_deg"], out["contrast"],
                  out["contrast_ratio_lo"], out["contrast_ratio_hi"],
                  out["theta_ratio_lo"], out["theta_ratio_hi"]))
    json.dump({"rows": rows, "fits": fits}, open(OUT / "n2_result.json", "w"), indent=2)
    print("-> {}".format(OUT / "n2_result.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    cs, jobs = cases()
    if not args.aggregate_only:
        i, n = (int(v) for v in args.part.split("/"))
        mine = jobs[i::n]
        print("{} of {} sims (part {})".format(len(mine), len(jobs), args.part), flush=True)
        t0, done = time.time(), 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for _ in as_completed([ex.submit(run, j) for j in mine]):
                done += 1
                print("  {}/{} ({:.0f}s)".format(done, len(mine), time.time() - t0), flush=True)
    aggregate(cs)


if __name__ == "__main__":
    main()
