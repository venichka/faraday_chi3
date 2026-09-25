#!/usr/bin/env python
"""What a bandpass in the probe arm does to the effect and the fringe: tau = 0, 4 carrier phases
plus a pump-off reference, with the per-bin Stokes output, for two cases:

  sin   the as-built SIN51 stack at the lab point (probe 800, pumps 1620 delayed / 1560), 1e12, 125 fs
  sic   the SiC L = 4.8 um sample at its recommended point (794.2 / 1515.2 / 1569.9), 1e11, 100 fs

The readout band is then narrowed in post-processing (any subset of the 15 bins, 2.0 nm apart,
covering +-18 nm): full band, +-6, +-4 (~ a 10 nm filter) and +-2 nm. For each band: the carrier-
averaged rotation (k = 0), the fringe amplitude (k = +-1), the contrast, and the both-pumps-chopped
observable [V-H](both) - [V-H](none) normalised by S0.

  python sim_filter.py --workers 10        # then aggregates
  python sim_filter.py --aggregate-only
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import common as K
import sim_labpoint as SL

C = K.dbr_harness()
HERE = Path(__file__).resolve().parent
OUT = HERE / "sim" / "runs_filter"
MEEP = HERE.parents[1] / "meep_project"
SIC_FLAGS = ["--cavity-material", "sic", "--cavity-fit", "sic.csv", "--cavity-n2", "5e-18"]
BANDS = {"full ±18 nm": 15, "±6 nm": 7, "±4 nm (10 nm filter)": 5, "±2 nm": 3}


def cases():
    fin = {f["label"]: f for f in json.load(open(MEEP / "chi5_sic_samples/runs/s3_finalists/finalists.json"))}
    op = fin["L4p8_nowbest"]["op"]
    sic_geom = json.load(open(MEEP / "chi5_sic_samples/runs/s3_finalists/3d/L4p8_nowbest/s0/geometry.json"))
    return {
        "sin": {"geom": SL.geom("best"), "freqs": SL.freqs(), "I": SL.I_PEAK,
                "label": SL.label_of(SL.TRACE_FWHM), "extra": []},
        "sic": {"geom": sic_geom, "freqs": {"probe": op["probe"], "pump1": op["pump1"], "pump2": op["pump2"]},
                "I": 1e11, "label": C.PULSE_LABEL_FS, "extra": SIC_FLAGS},
    }


def jobs():
    js = []
    for name, c in cases().items():
        for sub, t in enumerate(C.subsample_taus(c["freqs"]["pump1"], C.SUBSAMPLES, 0.0)):
            js.append((name, "s%d" % sub, t, c["I"]))
        js.append((name, "off", 0.0, 1e2))
    return js


def run(job):
    name, tag, t, I = job
    c = cases()[name]
    return C.run_case(OUT / name / tag, c["geom"], c["freqs"], tau_fs=t, pad_fs=SL.PAD_FS,
                      pump_intensity=I, pulse_label_fs=c["label"], extra=c["extra"])


def band_stokes(d, nb):
    b = d["probe_pulse_integrated_bins"]
    n = len(b["S0"])
    lo, hi = (n - nb) // 2, (n - nb) // 2 + nb
    return {k: float(np.mean(b[k][lo:hi])) for k in ("S0", "S1", "S2", "S3")}


def aggregate():
    res = {}
    for name in cases():
        subs = [json.load(open(OUT / name / ("s%d" % s) / "faraday_summary.json")) for s in range(C.SUBSAMPLES)
                if (OUT / name / ("s%d" % s) / "faraday_summary.json").exists()]
        offp = OUT / name / "off" / "faraday_summary.json"
        off = json.load(open(offp)) if offp.exists() else None
        if len(subs) < C.SUBSAMPLES or off is None:
            print(name, "incomplete:", len(subs), "phases, off", off is not None)
            continue
        res[name] = {}
        print("== {} ==".format(name))
        for bname, nb in BANDS.items():
            recs = [dict(band_stokes(d, nb), dolp=0.0, legacy=0.0) for d in subs]
            a = C.carrier_average(recs)
            so = band_stokes(off, nb)
            th_off = C.stokes_to_angles(so["S0"], so["S1"], so["S2"], so["S3"])[0]
            vmh_both = [-(r["S1"]) / r["S0"] for r in recs]                     # per phase
            vmh_off = -so["S1"] / so["S0"]
            chopped = np.mean(vmh_both) - vmh_off                                # both pumps chopped, k=0
            phi = 2 * np.pi * np.arange(len(vmh_both)) / len(vmh_both)
            chopped_fringe = 2 * abs(np.sum(np.array(vmh_both) * np.exp(-1j * phi))) / len(vmh_both)
            eff = a["theta_chi5_deg"] - th_off
            res[name][bname] = {"bins": nb, "effect_deg": float(eff), "fringe_deg": a["theta_fringe_amp_deg"],
                                "contrast": abs(eff) / max(a["theta_fringe_amp_deg"], 1e-12),
                                "S0_band_frac": float(np.mean([r["S0"] for r in recs]) / band_stokes(subs[0], 15)["S0"] * nb / 15),
                                "chopped_vmh_norm": float(chopped), "chopped_fringe_vmh_norm": float(chopped_fringe),
                                "dS0_over_S0": float(np.mean([r["S0"] for r in recs]) / so["S0"] - 1)}
            r_ = res[name][bname]
            print("  {:22s} effect {:+.5f}  fringe {:.5f}  contrast {:.3f} | chopped (V-H)/S0: k0 {:+.2e} fringe {:.2e} | "
                  "dS0/S0 {:+.3%} | band holds {:.0%} of the probe".format(
                      bname, r_["effect_deg"], r_["fringe_deg"], r_["contrast"], r_["chopped_vmh_norm"],
                      r_["chopped_fringe_vmh_norm"], r_["dS0_over_S0"], r_["S0_band_frac"]))
    json.dump(res, open(OUT / "filter_result.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if not a.aggregate_only:
        js = jobs()
        print(len(js), "sims", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(run, js))
    aggregate()


if __name__ == "__main__":
    main()
