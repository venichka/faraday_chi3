#!/usr/bin/env python
"""Single-pump runs: what each lock-in modulation scheme actually measures.

A pump-probe lock-in reads a DIFFERENCE of pump configurations, and the balanced-pump cancellation
of the direct chi3 rotation holds only for the two-pump total. With T = theta(both), a_p = theta(pump
p alone), all at tau = 0 and relative to the pumps-off run:

    one pump chopped (SIN51: the delayed one)    X = T - a_fixed
    both pumps on one chopper, bump over pedestal  = T - a_fixed   (a_fixed is the off-overlap pedestal)
    double modulation, product line              x = T - a_1 - a_2

Helicity: the simulator's pump1 is always sigma+ and pump2 sigma-, and --pump-imbalance 0 switches
pump2 off, so a single-pump run always has a sigma+ pump. For an isotropic stack the mirror through
the probe's 45 deg axis flips the helicity and the rotation, so a sigma- pump gives MINUS the sigma+
run at its wavelength. (The first version of this script took the sigma+ run as-is for the sigma-
pump; its chopped k = 0 of +0.0034 deg is corrected here to +0.0016 deg.)

  sin   SIN51 as-built at the lab point (sim pump1 = 1620 nm delayed, pump2 = 1560), 1e12, 125 fs
  sic   SiC L = 4.8 um at its recommended point (pump1 1515.2, pump2 1569.9), 1e11, 100 fs

The two-pump totals come from sim_filter.py (full band, carrier-averaged, off-subtracted).

  python sim_singlepump.py [--sample sin sic] [--workers 4]
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import common as K
import sim_filter as SF
import sim_labpoint as SL

C = K.dbr_harness()
OUT = Path(__file__).resolve().parent / "sim" / "runs_single"
FILTER = OUT.parent / "runs_filter"
# per sample: run name -> (which slot of the two-pump run this pump occupies, its frequency key)
RUNS = {"sin": {"only_1620": "pump1", "only_1560": "pump2"},
        "sic": {"sic_only_p1": "pump1", "sic_only_p2": "pump2"}}
SIGN = {"pump1": +1.0, "pump2": -1.0}      # pump2 is sigma- in the two-pump run


def run(job):
    sample, name = job
    c = SF.cases()[sample]
    slot = RUNS[sample][name]
    freqs = {"probe": c["freqs"]["probe"], "pump1": c["freqs"][slot], "pump2": c["freqs"]["pump2"]}
    return C.run_case(OUT / name / "s0", c["geom"], freqs, tau_fs=0.0, pad_fs=SL.PAD_FS,
                      pump_intensity=c["I"], pulse_label_fs=c["label"],
                      extra=list(c["extra"]) + ["--pump-imbalance", "0"])


def theta(path):
    r = C.read_case(path)
    return None if r is None else C.stokes_to_angles(r["S0"], r["S1"], r["S2"], r["S3"])[0]


def aggregate(samples):
    res = json.load(open(OUT / "single_result.json")) if (OUT / "single_result.json").exists() else {}
    flt = json.load(open(FILTER / "filter_result.json"))
    for sample in samples:
        th_off = theta(FILTER / sample / "off")
        a = {}
        for name, slot in RUNS[sample].items():
            th = theta(OUT / name / "s0")
            if th is None or th_off is None:
                print(sample, name, "missing")
                break
            a[slot] = SIGN[slot] * (th - th_off)
        else:
            full = flt[sample]["full ±18 nm"]
            T, fringe = full["effect_deg"], full["fringe_deg"]
            r = {"total_deg": T, "fringe_deg": fringe, "a_pump1_deg": a["pump1"], "a_pump2_deg": a["pump2"],
                 "theta_off_deg": th_off,
                 "chop_pump1_only_deg": T - a["pump2"], "chop_pump2_only_deg": T - a["pump1"],
                 "double_mod_deg": T - a["pump1"] - a["pump2"]}
            res[sample] = r
            print("{}: total {:+.5f}, pump1 alone {:+.5f}, pump2 alone {:+.5f} (off {:+.5f}) -> chop pump1 only "
                  "{:+.5f}, chop pump2 only {:+.5f}, double modulation {:+.5f}; fringe {:.5f}".format(
                      sample, T, a["pump1"], a["pump2"], th_off, r["chop_pump1_only_deg"],
                      r["chop_pump2_only_deg"], r["double_mod_deg"], fringe))
    for k in ("only_1620", "only_1560", "chopped_k0_deg"):     # superseded flat keys of the first version
        res.pop(k, None)
    json.dump(res, open(OUT / "single_result.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", nargs="+", default=["sin", "sic"], choices=list(RUNS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if not a.aggregate_only:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(run, [(s, n) for s in a.sample for n in RUNS[s]]))
    aggregate(a.sample)


if __name__ == "__main__":
    main()
