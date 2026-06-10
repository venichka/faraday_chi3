#!/usr/bin/env python
"""Phase-1 chi5 refinement: dense FDTD-light Sobol scan over geometries CLOSE to an existing
design (SiN best_absolute / SiC L3.2um), per the agreed plan. Writes the top-k geometries +
operating-point cavity_modes.json (pipeline format) for the TCMT(FaradayJL)+1D+3D-FDTD gate.

Run (under meep-mpi or any numpy+scipy env):
  python chi5_optimization/run_phase1.py --material sin --samples 1024
  python chi5_optimization/run_phase1.py --material sic --samples 1024
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
sys.path.insert(0, str(HERE))
import optimize as O   # noqa: E402

MATERIALS = {
    "sin": {"dir": "SiN_optimizations/best_absolute"},
    "sic": {"dir": "SiC_optimizations/sic_L3p2um"},
}


def write_candidate(outdir, rank, r):
    d = outdir / f"cand{rank:02d}"
    d.mkdir(parents=True, exist_ok=True)
    json.dump(r["geometry"], open(d / "geometry.json", "w"), indent=2)
    f = r["freqs"]
    delta = f["pump1"] - f["pump2"]
    modes = {
        "probe": {"frequency": f["probe"], "lambda_um": 1.0 / f["probe"]},
        "pump1": {"frequency": f["pump1"], "lambda_um": 1.0 / f["pump1"]},
        "pump2": {"frequency": f["pump2"], "lambda_um": 1.0 / f["pump2"]},
        "sidebands": {"frequency_plus": f["sb_plus"], "frequency_minus": f["sb_minus"],
                      "delta_frequency": delta,
                      "pump_separation_um": abs(1.0 / f["pump2"] - 1.0 / f["pump1"])},
    }
    json.dump(modes, open(d / "cavity_modes.json", "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", required=True, choices=("sin", "sic"))
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--scope", default="regime", choices=("local", "regime"))
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base_dir = MEEP / MATERIALS[args.material]["dir"]
    ex = json.load(open(base_dir / "tcmt_derivation_analysis/tcmt_extracted_params_derivation.json"))
    chi = float(ex["material_constants"]["chi_iso_meep"])

    t0 = time.time()
    top, (bs, bf, bp) = O.run(base_dir / "geometry.json", chi,
                              n_samples=args.samples, scope=args.scope,
                              seed=args.seed, topk=args.topk)
    dt = time.time() - t0

    outdir = HERE / "phase1" / args.material
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "material": args.material, "samples": args.samples, "scope": args.scope,
        "seconds": dt, "n_valid_top": len(top),
        "baseline": {"fom_rotation": bs["fom_rotation"] if bs else None,
                     "freqs": bf, "params": {"n_pairs": bp[0], "t_hi": bp[1],
                                             "t_lo": bp[2], "L_cav": bp[3]}},
        "top": [{k: r[k] for k in ("params", "freqs", "fom_rotation", "fom_ellipticity",
                                   "B1", "B2", "Bs", "Qprobe")} for r in top],
    }
    json.dump(summary, open(outdir / "phase1_summary.json", "w"), indent=2)
    for rank, r in enumerate(top):
        write_candidate(outdir, rank, r)

    nm = lambda x: 1000.0 / x
    base_fom = bs["fom_rotation"] if bs else float("nan")
    print(f"[{args.material}] {args.samples} candidates in {dt:.0f}s ({dt/args.samples:.2f}s ea), "
          f"{len(top)} in top -> {outdir}", flush=True)
    print(f"[{args.material}] baseline FoM={base_fom:.3e}  probe={nm(bf['probe']):.0f}nm "
          f"pumps={nm(bf['pump1']):.0f}/{nm(bf['pump2']):.0f}nm D={bf['pump1']-bf['pump2']:.4f}")
    for rank, r in enumerate(top[:5]):
        p, f = r["params"], r["freqs"]
        print(f"[{args.material}] #{rank} FoM={r['fom_rotation']:.3e} ({r['fom_rotation']/base_fom:.2f}x)  "
              f"n={p['n_pairs']} tH={p['t_hi']:.3f} tL={p['t_lo']:.3f} L={p['L_cav']:.3f}  "
              f"probe={nm(f['probe']):.0f}nm pumps={nm(f['pump1']):.0f}/{nm(f['pump2']):.0f}nm "
              f"D={f['pump1']-f['pump2']:.4f} Qs={r['Qprobe']:.0f}")


if __name__ == "__main__":
    main()
