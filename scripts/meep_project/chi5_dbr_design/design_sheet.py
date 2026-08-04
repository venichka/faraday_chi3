#!/usr/bin/env python
"""Emit a fabrication-ready spec sheet for the Stage-2 finalists.

Writes docs/design_sheets.md: for each finalist, the full deposition stack layer by layer
(material, thickness in nm, running total), the operating point in wavelengths the lab can dial
in, and the simulated performance.  Everything a PECVD run and an optical table need, with no
Meep units.

  python chi5_dbr_design/design_sheet.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import s3_validate as S3  # noqa: E402

S2 = HERE / "runs" / "s2_fdtd" / "s2_result.json"
DOCS = HERE / "docs"


def stack_rows(geom):
    """(index, material, thickness_nm, cumulative_nm) from the incident face inward."""
    rows, z = [], 0.0
    for l in geom["mirrors"]["left"]:
        z += float(l["thk_um"]) * 1000
        rows.append((len(rows) + 1, l["mat"], float(l["thk_um"]) * 1000, z))
    z += float(geom["cavity"]["L_um"]) * 1000
    rows.append((len(rows) + 1, geom["cavity"]["mat"] + "  (CAVITY)",
                 float(geom["cavity"]["L_um"]) * 1000, z))
    for l in geom["mirrors"]["right"]:
        z += float(l["thk_um"]) * 1000
        rows.append((len(rows) + 1, l["mat"], float(l["thk_um"]) * 1000, z))
    return rows


def _load(name):
    p = HERE / "runs" / "s3_validate" / name
    return json.load(open(p)) if p.exists() else {}


def main():
    res = json.load(open(S2))
    by = {r["label"]: r for r in res["ranking"]}
    fin = S3.load_finalists(S2)
    base_theta = abs(by["baseline"]["best"]["theta_chi5_deg"])
    inten, tol, d3 = _load("intensity_result.json"), _load("tolerance_result.json"), \
        _load("3d_result.json")

    out = ["# Fabrication spec sheets — chi5 DBR design campaign finalists", "",
           "Generated from `runs/s2_fdtd/s2_result.json` (1D FDTD, carrier-averaged,",
           "pulse-integrated objective at I_pump = 1e12 W/cm^2, 100 fs intensity FWHM).", "",
           "All stacks are SiN/SiO2 on an SiO2 substrate, deposited from the incident (air) face",
           "inward. Constraints honoured: <= 6 mirror pairs per side, every layer >= 80 nm,",
           "total stack <= 12 um.", "",
           "**Read the contrast column.** `theta` is how much rotation the design makes;",
           "`contrast = theta / fringe` is whether it can be separated from the coherent carrier",
           "fringe without an interferometrically stable delay line. They do not rank together.",
           ""]

    out += ["## Summary", "",
            "| design | theta_chi5 | vs fabricated | contrast | DoLP | probe (nm) | pump1/pump2 (nm) | Delta (1/um) | stack (um) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for label, geom, op in fin:
        r = by[label]
        t = abs(r["best"]["theta_chi5_deg"])
        f = r["best"]["theta_fringe_amp_deg"]
        out.append("| {} | {:.5f}° | {:.2f}× | **{:.2f}** | {:.3f} | {:.1f} | {:.1f} / {:.1f} | "
                   "{:.4f} | {:.2f} |".format(
                       label, t, t / base_theta, t / f, r["best"]["dolp"],
                       1000.0 / op["probe"], 1000.0 / op["pump1"], 1000.0 / op["pump2"],
                       op["delta"], r["params"]["stack_um"]))
    out.append("")

    for label, geom, op in fin:
        r = by[label]
        p = r["params"]
        t = abs(r["best"]["theta_chi5_deg"])
        f = r["best"]["theta_fringe_amp_deg"]
        out += ["---", "", "## {}{}".format(
            label, "  (the fabricated reference)" if label == "baseline" else ""), "",
            "* mirror pairs: **{} left / {} right**{}".format(
                p["n_left"], p["n_right"],
                "  (asymmetric)" if p["n_left"] != p["n_right"] else ""),
            "* SiN layer **{:.1f} nm**, SiO2 layer **{:.1f} nm**  (t_lo/t_hi = **{:.2f}**)".format(
                p["t_hi"] * 1000, p["t_lo"] * 1000, p["t_lo"] / p["t_hi"]),
            "* cavity (SiN): **{:.1f} nm**".format(p["L_cav"] * 1000),
            "* total deposited: **{:.3f} um**".format(p["stack_um"]),
            "",
            "**Operating point** — probe **{:.1f} nm**, pumps **{:.1f} / {:.1f} nm** "
            "(separation {:.1f} nm, Delta = {:.4f} /um), balanced sigma+ sigma-, "
            "counter-rotating.".format(
                1000.0 / op["probe"], 1000.0 / op["pump1"], 1000.0 / op["pump2"],
                abs(1000.0 / op["pump2"] - 1000.0 / op["pump1"]), op["delta"]),
            "",
            "**Simulated (1D)** — theta_chi5 = **{:.5f}°** ({:.2f}× the fabricated design), "
            "carrier-fringe amplitude {:.5f}°, contrast **{:.2f}**, DoLP {:.3f}.".format(
                t, t / base_theta, f, t / f, r["best"]["dolp"]),
            ""]

        # --- Stage-3 validation, where available -------------------------------------- #
        val = []
        d = inten.get(label)
        if d:
            sl = ", ".join("{:.2f}".format(s) for _a, _b, s in d["local_slopes"])
            hi = d["rows"][-1]
            val.append("* **intensity scaling** local log-log slopes {} "
                       "(chi5 => 2); {:.3f}° at {:.1e} W/cm² (DoLP {:.3f})".format(
                           sl, hi["theta"], hi["I"], hi["dolp"]))
        d = (tol.get(label) or {}).get("sigmas", {})
        if d:
            for sig in sorted(d, key=float):
                s = d[sig]
                val.append("* **tolerance sigma = {} nm**: median {:.0f}%, 10th pct {:.0f}%, "
                           "worst {:.0f}% of nominal (n={})".format(
                               sig, 100 * s["median_rel"], 100 * s["p10_rel"],
                               100 * s["min_rel"], s["n"]))
        d = d3.get(label)
        if d and d.get("theta_3d") is not None:
            val.append("* **3D**: theta_chi5 = **{:.5f}°** ({:.2f}× the 1D value), "
                       "contrast {:.2f}, DoLP {:.3f}".format(
                           d["theta_3d"], d["ratio"] or float("nan"),
                           d["theta_3d"] / max(d.get("fringe_3d") or 1e-12, 1e-12),
                           d["dolp_3d"]))
        if val:
            out += ["**Stage-3 validation**", ""] + val + [""]

        out += ["| # | material | thickness (nm) | cumulative (nm) |", "|---|---|---|---|"]
        for i, mat, thk, cum in stack_rows(geom):
            out.append("| {} | {} | {:.1f} | {:.1f} |".format(i, mat, thk, cum))
        out.append("")

    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "design_sheets.md"
    path.write_text("\n".join(out))
    print("->", path)


if __name__ == "__main__":
    main()
