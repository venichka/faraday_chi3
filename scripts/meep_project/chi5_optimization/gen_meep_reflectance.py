#!/usr/bin/env python
"""Generate Meep (FDTD) reflectance R(lambda) for a geometry -> CSV, to overlay on TMM
for visual validation. Reuses the existing pipeline (debug_reflectance + the same fit
materials the modes were extracted with). Run under meep-mpi.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
sys.path.insert(0, str(MEEP))

from optimize_cavity_geometry import debug_reflectance  # noqa: E402
from extract_tcmt_params_derivation import _build_materials  # noqa: E402
from nonlinear_materials import (  # noqa: E402
    canonical_high_index_material, resolve_high_index_index, resolve_high_index_kappa,
)


def materials_for(high: str, sin_csv: str, fit_poles: int):
    mat = canonical_high_index_material(high)
    a = argparse.Namespace(
        materials="fit", high_index_material=mat,
        nH=resolve_high_index_index(None, mat), kH=resolve_high_index_kappa(None, mat),
        nL=1.45, kappa_ref_lambda=1.55,
        sin_fit=str(MEEP / sin_csv), sio2_fit=str(MEEP / "sio2.csv"),
        fit_window=(600, 2000), fit_poles=fit_poles,
    )
    return _build_materials(a)


def gen(geom_path, out_csv, high, sin_csv, fit_poles, res=80, nfreq=801, wl=(0.6, 2.0)):
    spec = json.load(open(geom_path))
    mats = materials_for(high, sin_csv, fit_poles)
    wl_um, R = debug_reflectance(spec, mats, resolution=res, nfreq=nfreq,
                                 decay_threshold=1e-7, wl_min=wl[0], wl_max=wl[1])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, np.column_stack([np.asarray(wl_um) * 1000.0, np.asarray(R)]),
               delimiter=",", header="wavelength_nm,R", comments="")
    print(f"wrote {out_csv}  (R {np.min(R):.3f}..{np.max(R):.3f})", flush=True)


if __name__ == "__main__":
    out = HERE / "validation"
    gen(MEEP / "SiN_optimizations/best_absolute/geometry.json",
        out / "meep_refl_sin_best_absolute.csv", "sin", "si3n4.csv", fit_poles=2)
    gen(MEEP / "SiC_optimizations/sic_L3p2um/geometry.json",
        out / "meep_refl_sic_L3p2um.csv", "sic", "sic.csv", fit_poles=3)
