#!/usr/bin/env python
"""Shared configuration for the FABRICATED SiC-cavity samples (2026-08-03).

The lab fabricated two samples that are the SiN `best_absolute` stack with **only the cavity
spacer replaced by SiC**, at two cavity lengths:

    Air | [SiN 237.5 / SiO2 344.2 nm] x3 | SiC cavity (3.2 or 4.8 um) | [SiO2 / SiN] x3 | SiO2

⚠️ The mirrors are UNCHANGED -- same materials, same thicknesses, 3 pairs per side. This is NOT
the 2026-06 `SiC_optimizations/sic_L3p2um` construction, which swapped every high-index layer to
SiC and therefore has a completely different stopband. Do not reuse that study's operating points.

Both media are nonlinear: the SiC cavity at n2 = 5e-18 m^2/W and the SiN mirrors at their own
n2 = 5e-19 m^2/W. That needs the `--cavity-material` extension added to faraday_meep_fp_circ.py
on 2026-08-03; before it, the simulator could only express a TWO-material stack.

Everything about the measurement -- carrier averaging, the pulse-duration label, the Delta cap
set by the probe readout band, the run harness -- is imported unchanged from
`chi5_dbr_design/common.py` so numbers stay directly comparable to the SiN campaign.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
MEEP = HERE.parent
sys.path.insert(0, str(MEEP / "chi5_dbr_design"))
sys.path.insert(0, str(MEEP / "chi5_optimization"))

import common as C  # noqa: E402  -- the SiN campaign's harness, reused verbatim

# ------------------------------------------------------------------ the two samples --- #
CAVITY_LENGTHS_UM = {"L3p2": 3.2, "L4p8": 4.8}

# SiC Kerr coefficient. ⚠️ Flagged "user-specified" in nonlinear_materials.py, NOT literature.
# theta_chi5 scales ~n2^2, so a 2x error here is a 4x error in the predicted rotation. Every
# reported number is linear in CHI5_N2_SCALE = (n2_actual / SIC_N2)^2 -- see the report.
SIC_N2_M2_PER_W = 5.0e-18
SIN_N2_M2_PER_W = 5.0e-19

# ⚠️ 2 Lorentz poles, NOT 3. The 2026-06 SiC study used 3 and the meep-fit-stability note says
# SiC "needs" 3, but that was for a different fit window. Measured over the window this campaign
# uses (600-2000 nm), a 2-pole fit of sic.csv reproduces the ellipsometry to <= 0.0001 in n at
# both 800 and 1550 nm with no NaN or unphysical epsilon (checked at 2, 3 and 4 poles). Keeping 2
# means the SiN mirrors are fitted EXACTLY as in the SiN campaign, so the numbers stay directly
# comparable -- `--fit-poles` is global, so bumping it for SiC would silently change the mirrors.
SIC_FIT_POLES = 2

# Only the DELTA from chi5_dbr_design.common.FDTD_FLAGS, which C.run_case already passes. These
# are appended after it, and argparse last-wins, so listing the shared flags again would work but
# would be a silent trap if the base list ever changed. Keep this list minimal and additive.
FDTD_FLAGS_SIC = [
    "--cavity-material", "sic",              # the CAVITY is SiC (the mirrors stay SiN)
    "--cavity-fit", "sic.csv",
    "--cavity-n2", repr(SIC_N2_M2_PER_W),
]

# ------------------------------------------------------------------ probe scenarios --- #
# The user asked for BOTH: what is achievable today, and what an extended probe source buys.
PROBE_WINDOWS_NOW = [(0.790, 0.810), (0.850, 0.950)]      # um -- achievable now
PROBE_WINDOWS_FUTURE = [(0.600, 0.900)]                   # um -- possible later
PROBE_BAND_SCAN = (0.600, 1.000)      # scan wider than either, so the cost of the limit is known

# Pumps: 1400-2000 nm and "pretty broad" (user). Wide enough that FWM matching
# (2 f_pump ~ f_probe) is reachable for any probe from ~700 to ~1000 nm.
PUMP_BAND_SIC = (1.40, 2.00)

# Transmission floor for calling a TMM peak a usable probe mode. The SiC cavities have several
# high-Q modes at 640-700 nm with T_peak 0.2-0.5 -- strongly reflective, weak transmitted probe.
T_PEAK_MIN = 0.15


def probe_scenario_windows(scenario: str) -> List[Tuple[float, float]]:
    if scenario == "now":
        return list(PROBE_WINDOWS_NOW)
    if scenario == "future":
        return list(PROBE_WINDOWS_FUTURE)
    if scenario == "scan":
        return [PROBE_BAND_SCAN]
    raise ValueError("scenario must be one of: now, future, scan")


def in_windows(lam_um: float, windows: Sequence[Tuple[float, float]]) -> bool:
    return any(lo <= lam_um <= hi for lo, hi in windows)


# ------------------------------------------------------------------ geometry --- #
def base_geometry() -> dict:
    """The fabricated SiN best_absolute stack (mirrors + cavity all SiN)."""
    return C.load_base_geometry()


def sic_geometry(L_cav_um: float) -> dict:
    """SiN/SiO2 mirrors UNCHANGED; only the cavity spacer becomes SiC at the given length.

    The SiC entry carries its constant index for tooling that reads the JSON directly; the FDTD
    run overrides it with the dispersive sic.csv fit via --cavity-fit, and TMM looks the label up
    in its own CSV table (tmm.CSV_FOR["SiC"]).
    """
    g = copy.deepcopy(base_geometry())
    g["materials"]["SiC"] = {"type": "Medium", "params": {"index": 2.56}}
    g["cavity"] = {"mat": "SiC", "L_um": float(L_cav_um)}
    return g


def all_samples() -> Dict[str, dict]:
    return {k: sic_geometry(L) for k, L in CAVITY_LENGTHS_UM.items()}


def stack_um(geom: dict) -> float:
    return C.stack_thickness_um(geom)


# ------------------------------------------------------------------ modes --- #
def _tmm():
    import tmm as _t
    return _t


def probe_modes(geom: dict, windows: Optional[Sequence[Tuple[float, float]]] = None,
                t_min: float = T_PEAK_MIN) -> List[dict]:
    """Cavity modes in the given probe windows, low wavelength first (NOT sorted by Q).

    ⚠️ `chi5_dbr_design.common.probe_modes` sorts by Q, which repeatedly picked the wrong mode
    in the SiN campaign. Order here is by wavelength and selection is by window.
    """
    t = _tmm()
    idx, layers = t.index_map(), t.build_layers(geom)
    ms: List[dict] = []
    for lo, hi in (windows if windows is not None else [PROBE_BAND_SCAN]):
        ms += t.find_modes_in_band(layers, idx, 1.0 / hi, 1.0 / lo, "SiO2")
    ms = [m for m in ms if m["T_peak"] >= t_min]
    ms.sort(key=lambda m: m["lambda_um"])
    # de-duplicate overlapping windows
    out: List[dict] = []
    for m in ms:
        if not out or abs(m["freq"] - out[-1]["freq"]) > 1e-6:
            out.append(m)
    return out


def pump_modes(geom: dict) -> List[dict]:
    t = _tmm()
    idx, layers = t.index_map(), t.build_layers(geom)
    ms = t.find_modes_in_band(layers, idx, 1.0 / PUMP_BAND_SIC[1], 1.0 / PUMP_BAND_SIC[0], "SiO2")
    ms.sort(key=lambda m: m["lambda_um"])
    return ms


def stopband(geom: dict, threshold: float = 0.8) -> Optional[Tuple[float, float]]:
    """(lambda_min, lambda_max) in um where R > threshold, or None."""
    t = _tmm()
    idx, layers = t.index_map(), t.build_layers(geom)
    f = np.linspace(1.0 / 2.8, 1.0 / 1.0, 4000)
    R, _ = t.spectrum(layers, idx, f, sub_label="SiO2")
    lam = 1.0 / f
    m = R > threshold
    return (float(lam[m].min()), float(lam[m].max())) if m.any() else None


# ------------------------------------------------------------------ operating points --- #
def fwm_matched_center(f_probe: float) -> float:
    """Pump centre that satisfies 2 f_pump = f_probe exactly."""
    return 0.5 * float(f_probe)


def operating_points(geom: dict, windows: Sequence[Tuple[float, float]],
                     center_offsets: Sequence[float] = (-0.04, -0.02, 0.0, 0.02, 0.04),
                     deltas: Sequence[float] = (0.008, 0.011, 0.014, 0.017, 0.020, 0.023),
                     t_min: float = T_PEAK_MIN) -> List[dict]:
    """(probe mode) x (pump centre) x (Delta).

    Pump centres are laid out as FRACTIONAL offsets around the FWM-matched point rather than on
    cavity modes: every pump-band mode has Q = 33-185 against a 100 fs Q_cap of ~12, so the modes
    are unresolved by the pulse and the centre is a continuous knob. Offsets that fall outside
    the source's 1400-2000 nm range are dropped.
    """
    ops: List[dict] = []
    for m in probe_modes(geom, windows, t_min=t_min):
        fp = float(m["freq"])
        c0 = fwm_matched_center(fp)
        for off in center_offsets:
            c = c0 * (1.0 + float(off))
            lam_c = 1.0 / c
            if not (PUMP_BAND_SIC[0] <= lam_c <= PUMP_BAND_SIC[1]):
                continue
            for d in deltas:
                if d > C.DELTA_MAX_INBAND:
                    continue
                f1, f2 = c + 0.5 * d, c - 0.5 * d
                if not (PUMP_BAND_SIC[0] <= 1.0 / f1 <= PUMP_BAND_SIC[1]):
                    continue
                if not (PUMP_BAND_SIC[0] <= 1.0 / f2 <= PUMP_BAND_SIC[1]):
                    continue
                ops.append({
                    "probe": fp, "probe_nm": 1000.0 / fp,
                    "probe_Q": float(m["Q"]), "probe_T": float(m["T_peak"]),
                    "center": float(c), "center_nm": 1000.0 / c,
                    "center_offset": float(off),
                    "pump1": float(f1), "pump2": float(f2),
                    "pump1_nm": 1000.0 / f1, "pump2_nm": 1000.0 / f2,
                    "delta": float(d),
                    "fwm_mismatch": abs(2.0 * c - fp),
                    "fwm_mismatch_pct": 100.0 * abs(2.0 * c - fp) / fp,
                })
    return ops


def tag(op: dict) -> str:
    return "p{:.4f}_c{:.4f}_d{:.3f}".format(op["probe"], op["center"], op["delta"])


# ------------------------------------------------------------------ running --- #
def run_case(out: Path, geom: dict, op: dict, sub: int, tau_fs: float,
             res: int = C.RES_1D, decay: str = C.DECAY_1D, pad_fs: float = C.PAD_FS,
             dim: int = 1, pump_intensity: float = C.PUMP_INTENSITY,
             extra: Optional[Sequence[str]] = None) -> Path:
    """One carrier sub-sample of one operating point, with the SiC material flags."""
    flags = list(FDTD_FLAGS_SIC) + list(extra or [])
    return C.run_case(out, geom,
                      {"probe": op["probe"], "pump1": op["pump1"], "pump2": op["pump2"]},
                      tau_fs=tau_fs, res=res, decay=decay, pad_fs=pad_fs,
                      pump_intensity=pump_intensity,
                      extra=flags)


def collect(case_root: Path, op: dict, n_sub: int = C.SUBSAMPLES) -> Optional[dict]:
    recs = [C.read_case(case_root / tag(op) / "s{}".format(s)) for s in range(n_sub)]
    if any(r is None for r in recs):
        return None
    return C.carrier_average(recs)
