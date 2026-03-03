#!/usr/bin/env python3
"""Shared helpers for high-index Kerr material selection and parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Dict, Tuple

EPS0 = 8.854187817e-12
C0 = 299792458.0


@dataclass(frozen=True)
class HighIndexMaterialPreset:
    key: str
    display_name: str
    n_const: float
    k_const: float
    n2_m2_per_w: float
    reference_wavelength_um: float
    source: str


HIGH_INDEX_PRESETS: Dict[str, HighIndexMaterialPreset] = {
    # Baseline project value kept for backward-compatible behavior.
    "sin": HighIndexMaterialPreset(
        key="sin",
        display_name="Si3N4 (SiN)",
        n_const=2.0,
        k_const=0.0,
        n2_m2_per_w=5.0e-19,
        reference_wavelength_um=1.55,
        source="project baseline (scaled from 2.5e-19 m^2/W)",
    ),
    # TiO2 defaults from integrated-waveguide literature around 1550 nm.
    "tio2": HighIndexMaterialPreset(
        key="tio2",
        display_name="TiO2",
        n_const=2.31,
        k_const=8.0e-6,
        n2_m2_per_w=2.3e-18,
        reference_wavelength_um=1.55,
        source=(
            "n~2.31 at 1550 nm and n2~(2.3-3.6)e-18 m^2/W "
            "(Guan et al., Opt. Express 2018)"
        ),
    ),
}


def high_index_material_choices() -> Tuple[str, ...]:
    return tuple(HIGH_INDEX_PRESETS.keys())


def canonical_high_index_material(name: str | None) -> str:
    key = str(name or "sin").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "sin": "sin",
        "si3n4": "sin",
        "si3n": "sin",
        "s3n4": "sin",
        "siliconnitride": "sin",
        "tio2": "tio2",
        "titaniumdioxide": "tio2",
        "titania": "tio2",
    }
    canonical = aliases.get(key, key)
    if canonical not in HIGH_INDEX_PRESETS:
        allowed = ", ".join(sorted(HIGH_INDEX_PRESETS.keys()))
        raise ValueError(f"Unknown high-index material '{name}'. Allowed: {allowed}.")
    return canonical


def get_high_index_preset(name: str | None) -> HighIndexMaterialPreset:
    return HIGH_INDEX_PRESETS[canonical_high_index_material(name)]


def resolve_high_index_index(index_high: float | None, material: str | None) -> float:
    if index_high is not None and float(index_high) > 0.0:
        return float(index_high)
    return float(get_high_index_preset(material).n_const)


def resolve_high_index_kappa(kappa_high: float | None, material: str | None) -> float:
    if kappa_high is not None and float(kappa_high) >= 0.0:
        return float(kappa_high)
    return float(get_high_index_preset(material).k_const)


def resolve_high_index_n2(n2_high: float | None, material: str | None) -> float:
    if n2_high is not None and float(n2_high) > 0.0:
        return float(n2_high)
    return float(get_high_index_preset(material).n2_m2_per_w)


def build_constant_nk_medium(
    mp_module,
    *,
    n: float,
    k: float = 0.0,
    reference_wavelength_um: float = 1.55,
):
    """
    Build an mp.Medium for approximately constant n,k around a reference lambda.

    For k>0, convert complex epsilon to epsilon + D_conductivity using:
      eps_real = n^2 - k^2
      eps_imag = 2nk
      D_cond   = 2*pi*f_ref*(eps_imag/eps_real)
    This matches the Meep recommendation for approximating loss at one frequency.
    """
    n_val = float(max(n, 1e-9))
    k_val = float(max(k, 0.0))
    if k_val <= 0.0:
        return mp_module.Medium(index=n_val)

    eps_real = max(n_val * n_val - k_val * k_val, 1e-9)
    eps_imag = 2.0 * n_val * k_val
    f_ref = 1.0 / max(float(reference_wavelength_um), 1e-9)
    d_cond = 2.0 * pi * f_ref * (eps_imag / eps_real)
    return mp_module.Medium(epsilon=eps_real, D_conductivity=float(d_cond))


def n2_to_chi3_si(n2_m2_per_w: float, n_linear: float) -> float:
    n2 = float(max(n2_m2_per_w, 0.0))
    nlin = float(max(n_linear, 0.0))
    return float((4.0 / 3.0) * n2 * (nlin * nlin) * EPS0 * C0)


def chi3_si_to_meep_e_chi3(
    chi3_si: float, *, scale_e: float, nonlinear_scale: float = 1.0
) -> float:
    return float(chi3_si * (float(scale_e) ** 2) * float(nonlinear_scale))
