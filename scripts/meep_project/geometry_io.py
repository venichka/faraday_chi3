#!/usr/bin/env python3
"""
geometry_io.py

JSON schema for a 1D multilayer cavity (μm units):

{
  "materials": {
    "SiN":  {"type": "Medium", "params": {"index": 2.0}},
    "SiO2": {"type": "Medium", "params": {"index": 1.45}}
  },
  "pads": {"pml_um": 1.0, "air_um": 0.8, "substrate_um": 0.8},
  "spacers": {"left_um": 0.10, "right_um": 0.10},
  "cavity": {"mat": "SiN", "L_um": 12.8},
  "mirrors": {
    "left":  [ {"mat":"SiN","thk_um":0.260}, {"mat":"SiO2","thk_um":0.160} ],
    "right": [ {"mat":"SiO2","thk_um":0.160}, {"mat":"SiN","thk_um":0.260} ]
  },
  "meta": {"comment": "free text", "generated_on": "ISO8601"}
}

Functions:
- extract_stack_1d(geometry, mats, dpml, cell_z, spacer_detect="auto", tol_spacer_rel=0.05) -> dict
- build_geometry_from_json(spec, mp) -> (geometry_list, mats_dict, pads_dict)
- write_json(spec, path), read_json(path)
- material_factory(name, entry, mp) -> mp.Medium

New helpers integrated here so your scripts don't need bespoke IO code:
- load_params(report_json="optimize_report.json", geom_json="optimized_geometry.json", prefer="report", defaults=None) -> dict
- export_params_json(path, params)
- export_geometry_json(path, geometry, mats, dpml, cell_z, spacer_detect="never")
"""

from __future__ import annotations
import json
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path

try:
    import meep as mp
except Exception:
    mp = None  # allow import without Meep for pure JSON ops

try:
    import numpy as np
except Exception:
    np = None  # used for median in spacer detection


def _block_extent_z(block) -> Tuple[float, float]:
    """Return (zmin, zmax) of a Meep Block in 1D."""
    c = block.center.z
    thk = block.size.z
    return c - 0.5*thk, c + 0.5*thk


def _sort_by_z(geometry: List) -> List:
    """Sort objects by their lower z edge."""
    return sorted(geometry, key=lambda b: _block_extent_z(b)[0])


def _material_name(block, mats: Dict[str, object]) -> str:
    """Attempt to map a block.material to one of the provided materials dict keys by id()."""
    for name, m in mats.items():
        if block.material is m:
            return name
    # fallback: compare simple params if Medium
    try:
        for name, m in mats.items():
            if hasattr(block.material, 'epsilon') and hasattr(m, 'epsilon'):
                if abs(block.material.epsilon(1.0) - m.epsilon(1.0)) < 1e-9:
                    return name
    except Exception:
        pass
    return "UNKNOWN"


def extract_stack_1d(
    geometry: List,
    mats: Dict[str, object],
    dpml: float,
    cell_z: float,
    spacer_detect: str = "auto",
    tol_spacer_rel: float = 0.05,
) -> Dict:
    """
    Extract a stack description from a 1D Meep geometry of Blocks.
    Assumptions: all Blocks are infinite in x,y and piecewise-constant in z.

    spacer_detect:
      - "auto"  : classify an edge SiO2 as a spacer only if its thickness
                  significantly deviates from the typical SiO2 mirror layer.
      - "never" : never classify edge layers as spacers (keeps them in mirrors).
    tol_spacer_rel: relative deviation threshold for classifying spacers.
    """
    if mp is None:
        raise RuntimeError("Meep not available; cannot inspect Block objects.")

    spec = {
        "materials": {},
        "pads": {},
        "spacers": {"left_um": 0.0, "right_um": 0.0},
        "cavity": {"mat": "UNKNOWN", "L_um": 0.0},
        "mirrors": {"left": [], "right": []},
        "meta": {"generated_on": datetime.utcnow().isoformat() + "Z"}
    }

    # fill materials section from provided mats mapping
    for name, m in mats.items():
        params = {}
        if hasattr(m, "index"):
            try:
                params["index"] = float(m.index)
            except Exception:
                pass
        spec["materials"][name] = {"type": "Medium", "params": params}

    # sorted layers
    objs = _sort_by_z(
        [b for b in geometry if isinstance(b, getattr(mp, 'Block', object))])
    if not objs:
        return spec

    # infer left air pad as gap from dpml to first zmin
    first_zmin, _ = _block_extent_z(objs[0])
    pad_air = max(0.0, first_zmin - (-0.5*cell_z + dpml))
    spec["pads"]["air_um"] = pad_air
    spec["pads"]["pml_um"] = dpml

    # scan to find segments
    segments = []
    for b in objs:
        z0, z1 = _block_extent_z(b)
        segments.append((z0, z1, b))

    # merge adjacent blocks of the same material for robustness
    merged = []
    for z0, z1, b in segments:
        if not merged:
            merged.append([z0, z1, b.material])
            continue
        Z0, Z1, mat = merged[-1]
        if (abs(z0 - Z1) < 1e-9) and (b.material is mat):
            merged[-1][1] = z1
        else:
            merged.append([z0, z1, b.material])

    # choose cavity = longest SiN segment (fallback: longest segment)
    idx_cav = None
    longest = -1.0
    for i, (z0, z1, mat) in enumerate(merged):
        length = z1 - z0
        is_sin = False
        for name, m in mats.items():
            if name.lower().startswith("sin") and (mat is m):
                is_sin = True
        if (is_sin and length > longest) or (idx_cav is None and length > longest):
            longest = length
            idx_cav = i

    # split left/right around cavity
    if idx_cav is None:
        left_merged = merged
        cav_entry = None
        right_merged = []
    else:
        left_merged = merged[:idx_cav]
        cav_entry = merged[idx_cav]
        right_merged = merged[idx_cav+1:]

    # helper to map material to its schema name
    def name_of(material):
        return next((n for n, m in mats.items() if m is material), "UNKNOWN")

    # cavity
    if cav_entry is not None:
        z0, z1, cav_mat = cav_entry
        spec["cavity"]["mat"] = name_of(cav_mat)
        spec["cavity"]["L_um"] = float(z1 - z0)

    # Decide whether to mark edge SiO2 layers as spacers.
    # In the mirror-only (no-spacer) design, SiO2 layers touching the cavity are DBR layers.
    def maybe_edge_spacer_thickness(entries, side):
        """Return (spacer_thk_or_0, drop_count) where drop_count ∈ {0,1}."""
        if not entries:
            return 0.0, 0
        if side == "left":
            z0, z1, mat = entries[-1]
        else:  # right
            z0, z1, mat = entries[0]
        if name_of(mat) != "SiO2":
            return 0.0, 0
        thk_edge = float(z1 - z0)
        if spacer_detect == "never":
            return 0.0, 0
        if spacer_detect != "auto":
            # unknown mode: be conservative
            return 0.0, 0
        if np is None:
            # Without numpy for median, do not strip mirror layers
            return 0.0, 0

        # Build a pool of typical SiO2 mirror thicknesses excluding the two edge candidates
        def collect_sio2_thicknesses(seq):
            out = []
            for a, b, m in seq:
                if name_of(m) == "SiO2":
                    out.append(float(b - a))
            return out

        s_left = collect_sio2_thicknesses(
            entries[:-1]) if side == "left" else collect_sio2_thicknesses(entries)
        s_right = collect_sio2_thicknesses(
            entries[1:]) if side == "right" else collect_sio2_thicknesses(entries)
        pool = s_left + s_right
        if len(pool) == 0:
            # no internal SiO2 layers to compare — treat as mirror layer, not a spacer
            return 0.0, 0
        t_ref = float(np.median(pool))
        # classify as spacer only if edge deviates notably from typical mirror thickness
        if abs(thk_edge - t_ref) > tol_spacer_rel * max(t_ref, 1e-12):
            return thk_edge, 1
        return 0.0, 0

    # left/right spacer detection
    if spacer_detect == "never":
        sL_thk = sR_thk = 0.0
        left_drop = right_drop = 0
    else:
        sL_thk, left_drop = maybe_edge_spacer_thickness(left_merged,  "left")
        sR_thk, right_drop = maybe_edge_spacer_thickness(right_merged, "right")

    spec["spacers"]["left_um"] = sL_thk
    spec["spacers"]["right_um"] = sR_thk

    # mirrors: drop exactly one edge layer per side only if we've classified it as a spacer
    left_layers = left_merged[:-
                              1] if (left_merged and sL_thk > 0) else left_merged
    right_layers = right_merged[1:] if (
        right_merged and sR_thk > 0) else right_merged

    # convert to schema
    def to_layers(entries):
        layers = []
        for z0, z1, mat in entries:
            name = name_of(mat)
            thk = float(z1 - z0)
            layers.append({"mat": name, "thk_um": thk})
        return layers

    spec["mirrors"]["left"] = to_layers(left_layers)
    spec["mirrors"]["right"] = to_layers(right_layers)

    # infer substrate pad as gap from last zmax to +cell_z/2 - dpml
    last_zmin, last_zmax = _block_extent_z(objs[-1])
    pad_sub = max(0.0, (0.5*cell_z - dpml) - last_zmax)
    spec["pads"]["substrate_um"] = pad_sub

    return spec


def material_factory(name: str, entry: Dict, mp_module=None):
    """Create an mp.Medium (or future material) from a JSON entry."""
    mpm = mp_module or mp
    if mpm is None:
        raise RuntimeError("Meep not available to build materials.")
    typ = entry.get("type", "Medium")
    params = entry.get("params", {})
    if typ == "Medium":
        return mpm.Medium(**params)
    raise ValueError(f"Unsupported material type: {typ}")


def build_geometry_from_json(spec: Dict, mp_module=None):
    """Return (geometry_list, mats_dict, pads_dict)."""
    mpm = mp_module or mp
    if mpm is None:
        raise RuntimeError("Meep not available to build geometry.")

    # materials
    mats = {name: material_factory(name, ent, mpm)
            for name, ent in spec["materials"].items()}

    dpml = spec["pads"]["pml_um"]
    pad_air = spec["pads"]["air_um"]
    pad_sub = spec["pads"]["substrate_um"]

    sL = spec.get("spacers", {}).get("left_um", 0.0)
    sR = spec.get("spacers", {}).get("right_um", 0.0)

    cav_mat_name = spec["cavity"]["mat"]
    L_cav = spec["cavity"]["L_um"]
    cav_mat = mats[cav_mat_name]

    left_layers = spec["mirrors"]["left"]
    right_layers = spec["mirrors"]["right"]

    def sum_layers(layers):
        return sum(layer["thk_um"] for layer in layers)

    stack_len = pad_air + \
        sum_layers(left_layers) + sL + L_cav + sR + \
        sum_layers(right_layers) + pad_sub

    geometry = []
    z = -0.5*stack_len
    z += pad_air

    def add(thk, mat):
        nonlocal z
        c = z + 0.5*thk
        geometry.append(mpm.Block(center=mpm.Vector3(0, 0, c),
                                  size=mpm.Vector3(mpm.inf, mpm.inf, thk),
                                  material=mat))
        z += thk

    # left mirror
    for layer in left_layers:
        add(layer["thk_um"], mats[layer["mat"]])
    # spacers + cavity
    if sL > 0:
        add(sL, mats.get("SiO2", list(mats.values())[0]))
    add(L_cav, cav_mat)
    if sR > 0:
        add(sR, mats.get("SiO2", list(mats.values())[0]))
    # right mirror
    for layer in right_layers:
        add(layer["thk_um"], mats[layer["mat"]])
    # substrate spacer (assumed SiO2 in this schema; fall back to first mat if missing)
    add(pad_sub, mats.get("SiO2", list(mats.values())[0]))

    pads = {"pml_um": dpml, "air_um": pad_air, "substrate_um": pad_sub}
    return geometry, mats, pads


def write_json(spec: Dict, path: str):
    with open(path, "w") as f:
        json.dump(spec, f, indent=2)


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


# --------------------- New parameter IO helpers ---------------------

def _safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_params(
    report_json="optimize_report.json",
    geom_json="optimized_geometry.json",
    prefer="report",
    defaults=None,
):
    """
    Returns a dict with keys:
      N_per, t_SiN, t_SiO2, t_cav, pad_air, pad_sub, dpml, resolution, cell_margin

    Reads from:
      - optimize_report.json (new mirror+cavity or legacy spacer format), OR
      - geometry JSON (geometry_io schema).

    'prefer' controls which one is tried first when both exist.
    """
    # sensible defaults
    params = dict(N_per=3, t_SiN=0.260, t_SiO2=0.160, t_cav=1.543,
                  pad_air=0.8, pad_sub=3.0, dpml=1.0, resolution=100,
                  cell_margin=0.4)
    if defaults:
        params.update(defaults)

    report = _safe_load_json(report_json) if Path(
        report_json).exists() else None
    geom = read_json(geom_json) if (Path(geom_json).exists()) else None

    order = ("report", "geom") if prefer == "report" else ("geom", "report")

    for src in order:
        if src == "report" and report:
            bt = report.get("best_theta", {})
            sim = report.get("sim", {})

            # new mirror+cavity layout
            t_SiN = bt.get("t_SiN_um")
            t_SiO2 = bt.get("t_SiO2_um")
            L_cav = bt.get("L_cav_um")
            cell_margin = bt.get("cell_margin_um", params["cell_margin"])
            N_per = bt.get("N_per", params["N_per"])

            # legacy spacer layout (keep compatibility)
            if L_cav is None and "L_cav_um" in bt:
                L_cav = bt["L_cav_um"]

            if t_SiN:
                params["t_SiN"] = float(t_SiN)
            if t_SiO2:
                params["t_SiO2"] = float(t_SiO2)
            if L_cav:
                params["t_cav"] = float(L_cav)
            params["cell_margin"] = float(cell_margin)
            params["N_per"] = int(N_per)

            # optional simulation settings
            if "resolution_px_per_um" in sim:
                params["resolution"] = int(sim["resolution_px_per_um"])
            if "dpml_um" in sim:
                params["dpml"] = float(sim["dpml_um"])
            if "pad_air_um" in sim:
                params["pad_air"] = float(sim["pad_air_um"])
            if "pad_sub_um" in sim:
                params["pad_sub"] = float(sim["pad_sub_um"])

            return params

        if src == "geom" and geom:
            # geometry spec (geometry_io schema)
            pads = geom["pads"]
            mirL = geom["mirrors"]["left"]
            mirR = geom["mirrors"]["right"]
            cav = geom["cavity"]

            params["dpml"] = float(pads["pml_um"])
            params["pad_air"] = float(pads["air_um"])
            params["pad_sub"] = float(pads["substrate_um"])
            params["t_cav"] = float(cav["L_um"])

            # infer N_per & average layer thicknesses from mirrors
            tH = [l["thk_um"] for l in mirL if l["mat"] == "SiN"]
            tL = [l["thk_um"] for l in mirL if l["mat"] == "SiO2"]
            if not tH or not tL:
                tH = [l["thk_um"] for l in mirR if l["mat"] == "SiN"]
                tL = [l["thk_um"] for l in mirR if l["mat"] == "SiO2"]
            if tH:
                params["t_SiN"] = float(
                    np.mean(tH)) if np else float(sum(tH)/len(tH))
            if tL:
                params["t_SiO2"] = float(
                    np.mean(tL)) if np else float(sum(tL)/len(tL))
            params["N_per"] = int(min(len(tH), len(tL))) if (
                tH and tL) else params["N_per"]

            # geometry JSON doesn't store resolution/cell_margin — keep defaults
            return params

    return params  # defaults


def export_params_json(path, params):
    snap = dict(
        N_per=int(params["N_per"]),
        t_SiN_um=float(params["t_SiN"]),
        t_SiO2_um=float(params["t_SiO2"]),
        L_cav_um=float(params["t_cav"]),
        pad_air_um=float(params["pad_air"]),
        pad_sub_um=float(params["pad_sub"]),
        dpml_um=float(params["dpml"]),
        resolution_px_per_um=int(params.get("resolution", 100)),
        cell_margin_um=float(params.get("cell_margin", 0.0)),
        note="Parameter snapshot exported by fp_cavity_modes_spectrum.py",
    )
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    print(f"[params] wrote {path}")


def export_geometry_json(path, geometry, mats, dpml, cell_z, spacer_detect: str = "never"):
    """Serialize current Meep geometry to geometry_io schema JSON."""
    if mp is None:
        raise RuntimeError("Meep not available; cannot inspect Block objects.")
    spec = extract_stack_1d(geometry, mats, dpml=dpml,
                            cell_z=cell_z, spacer_detect=spacer_detect)
    write_json(spec, path)
    print(f"[geom] wrote {path}")
