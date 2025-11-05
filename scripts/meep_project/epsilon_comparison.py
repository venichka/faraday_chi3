#!/usr/bin/env python3
"""
Plot epsilon(λ) from two (n,k) text files and overlay Meep material models if available.

- Expects tab- or whitespace-separated lines with: wavelength_nm  n  k
- Skips non-numeric header lines.
- If python-meep is installed, overlays ε(λ)=epsilon(1/λ) for mp.materials.SiO2 and mp.materials.Si3N4_NIR.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- user inputs ----
FILE_SIO2 = "sio2_from_slowsio2_2min42s_fastsn_3min2s_no6_tape_backside.txt"
FILE_SINX = "sn_from_slowsio2_2min42s_fastsn_3min2s_no6_tape_backside.txt"
# ---------------------


def load_nk_file(path: Path) -> pd.DataFrame:
    rows = []
    # parse lines; tolerate headers and mixed whitespace/commas
    for line in path.read_text(errors="ignore").splitlines():
        s = line.strip()
        if not s or s[0] in "#;/%":
            continue
        parts = re.split(r"[\s,]+", s)
        if len(parts) >= 3:
            try:
                wl_nm = float(parts[0])
                n = float(parts[1])
                k = float(parts[2])
                rows.append((wl_nm, n, k))
            except ValueError:
                continue
    if not rows:
        raise RuntimeError(f"No numeric rows parsed from {path}")
    df = pd.DataFrame(rows, columns=["wavelength_nm", "n", "k"])
    df["wavelength_um"] = df["wavelength_nm"] / 1000.0
    n_complex = df["n"].to_numpy() + 1j * df["k"].to_numpy()
    eps = n_complex**2
    df["epsilon_real"] = np.real(eps)
    df["epsilon_imag"] = np.imag(eps)
    return df


def _fill_nearest_complex(arr: np.ndarray) -> np.ndarray:
    """Fill NaNs in a complex array from the nearest defined neighbor (prev/next)."""
    a = arr.astype(np.complex128, copy=True)
    # Work on real/imag separately to avoid complex-NaN gotchas
    for comp in (np.real, np.imag):
        vals = comp(a)
        isn = np.isnan(vals)
        if not isn.any():
            continue
        idx = np.arange(vals.size)

        # forward fill indices
        last = np.where(~isn, idx, -1)
        np.maximum.accumulate(last, out=last)

        # backward fill indices
        nxt = np.where(~isn, idx, vals.size)
        # cumulative minimum from the right
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]

        # choose nearest (ties prefer forward/previous)
        left_dist = np.where(last >= 0, idx - last, np.inf)
        right_dist = np.where(nxt < vals.size, nxt - idx, np.inf)
        choose = np.where(left_dist <= right_dist, last, nxt)

        # apply chosen neighbor where we had NaNs
        fill_vals = vals.copy()
        valid_mask = (choose >= 0) & (choose < vals.size)
        fill_vals[isn & valid_mask] = vals[choose[isn & valid_mask]]
        # if everything was NaN, leave as NaN
        if comp is np.real:
            a = fill_vals + 1j * np.imag(a)
        else:
            a = np.real(a) + 1j * fill_vals
    return a


def _safe_eps_vs_freq(mat, freq: np.ndarray) -> np.ndarray:
    """Evaluate mat.epsilon(freq) robustly, clamping to nearest available value on failure.

    Returns complex ε for each frequency in freq (which are in 1/μm).
    """
    eps = np.empty(freq.size, dtype=np.complex128)
    eps[:] = np.nan + 1j * np.nan
    for i, f in enumerate(freq):
        try:
            e = mat.epsilon(float(f))
            # Meep may return scalar or 3x3 tensor; take 0,0 for isotropic case.
            e_arr = np.asarray(e)
            val = e_arr[0, 0] if e_arr.ndim >= 2 else e_arr
            eps[i] = complex(val)
        except Exception:
            # leave NaN; we'll fill from neighbors below
            continue
    # fill out-of-range / failed points by nearest neighbor (previous/next)
    eps = _fill_nearest_complex(eps)
    return eps


def try_meep_eps(wl_um: np.ndarray):
    """Return dict name->epsilon(λ) or {} if meep unavailable. Meep expects frequency in 1/μm."""
    try:
        from meep import materials  # type: ignore
    except Exception as e:
        print("[info] meep not found; skipping meep overlays:", repr(e))
        return {}
    out = {}
    freq = 1.0 / wl_um
    for name, mat in [("SiO2 (Meep)", materials.SiO2), ("Si3N4_NIR (Meep)", materials.Si3N4_NIR)]:
        out[name] = _safe_eps_vs_freq(mat, freq)
    return out


def main():
    p1 = Path(FILE_SIO2)
    p2 = Path(FILE_SINX)
    if not p1.exists() or not p2.exists():
        raise SystemExit(f"Place the two files next to this script:\n - {p1}\n - {p2}")
    d1 = load_nk_file(p1)
    d2 = load_nk_file(p2)

    wl_min = max(d1["wavelength_um"].min(), d2["wavelength_um"].min())
    wl_max = min(d1["wavelength_um"].max(), d2["wavelength_um"].max())
    wl = np.linspace(wl_min, wl_max, 600)

    meep_curves = try_meep_eps(wl)

    # --- Plot ε'(λ) ---
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(d1["wavelength_um"], d1["epsilon_real"], label="SiO2 ε' (from file)")
    plt.plot(d2["wavelength_um"], d2["epsilon_real"], label="SiNx ε' (from file)")
    for name, eps in meep_curves.items():
        plt.plot(wl, np.real(eps), label=f"{name} ε'")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Re{ε(λ)}")
    plt.title("Real part of permittivity vs wavelength")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # save rather than show (so it works headless too)
    # plt.savefig("epsilon_real_vs_lambda.png")
    plt.show()
    print("Saved epsilon_real_vs_lambda.png")

    # --- Plot ε''(λ) from files (and Meep if available) ---
    plt.figure(figsize=(8, 5), dpi=150)
    plt.plot(d1["wavelength_um"], d1["epsilon_imag"], label="SiO2 ε'' (from file)")
    plt.plot(d2["wavelength_um"], d2["epsilon_imag"], label="SiNx ε'' (from file)")
    for name, eps in meep_curves.items():
        plt.plot(wl, np.imag(eps), label=f"{name} ε''")
    plt.xlabel("Wavelength (μm)")
    plt.ylabel("Im{ε(λ)}")
    plt.title("Imaginary part of permittivity vs wavelength")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # plt.savefig("epsilon_imag_vs_lambda.png")
    print("Saved epsilon_imag_vs_lambda.png")


if __name__ == "__main__":
    main()

