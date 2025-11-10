import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import meep as mp
import numpy as np

from geometry_io import load_params
from mode_targeting import get_cavity_materials, material_index_at_wavelength


@dataclass(frozen=True)
class CavityConfig:
    t_SiN: float
    t_SiO2: float
    t_cav: float
    N_per: int
    pad_air: float
    pad_sub: float
    dpml: float
    resolution: int
    cell_margin: float


MATERIAL_MODEL = dict(
    mode="fit",        # options: "library", "constant", "fit"
    sin_csv="si3n4.csv",
    sio2_csv="sio2.csv",
    lam_min=600,
    lam_max=2000,
    fit_poles=2,
    n_high=2.0,
    n_low=1.45,
)


def load_cavity_config(prefer: str = "report") -> CavityConfig:
    """Load the cavity definition produced by the optimizer."""
    params = load_params(prefer=prefer,
                         report_json="optimize_report_experiment.json",
                         geom_json="optimized_geometry_experiment.json")
    return CavityConfig(
        t_SiN=float(params["t_SiN"]),
        t_SiO2=float(params["t_SiO2"]),
        t_cav=float(params["t_cav"]),
        N_per=int(params["N_per"]),
        pad_air=float(params["pad_air"]),
        pad_sub=float(params["pad_sub"]),
        dpml=float(params["dpml"]),
        resolution=int(params["resolution"]),
        cell_margin=float(params["cell_margin"]),
    )




def stack_length(cfg: CavityConfig) -> float:
    """Total physical thickness of the cavity stack (excludes PMLs and margin)."""
    return (cfg.pad_air +
            cfg.N_per * (cfg.t_SiO2 + cfg.t_SiN) +
            cfg.t_cav +
            cfg.N_per * (cfg.t_SiO2 + cfg.t_SiN) +
            cfg.pad_sub)


def build_geometry(cfg: CavityConfig,
                   mat_sin: mp.Medium,
                   mat_sio2: mp.Medium) -> Tuple[List[mp.Block], float]:
    """
    Construct the 1D DBR cavity geometry:
      Air | (SiN/SiO2)^N | SiN cavity | (SiO2/SiN)^N | SiO2 substrate.

    Returns (geometry list, cell_z extent).
    """
    geom: List[mp.Block] = []
    stack_z = stack_length(cfg)
    cell_z = stack_z + 2*cfg.dpml + cfg.cell_margin

    # begin just inside the left PML, leaving half the cell_margin as buffer
    z = -0.5 * cell_z + cfg.dpml + 0.5 * cfg.cell_margin

    def add_block(thickness: float, material: mp.Medium):
        nonlocal z
        center = z + 0.5 * thickness
        geom.append(mp.Block(center=mp.Vector3(0, 0, center),
                             size=mp.Vector3(mp.inf, mp.inf, thickness),
                             material=material))
        z += thickness

    # left air pad
    z += cfg.pad_air

    # left mirror (SiN, SiO2)^N_per
    for _ in range(cfg.N_per):
        add_block(cfg.t_SiN, mat_sin)
        add_block(cfg.t_SiO2, mat_sio2)

    # cavity (SiN)
    add_block(cfg.t_cav, mat_sin)

    # right mirror (SiO2, SiN)^N_per
    for _ in range(cfg.N_per):
        add_block(cfg.t_SiO2, mat_sio2)
        add_block(cfg.t_SiN, mat_sin)

    # substrate pad (SiO2)
    add_block(cfg.pad_sub, mat_sio2)

    return geom, cell_z


def make_simulation(cfg: CavityConfig,
                    geometry: List[mp.Block],
                    cell_z: float,
                    sources: Sequence[mp.Source],
                    force_complex_fields: bool = False) -> mp.Simulation:
    """Helper to instantiate a 1D simulation with shared settings."""
    cell = mp.Vector3(0, 0, cell_z)
    return mp.Simulation(cell_size=cell,
                         geometry=geometry,
                         boundary_layers=[mp.PML(cfg.dpml)],
                         default_material=mp.air,
                         resolution=cfg.resolution,
                         dimensions=1,
                         sources=list(sources),
                         force_complex_fields=force_complex_fields)


# Spectrum setup (0.6–2.0 μm)
WL_MIN, WL_MAX = 0.6, 2.0
FMIN, FMAX = 1 / WL_MAX, 1 / WL_MIN
FCEN, DF = 0.5 * (FMIN + FMAX), (FMAX - FMIN)
N_FREQ = 800

# Time controls
HARMINV_RUN_TIME = 600
DESIRED_PUMP_SEPARATION_UM = 0.050
MIN_PUMP_SEPARATION_UM = 0.020


def plot_epsilon(cfg: CavityConfig,
                 geometry: List[mp.Block],
                 cell_z: float,
                 title: str = "ε(z)",
                 wavelength_um: float = 1.0) -> None:
    """Plot ε(z) using the dispersive materials' response at wavelength_um."""
    zgrid, eps_1d = epsilon_profile(geometry, cell_z, wavelength_um)
    plt.figure(figsize=(8, 3))
    plt.plot(zgrid, eps_1d)
    left_pml0 = -0.5 * cell_z
    right_pml0 = 0.5 * cell_z
    plt.axvspan(left_pml0, left_pml0 + cfg.dpml, color='k', alpha=0.05)
    plt.axvspan(right_pml0 - cfg.dpml, right_pml0, color='k', alpha=0.05)
    plt.xlabel("z (μm)")
    plt.ylabel("ε")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def find_modes(cfg: CavityConfig,
               geometry: List[mp.Block],
               cell_z: float,
               wavelength_um: float,
               bandwidth_fraction: float = 0.2,
               run_time: float = HARMINV_RUN_TIME,
               component=mp.Ex) -> List:
    """Run Harminv around the provided wavelength."""
    f0 = 1.0 / wavelength_um
    df_local = bandwidth_fraction * f0
    src_z = -0.5 * cell_z + cfg.dpml + 0.2
    src = [mp.Source(mp.GaussianSource(f0, fwidth=df_local),
                     component=component,
                     center=mp.Vector3(0, 0, src_z),
                     amplitude=1.0)]
    sim = make_simulation(cfg, geometry, cell_z, sources=src)
    monitor = mp.Harminv(component, mp.Vector3(), f0, df_local)
    sim.run(mp.after_sources(monitor), until=run_time)
    return list(monitor.modes)


HarminvMode = Any


def material_epsilon_at_wavelength(material: mp.Medium, wavelength_um: float) -> float:
    """Return real part of ε(λ) for plotting; fall back to index² if needed."""
    if wavelength_um <= 0:
        return 1.0
    freq = 1.0 / wavelength_um
    if hasattr(material, "epsilon"):
        try:
            eps_val = material.epsilon(freq)[0][0]
            eps_real = float(np.real(eps_val))
            if np.isfinite(eps_real) and eps_real > 0:
                return eps_real
        except Exception:
            pass
    idx = getattr(material, "index", None)
    if idx is not None:
        try:
            n = float(idx)
            if np.isfinite(n) and n > 0:
                return n * n
        except Exception:
            pass
    return 1.0


def epsilon_profile(geometry: Sequence[mp.Block],
                    cell_z: float,
                    wavelength_um: float,
                    samples: int = 1200) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ε(z) analytically for a dispersive stack at wavelength λ."""
    zmin, zmax = -0.5 * cell_z, 0.5 * cell_z
    zgrid = np.linspace(zmin, zmax, samples)
    eps = np.ones_like(zgrid)
    for block in geometry:
        mat = getattr(block, "material", None)
        if mat is None:
            continue
        half_z = 0.5 * block.size.z
        z0 = block.center.z - half_z
        z1 = block.center.z + half_z
        mask = (zgrid >= z0) & (zgrid <= z1)
        if not np.any(mask):
            continue
        eps_val = material_epsilon_at_wavelength(mat, wavelength_um)
        eps[mask] = eps_val
    return zgrid, eps


def find_reflectance_dips(wl: np.ndarray,
                          R: np.ndarray,
                          prominence: float = 0.003,
                          width: int = 3) -> List[Dict[str, float]]:
    dips: List[Dict[str, float]] = []
    try:
        from scipy.signal import find_peaks  # type: ignore

        peaks, props = find_peaks(-R, prominence=prominence, width=width)
        for idx in peaks:
            dips.append(dict(index=int(idx), lam=float(wl[idx]), R=float(R[idx])))
    except Exception:
        pass
    if len(dips) < 2:
        for idx in range(1, len(R) - 1):
            if R[idx] <= R[idx - 1] and R[idx] < R[idx + 1]:
                dips.append(dict(index=int(idx), lam=float(wl[idx]), R=float(R[idx])))
    dips.sort(key=lambda d: d["lam"])
    return dips


def nearest_dip(dips: Sequence[Dict[str, float]],
                target_lambda: float,
                default_lambda: float) -> Dict[str, float]:
    if not dips:
        return dict(lam=float(default_lambda), R=1.0, note="no_dips")
    best = min(dips, key=lambda d: abs(d["lam"] - target_lambda))
    return dict(best)


def select_pump_center(dips: Sequence[Dict[str, float]],
                       nominal: float = 1.60,
                       window: Tuple[float, float] = (1.45, 1.75)) -> float:
    if not dips:
        return nominal
    scoped = [d for d in dips if window[0] <= d["lam"] <= window[1]]
    pool = scoped if scoped else dips
    best = min(pool, key=lambda d: abs(d["lam"] - nominal))
    return float(best["lam"])


def select_pump_dips(dips: Sequence[Dict[str, float]],
                     lam_center: float = 1.60,
                     window_half: float = 0.15,
                     min_sep: float = MIN_PUMP_SEPARATION_UM,
                     desired_sep: float = DESIRED_PUMP_SEPARATION_UM) -> Tuple[Dict[str, float], Dict[str, float]]:
    low = lam_center - window_half
    high = lam_center + window_half
    ordered = sorted({d["lam"]: d for d in dips}.values(), key=lambda d: d["lam"])
    if len(ordered) < 2:
        raise RuntimeError("Not enough reflectance dips near pump band.")

    feasible: List[Tuple[float, Dict[str, float], Dict[str, float]]] = []
    for i in range(len(ordered) - 1):
        d1 = ordered[i]
        d2 = ordered[i + 1]
        lam1, lam2 = d1["lam"], d2["lam"]
        center = 0.5 * (lam1 + lam2)
        sep = lam2 - lam1
        within = (low <= lam1 <= high) or (low <= lam2 <= high) or (low <= center <= high)
        if not within:
            continue
        cost = 6.0 * abs(center - lam_center) + abs(sep - desired_sep)
        if sep < min_sep:
            cost += 200.0 * (min_sep - sep)
        feasible.append((cost, dict(d1), dict(d2)))

    if not feasible:
        for i in range(len(ordered) - 1):
            d1 = ordered[i]
            d2 = ordered[i + 1]
            lam1, lam2 = d1["lam"], d2["lam"]
            center = 0.5 * (lam1 + lam2)
            sep = lam2 - lam1
            cost = 10.0 * abs(center - lam_center) + abs(sep - desired_sep)
            if sep < min_sep:
                cost += 300.0 * (min_sep - sep)
            feasible.append((cost, dict(d1), dict(d2)))

    if not feasible:
        raise RuntimeError("Unable to choose pump dips.")

    _, d_low, d_high = min(feasible, key=lambda item: item[0])
    if d_low["lam"] > d_high["lam"]:
        d_low, d_high = d_high, d_low
    return d_low, d_high


def modes_to_freqs(modes: Sequence[HarminvMode], max_modes: int = 3) -> List[float]:
    """Extract up to max_modes non-growing resonant frequencies from Harminv modes."""
    selected: List[float] = []
    for mode in modes:
        if len(selected) >= max_modes:
            break
        if getattr(mode, "decay", 0.0) > 0:
            # skip exponentially growing artifacts
            continue
        freq = float(mode.freq)
        if not np.isfinite(freq) or freq <= 0:
            continue
        selected.append(freq)
    return selected


def plot_mode_profiles(cfg: CavityConfig,
                       geometry: List[mp.Block],
                       cell_z: float,
                       freqs: Sequence[float],
                       component=mp.Ex,
                       label: str = "|Ex|(z)") -> None:
    """Plot |field| along the cavity for the provided resonant frequencies."""
    if not freqs:
        print("[modes] No stable Harminv resonances to plot.")
        return

    src_z = -0.5 * cell_z + cfg.dpml + 0.2
    source = [mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                        component=component,
                        center=mp.Vector3(0, 0, src_z),
                        amplitude=1.0)]

    sim = make_simulation(cfg, geometry, cell_z, sources=source)
    monitor_len = cell_z - 2 * cfg.dpml - 0.02
    vol = mp.Volume(center=mp.Vector3(), size=mp.Vector3(0, 0, monitor_len))
    dft = sim.add_dft_fields([component], freqs, where=vol)

    settle_time = max(800, int(np.ceil(40 / min(freqs))))
    sim.run(until=settle_time)

    zgrid = np.linspace(-0.5 * monitor_len, 0.5 * monitor_len,
                        sim.get_dft_array(dft, component, 0).size)
    plt.figure(figsize=(8, 3))
    for idx, f in enumerate(freqs):
        field = sim.get_dft_array(dft, component, idx)
        lam_nm = 1e3 / f
        plt.plot(zgrid, np.abs(field), label=f"λ={lam_nm:.0f} nm")
    plt.legend()
    plt.xlabel("z (μm)")
    plt.ylabel(label)
    plt.tight_layout()
    plt.show()


def reflectance_spectrum(cfg: CavityConfig,
                         geometry: List[mp.Block],
                         cell_z: float,
                         component=mp.Ex) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the reflectance spectrum using the standard two-run normalization."""
    src_z = -0.5 * cell_z + cfg.dpml + 0.2
    refl_z = src_z + 0.1
    src = [mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                     component=component,
                     center=mp.Vector3(0, 0, src_z))]

    # Normalization run (air only)
    sim_ref = make_simulation(cfg, geometry=[], cell_z=cell_z, sources=src)
    refl_ref = sim_ref.add_flux(FCEN, DF, N_FREQ, mp.FluxRegion(
        center=mp.Vector3(0, 0, refl_z)))
    sim_ref.run(until_after_sources=mp.stop_when_fields_decayed(
        50, component, mp.Vector3(0, 0, refl_z), 1e-8))
    ref_data = sim_ref.get_flux_data(refl_ref)
    inc = np.array(mp.get_fluxes(refl_ref))
    freqs = np.array(mp.get_flux_freqs(refl_ref))

    # Structure run
    sim = make_simulation(cfg, geometry=geometry, cell_z=cell_z, sources=src)
    refl = sim.add_flux(FCEN, DF, N_FREQ, mp.FluxRegion(
        center=mp.Vector3(0, 0, refl_z)))
    sim.load_minus_flux_data(refl, ref_data)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, component, mp.Vector3(0, 0, refl_z), 1e-8))
    refl_flux = np.array(mp.get_fluxes(refl))
    R = np.maximum(0.0, -refl_flux / inc)
    return freqs, R


def summarize_modes(label: str, modes: Sequence[HarminvMode]) -> None:
    print(f"\n=== Harminv modes near {label} ===")
    if not modes:
        print("  (none found)")
        return
    for m in modes:
        lam_nm = 1e3 / m.freq if m.freq else float('inf')
        quality = getattr(m, "Q", None)
        decay = getattr(m, "decay", None)
        print(f"  λ={lam_nm:7.1f} nm | f={m.freq:.5f}  Q={quality:.2f}  decay={decay:.3e}")


def main():
    cfg = load_cavity_config()
    mat_sin, mat_sio2 = get_cavity_materials(
        model=MATERIAL_MODEL.get("mode", "library"),
        index_high=MATERIAL_MODEL.get("n_high", 2.0),
        index_low=MATERIAL_MODEL.get("n_low", 1.45),
        sin_csv=MATERIAL_MODEL.get("sin_csv"),
        sio2_csv=MATERIAL_MODEL.get("sio2_csv"),
        lam_min=MATERIAL_MODEL.get("lam_min", None),
        lam_max=MATERIAL_MODEL.get("lam_max", None),
        fit_poles=MATERIAL_MODEL.get("fit_poles", 2),
    )
    geometry, cell_z = build_geometry(cfg, mat_sin, mat_sio2)

    plot_epsilon(cfg, geometry, cell_z,
                 title="Dielectric profile of the cavity (λ=0.8 μm)",
                 wavelength_um=0.8)

    freqs, R = reflectance_spectrum(cfg, geometry, cell_z, component=mp.Ex)
    wl = 1.0 / freqs
    dips = find_reflectance_dips(wl, R)

    plt.figure(figsize=(7, 4))
    plt.plot(1e3 * wl, R, lw=1.5)
    plt.gca().invert_xaxis()
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("Cavity reflectance spectrum")

    selected_reflectance: Dict[str, Dict[str, float]] = {
        "probe": nearest_dip(dips, 0.800, 0.800),
    }
    pump_center = select_pump_center(dips, 1.7, window=[1.55,1.8])
    try:
        pump1_dip, pump2_dip = select_pump_dips(dips, lam_center=pump_center)
        pump_sep = pump2_dip["lam"] - pump1_dip["lam"]
        selected_reflectance["pump1"] = pump1_dip
        selected_reflectance["pump2"] = pump2_dip
        print(
            f"[INFO] Pump reflectance minima: {pump1_dip['lam']:.4f} μm & {pump2_dip['lam']:.4f} μm "
            f"(Δλ = {pump_sep*1e3:.1f} nm)"
        )
    except Exception as exc:
        print("[WARN] Pump dip selection fallback:", exc)
        selected_reflectance["pump1"] = nearest_dip(dips, 1.550, 1.550)
        selected_reflectance["pump2"] = nearest_dip(dips, 1.650, 1.650)
        pump_sep = selected_reflectance["pump2"]["lam"] - selected_reflectance["pump1"]["lam"]

    for role in ["probe", "pump1", "pump2"]:
        lam_min = selected_reflectance[role]["lam"]
        R_min = selected_reflectance[role].get("R", 0.0)
        plt.axvline(1e3 * lam_min, color='k', ls='--', lw=0.7)
        plt.scatter([1e3 * lam_min], [R_min], marker="o")
        plt.text(1e3 * lam_min + 5, R_min + 0.02, role, fontsize=8)

    plt.xlabel("wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.tight_layout()
    plt.show()

    harminv_targets = [
        ("probe", selected_reflectance["probe"]["lam"], 0.12),
        ("pump1", selected_reflectance["pump1"]["lam"], 0.12),
        ("pump2", selected_reflectance["pump2"]["lam"], 0.12),
    ]

    mode_sets: List[Sequence[HarminvMode]] = []
    for label, lam, bw in harminv_targets:
        modes = find_modes(cfg, geometry, cell_z, wavelength_um=lam,
                           bandwidth_fraction=bw)
        summarize_modes(label, modes)
        mode_sets.append(modes)

    short_labels = ["probe", "pump1", "pump2"]
    resonances: Dict[str, Dict[str, float]] = {}
    for short, (_, lam_target, _), modes in zip(short_labels, harminv_targets, mode_sets):
        reflectance_info = dict(
            lambda_um=float(selected_reflectance[short]["lam"]),
            R=float(selected_reflectance[short].get("R", 0.0)),
        )
        if modes:
            best_mode = min(modes, key=lambda m: abs(1.0 / m.freq - lam_target))
            freq = float(best_mode.freq)
            lam = float(1.0 / freq) if freq > 0 else float("nan")
            resonances[short] = dict(
                frequency=freq,
                lambda_um=lam,
                Q=float(getattr(best_mode, "Q", np.nan)),
                reflectance=reflectance_info,
            )
        else:
            fallback_freq = float(1.0 / reflectance_info["lambda_um"])
            resonances[short] = dict(
                frequency=fallback_freq,
                lambda_um=reflectance_info["lambda_um"],
                Q=float("nan"),
                reflectance=reflectance_info,
                note="no_harminv_mode_found",
            )

    freqs_to_plot: List[float] = []
    for modes in mode_sets:
        freqs_to_plot.extend(modes_to_freqs(modes, max_modes=3))
    if freqs_to_plot:
        plot_mode_profiles(cfg, geometry, cell_z, freqs_to_plot, component=mp.Ex)

    selected_freqs = [resonances[label]["frequency"]
                      for label in short_labels
                      if np.isfinite(resonances[label]["frequency"]) and resonances[label]["frequency"] > 0]
    if selected_freqs:
        print("[INFO] Plotting selected mode profiles (probe + pumps).")
        plot_mode_profiles(cfg, geometry, cell_z, selected_freqs, component=mp.Ex,
                           label="|Ex|(z) selected modes")

    # ---- Export resonance summary for nonlinear simulations ----
    if resonances.get("pump1") and resonances.get("pump2"):
        delta = abs(resonances["pump1"]["frequency"] - resonances["pump2"]["frequency"])
    else:
        delta = abs((1.0 / harminv_targets[1][1]) - (1.0 / harminv_targets[2][1]))
    pump_sep_um = selected_reflectance["pump2"]["lam"] - selected_reflectance["pump1"]["lam"]
    summary = dict(
        probe=resonances["probe"],
        pump1=resonances["pump1"],
        pump2=resonances["pump2"],
        sidebands=dict(
            frequency_plus=float(resonances["probe"]["frequency"] + delta),
            frequency_minus=float(max(resonances["probe"]["frequency"] - delta, 0.0)),
            delta_frequency=float(delta),
            pump_separation_um=float(pump_sep_um),
        ),
    )
    summary_path = Path("cavity_modes.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote resonance summary to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
