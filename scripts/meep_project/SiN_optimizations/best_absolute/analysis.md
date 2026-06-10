# Best Absolute Faraday Rotation — SiN/SiO2 Cavity

**Source:** `pipeline_cluster_20260323_104832`, optimizer `new` (Bayesian), profile `exact`

## Geometry

| Parameter | Value |
|---|---|
| Cavity material | Si3N4 (SiN) |
| Cavity length | 5.894 μm |
| Mirror pairs | 3 per side (6 layers each) |
| SiN layer thickness | 0.2375 μm |
| SiO2 layer thickness | 0.3442 μm |
| Mirror period | 0.5817 μm |
| Total structure length | ~13.4 μm |

The mirror layers are **asymmetric** — the SiO2 layer (0.344 μm) is 45% thicker than the SiN layer (0.238 μm). This differs significantly from a standard quarter-wave stack at the pump center wavelength (which would give t_SiN ≈ 0.186 μm, t_SiO2 ≈ 0.257 μm). The optimizer found that this detuned mirror design improves the simultaneous resonance alignment of probe and pump modes.

## Cavity Modes

| Mode | Wavelength (μm) | Frequency (μm⁻¹) | Cavity order (est.) |
|---|---|---|---|
| Probe | 0.8025 | 1.2461 | ~30 |
| Pump 1 | 1.5215 | 0.6573 | ~16 |
| Pump 2 | 1.5740 | 0.6353 | ~15 |
| Sideband + | 0.7886 | 1.2680 | — |
| Sideband − | 0.8169 | 1.2241 | — |

**Pump separation:** 52.5 nm (Δf = 0.0219 μm⁻¹)

The pump pair is nearly symmetric around 1.548 μm. The probe is at the second harmonic region. The sideband frequencies (f_probe ± Δf_pump) fall near adjacent cavity resonances, which is important for efficient four-wave mixing.

## Resonance Quality

| Mode | Q factor | Reflectance dip depth | Reflectance at resonance |
|---|---|---|---|
| Probe | 148 | 0.122 | 0.041 |
| Pump 1 | 70 | 0.098 | 0.001 |
| Pump 2 | 79 | 0.153 | 0.009 |

The probe has the highest Q (148), providing strong field enhancement for the weak probe signal. The pump Q factors are moderate (70–79), which is a good balance: high enough for cavity enhancement but low enough that the pump linewidths comfortably accommodate the pulsed source bandwidth (30 nm).

## Faraday Rotation Results

### Optimizer evaluation (resolution 50)

| Metric | Value |
|---|---|
| **Absolute rotation** | **0.1422°** |
| Probe DoLP | 0.9997 |
| Probe θ std | 0.0045° |
| SNR | 77.2 dB |
| S0 (probe intensity) | 0.195 |
| Pump balance (P2/P1) | 0.995 |
| Pump purity | 99.98% |

### 1D high-resolution simulation (resolution 100)

| Metric | Value |
|---|---|
| **Absolute rotation** | **0.1369°** |
| Probe DoLP | 0.9997 |
| Probe θ std | 0.0044° |
| SNR | 77.3 dB |
| S0 (probe intensity) | 0.187 |
| Pump balance (P2/P1) | 0.995 |
| Pump purity | 99.98% |

The rotation decreases by only 3.7% from resolution 50 to 100, confirming good convergence.

### 3D simulation (resolution 30, 24 MPI ranks)

| Metric | Value |
|---|---|
| **Absolute rotation** | **1.991°** |
| Probe DoLP | 0.723 |
| Probe ellipticity χ | -1.66° |
| Probe θ std | 0.0032° |
| S0 (probe intensity) | 0.235 |
| S0 rel max | 0.717 |
| Pump purity | 99.86% |

**Source:** `pipeline_cluster_20260325_114410/sims/new/dim3_high/`

The 3D simulation yields a dramatic **14.5× enhancement** over the 1D result (1.991° vs 0.137°). This is much larger than the ~2.5× ratio observed for the shorter 4.44 μm cavity (pipeline_20260317), indicating that the longer cavity with detuned mirrors couples much more effectively to the 3D transverse mode structure.

The probe DoLP drops from 0.9997 (1D) to 0.723 (3D), and the ellipticity grows to -1.66°. This reflects strong nonlinear polarization mixing in 3D where the transverse mode confinement enhances the intracavity field intensity and the cross-phase modulation between counter-rotating pump helicities.

### Dimension comparison

| Dimension | Rotation | DoLP | Ellipticity χ | Pump purity |
|---|---|---|---|---|
| 1D (res 50) | 0.142° | 0.9997 | -0.71° | 99.98% |
| 1D (res 100) | 0.137° | 0.9997 | -0.70° | 99.98% |
| **3D (res 30)** | **1.991°** | **0.723** | **-1.66°** | **99.86%** |

## Material Properties

| Property | Value |
|---|---|
| SiN refractive index (probe λ) | 2.077 |
| SiO2 refractive index | 1.457 |
| SiN n₂ | 5 × 10⁻¹⁹ m²/W |
| χ³ (SI) | 7.63 × 10⁻²¹ |
| χ³ (Meep diagonal) | 1.08 × 10⁻³ |
| Pump intensity | 10¹² W/cm² |
| Probe intensity | 5 × 10⁷ W/cm² |

## Simulation Parameters

- **Dimension:** 1D (quasi-1D collapsed transverse cell)
- **Pulse duration:** 100 fs
- **Pump bandwidth:** 30 nm
- **Probe bandwidth:** 10 nm
- **Runtime factor:** 6.0
- **Measurement method:** DFT forward-decomposed Jones coherent final window (64 points)

## Pump Analysis

Both pumps have excellent circular polarization purity (>99.97%). Pump 1 carries |σ⁻⟩ and Pump 2 carries |σ⁺⟩ (counter-rotating). The amplitude balance P2/P1 = 0.995 is near-ideal, ensuring symmetric nonlinear coupling.

## Plots

- `probe_polarization_zoom.png` — Probe rotation angle vs time (zoomed to steady state)
- `probe_dft_coherent.png` — DFT-based probe Stokes analysis
- `probe_band_heatmap_coherent.png` — Spectral heatmap across probe band
- `pumps_dft_coherent.png` — Pump circular components (DFT)
- `pumps_td.png` — Pump time-domain envelopes
- `optimize_debug_reflectance_marked.png` — Reflectance spectrum with mode markers
- `optimize_debug_epsilon_profile.png` — Dielectric profile of the cavity structure

## Comparison with Previous Optimizations

| Run | Cavity length | Mirrors | 1D rotation | 3D rotation | 3D/1D |
|---|---|---|---|---|---|
| pipeline_20260317 | 4.44 μm | 3 | 0.068° | 0.168° | 2.5× |
| **This run** | **5.89 μm** | **3** | **0.137°** | **1.991°** | **14.5×** |

The 33% increase in cavity length yielded a 2× improvement in 1D rotation and a **12× improvement in 3D rotation**. The 3D/1D enhancement ratio itself grew from 2.5× to 14.5×, suggesting a nonlinear cavity-mode coupling effect that amplifies with longer interaction length.

## Comparison with Other Geometries (3D)

| Geometry | Lc (μm) | 1D rotation | 3D rotation | 3D/1D |
|---|---|---|---|---|
| **Best absolute (this)** | **5.894** | **0.137°** | **1.991°** | **14.5×** |
| MF optimizer | 5.827 | 0.061° | 0.054° | 0.88× |
| Best rot/Lc ratio | 1.918 | 0.055° | 0.020° | 0.37× |

Only the best-absolute geometry benefits from the 1D→3D enhancement. The other two geometries show flat or reduced rotation in 3D, indicating that the detuned mirror design (t_SiO2/t_SiN = 1.45) is critical for 3D performance — standard quarter-wave mirrors do not achieve the same cavity-mode coupling in the full 3D transverse geometry.

## Files

- `geometry.json` — Simulation-ready geometry input
- `cavity_modes.json` — Mode frequencies and wavelengths
- `3d_sim/` — 3D simulation plots:
  - `probe_polarization_zoom.png` — Probe rotation angle vs time (3D, zoomed)
  - `probe_dft_coherent.png` — DFT probe Stokes analysis (3D)
  - `xz_snapshot.png` — Cross-section field snapshot
  - `pumps_dft_coherent.png` — Pump circular components (3D)
