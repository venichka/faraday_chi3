# Best Rotation/Length Efficiency — SiN/SiO2 Cavity

**Source:** `pipeline_cluster_20260323_104832`, optimizer `new` (Bayesian), profile `band`, eval 202

## Geometry

| Parameter | Value |
|---|---|
| Cavity material | Si3N4 (SiN) |
| Cavity length | 1.918 μm |
| Mirror pairs | 3 per side (6 layers each) |
| SiN layer thickness | ~0.171 μm (quarter-wave estimate) |
| SiO2 layer thickness | ~0.243 μm (quarter-wave estimate) |
| Mirror period | ~0.414 μm |
| Total structure length | ~7.4 μm |

**Note on layer thicknesses:** The exact Sobol-optimized layer thicknesses for this candidate were not preserved after the optimizer's eval cache cleanup. The values above are quarter-wave estimates at the pump center wavelength (1.414 μm). A re-optimization or reflectance scan to refine these thicknesses is recommended before running 3D simulations.

The mirror layers are **nearly quarter-wave** at the pump center frequency, unlike the best-absolute geometry which uses heavily detuned mirrors. This compact cavity relies on mode density rather than detuned mirror engineering.

## Cavity Modes

| Mode | Wavelength (μm) | Frequency (μm⁻¹) | Cavity order (est.) |
|---|---|---|---|
| Probe | 0.928 | 1.078 | ~9 |
| Pump 1 | 1.323 | 0.756 | ~6 |
| Pump 2 | 1.505 | 0.665 | ~5 |

**Pump separation:** 182 nm (Δf = 0.091 μm⁻¹)

This cavity has much wider pump separation (182 nm) compared to the best-absolute design (52.5 nm). The wider separation means stronger four-wave mixing sidebands but also requires the cavity to support resonances across a broader spectral range.

The low mode orders (5, 6, 9) mean fewer cavity modes compete for the pump/probe frequencies, which simplifies mode targeting. The probe is at approximately the 9th order — close to twice the pump orders (5+6)/2 ≈ 5.5, suggesting near-degenerate four-wave mixing.

## Resonance Quality

| Mode | Q factor | Reflectance dip depth | Reflectance at resonance |
|---|---|---|---|
| Probe | 46.7 | 0.137 | 0.008 |
| Pump 1 | 78.3 | 0.918 | 0.042 |
| Pump 2 | 45.0 | 0.757 | 0.144 |

The Q factors are notably lower than the best-absolute design (47–78 vs 70–148). This is expected for a shorter cavity with fewer round trips. However, the pump dip depths are remarkably high (0.92 and 0.76), indicating excellent coupling into the pump modes despite the lower Q. The probe dip depth (0.137) is comparable to the long-cavity design.

## Faraday Rotation Results

### Optimizer evaluation (resolution 50)

| Metric | Value |
|---|---|
| **Absolute rotation** | **0.0554°** |
| **Rotation per unit length** | **0.0289 °/μm** |

No 1D high-resolution simulation was performed for this specific candidate (only the overall-best geometry gets the full simulation pipeline).

### 3D simulation (resolution 30, 24 MPI ranks)

| Metric | Value |
|---|---|
| **Absolute rotation** | **0.020°** |
| Probe DoLP | 0.806 |
| Probe ellipticity χ | -0.011° |
| Probe θ std | 0.0099° |
| S0 (probe intensity) | 0.290 |
| S0 rel max | 0.430 |
| Pump purity | >99.99% |

**Source:** `pipeline_cluster_20260325_114430/sims/new/dim3_high/`

The 3D simulation shows a **significant degradation** compared to the 1D result — rotation drops from 0.055° to 0.020° (3D/1D ratio = 0.37×). This is in stark contrast to the best-absolute geometry which showed a 14.5× enhancement in 3D. The likely cause is that the quarter-wave-estimated layer thicknesses do not produce optimal resonances in the full 3D geometry. The exact Sobol-optimized values (lost during eval cache cleanup) would be needed to recover the 1D performance.

## Efficiency Comparison

### 1D efficiency (optimizer resolution 50)

| Design | Lc (μm) | Rotation (°) | rot/Lc (°/μm) | Relative efficiency |
|---|---|---|---|---|
| **This (best ratio)** | **1.918** | **0.055** | **0.0289** | **1.00×** |
| Best absolute | 5.894 | 0.142 | 0.0241 | 0.83× |
| Previous best (pipeline_20260317) | 4.439 | 0.068 | 0.0153 | 0.53× |

In 1D, the compact cavity achieves **20% higher rotation per unit length** than the best-absolute design.

### 3D comparison

| Design | Lc (μm) | 3D rotation (°) | 3D rot/Lc (°/μm) | 3D/1D ratio |
|---|---|---|---|---|
| This (best ratio) | 1.918 | 0.020 | 0.011 | 0.37× |
| MF optimizer | 5.827 | 0.054 | 0.009 | 0.88× |
| **Best absolute** | **5.894** | **1.991** | **0.338** | **14.5×** |

**The 1D efficiency advantage does not transfer to 3D.** The best-absolute geometry dominates in 3D by a factor of ~100× in rotation and ~30× in rot/Lc. This indicates that 3D transverse mode coupling and the detuned mirror design are far more important than cavity compactness for maximizing Faraday rotation.

## Material Properties

| Property | Value |
|---|---|
| SiN refractive index (probe λ) | 2.067 |
| SiO2 refractive index | 1.457 |
| SiN n₂ | 5 × 10⁻¹⁹ m²/W |
| Pump intensity | 10¹² W/cm² |

## Mode Analysis

### Why short cavities are more efficient

1. **Lower mode volume:** A 1.92 μm cavity with n=2.07 has an optical path length of ~3.97 μm. The pump field amplitude scales as E ∝ 1/√V_mode, so shorter cavities have higher intracavity field per unit input power.

2. **Better mode overlap:** With only 3 modes involved (orders 5, 6, 9), the spatial overlap integral between pump and probe modes is high. In longer cavities with higher-order modes (orders 15, 16, 30), the more rapid spatial oscillations reduce the overlap.

3. **Wider FSR:** The free spectral range of a 1.92 μm cavity is ~4× that of the 5.89 μm cavity. This means the pump separation (182 nm) naturally matches the FSR, leading to clean two-pump resonance without parasitic mode competition.

### Trade-offs

- **3D performance is poor:** The 1D efficiency advantage vanishes in 3D (0.020° vs 1.991° for best absolute). The compact cavity does not benefit from the 3D transverse mode enhancement.
- **Approximate layer thicknesses:** The quarter-wave estimates may not match the original Sobol-optimized values, contributing to the 3D degradation.
- **Lower Q factors** mean broader linewidths and less spectral selectivity.
- **Pump 2 reflectance** is higher (0.144 vs 0.009), indicating slightly worse coupling.

## Recommended Next Steps

1. **Re-optimize layer thicknesses:** Run a targeted optimization with `--cavity-min-length 1.8 --cavity-max-length 2.1 --bayes-iters 4` to recover exact t_sin, t_sio2 and verify whether the 1D performance is reproducible.
2. **Re-run 3D simulation** with recovered exact layer thicknesses to determine if the 3D degradation is due to the approximate geometry or a fundamental limitation of short cavities.
3. **Investigate 3D mode structure:** The 14.5× 3D enhancement seen in the best-absolute geometry appears linked to its detuned mirror design. A compact cavity with similarly detuned mirrors may recover some of the 3D advantage.

## Files

- `geometry.json` — Geometry input (quarter-wave layer estimate; see note above)
- `cavity_modes.json` — Mode frequencies and wavelengths
