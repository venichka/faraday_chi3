# SiC Cavity (from best_absolute geometry), L = 3.2 µm — Study Log

**Goal:** take the SiN `best_absolute` design, swap the nonlinear/high-index material to **SiC**
(linear n,k from `sic.csv` ellipsometry fit; Kerr n₂ = 5×10⁻¹⁸ m²/W) and set the **cavity length to 3.2 µm**,
keeping the mirror layer thicknesses unchanged. Then: mode analysis → 1D sim → 1D sweep → 3D sim.

> **Status:** mode analysis, 1D sim, 1D cold + resonant intensity sweeps, pump-frequency scan, and **3D sim
> all done.** Headline: the **cold/matched pumps (f₁=0.6037) are the best config** and the only clean χ⁽⁵⁾
> evidence (slope→2.07 over 10¹¹–8×10¹¹); the **3D sim shows no enhancement vs 1D** (365.8° ≈ 369.6°, because
> the pumps are non-resonant → bulk-χ³); and the earlier **"~13× at the 1.87 µm resonance" claim is retracted**
> (unwrapper artifact). Full corrected analysis in the scan / 3D / resonant-sweep / conclusions sections below.

## Geometry

| Parameter | Value |
|---|---|
| Cavity material | **SiC** (4H-SiC, ellipsometry fit) |
| Cavity length | **3.2 µm** (was 5.894 µm in best_absolute) |
| Mirror pairs | 3 per side (unchanged from best_absolute) |
| High-index (SiC) layer thickness | 0.2375 µm (unchanged) |
| SiO₂ layer thickness | 0.3442 µm (unchanged) |
| Low-index | SiO₂ (n≈1.457) |

Geometry file: `geometry.json` (labels are `SiC`/`SiO2`; linear n,k injected at runtime via
`--materials fit --sin-fit sic.csv`).

## Materials / fit notes

- SiC linear index from `sic.csv`: n≈2.674 @0.8 µm, n≈2.56 @1.5 µm; k≲8×10⁻⁴ in band (low loss).
- SiC n₂ = 5×10⁻¹⁸ m²/W — added as a `sic` preset in `nonlinear_materials.py`.
- **Lorentz-pole fit count matters for FDTD stability.** A 4-pole fit (via `material_fit.py`, whose
  `_fit_lorentz` does not bound pole frequencies) placed spurious poles at f₀≈93.7 and 9.78 (1/µm) —
  above the FDTD Lorentzian stability limit — and the sim diverged (NaN/Inf). **3 poles** is stable at
  resolution 100 (poles f₀ = 2.97 + two low-frequency/near-Drude terms). ⚠️ One SiO₂ pole sits at
  f₀≈23.2, which is *above* the res-30 stability cap (~19) — to be re-checked before the res-30 3D run.

## Mode analysis (resolution 100, Harminv + reflectance dips)

Tool: `fp_cavity_modes_spectrum.py` (constants temporarily pointed at this geometry + `sic.csv`, 3 poles).
Output: `cavity_modes.json`; reflectance spectrum in `mode_analysis/reflectance.{png,csv}`.

| Mode | λ (µm) | f (1/µm) | Q | Note |
|---|---|---|---|---|
| Probe | 0.824 | 1.213 | **144.5** | real cavity mode (≈ SiN baseline Q=148) |
| "Pump 1" | 1.657 | 0.604 | — (NaN) | **no Harminv mode**; bare reflectance min, R≈5×10⁻⁴ |
| "Pump 2" | 1.723 | 0.580 | — (NaN) | **no Harminv mode**; bare reflectance min, R≈1.5×10⁻³ |

FWM frequency-matching still holds (2×pump f ≈ probe f: 2×0.604 = 1.208 ≈ 1.213).

### Key finding — the DBR stopband shifted out of the pump band

SiC's higher index raised the high-index layer optical thickness (n·t: 0.49 → 0.61), pushing the DBR
**stopband to ~1.85–2.0 µm** (R→1, with a genuine high-Q cavity resonance at ~1.87 µm).

- The intended **pumps at ~1.66/1.72 µm land on the near-zero-reflectance blue edge** of the stopband →
  no mirror confinement → no pump cavity modes → no pump field enhancement.
- The **probe at 0.82 µm** sits in the high-order interference region and *does* have a Q≈145 mode.
- The real high-Q resonance is now at **~1.87 µm**, inside the shifted stopband.

**Implication:** with the mirrors left exactly as best_absolute, the SiC design has a resonant probe but
**non-resonant pumps** at the FWM-matched wavelengths. The nonlinear rotation would be weak. A design
choice is required before proceeding (re-tune SiC mirror thickness to move the stopband back to ~1.5 µm,
*or* move the operating wavelengths into the new ~1.87 µm stopband, *or* proceed as a documented null result).

**Decision (this run):** proceed **as-is** (non-resonant pumps), **1D only**, as a documented baseline;
revisit the design later.

### Field profiles & FWM overlap (`mode_plots.py` → `mode_analysis/`)

Plots: `epsilon_profile.png`, `mode_profiles.png`, `fwm_overlap.png`; numbers in `overlaps.json`.

- The probe field has a clear standing-wave pattern across the cavity; the **pumps are not cavity-confined**
  — their |Ex|(z) peaks in the mirror/edge regions, and the in-cavity field is only ~0.52 of the global max
  (vs the probe's structured cavity field).
- **FWM spatial-overlap density** |E_probe·E_pump1·E_pump2|(z) is **suppressed inside the cavity and peaks in
  the mirror layers** — so what FWM occurs happens in the (also-SiC, also-χ³) mirror layers, not in the
  cavity, and is not cavity-enhanced.
- Normalized overlap integrals: intensity overlaps probe–pump ≈ 0.143 / 0.155, pump–pump ≈ 0.156;
  triple FWM overlap ≈ 0.346 (these are baselines to compare against a re-tuned design later).

## 1D simulation

Run (canonical, full mode): `faraday_meep_fp_circ.py --dim 1 --mode full --materials fit
--sin-fit sic.csv --sio2-fit sio2.csv --fit-poles 3 --high-index-material sic
--resolution 100 --decay-threshold 1e-6 --pump-intensity 1e12` → `sim_1d/`.
(Convergence check: an earlier `--decay-threshold 1e-3` run gave the same settled value; numbers below are
from the tighter 1e-6 run.)

Confirmed material wiring: high-index slot `SiC`, `n2 = 5×10⁻¹⁸ m²/W` (from the new preset),
n_linear(probe) = 2.664, χ³_SI = 1.26×10⁻¹⁹, E_chi3_meep = 1.78×10⁻². Pumps at 1.657/1.723 µm,
probe at 0.824 µm. Pump intensity 10¹² W/cm², probe 5×10⁷ W/cm².

### Verified source / run parameters

Audited against the source-construction code (`circular_sources`, `linear_sources_45deg`,
`intensity_to_meep_amplitude`, `df_from_pulse_duration`) and the run summary.

| Quantity | Probe | Pump 1 | Pump 2 |
|---|---|---|---|
| Frequency (1/µm) | 1.21334 | 0.60367 | 0.58031 |
| Wavelength (µm) | 0.82417 | 1.65653 | 1.72322 |
| Polarization | 45° linear (Ex=Ey, in phase) | circular σ⁻ (Ey = +i·Ex) | circular σ⁺ (Ey = −i·Ex) |
| Intensity (W/cm²) | 5×10⁷ | 10¹² | 10¹² |
| Source amp (total) | 0.0515 | 7.286 | 7.286 |
| Source amp (per Ex,Ey) | 0.0364 | 5.152 | 5.152 |
| Measured circ. purity | DoLP 0.989 | 0.997 | 0.996 |

- **Counter-rotating pumps**, separation Δf = 0.02336 /µm (**Δλ = 66.7 nm**). FWM frequency-matching:
  2×pump ≈ probe (2×0.604 = 1.208 ≈ 1.213).
- **Temporal:** all three sources are 100 fs Gaussians, **cutoff 4.0**, **fwidth = 0.0462 /µm** (≈ 31 nm at
  the probe, ~127/137 nm at the pumps). ⚠️ The pump spectral width (~130 nm) **exceeds the pump separation
  (67 nm)** — the two pump spectra overlap heavily.
- **Sources injected in air** (n=1) at z ≈ −3.85 µm; `--calibrate-sources` **off**, so the intensities above
  are nominal plane-wave-in-air values used to set source amplitude, **not** calibrated at the cavity.
- **Numerics:** dim 1 (quasi-1D; 3D solver, collapsed transverse cell), resolution 100, PML 1.0 µm,
  cell_z 10.69 µm, complex fields, 3-pole dispersive SiC/SiO₂ fit over 600–2000 nm.

| Metric | Value (decay 1e-6) |
|---|---|
| **Settled probe rotation \|θ\|** | **9.57°** (steady value after pumps fade) |
| Accumulated (unwrapped) transient | 369.6° (probe spins ~1 full turn during the pump pulse) |
| Probe DoLP (tail) | 0.990 |
| Probe ellipticity χ | −3.97° |
| θ std (final window) | ~0.000° (fully settled) |
| Probe S0 / S0_rel_max | 0.034 / 0.131 |
| SNR | 153.4 dB |
| Pump balance P2/P1 | 1.018 |
| Pump purity | 0.997 / 0.996 |

**Convergence (decay 1e-3 → 1e-6):** the settled rotation moves only 9.69° → 9.57° (~1.3%), confirming the
result is converged. The tighter threshold (run time ~21 min vs ~3 min) mainly *cleans up* the measurement:
θ_std drops 0.064° → ~0.000° and SNR rises 50.8 → 153 dB as the residual field decays away.

**Dynamics (see `sim_1d/probe_polarization.png`):** while the pumps are on, the probe polarization
rotates continuously and wraps through ±90° several times (~370° accumulated); after the pumps decay it
rings down and **settles to a stable +9.6°** with DoLP 0.990. This is a **large-signal** result, not a
small perturbative rotation.

**Robustness / a dead-parameter note:** quick-mode and full-mode runs (both at decay 1e-3) give the
*identical* settled value (9.687°). The reason is **not** convergence coincidence: the per-source temporal width is derived solely
from the 100 fs pulse duration (fwidth = 0.0462 /µm for all three sources), because the
`pump_band_nm` / `probe_band_nm` preset fields are **unused dead code** — the `df_from_bandwidth` lines are
commented out at `faraday_meep_fp_circ.py:581–586`. So quick and full use *identical* source bandwidths,
and the only live 1D differences (`runtime_factor`, `src_buffer`) do not affect the settled value.

**Comparison with SiN best_absolute (1D):** 9.57° here vs 0.137° for SiN best_absolute — ~70× larger.
Caveats: (i) n₂ is 10× larger (5e-18 vs 5e-19); (ii) the pumps here are **non-resonant** (no cavity
buildup — the FWM happens in the SiC mirror layers, not the cavity); (iii) the large-signal/wrapping
regime means |θ| is likely **not linear in pump intensity** — a pump-intensity sweep (deferred) is needed
to characterize the scaling and confirm we are above the perturbative regime. So this is best read as a
strong **bulk-like χ³ rotation through the SiC stack**, not cavity-pump-enhanced FWM.

Plots in `sim_1d/`: `probe_polarization.png`, `probe_polarization_zoom.png`, `probe_dft_coherent.png`,
`pumps_dft_coherent.png`, `pumps_td.png`, `probe_band_heatmap_coherent.png`.

## Deep analysis: mechanism, the "70×", and the broadband probe

### Why the rotation is so large vs SiN — it's a regime change, not a clean enhancement

- **χ³ is 16.5× larger.** χ³ ∝ n₂·n²; n₂ is 10× (5e-18 vs 5e-19) and n² is 1.65× (2.66² vs 2.08²) → 16.5×
  (matches E_chi3 ratio 0.0178/0.00108 and χ³_SI 1.26e-19/7.63e-21).
- **The Kerr drive is extreme:** Δn = n₂·I = **0.050 (1.88% of n)** for SiC at I=10¹² W/cm², vs 0.0050 (0.24%)
  for SiN.
- **Perturbative → strong-conversion FWM.** In the probe DFT, SiN sidebands are ~7% of the carrier with
  balanced circular components (|e+|≈|e−|, θ=0.137°); SiC sidebands reach **~64%** of the carrier with a
  ~15% |e+|/|e−| imbalance. That is a *qualitative* jump, not a 16.5× linear scaling.
- **The probe spins, then freezes** (`sim_1d/probe_polarization.png`): while the pumps are on the
  polarization rotates continuously (~**369.6° accumulated**, wrapping through ±90°), then freezes when the
  pumps leave. The reported **9.57° is 369.6° mod 360°** — the wrapped remainder. This is why the settled
  value shifted 9.69°→9.57° between decay 1e-3 and 1e-6 (same unwrapped ~369.6°, different wrap point).
  **⇒ the wrapped settled angle is a poor figure of merit here; use the unwrapped accumulated rotation or a
  low-intensity linear slope.**
- **No bug found.** Units are consistent (χ³/E_chi3 ↔ n₂=5e-18), pumps are counter-rotating at 99.6% purity,
  pump ≫ probe (141× amplitude, undepleted), rotation is read at the probe carrier frequency, DoLP stays
  0.99, and the linear check below gives **exactly 0°**. The behavior is the correct physical consequence of
  a very large n₂·I.
- **Not apples-to-apples vs SiN:** n₂ (10×), geometry, cavity length, *and* pump resonance (SiN resonant vs
  SiC non-resonant) all differ simultaneously, so "70×" conflates several factors.

### Why the probe looks broadband — resolved, it is purely nonlinear

A near-zero-pump control run (`_linear_check/`, pump = 10⁶ W/cm² → Δn ~ 5×10⁻¹¹, decay 1e-4) gives:
**rotation = 0.000°, DoLP = 1.000**, and a probe-band heatmap that is a **single clean line at f=1.213**
(Q≈145) — *not* broadband. So:

- The broadband look in the nonlinear run is **100% nonlinear**, not a multimode-cavity artifact (multimode
  hypothesis **refuted**): it is **strong FWM sideband generation** (sidebands at f_probe ± Δf_pump, the
  lower one ~64% of the carrier), plus **carrier depletion** (probe-carrier S0 drops 0.16 → 0.034 as energy
  converts to sidebands), plus a **Kerr red-shift** of the probe resonance during the pulse (energy appears
  below the cold 1.213 line).
- SiN stays narrowband simply because it is in the weak-FWM (perturbative) regime.

### Recommendations

1. **Pump-intensity sweep** is now the key measurement: map θ(I). Expect a linear (perturbative) low-I region
   — whose slope is the clean, comparable FoM — then wrapping/saturation at high I. Also reveals the I at
   which sidebands stop being perturbative.
2. **Revisit n₂ = 5×10⁻¹⁸ m²/W and/or I = 10¹² W/cm².** n₂ is likely ~5–10× higher than realistic 4H-SiC;
   together with this intensity the run sits far past the perturbative Faraday regime.
3. Quote the **unwrapped accumulated rotation** (or the low-I slope), not the wrapped settled angle.

## 1D pump-intensity sweep (non-resonant pumps) — the χ⁽⁵⁾ test

Run: `pump_intensity_sweep.py --dim 1 --mode full ... --fit-poles 3 --high-index-material sic
--resolution 100 --decay-threshold 1e-4 --intensity-range 1e8 2e12 12 --range-scale log` at the cold
(non-resonant) pumps f₁=0.6037 → `sweep_1d/`.

Use the **wrapped/coherent net rotation** (the clean observable; the *unwrapped* column is corrupted — the
unwrapper miscounts the oscillatory transient when the net angle ≈ 0).

| I (W/cm²) | \|θ\| (deg) | local slope | DoLP | S0 |
|---|---|---|---|---|
| 1e8 | 0.00046 | — | 1.000 | 0.161 |
| 9.0e9 | 0.0119 | +0.89 | 1.000 | 0.159 |
| 5.5e10 | 0.0702 | +1.10 | 1.000 | 0.154 |
| 1.3e11 | 0.216 | +1.25 | 1.000 | 0.145 |
| 3.3e11 | 0.907 | +1.59 | 0.998 | 0.119 |
| **8.1e11** | **5.83** | **+2.07** | 0.990 | 0.052 |
| 2.0e12 | 16.55 | +1.16 | **0.616** | 0.010 |

**Key result — the log-log slope crosses over from ~0.6 (low I) to ~2.0 (8×10¹¹), then breaks down at
2×10¹² (DoLP 0.62).** This matches θ_net(I) ≈ **a·I** (shallow background: the ~1.7% pump imbalance gives a
residual *linear* χ³ carrier term, plus a measurement floor ~10⁻³°) **+ b·I²** (the cascaded **χ⁽⁵⁾** term).
The **I² (χ⁽⁵⁾) scaling is confirmed** (local slope → 2.07 while DoLP is still 0.99), but **not cleanly
isolated**: it only dominates in ~10¹¹–8×10¹¹, squeezed between the linear background below and the
non-perturbative breakdown above. The sweep script's single-power fit (p=1.05) averages the crossover and is
misleading. Plot: `sweep_1d/dim1/rotation_vs_intensity.png`.

**Attempted next step — resonant-pump sweep** (`sweep_1d_resonant/`): the plan was to move pumps onto the
1.87 µm stopband resonance via `cavity_modes_resonant.json` to get pump cavity buildup and a clean slope-2.
**This did not pan out** — see the corrected pump-frequency-scan reading and the resonant sweep below.

## Pump-frequency scan (1D, I=10¹², res 100) → `scan_pumpfreq/`

Seven points, f₁ ∈ {0.535 … 0.685} at fixed Δf, all at the over-driven I=10¹².

> **⚠️ Correction to an earlier claim.** The apparent "moving pumps toward the 1.87 µm resonance gives ~13×
> larger, sign-flipping rotation" was an **artifact of the unwrapper**: the `*_unwrapped_deg` column miscounts
> the oscillatory transient at this intensity (e.g. f₁=0.535 reports −4680°, impossible alongside its DoLP
> 0.999 and 80%-intact carrier). Read the scan through **carrier depletion** `S0_rel_max` (lower = stronger
> FWM conversion) and **intracavity pump balance** p2/p1 instead:

| f₁ (1/µm) | λ₁ (µm) | S0_rel_max | p2/p1 (intracavity) | wrapped θ̄ |
|---|---|---|---|---|
| 0.535 | 1.869 (resonance) | **0.805** (weak) | **0.080** (imbalanced) | 0.38° |
| 0.560 | 1.786 | 0.744 | 2.67 | 0.80° |
| 0.585 | 1.709 | 0.329 | 0.45 | 5.77° |
| **0.6037** | **1.657 (cold/matched)** | **0.131** (strongest) | **1.017** (balanced) | 9.63° |
| 0.625 | 1.600 | 0.232 | 1.23 | 5.01° |
| 0.655 | 1.527 | 0.227 | 0.81 | 3.97° |
| 0.685 | 1.460 | 0.157 | 1.12 | 4.72° |

**What the scan actually shows:** FWM conversion is **maximal at the cold/matched point f₁=0.6037**
(2f_pump ≈ f_probe, balanced pumps), *not* at the 1.87 µm cavity resonance. Detuning the pumps toward
1.87 µm is **doubly counterproductive**: it (a) breaks the 2f_pump≈f_probe frequency match and (b) **destroys
pump balance** — at f₁=0.535 only pump1 resonates, so p2/p1→0.08, which reintroduces the χ³ carrier term that
balanced σ⁺σ⁻ is meant to suppress ([[chi5-faraday-goal]]). The wrapped θ̄ column is itself over-driven
(I=10¹²) and not a clean FoM; the depletion column is the physical readout.

## 3D simulation → `sim_3d/`

Run (`sim_3d/run_sim3d.sbatch`, node005, **96 ranks, ~7.3 h**): `faraday_meep_fp_circ.py --dim 3 --mode full
--materials fit --sin-fit sic.csv --sio2-fit sio2.csv --fit-poles 3 --high-index-material sic
--courant 0.25 --resolution 30 --decay-threshold 1e-4 --pump-intensity 1e12`. `--courant 0.25` keeps the
3-pole fit's f₀≈23 pole FDTD-stable at res 30 ([[meep-fit-stability]]). Pumps stayed balanced and pure in 3D
(p2/p1=1.016, purity 0.99/0.95).

| Quantity | 3D (res 30) | 1D ref (res 100, decay 1e-6) | ratio |
|---|---|---|---|
| **Unwrapped accumulated \|θ\|** | **365.8°** | 369.6° | **0.99×** |
| Wrapped settled (forward-isolated) | −0.14° (DoLP 1.0) | 9.57° (DoLP 0.99) | — |
| Wrapped settled (total-field Stokes) | 5.79° (DoLP 0.78) | — | — |
| Sideband/carrier | ~26% | 64% | — |
| Carrier S0_rel_max | 0.20 | 0.13 | — |

**Headline — no 3D enhancement.** The 3D accumulated rotation (365.8°) equals the 1D value (369.6°) to within
~1%. Contrast SiN best_absolute, which had a **14.5× 3D enhancement** (1D 0.137° → 3D 1.991°). The difference
is mechanism: SiN's pumps were cavity-resonant (3D adds transverse field concentration to enhance); SiC's
pumps are **non-resonant** (stopband shifted to ~1.87 µm), so the FWM is a **bulk-χ³ accumulation in the SiC
layers** — dimension-independent, nothing to enhance in 3D. This confirms the non-resonant-pump diagnosis
from the mode analysis.

**Dynamics** (`sim_3d/probe_polarization.png`): identical to 1D — the probe spins through ±90° several times
during the pump pulse (0–500 fs), rings down, and settles by ~1500 fs. The settled circular components are
nearly balanced (|e⁺|≈|e⁻|≈0.042 in `probe_dft_coherent.png`), so net forward rotation ≈ 0; the spread
between −0.14° / 5.79° / 9.57° is just *where the ~366° accumulation wraps*. **The wrapped settled angle
remains a poor FoM; use the unwrapped accumulation.**

**Two readout methods disagree (−0.14° vs 5.79°) — same Stokes θ = ½·arctan2(S2,S1) on different fields.**
The forward-isolated estimate (`coherent_window_estimate`, method `dft_probe_center_forward_jones`) splits
the forward wave from the backward using E+H (`Hy=n·Ex`, `Hx=−n·Ey`) and keeps only the transmitted wave →
DoLP≈1.0, θ=−0.14° (this is what `objective_quality.abs_rotation` uses). The total-field Stokes
(`probe_stokes_dft`) feeds raw Ex,Ey including the ~16% back-reflected/scattered field (`forward_fraction
0.836`) → DoLP 0.78, θ=5.79°. In 1D both agreed (≈0% back-reflection, DoLP 0.99); in 3D they split because
there is real reflection **and** the true forward signal ≈0, so a small backward admixture swings the raw
read by degrees. Trust the forward-isolated number.

## 1D resonant-pump intensity sweep → `sweep_1d_resonant/`

Run: `pump_intensity_sweep.py --dim 1 --mode full ... --high-index-material sic --resolution 100
--decay-threshold 1e-4 --intensity-range 1e8 1e11 20 --range-scale log` with pumps on the 1.87 µm resonance
(f₁=0.535 / f₂=0.5116, λ 1.869 / 1.955 µm) via `cavity_modes_resonant.json`. **The intended "clean slope-2"
test failed**, for two structural reasons visible in the data:

- **Pump imbalance p2/p1 = 0.093, fixed** at all intensities (pump1 resonates ~11× stronger than pump2) → the
  χ³ carrier term is *not* isolated. (Each pump is individually circular-pure; they're just unequal.)
- **2·f₁ = 1.07 ≠ f_probe = 1.213** — moving onto the cavity resonance broke the FWM frequency match.

Result: a clean but **sublinear** power law **|θ| = 1.13×10⁻⁹·I^0.679 (R²=0.994)** — no slope-2 region — with
*tiny* net rotation (0.042° at 10¹¹). Meanwhile the min/max swing grows to **±24°**: the resonant buildup
*is* driving a large **AC** modulation of the probe polarization, but it **does not rectify to a DC rotation**
because the cascade is frequency-mismatched. (The `*_unwrapped_deg` column is again garbage — −24838° at
I=10⁸.) Plot: `sweep_1d_resonant/dim1/rotation_vs_intensity.png`.

**Comparison at matched intensity** (I=10¹¹): cold/matched sweep ≈ 0.1–0.2° vs resonant ≈ 0.042° — the **cold
(frequency-matched, balanced) config beats the resonant one by 3–5×**, *and* only the cold config shows the
I² χ⁵ signature. ⇒ **frequency matching + pump balance dominate over pump cavity-resonance.**

## Overall conclusions (SiC L=3.2 µm, mirrors unchanged from best_absolute)

1. **The cold/non-resonant-but-matched config (f₁=0.6037) is the best of what was tried** and the only
   legitimate χ⁽⁵⁾ evidence: balanced pumps (p2/p1≈1.0, purity≈1.0), 2f_pump≈f_probe, and a local log-log
   slope that crosses 0.6 → **2.07** over 10¹¹–8×10¹¹ before non-perturbative breakdown at 2×10¹².
2. **No 3D enhancement** (3D ≈ 1D, 0.99×) because the pumps are non-resonant → bulk-χ³, dimension-independent.
3. **The "13× at the 1.87 µm resonance" idea is retracted** — an unwrapper artifact; the careful resonant
   sweep and the depletion-based scan reading both show resonant pumps are *worse*.
4. **Design implication:** genuine resonant χ⁽⁵⁾ enhancement needs the SiC mirrors **re-tuned to move the
   stopband back to ~1.5 µm** so the cavity is *simultaneously* resonant for **both** balanced pumps, keeps
   2f_pump≈f_probe, and lands the sidebands on adjacent modes. The current SiC-swapped-mirror geometry
   satisfies none of these for the pumps.

### Open / next steps
- **Cold-pump intensity sweep in 3D** at the matched point (the real χ⁵-in-3D test; the current 3D run is a
  single over-driven I=10¹² point).
- **Re-tune SiC mirror thicknesses** (stopband → ~1.5 µm) and re-run mode analysis → 1D → 3D.
