# χ⁵ analytic model — pedantic review, audit, and bookkeeping

Status: 2026-06-11. Scope: the FDTD-light analytic FoM (`tmm.py` + `objective.py`) and how it is used in
`hybrid.py`. This document is the authoritative record of **what the model assumes, what it gets right, what
it gets wrong, and every constant/setup**. Read alongside `design_optimization_plan.md` (the roadmap) and the
memory note `chi5-design-optimization-plan`.

TL;DR: the corrected `chi5_score_v2` fixed the *sign* of the geometry scaling (Spearman vs FDTD flipped from
−0.92 to +0.87 on the L-family) and made the 100 fs buildup correct (flat). But the pedantic audit below finds
a **buildup normalization error** (pump buildup enters the FoM squared; the probe enters squared) and a
**CW-evaluated symmetry-break** that cannot pick the operating point. These are masked on the validated
L-family (buildup is flat there) but compromise cross-geometry pre-ranking. The hybrid mitigates by letting
1-D FDTD own the operating point and final ranking; a `v3` is recommended (§9).

---

## 1. Physical model & derivation chain

Source of truth: `faraday_chi3/tcmt_derivation.tex` (the trusted derivation), cross-checked against
`isotropic_derivation.tex`. Five carriers: pumps A₁(ω₁), A₂(ω₂), probe a± (ω_s, the two circular
components), sidebands b± (Ω± = ω_s ± Δ, Δ = ω₁ − ω₂). Balanced counter-rotating σ⁺σ⁻ pumps.

Time-domain TCMT (tcmt_derivation eq. 122–141), after adiabatic elimination of the sidebands (eq. 164–187):

    ȧ₊ = [iΔ_s − κ_s/2] a₊ + i(Φ₊ + G₊) a₊      G₊ = η₋ζ₋·|A₁|²|A₂|² / (½κ_{b−} − iΔ_{b−})
    ȧ₋ = [iΔ_s − κ_s/2] a₋ + i(Φ₋ + G₋) a₋      G₋ = η₊ζ₊·|A₁|²|A₂|² / (½κ_{b+} − iΔ_{b+})

with Φ± the direct (χ³) self/cross-phase shifts and G± the **cascaded χ⁵** mixing. Accumulated rotation and
ellipticity of the 45° probe:

    θ      = ½ ∫ Re(σ₊ − σ₋) dt        (rotation channel)
    ellipt = ½ ∫ Im(σ₊ − σ₋) dt        (ellipticity channel),   σ± = Φ± + G±

**Key facts that constrain any FoM:**
- *Balance nulls the direct χ³ rotation:* with σ⁺σ⁻ pumps Φ₊ = Φ₋, so the rotation comes only from the
  cascade G± (the χ⁵ effect we want to isolate). [[chi5-faraday-goal]]
- *Rotation = Re, ellipticity = Im.* A ω_s±Δ-**symmetric** cavity gives σ₊ = σ₋\* ⇒ Re(σ₊−σ₋)=0 = pure
  ellipticity. **Rotation requires breaking the ω_s±Δ symmetry** (cavity dispersion / asymmetric mode
  placement / probe detuning). This is the M5 lever.
- *Intensity order:* G ∝ |A₁|²|A₂|² ⇒ θ ∝ I_pump² (the χ⁵ signature; FDTD intensity scans confirm slope→2).
- *Rotation angle is independent of probe intensity* (it is a polarization phase): θ must NOT scale with the
  probe drive amplitude. (See the audit, §5.2 — v2 violates this.)
- *Adiabatic elimination assumes "fast" sidebands* (|Δ_b − iκ_b/2| large, eq. 164). In the validated
  **small-Δ** regime the sidebands sit *inside* the probe line (Δ ~ κ_s), so they are NOT fast — the discrete
  adiabatic formula is out of its stated regime. v2 sidesteps this by using the continuous cavity response
  (§3), which is strictly more general, but the η,ζ overlap structure it inherits came from that elimination.

Empirical anchor (FDTD, `decompose/`): θ ∝ L^+1.21, with intracavity **buildup FLAT in L** (pump |A₁|²|A₂|² ~
L^−0.08, probe ~ L^+0.05) and the growth carried entirely by the cascade/length. So the rotation is a
**propagation effect**: θ ≈ (pump buildup, flat) × (probe dwell) × (interaction length ∝ L) × (symmetry-break).

---

## 2. The TMM linear engine (`tmm.py`)

- 1-D characteristic-matrix TMM. Index **N = n − ik** (Meep e^{−iωt}; +ik gives R+T>1 gain — a fixed bug).
- `spectrum` (vectorized over frequency), `find_mode`/`find_modes_in_band` (transmission-peak comb; pick the
  peak **nearest** the target, not the tallest), `mode_Q` = transmission **group-delay** Q = (f₀/2)|dφ_t/df|.
- `field_profile`: complex E(z) normalized to unit **transmitted** wave.
- **`cav_field(f, z)` (keystone, added v2):** complex intracavity E(z) per unit **incident** amplitude
  (divides out E_inc = (E_air + H_air/n_inc)/2). |cav_field|² = physical intensity enhancement; the complex
  value carries the true cavity dispersion. For a symmetric Lorentzian cav_field(ω_s−Δ)=cav_field(ω_s+Δ)\*
  ⇒ Re difference = 0 (the symmetric-cavity null is built in).

Validated vs committed Meep modes: frequencies <0.3–0.7 %, SiC probe Q 123–165 vs FDTD 145; SiC pumps
non-resonant (Q nan) in BOTH TMM and FDTD (reproduces the documented SiC pump-buildup failure). Reflectance
overlay matches (RMS ≈ 0.12, sub-nm fringe phase from raw-CSV n(λ) vs Meep's Lorentz fit).

---

## 3. The v2 FoM, exactly as implemented (`objective.py`)

    chi5_score_v2:  fom_rotation = (B₁ · B₂ · Bs) · |Re Σ| · L_interaction
                    fom_ellipticity = (B₁ · B₂ · Bs) · |Im Σ| · L_interaction

- **Bᵢ = buildup_100fs(fᵢ)** = 100 fs pulse-spectrum-weighted, cavity-region-averaged |cav_field|²
  (§4). B₁,B₂ = pumps, Bs = probe.
- **Σ = corrected_cascade** = e_m·z_m − e_p·z_p, the two cascade arms, with (uⱼ = cav_field(fⱼ)):
      z_m = (3Ω₋/8)(2χ) ∫_cav ū_bm u₂ ū₁ u_s dz     e_m = (3ω_s/8)(2χ) ∫_cav ū_s u₁ ū₂ u_bm dz
      z_p = (3Ω₊/8)(2χ) ∫_cav ū_bp u₁ ū₂ u_s dz     e_p = (3ω_s/8)(2χ) ∫_cav ū_s u₂ ū₁ u_bp dz
  Same conjugation/phase structure as the legacy `counter_coefficients` (which matched the Meep extraction to
  1.1–1.4×, η=conj(ζ) to 2 %), but with **physical cav_field** instead of energy-normalized modes and the
  **true cavity response at ω_s±Δ** (u_bp, u_bm) instead of a snapped sideband Lorentzian 1/D.
- **L_interaction** = ∫ χ³-region dz (high-index cavity+mirror layers) ≈ ∝ L (the probe accumulates rotation
  over this length).
- χ enters as χ_iso (= 1 for ranking within a material; ranking-invariant. Real χ only needed cross-material).

---

## 4. The 100 fs treatment ("100 fs everywhere", per user)

Both pump and probe FDTD sources are **100 fs Gaussians** (`df_from_pulse_duration(100)` → fwidth ≈ 0.0462
1/µm; `faraday_meep_fp_circ`). Q_cap = f/fwidth ≈ 27 (probe band) / 14 (pump band). **All cavity modes have
Q > Q_cap** ⇒ the cavity line is narrower than the source ⇒ buildup **saturates** (does NOT scale 1/Q, nor
∝Q — it is FLAT in L). `buildup_100fs` reproduces this by weighting |cav_field(f)|² with the Gaussian source
spectrum (9-point, ±3σ): validated flat (Bp ~ L^−0.03) against FDTD (L^−0.08).

⚠️ The 100 fs limit is applied to the **buildup only**. The cascade Σ uses the **CW** cav_field at each
carrier; it is not a time-domain pulse convolution. Consequence: the symmetry-break's frequency/Δ dependence
is CW, not pulse-averaged (§5.3).

---

## 5. PEDANTIC AUDIT — known errors and their impact

### 5.1 Buildup over-counting (the main quantitative error)
Σ = e·z is a product of **eight** cav_fields → e_m·z_m = u_s² u₁² u₂² u_bm² ⇒ **|Σ| ∝ Bs·B₁·B₂·B_sb**
(one buildup power per carrier). The FoM then multiplies by B₁·B₂·Bs **again**, so

    fom_rotation ∝ B₁² · B₂² · Bs² · B_sb · L_interaction .

- **Pumps enter squared (B₁²B₂² ⇒ I_pump⁴ ⇒ χ⁹-like).** Correct χ⁵ is I_pump² (B₁B₂, once each). The explicit
  B₁·B₂ factor is a double-count and should be removed (the cascade already carries the pump buildup).
- **Probe enters squared (Bs²)** — see 5.2.
- *Impact:* on the L-family buildup is FLAT, so this is a constant multiplier and the L-validation (Spearman
  +0.87, exponent +1.67) **still holds**. **Cross-geometry it distorts the pre-rank**, over-rewarding
  high-Q/high-buildup geometries. Symptom: all top-10 SiN Stage-A candidates are n=4 (more pairs → higher
  buildup, amplified by the squared powers); a good n=3 could be crowded out of the top-K.

### 5.2 Spurious probe-intensity dependence
The rotation **angle** is independent of probe drive amplitude. But Σ carries u_s² and the FoM adds Bs, so
fom ∝ Bs² (probe buildup squared). Physically the probe should enter only as a **dwell/path** enhancement
(linear, ∝ the number of cavity passes ~ Q_s), not as an intensity². Over-counts probe-cavity coupling.

### 5.3 CW symmetry-break ⇒ cannot pick the operating point
Re Σ is a delicate **difference** of two nearly-equal arms (the symmetry-break). Evaluated CW (not
pulse-averaged) on the discrete cavity response, it is noisy in (center, Δ): v2's operating-point argmax is
center 0.643 / Δ 0.040, vs the FDTD optimum center 0.665 / Δ 0.022, and it does **not** reproduce the
FDTD/Δ-scan result that |θ| rises smoothly as Δ→small (the absorptive M5 lineshape ∝ Γ_s/(Γ_s²+Δ²)). ⇒ **v2
must not be trusted to choose the operating point.** (Hybrid Stage B = FDTD owns it.)

### 5.4 Other approximations (each a restriction, not necessarily an error)
- **Scalar fields.** cav_field is scalar; the σ± circular structure and the Maker–Terhune A,B,C tensor are
  collapsed to a single χ_iso with the (A+B)=2χ isotropic prefactor. No cross-polarization linear mixing.
- **1-D / quasi-1D.** No transverse mode, diffraction, or focusing. Matches the FDTD's collapsed-transverse
  quasi-1D, but the 3-D SiC result (NO enhancement, non-resonant→bulk-χ³; see [[sic-chi5-study]]) is **outside
  this model's scope** — 1-D cannot see the 3-D buildup failure.
- **Adiabatic-elimination heritage at small Δ** (§1) — formula used outside its stated "fast sideband" regime.
- **Balanced, perfect σ⁺σ⁻, |A₁|=|A₂|** assumed (FoM uses symmetric B₁·B₂).
- **Perturbative / linear** in the cascade self-energy — no saturation; a relative FoM for ranking χ⁵
  *potential*, not an absolute-θ predictor (consistent with the FaradayJL perturbative-vs-saturated gap).
- **Material dispersion mismatch:** TMM interpolates raw-CSV n(λ); FDTD uses a 2-pole (SiN) / 3-pole (SiC)
  Lorentz fit. Sub-nm fringe-phase differences (RMS ≈ 0.12 reflectance).
- **Buildup proxy:** cavity-region **average** of |cav_field|² (a point/avg proxy for the mode energy), not a
  true ∫ε|E|² mode energy.
- **Probe-mode choice:** highest-Q transmission peak in the probe windows (a heuristic, not guaranteed to be
  the experimentally relevant mode).

---

## 6. Validation evidence — v2 vs every FDTD study

| Quantity (SiN best_absolute family unless noted) | FDTD result | v2 analytic | verdict |
|---|---|---|---|
| Geometry scaling θ(L), Spearman | L^+1.21 | L^+1.67, **+0.87** | ✅ sign fixed (legacy −0.92) |
| 100 fs buildup vs L | flat (L^−0.08) | flat (L^−0.03) | ✅ |
| Operating point (center, Δ) | 0.665 / 0.022 | 0.643 / 0.040 | ❌ → FDTD owns it |
| Δ-dependence of |θ| | rises as Δ→small (M5) | noisy, no clean trend | ❌ |
| Intensity scaling | θ ∝ I² (slope→2.14) | ∝ I_pump² in B₁B₂ (but I⁴ via 5.1) | ⚠️ order ok, power over-counted |
| Material SiC ≫ SiN | yes | yes (via χ) | ✅ |
| Cross-geometry ranking (N, t_hi, t_lo) | — (FDTD-expensive) | **untested**; 5.1 suggests biased | ⚠️ hybrid Stage B covers it |
| Legacy proxy validation gate (Phase 1) | candidates 3–10× WORSE | (the bug v2 fixes) | — (hybrid re-test pending) |

---

## 7. Restrictions / domain of validity (summary)
Use v2 to **rank geometry** (N, thicknesses, L) within a material, at a fixed reasonable operating point.
Do NOT use it to (a) choose the operating point/Δ, (b) predict absolute θ, (c) compare geometries with very
different buildup without the 5.1 fix, (d) say anything about 3-D, transverse, or saturation physics.

---

## 8. Configuration / setups (all constants, one place)

**`objective.py`:** `FWIDTH_100FS = 0.04625` 1/µm (100 fs); `q_cap(f)=f/fwidth`; `buildup_100fs` 9-pt ±3σ.
**`optimize.py` / `hybrid.py`:** `PROBE_WINDOWS = [(0.790,0.810),(0.850,0.950)]` µm; `PUMP_BAND = (1.40,1.95)`
µm; pre-rank `PRERANK_DELTA = (0.012,0.025)`; Stage-B `FDTD_DELTAS = [0.015,0.022,0.030]`,
`FDTD_MAX_CENTERS = 5` (resonant pump modes by Q). Octave (f₁+f₂≈f_s) constraint **removed** (FDTD-refuted).
`phase1_bounds`: [n_pairs±1, t_hi·(1±0.30), t_lo·(1±0.30), L·(1±0.30)] (scope "regime"; 0.12 for "local").
**Materials (`MAT`):** sin = fit si3n4.csv+sio2.csv, 2 poles, window 600–2000 nm, I=1e12 W/cm²,
base SiN_optimizations/best_absolute. sic = sic.csv+sio2.csv, 3 poles, I=2e11, base SiC_optimizations/sic_L3p2um.
**FDTD:** `--dim 1 --mode full`, RES=80, DECAY=1e-4 (1e-5 for converged; see [[decay-threshold-defaults]]),
100 fs pulses, substrate SiO2. Run env: micromamba `meep-mpi`; parallel-1D fan-outs on node00x (90 workers).

---

## 9. v3 — clean normalization (IMPLEMENTED 2026-06-11)
`objective.cascade_normalized` + `chi5_score_v3`:

    θ_fom = B₁ · B₂ · Bs · |Re Σ̂| · L_interaction

- **Σ̂ uses normalized fields:** pumps & probe peak-normalized (shape, |peak|=1); **sidebands normalized by
  the PROBE peak** so their resonance *relative to* the probe (the M5 lever) and the ±Δ dispersion asymmetry
  (the symmetry-break/rotation) survive while the absolute buildup is removed. ⇒ Σ̂ carries NO buildup.
- Buildup appears **exactly once per carrier**: B₁·B₂ (pumps = I_pump²) × Bs (probe, LINEAR). Fixes §5.1/5.2.
- `hybrid.py` Stage-A pre-rank now calls `chi5_score_v3`.

**Validation:** L-family Spearman vs FDTD **+0.84** (v2 was +0.87) — the fix preserves the validated geometry
trend; exponent L^+1.76. The §5.1 over-count is removed by construction (cascade is shape-only).

⚠️ **Inconclusive empirical separation v2-vs-v3, and a deeper buildup-magnitude caveat:** the natural test
(N-sweep at fixed L) was inconclusive because in these designs the probe mode is **low-finesse** (Q≈86, mode
order ~30 ⇒ finesse ~3 ⇒ cav_field on-resonance |u|≈1.1, no strong enhancement — physical, not a bug) and the
probe mode **drifts relative to the fixed mirror stopband** as N changes, so `buildup_100fs` is weak and erratic
across N. This is a **buildup-magnitude modeling limitation shared by v2 and v3** (separate from the §5.1
normalization fix): `buildup_100fs` = cavity-region-averaged, 9-pt pulse-weighted |cav_field|², which (a) dilutes
the defect enhancement with the lower mirror-layer field and (b) under-resolves narrow lines (9-pt spacing 0.029
> line width 0.015). It is adequate where buildup is genuinely flat (the validated L-family) but is the weakest
link for cross-geometry ranking. FOLLOW-UP (not blocking): a defect-region buildup with adaptive on-line
sampling, validated against an FDTD N-sweep. The operating point (§5.3) stays with FDTD regardless.
