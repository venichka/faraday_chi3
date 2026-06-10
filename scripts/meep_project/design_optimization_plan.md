# χ⁽⁵⁾ Faraday-rotation design optimization — plan

Master plan for finding cavity geometries that **maximize the all-optical (effective χ⁽⁵⁾) probe
rotation**. Grounded in the trustworthy derivations `../../isotropic_derivation.tex` §1 +
`../../very_general_derivation.tex` (§1 FoM, re-derived) and `../FaradayJL/src/FaradayJL.jl` (cavity TCMT).
Companion to the project goal — the cascaded χ³·G·χ³ sideband effect under balanced σ⁺σ⁻ pumps, **not** the
χ³ carrier term.

> Status: **planning doc only** (2026-06-09, pedantically revised). No new pipeline built yet. Execution is
> staged below. Key constraint: **pumps fixed at ~100 fs** (§0) — this caps pump Q and reorders the levers (§3).

> ✅ **Theory grounding (re-derived 2026-06-09).** §1 and the §2b lineshape were **re-derived from the
> trustworthy docs** `../../isotropic_derivation.tex` (§1, counter-rotating σ⁺σ⁻ — exactly our config) and
> `../../very_general_derivation.tex` (basis-free tensor confirmation), replacing the earlier reliance on the
> possibly-erroneous `chi3_sideband_patch.tex`. **The FoM scaling survives:** the cascade is genuinely
> ∝ |E₁|²|E₂|² (χ⁵) with pump phases cancelling exactly, and the Q-buildup/sideband-Lorentzian structure holds.
> The patch's *formula* was consistent on these points; its loose *prose* ("dispersive, max at |Δ|~Γ_s") was the
> unreliable part. Two refinements the trusted derivation adds: (a) the pump factor is |E₁|²|E₂|² with **no**
> residual 2·arg phase (the old "(p₂p₁*)²" note was wrong); (b) **net rotation needs the ω_s±Δ sideband symmetry
> broken** — a perfectly symmetric, non-dispersive cavity gives pure ellipticity, not rotation (a new design
> lever, §1/§2b). **FaradayJL cross-check DONE (2026-06-10):** found & fixed a bug — the counter-rotating
> back-mixing used the un-conjugated pump dyad → cascade ∝ (p₂p₁*)² instead of |p₁|²|p₂|²; numerically it
> inflated rotation ~1.9× and mis-split rotation/ellipticity (demo `FaradayJL/examples/bug_demo_counter_backmix.jl`).
> The measured Δ-scan + intensity-scan remain the primary, derivation-independent evidence.

## Scope (agreed)

- **Two material systems**, low-index is **always SiO₂**:
  - high-index = **SiC** (4H-SiC, `sic.csv`, n₂ preset in `nonlinear_materials.py`) — **do first**,
  - high-index = **SiN** (Si₃N₄, `si3n4.csv`) — second.
- **Operating-point design variables (IMPORTANT — these ARE search variables, not fixed):**
  - **Probe wavelength** is constrained but not single-valued: **λ_s ≈ 800 nm (or very close)**, *or*
    **λ_s ∈ [850, 950] nm**. Pick to land the probe on a good cavity mode within these windows.
  - **Pump wavelengths are free and must be tuned** (IR, ~1.5–1.8 µm region for the prior designs; user calls
    it mid-IR), subject to the FWM matching (working assumption f₁+f₂ ≈ f_s, near-octave; confirm per design).
  - **Δ = |f₁−f₂| follows from the tuned pumps** — a real search direction; **measured optimum is small,
    ≈ Γ_s ≈ the cavity linewidth (§2b), NOT the source bandwidth.** Pulse *durations* are separately fixed at
    ~100 fs (§0) — distinct from the wavelength freedom here.
- **Per material, two steps:** (a) a **quick refinement** with the *existing* pipeline, then
  (b) run the **new χ⁵-targeted search machinery**.
- **Hard restriction — the existing pipeline stays fully functional and ideally untouched.** The new
  optimizer is a **separate entity** (new files) that may *import/reuse* existing modules
  (`geometry_io`, `mode_targeting` helpers, `faraday_meep_fp_circ.run_simulation`, the BO from
  `optimize_cavity_geometry`) but must not modify their behavior. `optimize_cavity_geometry.py`,
  `optimize_cavity_geometry_mf.py`, `pump_intensity_sweep.py`, `faraday_meep_fp_circ.py` keep working as-is.
- **Rich, meaningful output per optimized design** (see §6): mode analysis, field distributions, the
  three-θ rotation readouts, intensity sweep, and (optionally) a TCMT cross-check via `extract_tcmt_params*`
  → FaradayJL.

---

## 0. Hard constraint — fixed ~100 fs pumps (reshapes the whole strategy)

Pump pulse duration is **fixed at ~100 fs** (not a free variable). This is decisive because it **caps the
pump-mode quality factor**:

- A 100 fs pump has a large bandwidth: transform-limited FWHM Δν ≈ 0.44/τ ⇒ Δλ ≈ 30 nm at 1.5 µm, and the
  Meep source (`fwidth`, cutoff-truncated) is broader, ~120–130 nm (see the SiC `analysis.md`).
- A cavity mode of quality factor Q has linewidth δλ = λ/Q. A pump only builds up the **resonant slice** of
  its spectrum, so raising Q beyond **Q_cap ≈ λ/Δλ_pump ≈ O(10–40)** does **not** increase the in-cavity pump
  energy — it just rejects more of the (fixed) pulse. The intracavity intensity **saturates**.

**Consequences (carried into §1, §3, §8):**
- The "Q₁Q₂ buildup is the biggest lever" idea (still true in the CW limit) is **bandwidth-limited**: you
  **bandwidth-match** the pump modes (Q_pump ~ O(10)), you do not maximize them. "Lengthen the pumps to raise
  Q_pump" is **off the table**.
- **Asymmetry pump vs probe/sideband:** the *pump buildup* needs total in-cavity energy ⇒ bandwidth-capped.
  The *probe* rotation is read at the carrier DFT bin (the on-resonance slice), so a **high-Q probe mode is
  still useful** even with a 100 fs probe; likewise the **sidebands are generated inside the cavity** and
  benefit from their own Q_Ω. So Q_s and Q_Ω remain real levers; Q_pump does not.
- The biggest *practical* gain is therefore **recovering the buildup the SiC design threw away** (its pumps
  were entirely off-resonance) at bandwidth-matched Q, then winning on **mode volume, overlap, and the
  sideband/probe resonances** — not on pump Q.
- **Probe pulse is ALSO fixed at ~100 fs** (user, 2026-06-09) — *duration*, distinct from the probe-wavelength
  freedom (Scope). So **every** Q in the FoM (pump, probe, sideband) is bandwidth-capped at Q_cap ~ O(10–40):
  there is **no** Q lever and **no** narrow-linewidth regime to exploit (a mode sharper than the source
  bandwidth buys nothing). ⇒ The **only** intensity/coupling levers are (i) **alignment** (get all modes
  resonant at bandwidth-matched Q), (ii) **mode volume / field concentration**, (iii) **4-mode overlap**, and
  (iv) **material χ³**. **Q-chasing is spent.** Δ and the operating frequencies remain genuine **search
  variables** (Scope); the Δ-scan (§2b) shows **Δ_opt is small ≈ Γ_s (the cavity linewidth)** — drive Δ toward
  it (sidebands on the probe mode), *not* toward the source bandwidth.

## 1. Figure of merit, derived

**Source (re-derived):** `isotropic_derivation.tex` §1 (isotropic χ³, **counter-rotating σ⁺σ⁻** — exactly the
project configuration) gives the probe polarization in closed form; `very_general_derivation.tex` confirms the
same structure basis-free (no isotropy/Kleinman). Both are the trustworthy docs (top caveat). FaradayJL
`rhs_counter_derived!` is the same loop in cavity-TCMT form.

**Step 1 — the direct χ³ carrier cancels.** The direct SPM/XPM response is diagonal in the circular basis with
(isotropic_derivation, Eq. for χ₊₊−χ₋₋)

  χ₊₊ − χ₋₋ = |E₁|²(B_s⁽¹⁾ − C_s⁽¹⁾) − |E₂|²(B_s⁽²⁾ − C_s⁽²⁾).

For **balanced pumps |E₁| = |E₂|** (or B=C) this vanishes → **no χ³ Faraday rotation.** That is the whole
purpose of σ⁺σ⁻ balance: null the carrier, leaving only the cascade.

**Step 2 — the cascaded χ⁵ term.** Probe sidebands at Ω_± = ω_s±Δ are generated (∝ E₁E₂*·E∓), propagate via
the linear cavity Green function G(Ω_±), and back-mix with the *opposite* pump pair to return to ω_s. The
result (isotropic_derivation, boxed `P^casc`) is a **diagonal** circular correction:

  δχ₊₊ ∝ |E₁|²|E₂|² · Π₋(Ω₋),   δχ₋₋ ∝ |E₁|²|E₂|² · Π₊(Ω₊),
  Π_±(Ω_±) = (B_±^mx + C_±^mx)(B_±^sb + C_±^sb) · G_{±±}(Ω_±).

**Key:** the pump amplitudes enter as (E₂E₁*)(E₁E₂*) = **|E₁|²|E₂|²** — the phases cancel **exactly** (no
residual 2·arg phase; the earlier "(p₂p₁*)²" note was wrong — isotropic_derivation note (i): "all residual
phase sensitivity resides in G"). This is the χ⁵ / I_pump² scaling, manifest. `very_general` Eq. (Θ⁽⁵⁾)
reproduces it basis-free (its pump dyads M⁽ᵐⁿ⁾M⁽ᵐⁿ⁾ = |E₁|²|E₂|²).

**Step 3 — rotation.** The probe rotation is the real circular-birefringence (ellipticity = the imaginary part):

  θ_F = (k₀L/4) · Re[ χ₊₊ − χ₋₋ ] = (k₀L/4)·(¾ε₀)²·|E₁|²|E₂|² · Re[ Π₋(Ω₋) − Π₊(Ω₊) ].

The cavity enters via **G_{±±}(Ω_±) ∝ 1/(½κ_Ω − iΔ_Ω)** (a mode Lorentzian at the sideband), the pump buildup
**|E_i|²(intracavity) ∝ Q_i|S_i|²** (energy ∝ Q×input; on resonance |p_i|²=4|S_i|²/κ_i, the CW limit —
bandwidth-capped for 100 fs pumps, §0), and the probe interaction length **L_int ∝ Q_s**. With κ=ω/Q this gives
the master scaling (read with the §0 bandwidth caveat):

```
 θ_F  ∝   k0·Lint  ·  |S1|²|S2|²  ·  Q1·Q2  ·  Qs  ·  QΩ·ℒ(ΔΩ/κΩ)  ·  η·ζ
          └ k0L ┘     └ input ┘    └buildup*┘ └probe┘ └ sideband ┘   └overlap┘
   (*Q1·Q2 capped at ~Q_cap² by the fixed pump bandwidth — see §0;  η·ζ = the B,C overlap products in Π_±)
```

**Refinement the trusted derivation adds (new design lever).** With *real, non-dispersive* χ³ coefficients and
a *±Δ-symmetric* cavity (G₊₊ at ω_s+Δ the mirror image of G₋₋ at ω_s−Δ), Π₋(Ω₋)−Π₊(Ω₊) is **purely imaginary**
→ Re=0 → **pure ellipticity, zero net rotation.** Net rotation **requires breaking the ω_s±Δ symmetry**:
(i) cavity asymmetry — place the probe mode so the two sidebands see *different* G (different detuning/Q),
(ii) χ³ dispersion across the two arms, or (iii) a probe-carrier detuning. So a good design *deliberately* makes
the upper/lower sidebands inequivalent. (The empirical absorptive Δ-shape — |θ| max at Δ→0, §2b — means the SiC
cavity's asymmetry is **structural**, surviving Δ→0; smooth χ³ dispersion alone would vanish at Δ=0.)
k0 = ω_s/c also mildly favors the higher probe frequency. Everything below pushes one factor up **within §0**.

## 2. Resonance conditions (must hold simultaneously)

The pumps are a close doublet at ω₁, ω₂ (beat Δ = ω₁ − ω₂); the probe at ω_s acquires sidebands at
**ω_s ± Δ** (patch, "Cross-Tone" terms). So the cavity must place a **pump doublet + a probe triplet** on a
common comb of spacing Δ:

| band (~λ) | modes (spacing Δ) | role |
|---|---|---|
| pump (~1.5 µm) | ω₁, ω₂ | buildup → Q₁Q₂ (bandwidth-capped, §0) |
| probe (~0.8 µm) | ω_s−Δ, ω_s, ω_s+Δ | probe + its two sidebands |

> **Correction (pedantic):** an earlier draft wrote the triplet as {2ω₂, ω₁+ω₂, 2ω₁}. That identification only
> holds if the probe is placed *exactly* at the pump sum ω_s = ω₁+ω₂. The χ⁵ sideband mechanism does **not**
> require it — in the patch ω_s is arbitrary and the sidebands are simply ω_s ± Δ. The project puts the probe
> near the **octave** ω_s ≈ 2ω_p mainly to **spectrally separate** probe from pumps (clean readout) and for the
> small k0 = ω_s/c gain — it is a *design choice*, not a condition of the effect. (In the real runs 2ω₁ = 1.207
> vs f_probe = 1.213, i.e. ≈0.5% off the exact second harmonic — consistent with "near-octave, not exact-sum".)

- **M1 probe placement / dispersion:** put ω_s in a mirror stopband, well separated from the pumps; the
  comb just needs the triplet {ω_s, ω_s±Δ} on modes. Requires probe-band FSR ≈ pump-band FSR ≈ Δ (set by L + mirror phase).
- **M2 comb match:** pump spacing Δ = probe-band FSR (both ≈ c/2n_gL ⇒ a single FP defect gives this; set by **L**).
- **M3 balance:** Q₁≈Q₂, equal coupling ⇒ p₂/p₁≈1 (keeps the χ³ DC term cancelled).
- **M4 dual-band stopband:** high mirror reflectivity at **both** ~1.5 µm and ~0.8 µm.
- **M5 sideband-on-mode:** ω_s ± Δ land on high-Q probe-band modes (Δ_Ω→0). **The χ⁵ resonance the SiC design lacked.**

Cross-check with prior runs: best_absolute (SiN) won in 3D because *detuned* mirrors (t_SiO₂/t_SiN≈1.45)
open a stopband at the octave (a quarter-wave stack is transparent at λ/2 — half-wave "absentee" layers, so
its 2nd-order stopband is suppressed; detuning re-opens it) → M4 for the probe. The all-SiC design failed
because the stopband left the pumps (no buildup) and the retune broke M3 + M5.

## 2b. What the existing optimizer constrains (audit) + the Δ question

**Audit of `mode_targeting.py` / `optimize_cavity_geometry.py`:** the *current code* **hardwires** the
operating point (this is a limitation to fix in the new pipeline, not the physical constraint):
- ω_s, ω₁, ω₂ are **not searched** — they are module constants `LAM_PROBE=0.800`, `LAM_PUMP1=1.550`,
  `LAM_PUMP2=1.650` µm; the geometry is searched to land *modes* on them.
- The sum/near-octave relation is baked into those numbers: 1/0.800 = 1.250 ≈ 1/1.550 + 1/1.650 = 1.251, i.e.
  f_s ≈ f₁+f₂ *by construction*, not enforced dynamically.
- Δ = |f₁−f₂| is **derived** from the fixed pump wavelengths (`build_modes_spec`): 0.039 /µm (~100 nm in λ);
  **sidebands placed at f_s ± Δ** and scored against cavity resonances. `_comb_midpoint_penalty` pulls the two
  SH-band dips symmetric about the probe (≈M5/M1); `_fsr_penalty` softly targets 100 nm pump-band spacing.
- Weights: probe locked hard (`ALPHA_DETUNE` 80), pumps soft (25), FSR very soft (`ALPHA_FSR` 2) — realized
  designs drift (best_absolute 52 nm, best_ratio 182 nm) despite the 100 nm target.

⇒ The existing pipeline encodes M1–M5 reasonably **but the operating point is hardwired.** Per the Scope,
the **new pipeline must make the operating point a first-class search:** probe in {≈800 nm} ∪ [850, 950] nm,
**pumps tuned** in the IR under the matching, and **Δ a real search direction** — plus the overlap and
mode-volume terms.

**Is there an optimal Δ in the *real* system? — MEASURED (1D Δ-scans, `chi5_optimization/delta_scan/`,
2026-06-09). Δ_opt is SMALL ≈ Γ_s ≈ the probe cavity linewidth (~0.004–0.01 /µm, ~10–20 nm), NOT the source
bandwidth.** Two hypotheses I floated here were both wrong and are recorded so we don't repeat them:

> ⚠️ **Corrected twice.** (1) A "source-bandwidth Δ_opt ≈ 0.03–0.05 /µm" claim — **refuted** (|θ| at Δ=0.046 is
> ~30× below Δ=0.006). (2) A "the small-Δ rise is a run-length/DFT-leakage artifact" claim (the first scans
> stopped at T≈300 regardless of Δ, so at small Δ <1 beat was captured) — **tested and refuted**: a
> beat-resolved re-scan (T = 18/Δ, ≥18 beats/point; driver `--beats`) reproduces the rise *slightly larger*.
> The rise is **real**.

Measured on SiC L=3.2 µm (fix probe + pump-mean, balanced σ⁺σ⁻ at I=2×10¹¹, vary Δ; resolved scan, three-θ):
- **|θ| rises monotonically as Δ → 0** with pumps perfectly balanced (p2/p1=1.00 for Δ≤0.023): θ = 2.22°
  (Δ=0.004) → 1.53° (0.010) → 0.43° (0.0234), sign-flips ~0.033, small/negative beyond (where p2/p1 also
  degrades to 0.5–0.8 as the pumps separate — a second penalty on large Δ).
- **Theory-consistent (re-derived from `isotropic_derivation` §1 — see §1).** θ_F ∝ Re[Π₋(ω_s−Δ) − Π₊(ω_s+Δ)],
  the Π_± carrying the sideband Green function G(ω_s±Δ) ∝ 1/(½κ_s ∓ iΔ) (sidebands inside the probe mode, M5).
  The **magnitude** peaks at Δ→0 with half-width Γ_s=½κ_s; the data fit θ=Re{A/(Γ_s−iΔ)} gives **Γ_s≈0.011/µm**
  (a few × the probe ½-linewidth), sensible. The fact that the measured shape is **absorptive (max at Δ=0)**,
  not dispersive, means the cavity's upper/lower-sideband asymmetry is **structural** (survives Δ→0); a
  perfectly symmetric, non-dispersive cavity would instead give pure ellipticity (§1 refinement). This now rests
  on the trustworthy docs, **not** the patch — whose formula agreed but whose prose ("max at |Δ|~Γ_s") was wrong.
- **Physical picture:** for Δ ≲ κ_s the sidebands at f_s±Δ fall *inside the probe mode* → the cascade is
  resonantly enhanced (the **M5 lever**). Balanced pumps zero the DC χ³, so this is the χ⁵ loop.
- **Caveats / open:** DoLP sags to ~0.97 at small Δ (the dichroism/ellipticity Im δχ that accompanies the
  resonant cascade, plus sidebands merging into the carrier) → **clean window Δ ≈ 0.01–0.023** (θ ~0.4–1.5°,
  DoLP 0.984–0.999).
- **χ⁵ origin CONFIRMED by intensity scaling** (`chi5_optimization/intensity_scaling/sic_L3p2um/`, 2026-06-09;
  balanced σ⁺σ⁻, beat-resolved T=18/Δ, res 60, I=5e10…8e11). It is a **χ³→χ⁵ crossover, not a single power
  law:** the local log-log slope climbs monotonically with intensity at *both* Δ. **Native Δ=0.0234 is the
  cleanest χ⁵ signature** — slope 1.21→1.42→1.72→**2.14** while **DoLP stays ≥0.99** the whole sweep (reaches
  the χ⁵ regime with no depolarization). **Small Δ=0.006 gives 3.3–5.5× larger |θ|** (M5 enhancement, matches
  the Δ-scan ratio) **but its χ⁵ regime is entangled** — slope only 1.14→1.26→1.51→1.87 *and* DoLP collapses
  0.998→**0.80**, plus a residual-χ³ floor at low I (slope→1.1 from the tiny p2/p1≈0.999 imbalance). Global
  single-power fits (p=1.43 small, 1.61 native) **average the crossover and understate the high-I χ⁵ slope** —
  read the *local* slope, as the SiC study warned. **Two regimes:** max raw |θ| → small Δ + moderate I
  (~2e11, DoLP 0.976); cleanest χ⁵ *proof* → native Δ + high I (slope 2.14 @ DoLP 0.99).
- **Design takeaway:** drive Δ **down toward Γ_s** (sidebands on the probe mode), bounded below by the DoLP/
  resolvability floor — Δ_opt scales with κ_s (∝1/Q_s). The hardwired 100 nm and best_ratio (182 nm) are far
  past optimum; best_absolute (52 nm) is closer but still high. **Methodology rule for any Δ-scan:** set run
  length to resolve the beat (T ≫ 1/Δ; `--beats` ≳ 18). Plots/CSVs:
  `chi5_optimization/delta_scan/{sic_L3p2um, sic_L3p2um_resolved}/`.

## 3. Design levers, ranked **for fixed 100 fs pumps + probe** (§0)

With pump-Q bandwidth-capped, the priority order is **not** "maximize all Q's"; it is alignment + intensity +
overlap, with Q's bandwidth-matched:

1. **Pump resonance *alignment* (not maximization).** Get both pumps actually onto bandwidth-matched modes
   (Q_pump ~ Q_cap, §0), balanced (M3). This *recovers the buildup the SiC design lost* (pumps were fully
   off-resonance) — the single largest *practical* gain, but it tops out at the bandwidth cap, so it's an
   alignment problem, not a Q-race.
2. **Mode volume / field concentration.** Once Q_pump is capped, intracavity **intensity per input power** is
   set by the mode volume — tighter transverse (3D) and longitudinal confinement raises |E₁|²|E₂)|² directly.
   This is the real 3D lever (consistent with best_absolute's 14.5× 3D gain).
3. **Sideband-on-resonance Q_Ω (M5).** The sidebands are generated *inside* the cavity, so they still benefit
   from their own Q_Ω and from landing on modes — a genuine multiplicative lever the SiC run never used.
4. **Probe Q_s + readout.** Read at the carrier DFT bin, so a high-Q probe mode helps even with a 100 fs probe
   (stronger if the probe pulse can be lengthened — §8).
5. **4-mode overlap η·ζ (×few).** χ³ material in the **defect only**, all modes co-peaking; **linear low-loss
   mirrors** so FWM doesn't leak into mirror layers (the all-SiC failure mode).
6. **Dispersive tuning Δ ~ Γ_s.** Set the pump beat Δ near the sideband-mode linewidth so Re{1/D} (the ℒ factor)
   is near its dispersive maximum.

## 4. Geometry strategies (in order of preference)

- **A — Detuned single-defect DBR** (refine best_absolute). Params: optical-thickness ratio r = n_H t_H/n_L t_L,
  period count N, defect L. r sets relative pump/probe stopband positions (M4); N sets the Q's; L sets Δ and
  octave alignment (M1/M2). Cheapest; proven by best_absolute.
- **B — Dual-band mirror.** Stack a quarter-wave section at 1.5 µm with one at 0.8 µm → tune Q_pump and
  Q_probe/sb **independently** (A couples them through one ratio). More layers, thicker, more DOF.
- **C — Coupled triple-defect (photonic molecule)** in the SH band → force the 2ω₂, ω₁+ω₂, 2ω₁ triplet onto
  three engineered high-Q modes of spacing Δ (M5 by construction). Strongest comb control; hardest to make.
- **D — Material split.** Linear DBR + high-χ³ defect (already implied for SiN/SiC over SiO₂) — maximizes η,
  removes mirror-FWM contamination.

## 5. The new search machinery — architecture (FDTD-light: TMM + TCMT)

**Core principle (user, 2026-06-10): minimize FDTD.** The linear optics of a 1-D DBR+defect stack is solved
**exactly and analytically by the transfer-matrix method (TMM)** — R/T(λ), mode frequencies, Q's, complex field
profiles E(z), and mode volume — with **no FDTD per candidate**. The rotation is the **derived FoM / TCMT**
evaluated from those linear quantities + the χ³ overlaps. **FDTD (`faraday_meep_fp_circ`) is used ONLY to
validate winners**, never in the search loop. Each TMM eval is ~ms vs ~minutes for FDTD ⇒ a broad
geometry+operating-point search becomes affordable. This is the whole point of the new strategy.

**Separate entity.** New package `chi5_optimization/` — no edits to the existing FDTD pipeline. Modules:
- `tmm.py` — the analytic 1-D TMM linear engine: R/T(λ) from the layer stack + dispersive n(λ) (interpolated
  from the ellipsometry CSVs, no Meep); resonance finder (mode f₀ + Q from the transmission-peak Lorentzian /
  group delay); complex field profile E(z) per mode; mode volume V_mode. **Validated against the committed FDTD
  modes (best_absolute, SiC) before anything is built on it.** This is the keystone — it replaces FDTD in-loop.
- `objective.py` — from the TMM modes/fields: the 4-mode overlaps η, ζ (∫ over the χ³ region), the pump buildup
  B_i = min(Q_i, Q_cap), and the **derived FoM (§1)** and/or a direct **TCMT rotation** (assemble the
  FaradayJL-style coupled-mode coefficients analytically and integrate the fixed counter-rotating ODE).
- `optimize.py` — the search (reuse the Sobol+ARD+Kriging-Believer BO from `optimize_cavity_geometry` by import).
- `report.py` — the rich §6 output.

**Proxy FoM (fully analytic, per candidate, no FDTD):**

```
 FoM = B1·B2 · Qs · QΩ·ℒ(ΔΩ/κΩ) · (1/Vmode) · |η·ζ| · R_sym
       · exp[-(comb_mismatch/σ1)²]   · exp[-(log10(Q1/Q2)/σ2)²]      Bi = min(Qi, Q_cap)
```
All terms come from TMM (Q's, f's, E(z), V_mode) + the CSV n(λ). **B_i clamps the pump buildup at Q_cap** (§0:
the optimizer must align, not over-Q). **R_sym is the symmetry-break factor** (§1 refinement: net rotation needs
the ω_s±Δ sidebands inequivalent) — computed from the two sideband Green functions G(ω_s±Δ); a symmetric, real
cavity scores ≈0, steering the search toward asymmetric mode placement. Penalties enforce M1/M2/M3; M4 is
implicit in finite Q's. **Validate the weights against best_absolute (known |θ|) before trusting.**

**Search space:** geometry (r = optical-thickness ratio, N, defect L, material placement, optional dual-band
split for Strategy B) **plus the operating point** (probe window {≈800}∪[850, 950] nm, pump frequencies/Δ under
the matching — §2b). Δ targeted small ≈ Γ_s.

**Validation gate:** every candidate that tops the TMM proxy is re-checked by **(i) TCMT** (FaradayJL, fixed)
and **(ii) full FDTD** (`faraday_meep_fp_circ`, three-θ) — both must agree before a design is trusted. (This is
exactly the Phase-1 acceptance test the user asked for.)

## 6. Output per optimized design (rich, by request)

Every reported candidate should emit, into a self-contained dir:
- **Mode analysis:** reflectance spectrum (both bands), Harminv mode table (the 5 modes + Q's, FSR/Δ),
  octave-mismatch and balance numbers, the matching penalties — i.e. *why* it scored what it did.
- **Field distributions:** ε(z) profile; |E|(z) for all 5 modes; the 4-mode overlap density in the χ³ region
  (the η4 integrand) so overlap quality is visible. (3D: xz/xy snapshots for the winner.)
- **Rotation:** the full nonlinear `faraday_meep_fp_circ` outputs with the **three-θ** comparison
  (forward-coherent / forward-incoherent / total-field), DFT sidebands, polarization-vs-time.
- **Intensity sweep:** θ(I) on the winner to confirm the I² (χ⁵) slope and locate the perturbative range.
- **TCMT cross-check (optional but desired):** run `extract_tcmt_params_derivation.py` on the winner →
  FaradayJL case, integrate the coupled-mode ODEs, and compare the TCMT rotation against the FDTD — a
  reduced-order validation and a bridge to the analytic model.
- **Scorecard:** one JSON + one markdown summarizing FoM terms, all five conditions, and pass/fail vs targets.

## 7. Execution roadmap

> **Reframed 2026-06-10 (user):** the new optimizer is a *physics-based, FDTD-light* strategy (TMM + TCMT, §5),
> built first and used for BOTH phases. Phase 1 refines the existing DBRs (bounds close to them); Phase 2 is an
> unconstrained redesign. FDTD only validates winners.

- **Phase 0 — this doc + re-derived FoM (§1) + fixed TCMT (FaradayJL).** ✅
- **Build the FDTD-light machinery (§5), material-agnostic, once:** `tmm.py` (validate vs committed modes) →
  `objective.py` (overlaps + derived FoM + TCMT rotation) → `optimize.py` → `report.py`.
- **Phase 1 — refine the EXISTING SiN + SiC DBRs**, bounds **close to** best_absolute / SiC-L3.2µm. Search with
  the TMM+TCMT proxy; **validate the top refinements with BOTH TCMT and FDTD** (the acceptance gate). Goal: a
  better rotation near the known designs, fast. **Both materials in parallel.**
- **Phase 2 — complete new search**, bounds **unconstrained** (not tied to the existing geometry): SiC and SiN,
  TMM+TCMT proxy → FDTD-validate winners → 3D check.
- **Phase 3 — report & compare:** SiC vs SiN, TCMT-vs-FDTD agreement, recommended geometry.

## 8. Decisions

- **Pump + probe pulse durations — RESOLVED: both fixed at ~100 fs.** ⇒ every Q bandwidth-capped (§0); the §3
  ranking and the §5 B_i = min(Q, Q_cap) clamp follow. (Δ_opt ≈ Γ_s is set by the cavity Q, not the duration — §2b.)
- **Operating point — RESOLVED as constrained search variables (Scope):** probe λ_s ∈ {≈800 nm} ∪ [850, 950] nm;
  pump frequencies **tuned** in the IR under the matching (f₁+f₂ ≈ f_s, working assumption — confirm the exact
  FWM relation per design); Δ = f₁−f₂ searched, targeted **small ≈ Γ_s (cavity linewidth)** — §2b.
- **FDTD usage — RESOLVED (user, 2026-06-10): FDTD-light.** Search loop is **TMM (linear) + TCMT (rotation)**,
  analytic; FDTD only validates winners (§5). Build the TMM engine first; validate it vs the committed modes.
- **Fabrication bounds — RESOLVED (user):** **Phase 1 = close to the existing designs** (small perturbations of
  best_absolute / SiC-L3.2µm); **Phase 2 = unconstrained** (any pairs/L/ratio). Still set: layer-thickness
  tolerance (robustness of the detuning ratio r) and min/max defect length for the Phase-2 search box.
- **TCMT in the loop — RESOLVED (user): yes.** TCMT is both an in-loop rotation evaluator (cheap) AND part of
  the Phase-1 acceptance gate (TCMT *and* FDTD must agree).
- **Q_cap value** — compute from the actual 100 fs source bandwidth per band (≈ λ/Δλ_source); set it once.
- **Strategy scope** — Phase 1 is A (detuned single defect, by construction = refining existing); Phase 2 may
  open B (dual-band) / C (photonic molecule) in the search space.
- **Proxy weights/targets** — w's and σ penalties; validate against best_absolute (known |θ|) before trusting.

---

See `SiC_optimizations/sic_L3p2um/analysis.md` for the SiC study that motivated this, the README/CLAUDE.md
for the existing pipeline, and `../FaradayJL/` for the TCMT solver this plan's FoM is derived from.
