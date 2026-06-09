# χ⁽⁵⁾ Faraday-rotation design optimization — plan

Master plan for finding cavity geometries that **maximize the all-optical (effective χ⁽⁵⁾) probe
rotation**. Grounded in the repo theory: `../../chi3_sideband_patch.tex` (bulk sideband-dressed
susceptibility) and `../FaradayJL/src/FaradayJL.jl` (cavity TCMT). Companion to the project goal — the
cascaded χ³·G·χ³ sideband effect under balanced σ⁺σ⁻ pumps, **not** the χ³ carrier term.

> Status: **planning doc only** (2026-06-09, pedantically revised). No new pipeline built yet. Execution is
> staged below. Key constraint: **pumps fixed at ~100 fs** (§0) — this caps pump Q and reorders the levers (§3).

## Scope (agreed)

- **Two material systems**, low-index is **always SiO₂**:
  - high-index = **SiC** (4H-SiC, `sic.csv`, n₂ preset in `nonlinear_materials.py`) — **do first**,
  - high-index = **SiN** (Si₃N₄, `si3n4.csv`) — second.
- **Operating-point design variables (IMPORTANT — these ARE search variables, not fixed):**
  - **Probe wavelength** is constrained but not single-valued: **λ_s ≈ 800 nm (or very close)**, *or*
    **λ_s ∈ [850, 950] nm**. Pick to land the probe on a good cavity mode within these windows.
  - **Pump wavelengths are free and must be tuned** (IR, ~1.5–1.8 µm region for the prior designs; user calls
    it mid-IR), subject to the FWM matching (working assumption f₁+f₂ ≈ f_s, near-octave; confirm per design).
  - **Δ = |f₁−f₂| follows from the tuned pumps** — a real search direction; its optimum is analyzed in §2b
    (≈ source bandwidth). Pulse *durations* are separately fixed at ~100 fs (§0) — distinct from the
    wavelength freedom here.
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
  variables** (Scope), but Δ's *optimum* is pinned ≈ the source bandwidth (§2b) — set it once, don't expect
  further gain from sweeping it.

## 1. Figure of merit, derived

**Bulk** (`chi3_sideband_patch.tex`, Eq. θF-final): for balanced σ⁺σ⁻ the DC χ³ term cancels, leaving

  θ_F = (k₀L/4) · Re Σ_± [ δχ^(±)_++ − δχ^(±)_-- ],   δχ^(±) ∝ |E₁|²|E₂)|² / (Γ_s ∓ iΔ).

**Cavity** (FaradayJL `rhs_counter_derived!`): eliminate the sidebands in steady state. From dy[6],
b₋ = i ζ₋ (p₂p₁*) a₊ / (½κ_Ω − iΔ_Ω); substituting into dy[3] gives a sideband-loop self-energy on a₊:

  Σ₊ = − η₋ ζ₋ (p₂ p₁*)² / ( ½κ_Ω − i Δ_Ω ),   and analogously Σ₋.

Two pedantic points: (i) the loop carries (p₂p₁*)² — its **magnitude** is |p₁|²|p₂|² (consistent with the
bulk δχ ∝ |E₁|²|E₂|²); the residual phase 2·arg(p₂p₁*) is a rotating-frame convention and the **dispersive**
character — what makes it rotation rather than loss — is the Re{1/D} of the bulk result, D = Γ_s ∓ iΔ →
(½κ_Ω − iΔ_Ω) in the cavity. (ii) The on-resonance pump buildup |p_i|² = 4|S_i|²/κ_i is the **CW / narrowband-
drive limit**; for the fixed 100 fs pumps it is bandwidth-capped (§0) and the FaradayJL Gaussian drives capture
the reduced, pulsed value.

The rotation is the differential **real** frequency pull between a₊ and a₋; with κ = ω/Q this gives the
master scaling (CW limit; read with the §0 bandwidth caveat):

```
 θ_F  ∝   k0·Lint  ·  |S1|²|S2|²  ·  Q1·Q2  ·  Qs  ·  QΩ·ℒ(ΔΩ/κΩ)  ·  η·ζ
          └ k0L ┘     └ input ┘    └buildup*┘ └probe┘ └ sideband ┘   └overlap┘
   (*Q1·Q2 capped at ~Q_cap² by the fixed pump bandwidth — see §0)
```

k0 = ω_s/c favors the higher probe frequency (another minor reason to place the probe in the octave).
Everything below is a way to push one of these factors up **within the §0 constraint**.

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

**Is there an optimal Δ in the *real* system (beyond CMT)?** Yes — and it is set by the **fixed 100 fs source
bandwidth, not the cavity linewidth:**
- *Naive CMT:* rotation ∝ dispersive Δ/(Γ²+Δ²), peaked at Δ = Γ. If Γ were the cavity linewidth κ_s/2 ≈
  0.004 /µm (Q_s~150), the optimum would be a *tiny* Δ ≈ 0.004 /µm.
- *Real system:* the sideband at f_s±Δ is **generated by the E₁E₂*E_s product**, whose bandwidth is ~(2–3)×
  the source bandwidth. With 100 fs sources the source bandwidth (~0.015 /µm transform-limited; ~0.046 /µm for
  the Meep cutoff-Gaussian) **dominates the cavity linewidth (~0.004–0.008 /µm)**, so **Γ_eff ≈ source
  bandwidth** and the dispersive optimum moves to **Δ_opt ≈ 0.03–0.05 /µm (~60–100 nm separation).**
- *Hard lower bound:* Δ must exceed the source bandwidth or the sidebands **merge into the carrier** (the
  "rotation" degenerates into carrier self-phase modulation). So Δ ≳ source bandwidth.
- *Upper bound:* the dispersive 1/Δ falloff for Δ ≫ Γ_eff, group-velocity walk-off, and the need for all five
  modes to fit the stopbands ⇒ Δ ≲ ~0.1 /µm.
- **Conclusion:** Δ_opt ≈ the 100 fs source bandwidth ≈ **0.03–0.05 /µm (~60–100 nm)** — where the hardwired
  100 nm guess already sits. The CMT small-Δ optimum is **inaccessible** (broadband sources floor Γ_eff), so
  **Δ is bounded, not free to push:** tune the pumps so Δ ≈ source bandwidth and stop. best_absolute (52 nm)
  sits at the lower edge (sidebands marginally resolved); best_ratio (182 nm) is past the dispersive peak.
- **Verification experiment (proposed):** a 1D **Δ-scan** — fix the probe and the pump *mean* frequency, vary
  Δ = f₁−f₂ symmetrically at moderate *balanced* intensity, read the three-θ — to map θ(Δ) and confirm the
  peak ≈ source bandwidth. (The earlier SiC "pump-frequency scan" varied the *common* pump frequency, not Δ,
  so it did not measure this.)

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

## 5. The new search machinery (Phase 2) — architecture

**Separate entity.** New files, no edits to existing pipeline. Proposed (names provisional):
- `chi5_objective.py` — the proxy FoM + a thin wrapper around `faraday_meep_fp_circ.run_simulation` for the
  full nonlinear objective. Imports `mode_targeting`, `geometry_io`.
- `optimize_chi5.py` — the new optimizer CLI (reuses the Sobol+ARD+Kriging-Believer BO from
  `optimize_cavity_geometry` by importing it, not copying), multi-fidelity staged like
  `optimize_cavity_geometry_mf.py`.
- `chi5_report.py` — the rich-output generator (§6).

**Proxy FoM (cheap, linear-optics; one linear FDTD + Harminv per candidate):**

```
 FoM_proxy = B1·B2 · Qs^w2 · (QΩ+·QΩ−)^w3 · (1/Vmode)^w4 · η4
             · exp[-(comb_mismatch/σ1)²]        # M1/M2: triplet & doublet on a Δ-comb
             · exp[-(log10(Q1/Q2)/σ2)²]         # M3: balance
             · ℒ(ΔΩ/κΩ)                          # M5: sideband on its mode
   with  Bi = min(Qi, Q_cap)                     # §0: pump buildup SATURATES at the 100 fs bandwidth cap
```
Q's come from Harminv on a linear run; **B_i clamps the pump buildup at Q_cap** (so the optimizer cannot win
by chasing unphysical pump Q — it must align, not over-Q); 1/V_mode rewards field concentration (lever #2);
η4 = 4-mode overlap ∫ u_s* u₁ u₂* u_s in the χ³ region from the mode profiles; the penalties enforce
M1/M2/M3; M4 is implicit in the Q's being finite. Validate the weights against best_absolute before trusting.

**Search space:** geometry (r = optical-thickness ratio, N, defect L, material placement, optional dual-band
split for Strategy B) **plus the operating point** (probe window {≈800}∪[850, 950] nm, pump frequencies/Δ under
the matching — §2b). Multi-fidelity: proxy as stage A/B; full nonlinear |θ| (three-θ readout) on the top-k as
stage C; 3D on the winner.

**Why a proxy:** the master FoM (§1) is computable from linear quantities, far cheaper than the nonlinear
FDTD, and it directly targets the five resonances — which a raw |θ| objective only sees indirectly.

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

- **Phase 0 — this doc.** ✅
- **Phase 1 — quick refinement (existing pipeline, no new code):**
  - 1a. **SiC**: re-optimize the detuned-mirror design over SiO₂ with `optimize_cavity_geometry(_mf).py`,
    bounded near the best_absolute regime, reading the new three-θ outputs. Goal: a better SiC baseline fast.
  - 1b. **SiN**: same, refining best_absolute.
- **Phase 2 — build the new machinery (material-agnostic; built once):** `chi5_objective.py`,
  `optimize_chi5.py`, `chi5_report.py` + the proxy FoM and rich outputs (§5–6). Validate on a known design
  (best_absolute should score well).
- **Phase 3 — run the new search:** **SiC first**, then **SiN**. Stage A/B proxy → stage C nonlinear → 3D winner.
- **Phase 4 — report & compare:** SiC vs SiN best designs, TCMT cross-checks, recommended geometry.

## 8. Decisions

- **Pump + probe pulse durations — RESOLVED: both fixed at ~100 fs.** ⇒ every Q bandwidth-capped (§0); the §3
  ranking, the §5 B_i = min(Q, Q_cap) clamp, and the §2b Δ_opt ≈ source-bandwidth result all follow.
- **Operating point — RESOLVED as constrained search variables (Scope):** probe λ_s ∈ {≈800 nm} ∪ [850, 950] nm;
  pump frequencies **tuned** in the IR under the matching (f₁+f₂ ≈ f_s, working assumption — confirm the exact
  FWM relation per design); Δ = f₁−f₂ searched, targeted near the source bandwidth (§2b).
- **Q_cap value** — compute from the actual 100 fs Meep source bandwidth per band (≈ λ/Δλ_source); set it once.
- **Fabrication limits** — max DBR pairs, layer-thickness tolerance (matters for the detuning ratio r), min/max defect length.
- **1D-vs-3D in the loop** — proxy + stage-C in 1D, 3D only on winners (assumed), or 3D earlier? (Mode volume
  lever #2 is partly a 3D effect, so a 3D check on finalists matters.)
- **Strategy scope** — start with A (detuned single defect) only, or include B (dual-band) in the search space?
- **Proxy weights/targets** — w2,w3,w4 and the σ penalties; validate against best_absolute before trusting.
- **TCMT in the loop** — cross-check only on winners (assumed), or as an additional cheap proxy?

---

See `SiC_optimizations/sic_L3p2um/analysis.md` for the SiC study that motivated this, the README/CLAUDE.md
for the existing pipeline, and `../FaradayJL/` for the TCMT solver this plan's FoM is derived from.
