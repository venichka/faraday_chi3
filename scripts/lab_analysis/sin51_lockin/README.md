# SIN51 lock-in delay scans — analysis

What this analysis means for the next experiment: the SiC run plan in
[`../sic_run_plan/`](../sic_run_plan/run_plan.md) (short: [`run_plan_short.md`](../sic_run_plan/run_plan_short.md)).

Data (repo-root [`data/`](../../../data/); file names are acquisition times, HHMMSS):

| tag | file | what | points |
|---|---|---|---|
| **146** | `SIN51_lockin_180146.csv` | pump₂-delay scan, **29.85 fs** step | 68 |
| **235** | `SIN51_lockin_180235.csv` | pump₂-delay scan, **20 fs** step | 101 |
| **626** | `SIN51_lockin_180626.csv` | "both pumps fixed, no delay" reference | 101 |

All three span the same stage positions (185.7053 → 186.0052 mm), so they share one delay axis:
τ = −1500 fs + 2(x − 185.70530 mm)/c. Columns: set-point delay, stage read-back, lock-in X, Y,
R, phase.

**Setup (user, 2026-09-23/24).**
- **Sample:** SIN51, the fabricated SiN `best_absolute` cavity. As built, it differs slightly
  from the design (see §4). That is why the pumps had to move from the design's 1521.5 / 1574.0 nm
  to **1560 / 1620 nm**, to sit on the actual resonances.
- **Beams:** the **1620 nm pump is delayed** and chopped at 500 Hz on a 1 kHz laser. Pump₁
  (1560 nm) and the probe (**800 nm**) travel together. So X is the pump₂-induced change in the
  balanced V−H signal.
- **Pulses:** ~100–150 fs.
- **Peak intensity:** **1 TW/cm² = 10¹² W/cm²** per pump (confirmed), the simulations'
  reference intensity.
- **The two pumps come from different OPAs**, so their relative optical path — and with it the
  fringe phase — is not interferometrically stable.
- **Delay zero:** τ = 0 in the files is nominal, not a measured overlap.
- **Calibration:** total probe signal S₀ = 9 mV on the V+H output. The V+H and V−H gains differ,
  but their ratio is **taken as 1 for now** (user); every angle below scales with it.

---

## Results (θ at gain ratio 1, 10¹² W/cm²)

| quantity | 146 | 235 | **joint (146 + 235)** | 626 (reference) |
|---|---|---|---|---|
| pedestal (all delays) | 0.591° | 0.582° | — | 0.309° |
| noise per point | 0.066° | 0.056° | — | 0.061° |
| overlap centre τ₀ | −1067 fs | −1054 fs | **−1061 fs [−1066, −1056]** | −1030 fs |
| envelope FWHM | 160 fs | 160 fs | **160–170 fs** | ~100 fs |
| **fringe amplitude A** | 2.44° [1.82, 3.28] | 2.28° [1.86, 2.92] | **2.32° [1.96, 2.70]** | 0.10° (4%) |
| effect E (k = 0), shared-envelope random-phase fit (s5/s6) | +0.12° [−0.44, +0.63] | +0.71° [+0.32, +1.10] | +0.50° [+0.19, +0.80] | +0.09° ± 0.04° |
| ⭐ **effect E (k = 0), fringe-model review (s7, §6)** | 1.5–2.4° | 1.7–2.4° | **+1.8° (tied models 1.73–1.89°)** | — |
| effect / fringe | | | **≈ 1** (simulated χ⁽⁵⁾: 0.01–0.09) | |

Intervals are 68% profile-likelihood intervals. The s5/s6 effect estimate is **superseded** by
§6: forcing the effect onto the fringe's envelope biases it low (the data reject a shared
envelope, ΔBIC 154). **The overlap is at stage position
185.7712 mm**, 159 µm from the nominal zero (185.9303 mm).

### 1. A strong, reproducible pulse-overlap feature
Both delay scans show the same feature at the same stage position:
- **Location:** centred at τ₀ ≈ −1061 fs.
- **Envelope:** ~160 fs wide, consistent with 100–150 fs pulses.
- **Size:** the rotation swings from ~0° up to 3.5°, 50× the per-point noise, on a 0.58° pedestal.

It is a genuine chopper-synchronous optical signal. Its lock-in phase matches the pedestal's to
within ±3° in every file, and the quadrature channel (rotated to that phase) shows no excess (s4).

### 2. The fringe dominates, and it is not phase-stable
- **The two scans disagree at the same delay.** At −1020 fs, scan 235 reads 3.53° and scan 146
  reads 0.17°, while the envelope reproduces. The fine structure is a fast component — the
  ~5.4 fs carrier fringe of the delayed 1620 nm pump.
- **A single coherent fringe model misfits badly** (reduced χ² 120–190 inside the envelope,
  s3). Its phase is therefore not stable from point to point, so subtracting a fitted fringe is
  not valid.
- **The valid route is to average over the fringe phase** (s5, s6). Each point is modelled as
  pedestal + E·G(τ) plus a random-phase fringe of amplitude A·G(τ), with the same envelope G,
  fitted by maximum likelihood.
- **First-pass result:** fringe A = 2.32°, effect E = +0.50° ± 0.3°. §6 revisits this with a
  full model review: the effect has its own, narrower envelope, and **E ≈ +1.8°**, about as
  large as the fringe.

⚠️ **Open alternative.** Shift scan 235 by +36 fs and the two scans correlate at r = 0.86. The
cross-correlation oscillates with a ~96 fs period, but a permutation test gives only p ≈ 0.05.
§6 finds that models with a ~100 fs apparent oscillation fit both scans with the *same* overlap
position, so no timing drift is required. Two scans cannot say whether that oscillation is slow
or an aliased fast fringe. A scan with **≤ 2.5 fs steps** would settle it at once, because it
resolves the fringe directly.

### 3. The "pumps fixed" reference is not a clean null
File 626 carries a weak feature at the same stage position:
- **Size:** fringe ≈ 0.10°, 4% of the delay scans; effect +0.09° ± 0.04°.
- **Significance:** local variance 6.9×, p < 0.001.
- **Pedestal:** 0.31°, half the delay scans'.

**Nothing moved optically in that run** (user; the lab suspects pump–probe interference). But its
stage read-back still spans the same 185.705 → 186.005 mm as the scans. So its x-axis is really
*time*, and an interference whose phase drifts in time can raise the scatter at any point. Its
landing within ~30 fs of the overlap position would then be a coincidence, a few percent likely.
If the pumps had been parked *at* the overlap, the whole trace would carry the ~1.8° k = 0 term
and ~2° fringe fluctuations. It shows neither (flat 0.31°), so in 626 the pumps were most likely
not overlapped with the probe.

### 4. The as-built sample (`asbuilt_geometry.py`)
The transfer-matrix mode solver fits per-material thickness errors to the resonances the lab
uses (1560, 1620, 800 nm). **Best fit: SiN layers +3.0%, SiO₂ layers +0.95%, 1.5 nm rms.** A
uniform +2.5% gives 2.6 nm. The fit has a long, degenerate valley (SiN and SiO₂ errors trade
off), so the robust statement is **the stack is ~2.5–3% thicker than designed, mostly in SiN**,
including the 5.9 µm cavity (→ ~6.07 µm).

Consequences:
- **Pumps:** the lab pumps sit on the **same two modes** the design used (designed at
  1525.1 / 1577.5 nm, now at 1560 / 1619).
- **Probe:** the **800 nm probe sits on the mode designed at 781.8 nm**. The mode designed for
  the probe (800.1 nm) is now at ~820 nm.
- **Frequency matching:** this combination is ~0.9% off the frequency-matching (octave)
  condition, against 3.3% at the design point. In the earlier retuning map of this sample, the
  781.8 nm mode gave ~2.4× more rotation than the 800.1 nm one.

### 5. Simulation of the as-built sample at the lab's operating point
`sim_labpoint.py` runs the simulations below; results are aggregated in `sim/sim_result.json` and
compared in `s6_compare.py` and `s7_models.py`. Setup: probe 800 nm, pumps 1560 / 1620 nm with
1620 delayed, 10¹² W/cm².
- **1D predicted delay trace:** ±240 fs, 20 fs steps, 125 fs pulses, 4 carrier phases per delay.
- **1D τ = 0:** 100 / 125 / 150 fs pulses, on both ends of the geometry valley.
- **3D τ = 0:** 125 fs pulses.

| 1D, τ = 0 | effect | fringe | effect / fringe |
|---|---|---|---|
| best-fit stack, 100 / 125 / 150 fs | 0.004 / 0.002 / 0.0005° | 0.094 / 0.080 / 0.057° | 0.05 / 0.03 / 0.01 |
| uniform-scale stack, 100 / 125 / 150 fs | 0.006 / 0.004 / 0.002° | 0.062 / 0.052 / 0.039° | 0.09 / 0.09 / 0.06 |
| **3D**, best-fit stack, 125 fs | **0.021°** (×8.4 vs 1D) | **0.265°** (×3.3) | **0.08** |

- **Fringe:** a single fast fringe with a ~180 fs envelope. Its carrier is pulled by the cavity
  from 1620 to **~1602–1612 nm**, with a mild chirp. There is no slow (~100 fs) structure.
- **Envelope shape:** scaled to the data, the simulated fringe envelope matches the measured
  fringe-rms envelope (`s6_joint.png`, panel 3).
- **Effect:** the simulated χ⁽⁵⁾ effect is **1–9% of the fringe in 1D and 8% in 3D**. The
  measured k = 0 term is about as large as the fringe (§6), ~12× more than χ⁽⁵⁾ predicts relative
  to the fringe.
- **Size (3D):** the predicted fringe is **0.265°** and the effect **0.021°** (DoLP 0.63). The
  measured fringe (~1.9° at gain ratio 1) would match if G(V−H)/G(V+H) ≈ 7 — a typical
  difference-vs-monitor gain split on balanced detectors, so worth checking. The effect/fringe
  ratio is gain-independent: **0.08 predicted vs ~1 measured.**

### 6. Review of the fringe model (`s7_models.py`)
The simple model — one Gaussian envelope carrying the effect and a fringe at the delayed pump's
1620 nm carrier — misfits badly (χ²/point ≈ 14). Stage 7 builds the model up one ingredient at a
time. Every variant is fitted to **both delay scans jointly**, and all are scored on the same
footing:
- **Scores:** the likelihood (χ² for known noise), AIC/BIC, and **cross-validation between the
  scans**. The shared physics is fitted on one scan and must predict the other, with only that
  scan's pedestal, overlap position and fringe phase refitted.
- **Frequency search:** the carrier frequency (and the template's delay scale κ) is highly
  multimodal. It is therefore profiled on a fine grid first, and every model is then optimised
  inside each candidate basin.
- **Aliasing-aware validation:** a single scan fixes a fast carrier only up to its aliasing
  partners, so cross-validation tries all of them.
- **Tied variants:** because an unconstrained fit could misuse its freedom (see M11 below),
  each important model also has a physically constrained version. It has **one effect amplitude
  and one amplitude per fringe component shared by both scans** (same sample, same powers, a
  minute apart); only pedestals, overlap positions and fringe phases differ. Envelopes can be
  no shorter than 80 fs, given the 100–150 fs pulses.

| model (joint, 169 points) | −2lnL | params | ΔBIC | χ²/pt | effect E (146 / 235) |
|---|---|---|---|---|---|
| M0 fringe at 1620 nm, one envelope | 2313 | 11 | +2071 | 13.7 | — (misfit) |
| T1 simulated fringe exactly as computed (κ = 1) | 2949 | 12 | +2713 | 17.5 | — (misfit) |
| H1 fringes at ω₂ and 2ω₂, 1620 nm fixed | 1588 | 17 | +1376 | 9.4 | — (misfit) |
| M1 carrier frequency free | 639 | 12 | +402 | 3.8 | 1.24 / 1.61° |
| T4 simulated fringe, delay axis ×κ, **κ = 1.122** | 535 | 13 | +303 | 3.2 | 1.53 / 1.74° |
| M2 + separate effect envelope | 475 | 14 | +248 | 2.8 | 1.62 / 1.92° |
| M3–M6 + chirp / ring-down / lock-in lag / cavity echoes | 451–475 | 15–16 | +230…+253 | 2.7–2.8 | 1.5–2.4° |
| M8 + partial coherence | 251 | 15 | +29 | 1.5 | 1.62 / 1.95° |
| M7 + second oscillation (104 fs + 62 fs apparent) | 217 | 19 | +16 | 1.28 | 1.59 / 1.93° |
| M11 two free fast fringes (1430 + 1628 nm) | 201 | 19 | 0 | 1.19 | −0.27 / +2.03° ⚠️ |
| **M7t** second oscillation, **tied** | **254** | **16** | **+37** | **1.50** | **+1.81° ± 0.03° (shared)** |
| M11t two fast fringes, tied (1423 + 1611 nm) | 279 | 16 | +63 | 1.65 | +1.78° ± 0.03° |
| T4t simulated fringe κ = 1.122, tied | 578 | 11 | +336 | 3.4 | +1.73° ± 0.05° |

Figures: `figs/combined/s7_ladder.png` (every model), `s7_best_fits.png` (M7t, M7, T4t with
residuals), `s7_frequency.png` (carrier and κ profiles), `s7_effect.png` (effect vs model).

**What the data need:**
- **A carrier other than the nominal 1620 nm.** Freeing it cuts χ²/pt from 13.7 to 3.8. Fixing
  the physically expected carriers — the delayed pump at 1620 nm, its second harmonic, or the
  simulated cavity-pulled fringe — fails (χ²/pt 9–18).
- **Separate envelopes for the effect and the fringe.** The effect (k = 0) envelope, **~95 fs**,
  is narrower than the fringe's, **~140–150 fs** (width ratio 0.53–0.64 across models). For
  Gaussian pulses a χ⁽⁵⁾ k = 0 term should be 0.75× the fringe's width, and a pump-SFG one 0.82×.
  The measured term is narrower than both, so **the width ratio does not tell χ⁽⁵⁾ from SFG**. It
  does confirm the k = 0 term is a higher-order overlap signal, not a fitting artefact. The
  implied pulses are 67–92 fs on the nominal delay axis, or 75–103 fs if the axis is ×1.12 (below).
- **A second oscillatory component or partial coherence.** Either brings χ²/pt to 1.2–1.5, near
  the noise.
- **Not needed:** chirp, a ring-down tail, lock-in lag and cavity echoes add nothing significant
  (ΔBIC within ±25 of M2).

**What does not survive scrutiny:**
- **M11, the raw BIC winner, is unphysical.** It gives scan 146 a 15° second fringe, hidden
  because the 29.85 fs step samples it at its zero crossings, and an effect of the opposite sign
  to scan 235's.
- **Partial-coherence models (M8, T5/T6) "fit" only by declaring the overlap centre noisy.** They
  leave 10–15σ residuals at the peaks; physically plausible, but not predictive.
- **Envelopes shorter than the pulses give spurious fits.** An unconstrained H1 reached
  −2lnL = 530 with a 40 fs envelope and 64° amplitudes.

**Robust across every adequate model:**
- **Overlap position:** τ₀ = −1063 ± 5 fs.
- **Envelopes:** effect ~95 fs, fringe ~140–150 fs.
- **Dominant oscillation:** amplitude ~1.9°.
- ⭐ **k = 0 term E = +1.8°:** tied models give +1.73 to +1.89°, and adequate untied ones 1.5–2.4°.
  The tied models' quoted ±0.03° are conditional on the model; see the verification below.

**Carrier-free estimate (`s8_phase_marginal.py`, 2026-09-25).** Each point is pedestal + E·G_e +
A·G_f·cos φ + noise with φ uniform and integrated out exactly (an arcsine law convolved with the
noise), so no fringe carrier is assumed at all. Both scans share E, A and the envelopes.
**E = +1.73° [1.60, 1.90] (68%), [1.50, 2.30] (95%); E = 0 is excluded at 4.9σ.** Fringe
A = 1.83°, envelopes 115 fs (effect) and 175 fs (fringe). This is the number to quote for the
k = 0 term from these two scans. `figs/combined/s8_phase_marginal.png`.

**What the chopped lock-in actually measures.** Only pump₂ is chopped, so X ∝ θ(both pumps) −
θ(pump₁ only). The balanced-helicity cancellation of the direct χ⁽³⁾ rotation holds for the
two-pump total, not for this difference: **pump₂'s own χ⁽³⁾ rotation survives as a k = 0,
overlap-only term**, linear in I₂ and independent of pump₁. Any pump₂-induced transmission
change also leaks into V−H through imperfect balance. Single-pump simulations of the as-built
stack (`sim_singlepump.py`, 1D, τ = 0, 10¹² W/cm², 125 fs) quantify the first: the 1620 nm
pump (σ⁺) alone rotates the probe by −0.0018°, the 1560 nm pump (σ⁻) alone by +0.0009°, both
pumps (carrier-averaged) by +0.0025°. **The chopped k = 0 term is therefore +0.0016° against a
fringe of 0.080°: a ratio of 0.02.** Including the direct χ⁽³⁾ term does not change the
verdict; the measured ratio (~1) is ~50× larger than any Kerr mechanism gives at this
intensity. The transmission-leak term needs V+H recorded (SiC plan).

*Helicity (fixed 2026-09-25).* The simulator's pump1 is always σ⁺ and `--pump-imbalance 0`
switches pump2 off, so every single-pump run has a σ⁺ pump. The 1560 nm pump is σ⁻ in the
two-pump run, and for an isotropic stack a σ⁻ pump gives minus the σ⁺ rotation (mirror through
the probe's 45° axis). The first version took −0.0009° as-is and quoted +0.0034° / 0.04.

**Which pump to chop.** With T = θ(both), a₁ = θ(pump₁ alone), a₂(τ) = θ(delayed pump alone):
- chop the delayed pump (SIN51): X = T − a₁ → the bump holds x + a₂(τ);
- chop both pumps on one chopper: X = T → the bump over the off-overlap pedestal a₁ is again x + a₂(τ);
- **chop the pump that travels with the probe: X = T − a₂(τ) = a₁ + x(τ)** → a flat pedestal a₁
  and a bump that is the two-pump term x = T − a₁ − a₂ alone, at the full lock-in amplitude.

On SIN51 (1D) x = +0.0034°. On SiC L = 4.8 at its recommended point (1e11, 1D) the single-pump
rotations are −0.0001° (pump1) and +0.0002° (pump2) against a total of −0.0074°, so x = −0.0075°
≈ T: there the Kerr part is scheme-independent and the choice guards against single-pump
artefacts (SIN51's pedestal, pump₂ SHG at 785 nm). Double modulation also isolates x but, on a 1 kHz train, only for
the right frequencies (checked numerically on a pulse train): 500 + 250 Hz synchronised puts
pump₂'s own signal on the 250/750 Hz lines (0.50 against 0.35 for x); 500 + 333.3 Hz with the
reference at 166.7 Hz is clean but at 1/3 of the amplitude (θ = 3X/(√2 S₀) × gain ratio);
per-pulse recording of the four states at 500 + 250 Hz costs ×1.9 in noise. For 5σ on a
0.07° k = 0 term at the SIN51 noise (±200 fs at 10 fs steps): 5–6 scans with the recommended
chopping, 14–16 per-pulse, 34–39 on the 166.7 Hz line.

**Two more candidates checked against the simulation (2026-09-25).**
- *Transmission leak.* With both pumps on, the simulated probe transmission drops by 2.2 % at
  the overlap (single pump: +0.1 %). A leak ε·ΔS₀/S₀ into V−H with |ε| ≤ 1 therefore reaches at
  most 0.6°, not 1.7°. The candidate needs a real transmission dip ≥ 6 %, which V+H shows directly.
- *A 10 nm bandpass in the probe arm* (`sim_filter.py`, per-bin Stokes added to the simulator):
  it does **not** remove the fringe. Narrowing the band from ±18 to ±4 nm raises the fringe from
  0.080° to 0.124° (SIN51) and 0.050° to 0.069° (SiC L = 4.8) while the effect stays put, so the
  contrast falls (0.031 → 0.008; 0.149 → 0.131). The fringe field spans the whole probe spectrum,
  not only the ±14 nm sidebands. Keep the detection broadband.
- *10 fs steps* alias the 5.4 fs fringe to the same ~67 fs apparent period as 20 fs steps: they
  double the points in the overlap (√2 better carrier-free estimate) but do not resolve it.

**Verification with no fringe model (2026-09-25):**
- **Calibration:** θ = X/(√2 S₀) confirmed numerically for shot-by-shot chopping.
- **Pedestal and overlap position:** 0.592 / 0.580° and −1064 / −1048 fs from simple statistics.
- **Fringe amplitude:** ≈ √2 × the scatter at the centre = **1.7–1.9°**.
- **k = 0 term:** the integrated excess over the pedestal is +140 and +190 deg·fs. Treating the
  fringe as random-phase noise, that is **+1.8σ and +3.0σ, +3.3σ combined** — positive,
  independent of any fringe model. As an amplitude it is **+0.9 to +1.9°**, depending on the
  assumed envelope (150 → 95 fs).
- **Central points:** 6 of the 7 within ±40 fs sit above the pedestal, mean +1.6° (68% CI
  1.1–2.1°).

The stage-6 random-phase estimate (+0.50°) is biased low. It forces the effect onto the fringe's
envelope and down-weights the centre, and the data reject a shared envelope (M1 vs M2: ΔBIC 154).

**What the data cannot tell (two scans at 20–30 fs steps):**
- **Which carrier.** The structure can be written either as slow modulations (104 + 62 fs) or as
  fast fringes. The fast solutions include a component at **~1611 nm — the simulated physical
  fringe — plus a stronger one at ~1423–1430 nm**, which no physical process here produces.
- **Nothing transfers between scans.** No model predicts one scan from the other: strict
  cross-validation stays at ≥ 3–5 per point. The fine structure is not a fixed function of delay.
- **A delay-axis hypothesis.** The simulated fringe fits only if the lab's delay axis is **~12%
  longer than 2Δx/c (κ = 1.122; κ = 1 is rejected by Δ(−2lnL) ≈ 2400)**. This is the same
  finding as the free-carrier model's apparent 1429 nm (≈ 1603 nm / 1.122). A mis-set stage
  scale would do it, and that is quick to check (below).

**What it means for χ⁽⁵⁾.** The measured k = 0 term (~1.8°) is about as large as the fringe
(~1.9°). The simulated χ⁽⁵⁾ effect is only 8% of the fringe in 3D (1–9% in 1D). So most of the measured k = 0
signal is probably **not** χ⁽⁵⁾ rotation. The prime suspect is **light generated at the probe
wavelength**, most plausibly pump₁ + pump₂ sum frequency at 1/(1/1560 + 1/1620) = **794.7 nm**:
- It appears only when the two pumps overlap, and it is chopped with pump₂.
- It goes as |E₁E₂|², hence the narrower envelope.
- It needs no probe, and it adds to V−H whenever its polarisation is not at 45°.

Blocking the probe tells them apart.

### Calibration (applies to every angle)
θ = X / (√2·S₀) × G(V+H)/G(V−H) = **4502 °/V × G(V+H)/G(V−H)**. This assumes:
- (a) S₀ is the time-averaged V+H level;
- (b) the lock-in reports RMS;
- (c) the detector response is flat from DC to 500 Hz.

The √2 comes from chopping a pulse train shot by shot: the 500 Hz component is half the on/off
difference. A square-wave model would give 1.57× larger angles. To apply the real gain ratio,
set `GAIN_SUM_OVER_DIFF` in `common.py` and re-run all stages.

---

## Recommended next measurements
0. **Probe blocked, same delay scan.** Any overlap signal that survives is light generated at the
   probe wavelength (sum frequency, §6), not rotation. This decides what the +1.8° k = 0 term is.
   Also run the σ⁺σ⁺ null: χ⁽⁵⁾ rotation must vanish for co-rotating pumps.
1. **Chop the pump that travels with the probe (1560 nm), not the delayed one.** The overlap
   bump is then the two-pump term alone (see "Which pump to chop" above).
1a. **Steps ≤ 2.5 fs are not available (10 and 20 fs only, 2026-09-25)**, and both alias the
   5.40 fs fringe to the same ~67 fs. Treat the fringe as random-phase noise instead: 10 fs
   steps over ±200 fs, ≥ 6 back-to-back scans, carrier-free fit (s8). A piezo dither of
   ≥ 0.81 µm in the delayed arm would average it at acquisition.
1b. **Check the delay-axis calibration** (§6: the simulated fringe fits only with a delay axis
   ~12% longer than 2Δx/c). Scan any known-wavelength interference on the same stage, e.g. a
   HeNe or the pump in a Michelson, and compare its fringe period with λ/c.
2. **Repeat scans back-to-back.** The repeats beat the random-phase fringe down and measure any
   timing drift between scans; no model predicts one of the two present scans from the other.
3. **What moved in the "pumps fixed" run (626)?** Its weak feature sits exactly at the overlap.
4. **Pedestal controls** (0.58° at all delays):
   - block the probe: scatter or pickup survives;
   - block pump₁: is it needed at all?
   - vary the pump₂ power: a linear dependence points to χ⁽³⁾ or heating.
5. **Measure the gain ratio**, and confirm the intensity (10¹² W/cm²?), so the angles become
   absolute.

## Stages

Per dataset: `python sN.py [146 235 626]` (default: all). Outputs go to `figs/<tag>/` and
`results/<tag>/`.

| script | what it does | figures |
|---|---|---|
| `s0_raw.py` | raw channels; stage calibration (double pass to 0.01%); actual vs nominal delay; X/Y/R consistency; noise autocorrelation | `s0_raw.png`, `s0_checks.png` |
| `s1_rotation.py` | X → θ; the overlap is located and excluded; pedestal, noise, drift and step models (AR(1) least squares) | `s1_rotation.png` |
| `s2_fringe.py` | the fringe's aliasing at this dataset's step; wavelengths with identical traces; how well λ₂ could be pinned | `s2_fringe.png` |
| `s3_search.py` | effect + fringe matched filter over τ₀ at λ₂ = 1620 nm (1560 control, λ₂ free), look-elsewhere-corrected; coherent fringe fit and subtraction; λ₂ profile | `s3_search.png`, `s3_decomposition.png` |
| `s4_quadrature.py` | lock-in rotated to the pedestal phase; phase of the excess signal; sliding variance; noise budget | `s4_quadrature.png` |
| `s5_net.py` | random-phase-fringe maximum-likelihood fit: effect (k = 0) and fringe amplitude, net rotation | `s5_net.png` |
| `s6_compare.py` | all files on one axis; joint fit of the two delay scans; pooled profile; the reference file; the simulation | `figs/combined/s6_*.png` |
| `s7_models.py` | **review of the fringe model**: 20+ variants fitted jointly, basin-profiled carriers, tied (shared-amplitude) versions, cross-scan validation, the simulated fringe as a template | `figs/combined/s7_*.png` |
| `s8_phase_marginal.py` | **carrier-free k = 0 estimate**: fringe phase integrated out exactly (arcsine likelihood); profile-likelihood interval for E | `figs/combined/s8_phase_marginal.png` |
| `sim_singlepump.py` (+ `.sbatch`) | single-pump runs (SIN51 as-built, SiC L = 4.8): what each chopping scheme measures; σ⁻ pump = minus the σ⁺ run | `sim/runs_single/single_result.json` |
| `sim_filter.py` (+ `.sbatch`) | effect and fringe versus detection bandwidth (a 10 nm filter), SIN51 as-built and SiC L = 4.8; needs the per-bin Stokes output | `sim/runs_filter/filter_result.json` |
| `asbuilt_geometry.py` | as-built layer thicknesses from the lab's resonances (TMM) | `figs/asbuilt_geometry.png` |
| `sim_labpoint.py` (+ `.sbatch`, `_3d.sbatch`) | simulations of the as-built sample at the lab's operating point | — |

~15 min for s0–s6 in total (numpy and matplotlib only; s3 dominates).

**Method notes.**
- **Delays come from the stage read-back**, not the set point: a sawtooth of up to 0.5 fs,
  ~10% of a fringe period.
- **Noise and ρ come from outside the overlap.** Measured on the whole trace, the strong feature
  would inflate them ~10×. Fit errors are further scaled by √(reduced χ²) where the model misfits.
- **Significance nulls:**
  - AR(1) Gaussian surrogates;
  - a block bootstrap of the residuals with the candidate region excluded. A circular-shift null
    carries the candidate into every surrogate.

  Every surrogate is normalised by its own noise estimate.
- **In the free-λ₂ search, wavelengths where the fringe freezes into a smooth bump are masked**
  (1499 nm at a 20 fs step; 1492 / 1790 nm at 29.85 fs). There it cannot be told apart from the
  effect.
- **`common.dbr_harness()`** loads `meep_project/chi5_dbr_design/common.py` under another module
  name. Both files are called `common`, and a plain import returns the wrong one.
