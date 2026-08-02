# Pump–probe delay simulation — physics-first audit and rebuild

Written 2026-08-01, after the first delay campaign (`delay_scan.py`, 202 sims) turned out to be
dominated by effects nobody had budgeted for. This document derives what the simulation *should*
show before looking at any output, then lists every place the implementation departed from it.

Deliverables of the rebuild: `delay_physics.py` (driver), `plot_delay_physics.py` (figures),
`delay_3d.py` + `delay_3d.sbatch` (3D). `delay_scan.py` is left untouched — it documents the
earlier campaign.

---

## 1. What is actually measured

The probe leaves the sample, a polarizing beamsplitter splits it into V and H, and a detector
integrates each arm over the pulse. For a probe launched at azimuth ψ with ellipticity χ,

    V − H = −S1 = S0 · cos(2χ) · sin(2θ) ≈ 2 S0 θ.

So balanced detection *is* the rotation. Two consequences that decide which simulator output to
quote:

* The detector integrates **pulse energy**: ∫|E_V(t)|²dt − ∫|E_H(t)|²dt. By Parseval this is the
  run-accumulated DFT summed over the probe band — exactly `probe_pulse_integrated`. This is the
  lab observable.
* The alternative readout, `probe_rotation_deg.final_relative_deg` ("legacy"), is the polarization
  azimuth averaged over the **last M time samples** of the run, i.e. of whatever is still ringing
  after the probe energy has decayed to ~the decay threshold. It is a settled-state property of the
  longest-lived mode, not an energy measurement, and no detector can read it without gating ~1e−4
  of the pulse energy. It is still useful as a design-comparison scalar — it is the estimator behind
  the published 0.137° — and both are reported everywhere, but they are not the same quantity and
  differ by ~5× on this design.

Because the pulse-integrated Stokes vector is a **sum over frequency bins of same-frequency
products**, it contains no cross-frequency terms: the readout itself introduces no interferometric
artifact, and a global time shift must cancel from it exactly. Both facts are used as tests below.

## 2. What the delay can possibly do

Only pump1 moves. Shifting it by τ multiplies its field by exp(iω₁τ). Every observable is a sum of
products of fields, so a term carrying n powers of E₁ and m of E₁* picks up exp(i(n−m)ω₁τ):

| n − m | τ-dependence | what it is |
|---|---|---|
| 0 | envelope overlap only | the genuine χ³ / χ⁵ response |
| ±1 | fringe at T₁ = 5.075 fs | interference of terms of different order in E₁ |
| ±2 | fringe at T₁/2 = 2.54 fs | two-pump1-photon path (near-octave: 2f₁ = 1.3145 vs f_probe = 1.2461) |

**Therefore: averaging the Stokes vector uniformly over one pump1 optical period annihilates every
n ≠ m term and leaves exactly the physical rotation.** This is the single most important design
consequence, and it is also what an experiment measures when its delay line is not
interferometrically phase-stable. Sampling at *whole multiples* of T₁ — what the old stage 1 did —
does **not** average; it freezes the fringe at one phase, i.e. measures the fringe maximum.

Measured on the old campaign: the fringe amplitude is ~18× the carrier-averaged mean, and the
fundamental needs a 26% second harmonic to fit (R² = 0.996). So N = 4 sub-samples (which cancel
harmonics 1, 2, 3) is the minimum defensible choice.

## 3. What sets the slow (envelope-scale) periodicity

Not the "pump beat". A Δ beat oscillates in **time**, not in delay:

    2 Re[E₁(t−τ)E₂*(t)] = 2|E₁||E₂| cos(Δ·t − **ω₁**τ + φ)

— in τ that is ω₁, i.e. 5.075 fs. A delay-domain periodicity instead comes from the **cavity mode
comb inside the bandwidth of the delayed pulse**: pump1 (fwidth 0.0462 /µm) excites several
pump-band modes, each intracavity amplitude carries exp(iω_nτ), and terms *quadratic* in the pump1
field (the intensity-like χ³/χ⁵ response) give cross terms exp(i(ω_n−ω_m)τ) at mode **spacings**.

TMM comb of this geometry:

| band | modes (nm) | Q | spacings → delay periods |
|---|---|---|---|
| pump | 1656.1 / 1577.6 / 1525.1 / 1492.7 | 67 / 46 / 41 / 41 | **152.8**, 234.6, 92.6 fs |
| probe | 844.8 / 816.2 / 800.1 / 781.8 / 763.6 | 76–94 | 80.4, 135.3, 113.7, 109.7 fs |

The two design pumps *are* the 1577.6/1525.1 pair, which is why 152.8 fs coincides with the nominal
pump beat. Note probe-band FSR / pump-band FSR = **1.43**: the cavity is dispersive, there is no
single "round-trip time", and which comb appears depends on **which pulse is delayed**.

This sets the grid: the step must resolve 92.6 fs (25 fs → 3.7 points) and the span must outlast the
~35–40 fs energy ring-down (±400 fs ≈ 10 lifetimes).

## 4. Bug found: the delay convention made the two halves different experiments

In the lab pump1 moves and pump2+probe stay locked. The old code kept the relative *timing* but, to
keep every source causal, shifted **pump2 and the probe** for τ<0 instead of pump1:

    t_start_pump1 = max(0, τ);   t_start_rest = max(0, −τ)

For τ>0 that is right. For τ<0 pump2 and the probe pick up exp(iω₂|τ|) and exp(iω_s|τ|) rather than
pump1 picking up exp(−iω₁|τ|) — a different experiment, and by §3 it exposes the *probe*-band comb
(135.3 / 113.7 fs) rather than the pump-band comb (152.8 fs). The observed ±τ asymmetry of the old
scan is explained by this, not by physics.

**Fix** (`--delay-pad-fs`): apply one *fixed* start-time pad to every source, held constant across
the scan, so pump1 is the only source whose timing changes for both signs of τ:

    t_start_pump1 = pad + τ;   t_start_rest = pad,   pad ≥ |τ_min|

A common offset is physically harmless — it shifts all fields in time, and the Stokes parameters are
built from same-frequency products, so the global phase cancels. The legacy path is kept (pad
unset) so earlier runs still reproduce. For τ ≥ 0 the two conventions are identical, which the 3D
scan exploits to avoid paying for the pad.

## 5. Definition issues (not bugs, but they change how numbers are read)

* **`pulse_duration_fs = 100` is not a 100 fs pulse.** `df_from_pulse_duration` sets
  width = T/(2 ln2), so σ = 72.13 fs and the **intensity FWHM is 120.1 fs** (field FWHM 169.9 fs).
  The pulse is properly transform-limited (TBP = 0.4413), just 20% longer than its label. Envelope
  widths must be compared to 120 fs, not 100 fs.
* **The probe DFT band only just contains the sidebands.** The band is f_probe ± df/2 = ±0.02312
  /µm and the sidebands sit at ±Δ = ±0.021944 — at **94.9%** of the half-width. It works for this
  design but leaves no margin if Δ grows or the operating point moves; worth widening before reusing
  this readout elsewhere.

## 6. Numerics — what was checked

* **Convergence in run length: passed.** Re-running at `--decay-threshold 1e-6` (run length ×3–4,
  t_stop 1245–1358 vs 268–386) moved V−H by ≤4.4e−5 and typically ~1e−7, i.e. ~4% of the
  point-to-point scatter. The 1e−4 runs are converged; the structure in the old trace is real,
  reproducible physics, not truncation.
* **Stokes bookkeeping: passed.** The aggregation helper reproduces the simulator's own
  θ, χ, DoLP and V−H to 0.00e+00, including for a 20° elliptical probe.
* **Global-shift invariance: see §7.**

## 7. Global-shift test — invariance holds, and t=0 is anomalous

τ = 0 repeated at pads of 0, 12.5, 25, 50, 100, 200 fs. Analytically the answer must be
pad-independent (§1).

| pad (fs) | pad mod T₁ | V−H | θ_pulse (°) | θ_legacy (°) | t_stop |
|---|---|---|---|---|---|
| 0 | 0.000 | 9.008846e−4 | 0.025892 | 0.138299 | 267.63 |
| 12.5 | 2.350 | 8.830526e−4 | 0.025380 | 0.143040 | 271.38 |
| 25 | 4.700 | 8.829873e−4 | 0.025378 | 0.142645 | 275.13 |
| 50 | 4.325 | 8.826331e−4 | 0.025368 | 0.142814 | 282.37 |
| 100 | 3.574 | 8.810378e−4 | 0.025322 | 0.143191 | 297.01 |
| 200 | 2.073 | 8.829890e−4 | 0.025378 | 0.142691 | 327.58 |

**Invariance holds for every pad > 0: V−H reproducible to 0.23%, θ_legacy to 0.38%**, even though
`pad mod T₁` ranges over most of a carrier period and t_stop varies by 56 Meep units. The
correlation of V−H with `pad mod T₁` among those runs is −0.18, so there is **no carrier-phase
(CEP) dependence** — the nonlinearity is not resolving the absolute optical phase, as it should not
for a ~24-cycle pulse.

**pad = 0 is the outlier**: +2.08% in V−H and −3.20% in θ_legacy relative to the mean of the padded
runs. The only thing special about it is that the sources turn on at exactly t = 0, coincident with
the start of the simulation, so this is a start-up transient of the t = 0 turn-on rather than
anything physical. Two consequences:

* **The per-point systematic of this readout is 0.23%** (0.38% for the legacy estimator). That is
  the honest error bar on every absolute number quoted from a padded run, and it is far smaller than
  any effect the study is after.
* **The historical numbers, including the published 0.137°, were all taken at pad = 0** and
  therefore carry that ~2–3% start-up offset. Small, but it explains why a padded τ = 0 run gives
  0.1428° rather than 0.1383°, and it means padded and unpadded runs should not be mixed in one
  comparison.

The pad is held **fixed across a scan**, so this systematic is common-mode and cannot distort the
τ-dependence, which is what the study measures.

## 8. Campaign as run

1D (`delay_physics.py`, 336 sims): τ ∈ [−400, +400] fs, step 25 fs (χ₀ = 0) / 50 fs (ellipticity
families χ₀ = 5, 10, 20°), 4 carrier sub-samples each, pad 500 fs, res 80, decay 1e−4,
I_pump = 1e12, I_probe = 5e7 W/cm².

3D (`delay_3d.py`, 32 sims): τ ∈ {0 … 400} fs, τ ≥ 0 only (no pad needed), 4 sub-samples,
24 MPI ranks, res 30, decay 1e−4 — same geometry, modes, materials and n₂ as the 1D study and as
the 1.991° reference run, so 3D/1D ratios are directly comparable.
