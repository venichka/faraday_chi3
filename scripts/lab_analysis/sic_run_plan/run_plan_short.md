# SiC χ⁽⁵⁾ Run Plan — Short Version

As of 2026-09-25. Snapshot of the Claude Doc [SiC χ⁽⁵⁾ Run Plan — Short Version](https://claude.ai/code/artifact/21d4934d-826b-4ea9-be22-0400bf186505), which stays the living version.

## Goal

Show the χ⁽⁵⁾ all-optical Faraday rotation on the SiC L = 4.8 µm sample: an overlap-only rotation of about 0.074° (2.6 × 10⁻³ of V+H) that needs both pumps and vanishes for co-rotating pumps.

The SIN51 run found a 1.7° overlap signal, about 12× larger than χ⁽⁵⁾ allows; this run is built to tell χ⁽⁵⁾ apart from that. It succeeds when the overlap bump repeats, scales as I², and is absent with the probe blocked and with σ⁺σ⁺ pumps.

Reasons and full numbers: [SiC χ⁽⁵⁾ Run Plan](run_plan.md).

## Settings per sample

Use L = 4.8 µm at the first column; it is the only point where the effect beats the fringe in 3D. Values are design-stack simulations; the wavelengths move once the linear spectrum is fitted (step 0).

|  | L = 4.8 µm — use | L = 4.8 µm — fallback | L = 3.2 µm — control |
|---|---|---|---|
| probe | 794.2 nm | 759.4 nm | 850.2 nm |
| pumps | 1515.2 / 1569.9 nm | 1492.5 / 1545.6 nm | 1655.9 / 1695.2 nm |
| peak intensity per pump | 1 × 10¹¹ W/cm² (max 2 × 10¹¹) | same | same |
| expected χ⁽⁵⁾ rotation | 0.074° | 0.120° | 0.017° |
| carrier fringe amplitude | 0.047° | 0.132° | 0.25° |
| effect / fringe | 1.59 | 0.91 | 0.07 |
| fringe period, long / short pump delayed | 5.24 / 5.05 fs | 5.16 / 4.98 fs | 5.65 / 5.52 fs |

The a-SiC n₂ is uncertain (3.0–6.7 × 10⁻¹⁸ m²/W); if the signal is weak, raise both pumps together to 1.6 × 10¹¹ before anything else.

## Run order

Each step is a delay scan with the same settings and file format unless noted; do them in this order.

1. **Linear spectrum** of each sample at the spot to be used, 700–1900 nm at ≤ 1 nm. Send it before tuning: the as-built stack is fitted to it and the probe and pump wavelengths are returned within a day.
2. **Set the point**: pumps at the returned wavelengths, 1 × 10¹¹ W/cm² each, σ⁺ and σ⁻; probe 45° linear and weak; the chopper on the pump that travels with the probe, synchronised to the laser (see Acquisition).
3. **Delay-axis calibration**: scan a known-wavelength interference on the same stage and compare its period with λ/2 of travel. SIN51 fitted only with the axis stretched ×1.12.
4. **Fringe-phase drift**: park near the overlap and log X and Y for 1 minute.
5. **Coarse scan**: ±2 ps at 20 fs to find the overlap centre τ₀ (on SIN51 it was 159 µm from the nominal zero).
6. **Probe blocked** and **both pumps blocked**, over τ₀ ± 200 fs. Any overlap signal with the probe blocked is generated light, not rotation.
7. **Main measurement**: τ₀ ± 200 fs at 10 fs steps (41 points), σ⁺σ⁻ pumps. Repeat back to back, at least 6 scans; at the SIN51 noise that fixes the k = 0 term to about 0.013°.
8. **σ⁺σ⁺ null**: the same scan set with co-rotating pumps; χ⁽⁵⁾ must vanish.
9. **Intensity series**: 6 × 10¹⁰ and 2 × 10¹¹ W/cm² besides the main 1 × 10¹¹, both pumps scaled together, at least 6 scans each. χ⁽⁵⁾ grows as I²; this fixes the exponent to about ±0.2.
10. **Blank spot**, then the L = 3.2 µm sample as a control if time allows.

## Acquisition and what to record

Chop the pump that travels with the probe, not the delayed one. The delayed pump's own effects are then present in both chopper states and cancel, so the overlap bump is the two-pump term alone on a flat pedestal. SIN51 chopped the delayed pump, which leaves that pump's own rotation in the bump.

- **Recommended**: chopper on the pump that travels with the probe, 500 Hz, synchronised to the laser. Same lock-in calibration as SIN51: θ = X / (√2 S₀) × G(V+H)/G(V−H).
- **Optional double modulation** also removes the flat pedestal, at a cost in scans. Per-pulse recording: pump₁ 500 Hz, pump₂ 250 Hz, both synchronised; the four states cycle every four pulses, about 3× more scans. Lock-in only: 500 and 333.3 Hz, reference 166.7 Hz (laser ÷ 6), sine demodulation, θ = 3X / (√2 S₀) × gain ratio, about 8× more scans.
- **Not 500 with 250 Hz on a lock-in**: pump₂'s own signal then appears at the sum and difference frequencies, larger than the two-pump term.

Settings:

- Broadband balanced detection, no filter in the probe path.
- Probe 45° linear, ≤ 10⁻⁴ of the pump intensity; log its ellipticity.
- Lock-in time constant and wait per point (≥ 5 time constants) fixed for a series and logged.
- Each scan saved as its own file, never averaged in the instrument.

Per point: nominal delay, stage read-back, X and Y (or per-pulse V−H with the chopper states), V+H, time stamp.

Per file: sample and spot; which pump is delayed; modulation scheme and frequencies; pump and probe wavelengths with spectra; powers, spot sizes, pulse durations and helicities; V−H and V+H gains; lock-in settings; stage model and calibration factor.

## Do not

Each item cost the SIN51 run something.

- Chop the delayed pump, alone or together with the other on one chopper: its own rotation then stays in the overlap bump.
- Put the 10 nm filter in the probe path: in simulation the fringe grows (0.050° → 0.069° on SiC) and the effect does not.
- Expect 10 fs steps to resolve the 5 fs fringe: 10 and 20 fs alias it to the same ~67 fs period. Beat it with repeats.
- Go above 2 × 10¹¹ W/cm² per pump: the SiC probe depolarises and the I² law breaks. SIN51 ran at 10¹².
- Set the probe by wavelength alone: find the mode by counting from the design comb. On SIN51 the 800 nm probe sat on the mode designed for 781.8 nm.
- Use the 689–695 nm probe modes: the probe comes back half depolarised there.
- Trust the nominal τ = 0, or skip the probe-blocked scan.

## Before the run

- [ ] Which pump travels with the probe on SiC (it takes the chopper), and which is delayed?
- [ ] Can the chopper be synchronised to the laser trigger? For optional double modulation: a second chopper at 333.3 Hz, or per-pulse recording?
- [ ] V−H and V+H channel gains.
- [ ] Can the probe reach 794 nm, and 759 nm for the fallback?
- [ ] Pulse durations and spot sizes at the sample.
- [ ] Lock-in time constant and wait per point used for SIN51.
- [ ] Delay stage model; is the read-back an encoder reading?
- [ ] Detector model.
