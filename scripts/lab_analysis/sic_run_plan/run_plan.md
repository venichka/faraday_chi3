# SiC χ⁽⁵⁾ Run Plan

As of 2026-09-25. Snapshot of the Claude Doc [SiC χ⁽⁵⁾ Run Plan](https://claude.ai/code/artifact/d9a20520-6897-4eed-94d0-2a1660942393), which stays the living version; the one-page bench version is [run_plan_short.md](run_plan_short.md).

Run the L = 4.8 µm sample at probe 794.2 nm and pumps 1515.2 / 1569.9 nm, 10¹¹ W/cm² per pump, with the chopper on the pump that travels with the probe. The simulated χ⁽⁵⁾ rotation there is 0.074°, larger than its carrier fringe. The one-page bench version is [SiC χ⁽⁵⁾ Run Plan — Short Version](run_plan_short.md).

## Why this plan

The SIN51 scans found the pump–probe overlap and a large signal, but could not isolate the χ⁽⁵⁾ rotation. Each rule below removes one of the reasons.

| Seen in SIN51 | Consequence | Rule here |
|---|---|---|
| Only pump₂ was chopped | The lock-in difference keeps pump₂'s own χ⁽³⁾ rotation and any pump₂-induced transmission change; the balanced-pump cancellation does not apply to it | Chop the pump that travels with the probe instead ([Acquisition rules](#acquisition-rules)) |
| Delay step 20–30 fs | The 5.4 fs carrier fringe was aliased; effect and fringe could not be separated | Treat the fringe as random-phase noise: 10 fs steps, ≥ 6 repeats, carrier-free fit; or scramble the fringe phase ([Acquisition rules](#acquisition-rules)) |
| Pumps from two OPAs, not phase-locked | The fine structure did not repeat between scans; no model predicted one scan from the other | Piezo phase scrambling at acquisition; repeated scans ([Acquisition rules](#acquisition-rules)) |
| V+H not recorded; V+H and V−H gains differ | Angles known only up to a gain factor (~7 suspected); transmission changes leak into V−H | Record V+H with V−H and log both gains ([Acquisition rules](#acquisition-rules), [Data and metadata](#data-and-metadata-to-record)) |
| Sample differed from design (SiN +3 %, SiO₂ +1 %) | Pumps retuned by hand; the probe sat on a different mode than designed | Linear spectrum first; operating point re-derived for the as-built stack ([Step 0](#step-0-measure-the-sample-before-any-nonlinear-run)) |
| τ = 0 nominal; overlap 159 µm away | Most of each scan spent far from the overlap | Coarse scan to locate the overlap, then fine scans around it ([Acquisition rules](#acquisition-rules)) |
| Simulated fringe fitted only with the delay axis ×1.12 | Stage calibration in doubt | Calibrate the delay axis with a known fringe ([Controls](#controls-and-diagnostics)) |
| 0.58° pedestal at all delays, origin unknown | Rotation, generated light and pickup indistinguishable | Probe-blocked and pump-blocked scans ([Controls](#controls-and-diagnostics)) |

Three SiC-specific points add to these. The pumps must be 10× weaker than for SiN (10¹¹ W/cm²), the a-SiC n₂ is uncertain by ×0.6–1.3, and two-photon absorption at the probe is allowed and unmodelled, so the total transmission must be recorded too ([Controls](#controls-and-diagnostics)).

## What the SIN51 data showed

The two SIN51 delay scans hold a real overlap-only signal of +1.73°. It is about 12× larger, relative to the fringe, than χ⁽⁵⁾ allows at that setting, so it is most likely not the χ⁽⁵⁾ rotation. Angles below assume a gain ratio of 1.

| Quantity | Measured (scans 146 and 235) | Simulated 1D | Simulated 3D |
|---|---|---|---|
| overlap position | stage 185.771 mm (τ = −1063 fs on the nominal axis), same in both scans | — | — |
| pedestal, all delays | 0.58° | — | — |
| carrier fringe amplitude | 1.83° | 0.080° | 0.265° |
| k = 0 overlap term | +1.73° [1.60, 1.90], 4.9σ | +0.0025° | +0.021° |
| k = 0 / fringe | 0.95 | 0.03 | 0.08 |
| envelope FWHM, k = 0 / fringe | 115 / 175 fs | — | — |

Simulations use the as-built stack at the lab setting: probe 800 nm, pumps 1560 / 1620 nm (1620 delayed), 10¹² W/cm², 125 fs. The k = 0 value comes from a carrier-free fit that integrates the random fringe phase out. The ratio in the fifth row does not depend on the unknown channel gains; the measured fringe alone would match the 3D prediction if G(V−H)/G(V+H) ≈ 7.

| Candidate for the +1.73° term | Status | Test that decides |
|---|---|---|
| χ⁽⁵⁾ rotation | Too small: k = 0 / fringe 0.08 simulated against 0.95 measured | σ⁺σ⁺ null, I² scaling |
| Delayed pump's own χ⁽³⁾ rotation, kept because only that pump was chopped | Too small: +0.0016° in 1D at the lab setting | Chop the pump that travels with the probe |
| Pump-induced transmission change leaking into V−H | At most 0.6° even with full leakage (simulated dip −2.2 %) | Record V+H |
| Pump light generated in the probe band: 1560 + 1620 nm sum frequency at 794.7 nm | Not excluded: overlap-only, chopped with the delayed pump, needs no probe | Probe-blocked scan |
| Delay-axis error | Simulated fringe fits only with the axis stretched ×1.12 | Stage calibration |

- As-built stack from the three resonances: SiN layers +3.0 %, SiO₂ +0.95 % (1.5 nm rms). The 800 nm probe sat on the mode designed at 781.8 nm.
- Over 20 fringe models were fitted. Carriers fixed at the physical values fail (χ² per point 9–17); the best physical model reaches 1.5, and no model predicts one scan from the other. Steps of 20–30 fs cannot identify the carrier.
- File 626 (pumps fixed) is not a clean null: a flat 0.31° and a weak fringe (0.10°) at the same stage position as the overlap.

Analysis code and figures: [scripts/lab_analysis/sin51_lockin](../sin51_lockin/), stages s0–s8.

## Operating points per sample

Use the L = 4.8 µm sample at probe 794.2 nm and pumps 1515.2 / 1569.9 nm: it is the only point where the χ⁽⁵⁾ effect exceeds the carrier fringe in 3D. The L = 3.2 µm sample gives 4× less rotation and 23× worse contrast; keep it as a control sample. All values are design-stack simulations (1640 runs, 1D + 3D) at 10¹¹ W/cm² peak per pump and 100 fs pulses; [Step 0](#step-0-measure-the-sample-before-any-nonlinear-run) re-derives the wavelengths for the as-built stacks.

|  | L = 4.8 µm — use | L = 4.8 µm — fallback (needs probe < 790 nm) | L = 3.2 µm — control only |
|---|---|---|---|
| probe (cavity mode Q, transmission) | 794.2 nm (Q 94, T 0.91) | 759.4 nm (Q 114, T 0.83) | 850.2 nm (Q 69, T 0.92) |
| pumps | 1515.2 / 1569.9 nm | 1492.5 / 1545.6 nm | 1655.9 / 1695.2 nm |
| pump centre, splitting Δ | 1542.0 nm, 0.0230 µm⁻¹ (6.9 THz) | 1518.6 nm, 0.0230 µm⁻¹ | 1675.3 nm, 0.0140 µm⁻¹ (4.2 THz) |
| frequency matching, 2f_pump vs f_probe | 3.0 % off | 0.01 % off | 1.5 % off |
| peak intensity per pump | 1 × 10¹¹ W/cm² | 1 × 10¹¹ W/cm² | 1 × 10¹¹ W/cm² |
| θ_χ5, 3D | 0.074° | 0.120° | 0.017° |
| carrier fringe amplitude, 3D | 0.047° | 0.132° | 0.25° |
| contrast = effect / fringe, 3D | 1.59 | 0.91 | 0.07 |
| degree of linear polarisation, 3D | 0.72 | 0.77 | 0.82 |
| θ_χ5 / fringe, 1D plane wave | 0.0074° / 0.050° | 0.0140° / 0.057° | 0.0060° / 0.0056° |
| fringe period if the long pump is delayed | 5.24 fs = 0.785 µm stage travel | 5.16 fs = 0.773 µm | 5.65 fs = 0.848 µm |
| fringe period if the short pump is delayed | 5.05 fs = 0.758 µm | 4.98 fs = 0.746 µm | 5.52 fs = 0.828 µm |
| quarter-fringe dither step (N = 4) | 0.196 / 0.189 µm | 0.193 / 0.187 µm | 0.212 / 0.207 µm |

Stage travel assumes a double-pass delay line (travel = λ/2 per fringe).

- Intensity window: the response is clean χ⁽⁵⁾ (log-log slope ≈ 2, polarisation preserved) up to 2 × 10¹¹ W/cm². At 10¹², the SiN setting, the probe depolarises (DoLP 0.81) and the I² law breaks (slope 1.42).
- n₂ margin: measured a-SiC films span 3.0–6.7 × 10⁻¹⁸ m²/W (5 × 10⁻¹⁸ assumed). Over that range θ changes ×0.4–1.7 and the contrast ×0.67–1.25 (3D contrast 1.1–2.0). Only the product n₂·I matters, so a weak signal is first answered by raising I to ≤ 1.6 × 10¹¹ W/cm².
- Do not use the 689–695 nm probe modes on either sample: the largest 1D rotations sit there, but in 3D the probe comes back half depolarised (DoLP 0.40–0.45), so the azimuth is not a rotation.
- The L = 3.2 µm sample looked best in 1D (contrast 1.08 at 850 nm) and collapses in 3D because its fringe grows 43×; that is why it is not the demonstration sample.
- Generated light to watch for with the L = 4.8 pumps: pump sum frequency at 771 nm (outside the ±18 nm probe band) and pump second harmonics at 757.6 and 785.0 nm (the second is inside the band).

## Step 0: measure the sample before any nonlinear run

Record the linear transmission spectrum of each sample at the spot to be used, 700–1900 nm at ≤ 1 nm resolution, normal incidence, and send it before tuning the lasers. The as-built layer thicknesses will be fitted to it (as done for SIN51) and the operating point re-derived within a day; the design wavelengths under [Operating points](#operating-points-per-sample) will move.

The mirrors are the SIN51 mirror recipe. If they carry the same error (SiN +3.0 %, SiO₂ +0.95 %), the cavity modes shift as below; the SiC cavity's own thickness error is unknown, so a +2 % case is shown as well. Identify modes by counting from the design comb, not by wavelength.

| L = 4.8 µm | design | mirrors as SIN51 | mirrors as SIN51 + cavity +2 % |
|---|---|---|---|
| probe-band modes (nm), design probe in bold | 759.3, 776.3, **794.1**, 810.3, 828.2, 850.0 | 761.4, 778.4, **796.3**, 814.9, 830.2, 852.3 | 772.7, **790.0**, 808.0, 825.0, 843.4 |
| pump-band modes (nm) | 1449.5, 1538.1, 1590.9, 1672.9 | 1454.3, 1527.6, 1564.4, 1599.2, 1678.9 | 1477.5, 1567.5, 1621.1, 1705.1 |

| L = 3.2 µm | design | mirrors as SIN51 | mirrors as SIN51 + cavity +2 % |
|---|---|---|---|
| probe-band modes (nm) | 774.7, 799.5, 818.6, **850.2**, 882.3 | 777.6, 802.6, 825.8, **853.4**, 885.4 | 788.3, 813.4, 833.6, **865.7**, 898.1 |
| pump-band modes (nm) | 1482.6, 1536.7, 1577.7, 1687.9 | 1489.4, 1590.6, 1695.6 | 1511.5, 1566.5, 1607.4, 1720.1 |

Rules for setting the operating point on the as-built sample:

1. Probe: the mode that corresponds to the design one (L = 4.8: the third mode from the short-wavelength end of the 740–900 nm comb). Wavelength alone is not enough: on SIN51 the 800 nm probe sat on the mode designed at 781.8 nm.
2. Pump centre: half the probe frequency, i.e. 2 f_pump = f_probe within 3 %. The pump-band modes are broad (Q 40–140 against Q ≈ 12 for a 100 fs pulse), so the pumps need not sit on a mode.
3. Splitting: keep Δ = 0.023 µm⁻¹ (55 nm apart at 1540 nm) for L = 4.8. Larger Δ pushes the sidebands out of the ±18 nm readout band.
4. Also record before the run: both pump spectra, the probe spectrum, pulse durations at the sample, spot sizes and powers (for the intensity), and the probe ellipticity.

## Acquisition rules

The measurement must isolate the two-pump signal, resolve or remove the carrier fringe, and record the total transmission alongside the difference.

1. Modulation: chop the pump that travels with the probe, at 500 Hz synchronised to the laser. The lock-in then reads θ(both pumps) − θ(delayed pump only). The delayed pump's own rotation, generated light and heating are present in both chopper states and cancel, so the overlap bump is the two-pump term alone, on a flat pedestal. Do not chop the delayed pump alone, as for SIN51, and do not chop both pumps with the same chopper: either way the delayed pump's own rotation stays in the overlap bump. In the SIN51 as-built simulation that setting reads +0.0016°, against +0.0034° for the two-pump term. What survives is everything that needs both pumps: the χ⁽⁵⁾ rotation, the four-wave-mixing fringe and any pump sum-frequency light. Double modulation also removes the pedestal, at a cost in scans; on a 1 kHz laser its frequencies must be chosen as in the table below.
2. Pump intensity: 1 × 10¹¹ W/cm² peak per pump, never above 2 × 10¹¹. Both pumps at the same intensity (imbalance re-introduces the χ⁽³⁾ term); helicities σ⁺ and σ⁻.
3. Find the overlap: one coarse scan, ±2 ps at 20 fs. Every fine scan then covers ±200 fs around the overlap centre, and the stage start position is the same for every file of a series.
4. Delay step and repeats: the 5.0–5.7 fs fringe cannot be resolved at 10 or 20 fs steps (both alias it to the same ~67 fs apparent period), so it is treated as random-phase noise and averaged down by statistics. Use 10 fs steps over ±200 fs (41 points) and repeat each scan at least six times back to back. At the SIN51 noise the carrier-free estimator then gives the k = 0 term to about 0.013° (expected 0.074°); ±300 fs would cost half as much time again for the same precision. If steps ≤ 2.7 fs ever become available, one such scan resolves the fringe directly and calibrates the delay axis from its period.
5. Fringe-phase scrambling, if a piezo mirror is available: dither the delayed pump's path by at least one fringe (≥ 0.79 µm of double-pass travel) at a rate well above 1/τ_lock-in and not a multiple of the chopper frequency. The lock-in then reads the phase-averaged rotation directly and fewer repeats are needed. The stage cannot make the 0.19 µm quarter-fringe sub-steps, so the N = 4 dither under [Operating points](#operating-points-per-sample) is not an option here.
6. Channels: V−H and V+H recorded together at every point, each with its gain noted; lock-in X and Y (not only R), the lock-in phase, and the stage read-back position.
7. Lock-in: time constant and wait per point (≥ 5 time constants) fixed for a series and logged. Both were unknown for SIN51.
8. Repeats: every fine scan at least five times back to back (six for the main measurement), saved as separate files, never averaged in the instrument.
9. Detection: broadband balanced detection, no bandpass in the probe arm. Simulated for both samples, a 10 nm filter leaves the fringe in place — it rises from 0.080° to 0.124° on SIN51 and from 0.050° to 0.069° on SiC L = 4.8, because the fringe field spans the whole probe spectrum, not only the ±14 nm sidebands — while the effect does not grow, so the contrast falls (SIN51 0.031 → 0.008, SiC 0.149 → 0.131).
10. Probe: 45° linear, weak (≤ 10⁻⁴ of the pump intensity), fixed in power and wavelength for a series. A 5° ellipticity adds a delay-independent offset; measure and log it.

Modulation schemes on the 1 kHz laser. Scan counts are for 5σ on a 0.07° k = 0 term at the SIN51 noise per point, ±200 fs at 10 fs steps; the chopper patterns were checked numerically on a pulse train.

| Scheme | What the overlap bump holds | Signal per point vs SIN51 | Scans needed |
|---|---|---|---|
| Chop the pump that travels with the probe, 500 Hz (recommended) | two-pump term only, on a flat pedestal | 1 | 5–6 |
| Chop the delayed pump, 500 Hz (as SIN51) | two-pump term + delayed pump's own rotation | 1 | not usable |
| Both pumps on one chopper | two-pump term + delayed pump's own rotation | 1 | not usable |
| Per-pulse recording, pump₁ 500 Hz + pump₂ 250 Hz, synchronised | two-pump term only; all four pump states recorded | noise ×1.9 | 14–16 |
| Lock-in, pump₁ 500 Hz + pump₂ 333.3 Hz, synchronised, reference 166.7 Hz | two-pump term only, no pedestal | 1/3; θ = 3X / (√2 S₀) × gain ratio | 34–39 |
| Lock-in, 500 Hz + 250 Hz, reference 250 or 750 Hz | two-pump term + pump₂ alone (0.50 against 0.35) | — | not usable |

On SiC the simulated single-pump rotations are 1–3 % of the two-pump term, so the Kerr part barely depends on the scheme. The choice matters for what a single pump can produce that no simulation contains: SIN51's 0.58° pedestal came from the delayed pump alone, and pump₂'s second harmonic (785.0 nm) falls inside the probe band.

## Controls and diagnostics

Each control is a delay scan over the same stage range, with the same settings and file format as the measurement, unless noted. Together they decide what the overlap signal is.

| Control | What it tests | Expected if the signal is χ⁽⁵⁾ rotation |
|---|---|---|
| Probe blocked | Light generated by the pumps at the probe wavelength (sum frequency, second harmonics) reaching the detectors | Nothing at the overlap and no pedestal |
| Delayed pump alone (other pump blocked, chopper moved to the delayed pump) | That pump's own χ⁽³⁾ rotation and transmission change | Small overlap bump, linear in its intensity; no fringe |
| Pump that travels with the probe alone (delayed pump blocked) | Its rotation and any pedestal it causes | Flat trace |
| Both pumps blocked | Chopper pickup, electrical offsets | Flat zero |
| Same helicity, σ⁺σ⁺ | The cascade and the four-wave-mixing fringe both vanish for co-rotating pumps | Overlap signal disappears |
| Intensity series, 3–4 levels from 3 × 10¹⁰ to 2 × 10¹¹ W/cm², both pumps scaled together | Order of the process | k = 0 term ∝ I² (exponent 1.8–2.2), fringe ∝ I^1.3–1.7 |
| V+H (total transmission) versus delay, analyser removed or from the V+H channel | Two-photon absorption at the probe (pump + probe photons exceed the a-SiC gap; not in any simulation) | Flat; a dip at the overlap means part of V−H is absorption, not rotation |
| Blank spot or substrate | Contributions from outside the cavity | Flat |

Two instrument diagnostics, once per session:

- Delay-axis calibration: scan a known-wavelength interference (a HeNe Michelson on the stage, or the probe's own fringe) and compare its period with λ/2 per micrometre of travel. The SIN51 fringe fitted the simulated physics only with the axis stretched by 1.12.
- Fringe-phase drift: park the stage at the overlap and log X and Y for at least one minute at the working time constant. The fluctuation time scale sets whether the phase must be scrambled within each reading ([Acquisition rules](#acquisition-rules), rule 5) or averages out between points.

## Run sequence

One SiC session in order. The probe-blocked scan comes before the main measurement because it decides what the overlap signal is.

| # | Step | Range and step | Repeats | Decides |
|---|---|---|---|---|
| 0 | Linear transmission spectrum at the spot ([Step 0](#step-0-measure-the-sample-before-any-nonlinear-run)) | 700–1900 nm, ≤ 1 nm | 1 | the as-built operating point |
| 1 | Delay-axis calibration with a known-wavelength fringe ([Controls](#controls-and-diagnostics)) | a few fringes | 1 | the fs-per-mm factor |
| 2 | Fringe-phase drift log, parked near the overlap ([Controls](#controls-and-diagnostics)) | 1 min, fixed delay | 1 | whether the fringe averages between points |
| 3 | Coarse scan to find the overlap | ±2 ps at 20 fs | 1 | τ₀ |
| 4 | Probe blocked; both pumps blocked | τ₀ ± 200 fs at 10 fs | 2 each | generated light, pickup |
| 5 | Main measurement, σ⁺σ⁻ | τ₀ ± 200 fs at 10 fs | ≥ 6 | the k = 0 term |
| 6 | σ⁺σ⁺ null | as 5 | ≥ 6 | whether k = 0 is χ⁽⁵⁾ |
| 7 | Intensity series: 6 × 10¹⁰ and 2 × 10¹¹ W/cm², both pumps together (1 × 10¹¹ is step 5) | as 5 | ≥ 6 each | the order of the process |
| 8 | Single-pump scans, chopper moved to the pump that is on | as 5 | 2 each | each pump's own rotation |
| 9 | Blank spot; L = 3.2 µm sample if time allows | as 5 | 2 | contributions outside the cavity |

Six scans at each of 6 × 10¹⁰, 10¹¹ and 2 × 10¹¹ W/cm² fix the exponent to about ±0.2. That separates I² from the fringe's I^1.3–1.7 only at about 2σ, so the σ⁺σ⁺ null is the stronger test. A 3 × 10¹⁰ point carries a tenth of the signal and adds almost nothing. Record V+H alongside V−H in every step.

## Data and metadata to record

One CSV per scan, one row per point, with a header block that makes the file self-describing. The SIN51 files lacked the header items marked new; without them the angles stay uncalibrated.

Per point:

| Column | Unit | Note |
|---|---|---|
| nominal delay | fs | the set point |
| stage read-back | mm | from the encoder, not the command |
| X, Y | V | both lock-in quadratures; R and phase may be added |
| V+H | V | total transmission, same point, gain noted (new) |
| time stamp | s | wall-clock, for drift checks (new) |
| dither sub-step | — | 0–3 when the quarter-fringe dither is used (new) |

Per file:

- sample ID and spot; which pump is delayed; which beams are chopped and how (single, both, double modulation); chopper and laser repetition rates
- probe wavelength and spectrum file, power, polarisation state (azimuth, ellipticity)
- pump wavelengths and spectrum files, powers, spot sizes, pulse durations, helicities, computed peak intensities
- lock-in: time constant, sensitivity, wait per point, reference phase
- detector model; gains of the V−H and V+H outputs; any filter in the probe path
- piezo dither: amplitude and frequency, or none
- delay stage: model, calibration factor from the fringe test ([Controls](#controls-and-diagnostics)), direction convention
- lab temperature and time; sample changes since the last file

File names: sample, configuration, start time (for example L4p8_both-chopped_2fs_HHMMSS.csv). Controls use the same name pattern with the control in the configuration field.

## Expected signals and success criteria

At the recommended L = 4.8 µm point the effect should be visible in the raw trace: a 0.074° bump with a 0.047° fringe riding on it, so the trace stays above the pedestal over the overlap. In V−H terms, 0.074° is 2.6 × 10⁻³ of V+H. With the SIN51 noise (0.06° per point at a 9 mV total signal) six fine scans give the k = 0 term to about 0.013°. The simulated 0.074° is the two-pump total at τ = 0; the lock-in bump is that total minus both single-pump rotations. On SiC L = 4.8 the single-pump rotations are only 1–3 % of the total (1D: −0.0001° and +0.0002° against −0.0074°), so the expected bump is the 0.074° to within 2 %. On SIN51 at 10¹² W/cm² they were comparable to it (0.0034° against 0.0025°).

The χ⁽⁵⁾ rotation is established when all of these hold:

1. The overlap bump has the same position and amplitude in three back-to-back scans.
2. Its k = 0 part scales as I² (exponent 1.8–2.2) over the intensity series, while the fringe scales as I^1.3–1.7.
3. It vanishes for σ⁺σ⁺ pumps.
4. It is absent with the probe blocked.
5. It survives normalisation by V+H, i.e. it is not a transmission change.
6. The fringe period equals λ_delayed / c on the calibrated delay axis (5.24 fs for the 1569.9 nm pump).
7. Its envelope is 100–150 fs wide, the pulse-overlap width.

Analysis: the SIN51 pipeline ([scripts/lab_analysis/sin51_lockin](../sin51_lockin/), stages s0–s8) is re-pointed at the new files. At 10–20 fs steps the fringe is random-phase noise and the k = 0 term comes from the carrier-free (phase-marginal) estimator over all repeats; chopping the pump that travels with the probe removes the delayed pump's own terms by construction; with a piezo scramble the lock-in output is the k = 0 term directly. The same as-built stack fit used for SIN51 then gives the 1D and 3D predictions the measured numbers are compared with.

## Open questions for the lab

Confirmed so far: double modulation of the two pumps is possible, V+H can be recorded, delay steps of 10 and 20 fs are available (not finer), and a 10 nm probe filter exists but should not be used ([Acquisition rules](#acquisition-rules)). The items below decide the rest.

- [x] Can the two pumps be chopped at different frequencies for double modulation? Yes.
- [x] Can V+H be recorded at the same time as V−H? Yes.
- [x] Delay steps available: 10 and 20 fs.
- [ ] Which pump travels with the probe (it takes the chopper), and which is delayed? The delayed one sets the fringe period ([Operating points](#operating-points-per-sample)).
- [ ] Can the chopper be synchronised to the laser trigger? For optional double modulation: a second chopper at 333.3 Hz, or per-pulse recording (boxcar or fast digitiser)?
- [ ] What are the V−H and V+H channel gains? (Needed also to calibrate the SIN51 angles.)
- [ ] Is a piezo-mounted mirror available in the delayed pump's path for the phase scramble?
- [ ] Can the probe be tuned to 794 nm, and to 759 nm for the fallback point?
- [ ] Pulse durations and spot sizes at the sample, for the intensity calculation.
- [ ] Lock-in time constant and wait per point used for SIN51.
- [ ] Delay stage model and whether the read-back column is an encoder reading.
- [ ] Detector model.
