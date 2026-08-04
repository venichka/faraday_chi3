# Fabricated SiC-cavity samples — χ⁽⁵⁾ operating-point analysis

Two samples were fabricated (2026-08-03) and this directory answers, for each of them:

1. **which probe / pump₁ / pump₂ wavelengths to use**,
2. **what rotation to expect**, and
3. **whether the effect is visible above the carrier fringe**, or whether the delay-dither
   procedure is still required.

---

## 1. What the samples are

```
Air | [SiN 237.5 / SiO₂ 344.2 nm] ×3 | SiC cavity (3.2 or 4.8 µm) | [SiO₂ / SiN] ×3 | SiO₂ sub
```

⚠️ **SiC replaces only the cavity spacer.** The mirrors are the fabricated `best_absolute`
SiN/SiO₂ stack, unchanged — same materials, same thicknesses, 3 pairs per side. This is *not*
the construction in `../SiC_optimizations/sic_L3p2um/` (2026-06), which swapped every
high-index layer to SiC and therefore has a completely different stopband; none of that study's
operating points transfer.

| | optical n·L @1.55 µm | stack | stopband (R>0.8) |
|---|---|---|---|
| SiN L=5.894 µm (the earlier fabricated sample) | 12.04 µm | 8.38 µm | 1685–2281 nm |
| **SiC L=3.2 µm** | 8.18 µm | 6.69 µm | 1713–2403 nm |
| **SiC L=4.8 µm** | 12.28 µm | 8.29 µm | 1694–2322 nm |

Because the mirrors are untouched, the stopband barely moves. **L=4.8 is close to an optical
drop-in for the earlier SiN sample** (n·L within 2%), so it has a similar mode comb but ~10× the
nonlinearity where the field is strongest.

**Materials.** SiC cavity n₂ = 5×10⁻¹⁸ m²/W; SiN mirrors n₂ = 5×10⁻¹⁹ m²/W and **still
nonlinear**. Linear n,k for all three layers come from the measured ellipsometry CSVs
(`sic.csv`, `si3n4.csv`, `sio2.csv`) as 2-pole Lorentz fits.

⚠️ **The SiC n₂ is not a measured or literature value** — it is flagged "user-specified" in
`nonlinear_materials.py`. θ_χ5 scales roughly as n₂², so a 2× error in n₂ is a 4× error in the
predicted rotation. Every number here is reported so it can be rescaled.

---

## 2. The simulator needed a three-material extension

`faraday_meep_fp_circ.py` assumed a **two-material stack**: one high-index medium was assigned to
both the mirror layers and the cavity (`materials[high_slot] = mat_sin`), and `geometry_io`
inferred the mirror thickness by matching layers *against the cavity material*. A SiC cavity
inside SiN mirrors could not be expressed.

Added (all purely additive, defaults reproduce the old behaviour exactly):

| flag | meaning |
|---|---|
| `--cavity-material {sin,tio2,sic}` | preset for the cavity when it differs from the mirrors |
| `--cavity-fit CSV` | ellipsometry CSV for the cavity under `--materials fit` |
| `--n-cav`, `--k-cav` | constant-index overrides for the cavity |
| `--cavity-n2` | cavity Kerr n₂; the mirrors keep their own |

χ³ is now set on **both** media, each from its own n₂ and its own linear index.

### Validation

- **Regression — bit-identical.** The stored SiN `best_absolute` case re-run with the new code
  matches to `rel 0.00e+00` on S₀, S₁, S₂, S₃, DoLP and both estimators.
- **Layer stack** — mirrors n = 2.0773 / 2.0431 (SiN), cavity n = 2.6742 / 2.5574 (SiC),
  spacers and substrate SiO₂, all matching the ellipsometry at 800 / 1550 nm.
- **χ³ assignment** — χ³(SiC)/χ³(SiN) = 16.594 against the expected 10·(n_SiC/n_SiN)² = 16.579.

### ⚠️ 2 Lorentz poles, not 3
The 2026-06 study used 3 poles for SiC and the project notes say it "needs" 3. Over the window
this campaign uses (600–2000 nm) a **2-pole fit reproduces `sic.csv` to ≤ 0.0001 in n** at both
800 and 1550 nm, with no NaN or unphysical ε (checked at 2, 3 and 4 poles). Keeping 2 matters
because `--fit-poles` is **global** — raising it for SiC would silently re-fit the SiN mirrors
too and break comparability with the SiN campaign.

---

## 3. Operating-point space

The objective is the **carrier-averaged, pulse-integrated** rotation (4 pump-1 phases spanning
one optical period) — the physical observable, not the tail-window azimuth. The full derivation
and the fringe/effect separation are in
[`../chi5_dbr_design/docs/fringe_vs_effect.md`](../chi5_dbr_design/docs/fringe_vs_effect.md);
the harness is imported unchanged from `chi5_dbr_design/common.py`.

**Lab constraints (user, 2026-08-03):**
- probe `{~800} ∪ [850–950] nm` **today**, `[600–900] nm` **possible later** — both analysed;
- pumps **1400–2000 nm**, broad.

Pump centres are laid out as fractional offsets around the FWM-matched point 2·f_pump = f_probe,
**not** snapped to cavity modes: every pump-band mode has Q = 33–185 against a 100 fs Q_cap of
≈12, so the modes are unresolved by the pulse and the centre is a continuous knob.

### ⚠️ Extending the probe below ~700 nm buys nothing on its own
A probe mode is only usable if its FWM-matched pump, at 2×λ_probe, is inside the 1400–2000 nm
pump range. That puts a hard floor at **λ_probe ≈ 700 nm**:

| sample | probe modes in 600–1000 nm | usable | dropped (matched pump < 1400 nm) |
|---|---|---|---|
| L=3.2 | 18 | 13 (676–985 nm) | 604, 616, 627, 641, 658 nm |
| L=4.8 | 22 | 17 (676–970 nm) | 605, 614, 620, 630, 640 nm |

So the "future" `[600–900] nm` probe window is effectively `[700–900] nm` **unless the pump
source is also extended below 1400 nm.**

---

## 4. Stages

| stage | script | what it does |
|---|---|---|
| 0 | `s0_intensity.py` | intensity ladder → pick the reference pump intensity (see below) |
| 1A | `s1_scan.py --phase A` | every probe mode × 3 centre offsets × 3 Δ (868 sims) |
| 1B | `s1_scan.py --phase B` | top 2 probe modes per sample, fine grid (672 sims) |
| 2 | `s2_analyze.py` | applies the "now" / "future" probe filters, ranks both axes |

Aggregation walks the run directory rather than the current phase's op list, so a phase-B run
cannot truncate the phase-A map.

### ⚠️ Why stage 0 exists
SiC's ~10× n₂ means the intensity that keeps the SiN sample comfortably perturbative
(10¹² W/cm²) does **not** here. A single-phase probe on L=4.8 at 10¹² W/cm² comes back with
**DoLP 0.72 and ≈0.9° rotation** — deep in the large-signal regime, where the azimuth is barely
defined and θ ∝ I² no longer holds. That is the same trap the 2026-06 SiC study fell into
("the probe spins ~370° during the pulse then freezes"). Stage 0 finds the intensity where
DoLP ≈ 1 and the local log-log slope is ≈ 2, so the reported numbers mean what they say.

---

## 5. Running

```bash
micromamba activate meep-mpi     # cluster;  conda activate mp  locally

sbatch chi5_sic_samples/s0_intensity.sbatch                              # pick I_ref
sbatch --export=ALL,PHASE=A chi5_sic_samples/s1_scan.sbatch              # coarse map
sbatch --export=ALL,PHASE=B chi5_sic_samples/s1_scan.sbatch              # refine winners
python  chi5_sic_samples/s2_analyze.py                                   # the answer
```

⚠️ `sbatch --export` splits on commas — never put a comma inside an exported value.
