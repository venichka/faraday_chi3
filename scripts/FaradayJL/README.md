# FaradayJL

`FaradayJL` is the Julia TCMT package used in this project to model all-optical Faraday rotation of a weak linearly polarized probe driven by two slightly detuned circularly polarized pumps in a nonlinear cavity.

The package sits downstream of the Meep workflow:

- `meep_project` computes cavity spectra, modal field profiles, linewidths, detunings, and overlap-derived nonlinear coefficients.
- `FaradayJL` turns those extracted parameters into reduced time-domain TCMT simulations for:
  - counter-rotating pumps
  - co-rotating pumps
  - the legacy overlap-proxy model
  - the derivation-consistent extracted-coefficient model

The current package is designed for interactive model inspection and side-by-side comparison against the earlier Julia implementation, not as a general-purpose published library.

## Physical Model

The intended process is a fifth-order effective nonlinear interaction built from a cascaded `χ^(3)` four-wave-mixing process:

- two pump pulses at frequencies `ω1` and `ω2`
- beat detuning `Δ = |ω1 - ω2|`
- a weak probe at `ωs`
- generated sidebands at `ωs + Δ` and `ωs - Δ`

Those sidebands couple differently to the `σ+` and `σ-` components of the probe and produce a polarization rotation in the transmitted probe field. The rotation is nonreciprocal because the pumps define a handed dynamic bias.

`FaradayJL` contains two levels of description:

- `legacy`: the original simplified model using scalar overlap proxies stored in `Norms` plus isotropic `χiso = χ3 / 3`
- `derived`: a derivation-consistent model using directly extracted coefficients `α`, `ζ`, `η`, and optional sideband mixing `Λ`

Both models can be run for:

- `:counter`: counter-rotating pumps
- `:coro`: co-rotating pumps

## Repository Layout

Key files:

- [src/FaradayJL.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/src/FaradayJL.jl)
  Core types, ODE right-hand sides, simulation helpers, and polarization observables.
- [examples/demo.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/demo.jl)
  Batch entry point that loads one generated case, runs all four model comparisons, and saves figures and JSON summaries.
- [examples/interactive_tcmt_explorer.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/interactive_tcmt_explorer.jl)
  Notebook-style script for interactive use in VS Code.
- [examples/tcmt_example_utils.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/tcmt_example_utils.jl)
  Case loading, pulse conversion, plotting, and convenience wrappers.
- [examples/generated/tcmt_case_sin_090326_mf.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/generated/tcmt_case_sin_090326_mf.jl)
  Example auto-generated case from the Meep parameter extraction pipeline.
- [examples/generated/tcmt_case_sin_090326_new.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/generated/tcmt_case_sin_090326_new.jl)
  Another extracted case with the same schema.

## Requirements

- Julia `1.12` is the version currently used in this checkout.
- Dependencies are listed in [Project.toml](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/Project.toml).

Current direct dependencies:

- `DifferentialEquations`
- `CairoMakie`
- `LinearAlgebra`
- `Revise`

## Installation

From the `scripts` directory:

```bash
julia --project=FaradayJL -e 'using Pkg; Pkg.instantiate()'
```

If your default `julia` launcher is misconfigured and routes through `juliaup`, use the direct binary instead:

```bash
'/Users/nikita/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia' \
  --project=FaradayJL \
  -e 'using Pkg; Pkg.instantiate()'
```

To use the package programmatically:

```julia
using FaradayJL
```

## Quick Start

### 1. Run the standard batch demo

This loads a generated case file, runs:

- legacy counter-rotating
- legacy co-rotating
- derived counter-rotating
- derived co-rotating

and saves figures plus summary JSON files.

From `scripts`:

```bash
julia --project=FaradayJL FaradayJL/examples/demo.jl
```

By default this uses:

- [tcmt_case_sin_090326_new.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/generated/tcmt_case_sin_090326_new.jl)

and writes to:

- [examples/runs/sin_090326_new](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/runs/sin_090326_new)

To run a different case and output folder:

```bash
julia --project=FaradayJL FaradayJL/examples/demo.jl \
  FaradayJL/examples/generated/tcmt_case_sin_090326_mf.jl \
  FaradayJL/examples/runs/sin_090326_mf
```

You can also supply the case path through `FARADAYJL_CASE_FILE`.

### 2. Use the interactive script

Open [interactive_tcmt_explorer.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/interactive_tcmt_explorer.jl) in VS Code and run it cell by cell.

The control section lets you change:

- `selected_case_name`
- `pump_intensity_w_cm2`
- `probe_intensity_w_cm2`
- `pulse_fwhm_intensity_fs`
- `t_window_fs`
- `saveat_fs`
- `detail_variant`
- `detail_mode`
- `swap_pump_order`

The last flag is the correct way to flip the pump ordering and the sign of the beat detuning inside the interactive workflow. It swaps pump-associated rates and coefficients consistently. Do not manually negate only `Δ1` and `Δ2`.

You can also run the full interactive script once from the terminal:

```bash
julia --project=FaradayJL FaradayJL/examples/interactive_tcmt_explorer.jl
```

## Data Flow From Meep

The Julia package expects parameter files generated by the Python extraction workflow. A generated case file, for example [tcmt_case_sin_090326_mf.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/generated/tcmt_case_sin_090326_mf.jl), contains:

- source file paths
  - geometry JSON
  - modes JSON
  - extracted parameter JSON
- material data
  - `n2`
  - `chi3_si`
  - `chi3_meep`
  - `chi_iso_meep`
- pulse defaults
- the probe normalization scale `kappa_probe_meep`
- legacy overlap proxies
- modal rates and detunings
- derivation-consistent counter-rotating coefficients
- derivation-consistent co-rotating coefficients
- optional probe output mapping
- paths to diagnostic plots produced by the extractor

This means `FaradayJL` itself does not compute cavity modes or overlap integrals. It consumes an extracted case and runs the reduced dynamical model.

## Public API

The package exports the following core types:

- `Norms`
- `ChiDirect`
- `ChiSideband`
- `Rates`
- `Drives`
- `CounterDerived`
- `CoroDerived`
- `ProbeOutput`

and the following main functions:

- `gauss`
- `chi_eff3_counter`
- `chi_eff3_coro`
- `couplings_counter`
- `couplings_coro`
- `make_parameters`
- `make_parameters_derived`
- `run_sim`
- `rotation_ellipticity`
- `stokes_from_circular`
- `output_probe_fields`
- `rotation_ellipticity_physical`

### Core Types

#### `Norms`

Legacy overlap-proxy factors used to convert scalar `χ^(3)` terms into effective frequency pulls and cascade couplings.

Fields:

- `ηs_u1_us`
- `ηs_u2_us`
- `ηp1`
- `ηp2`
- `ηΩp`
- `ηΩm`

#### `ChiDirect`

Direct `χ^(3)` coefficients for probe-frequency Kerr shifts.

Fields:

- `A1`, `B1`, `C1`
- `A2`, `B2`, `C2`

#### `ChiSideband`

Legacy sideband-generation and back-mixing coefficients.

Fields:

- generation at `Ω+`
  - `A_sb_p`, `B_sb_p`, `C_sb_p`
- generation at `Ω-`
  - `A_sb_m`, `B_sb_m`, `C_sb_m`
- back-mixing from `Ω+`
  - `A_mx_p`, `B_mx_p`, `C_mx_p`
- back-mixing from `Ω-`
  - `A_mx_m`, `B_mx_m`, `C_mx_m`

#### `Rates`

Loaded linewidths and detunings for pumps, probe, and sidebands.

Fields:

- pumps
  - `κ1`, `Δ1`
  - `κ2`, `Δ2`
- probe
  - `κs`, `Δs`
- aggregate sidebands
  - `κΩp`, `ΔΩp`
  - `κΩm`, `ΔΩm`
- sideband-resolved co-rotating fields
  - `κΩp_p`, `ΔΩp_p`
  - `κΩp_m`, `ΔΩp_m`
  - `κΩm_p`, `ΔΩm_p`
  - `κΩm_m`, `ΔΩm_m`

#### `CounterDerived`

Counter-rotating derivation-consistent coefficients.

Fields:

- direct Kerr pulls
  - `α1_plus`, `α2_plus`
  - `α1_minus`, `α2_minus`
- sideband generation
  - `ζ_plus`, `ζ_minus`
- back-mixing
  - `η_plus`, `η_minus`

#### `CoroDerived`

Co-rotating derivation-consistent coefficients.

Fields:

- direct Kerr pulls
  - `α1_plus`, `α2_plus`
  - `α1_minus`, `α2_minus`
- generation
  - `ζ_pp`, `ζ_pm`, `ζ_mp`, `ζ_mm`
- back-mixing
  - `η_pp`, `η_pm`, `η_mp`, `η_mm`
- optional linear sideband mixing
  - `ΛΩp`, `ΛΩm`

#### `ProbeOutput`

Simple output-field map used to compute physical probe observables from the simulated intracavity fields.

Fields:

- `κ_out_plus`
- `κ_out_minus`
- `c_plus`
- `c_minus`

With the current default extracted cases:

- `κ_out_plus = κ_out_minus = 1`
- `c_plus = c_minus = 0`

so the physical output rotation matches the old intracavity `a+/a-` rotation.

### Building and Running a Simulation

Minimal example:

```julia
using FaradayJL

norms = Norms(
    ηs_u1_us = 0.3 + 0im,
    ηs_u2_us = 0.3 + 0im,
    ηp1 = 0 + 0im,
    ηp2 = 0 + 0im,
    ηΩp = 0.25 + 0im,
    ηΩm = 0.25 + 0im,
)

rates = Rates(
    κ1 = 0.3, Δ1 = -0.08,
    κ2 = 0.8, Δ2 = 0.08,
    κs = 1.0, Δs = 0.6,
    κΩp = 0.4, ΔΩp = 0.01,
    κΩm = 1.0, ΔΩm = -1.3,
)

χ = 3.6e-4 + 0im
chidir = ChiDirect(A1 = χ, B1 = χ, C1 = χ, A2 = χ, B2 = χ, C2 = χ)
chisb = ChiSideband(
    A_sb_p = χ, B_sb_p = χ, C_sb_p = χ,
    A_sb_m = χ, B_sb_m = χ, C_sb_m = χ,
    A_mx_p = χ, B_mx_p = χ, C_mx_p = χ,
    A_mx_m = χ, B_mx_m = χ, C_mx_m = χ,
)

s1!(t) = gauss(t; A = 1.0, t0 = 10.0, τ = 3.0)
s2!(t) = gauss(t; A = 1.0, t0 = 10.0, τ = 3.0)
splus!(t) = gauss(t; A = 1e-2, t0 = 10.0, τ = 3.0)
sminus!(t) = gauss(t; A = 1e-2, t0 = 10.0, τ = 3.0)
drives = Drives(s1!, s2!, splus!, sminus!)

params = make_parameters(
    case = :counter,
    norms = norms,
    rates = rates,
    chidir = chidir,
    chisb = chisb,
    drives = drives,
)

t, sol = run_sim(:counter, params; T = (0.0, 40.0), saveat = 0.05)
a_plus = sol[3, :]
a_minus = sol[4, :]
theta, eps = rotation_ellipticity(a_plus, a_minus)
```

For the derivation-consistent model, use `CounterDerived` or `CoroDerived` with `make_parameters_derived`.

## Observables

Two probe observables are currently kept on purpose.

### 1. Legacy intracavity observable

`rotation_ellipticity(a_plus, a_minus)` computes:

- rotation from the phase ratio of `a_plus / a_minus`
- a legacy ellipticity-like quantity from the log amplitude ratio

This preserves the original Julia workflow and older comparisons.

### 2. Physical output observable

`output_probe_fields(...)` and `rotation_ellipticity_physical(...)` compute:

- output circular fields `E_out,+`, `E_out,-`
- physical rotation `ψ`
- ellipticity angle `χ`

These functions are the correct place to compare against transmitted probe observables from Meep. In the currently generated cases, the output map is intentionally simplified so the final rotation numerically matches the old `a+/a-` result.

## What the Example Scripts Do

### `demo.jl`

Workflow:

1. Load a generated `TCMT_CASE`
2. Run `run_demo`
3. Simulate:
   - `legacy/counter`
   - `legacy/coro`
   - `derived/counter`
   - `derived/coro`
4. Save:
   - per-run figures
   - per-run summary JSON files
   - `counter_rotation_compare.png`
   - `coro_rotation_compare.png`

Per-run figures include:

- pump amplitudes
- intracavity probe amplitudes
- sideband amplitudes
- legacy and physical rotation
- legacy and physical ellipticity
- output probe amplitudes

### `interactive_tcmt_explorer.jl`

This script is meant for manual exploration in an IDE.

Typical use cases:

- switch between generated cases
- vary pump/probe intensities
- vary pulse width
- inspect `counter` vs `coro`
- inspect `legacy` vs `derived`
- flip pump ordering with `swap_pump_order`
- optionally save the current run

### `tcmt_example_utils.jl`

This file contains the practical glue around the package:

- loading generated case files
- converting optical intensity to Meep-style field amplitude
- building Gaussian drives
- converting Meep-normalized rates into Julia `Rates`
- building `legacy` or `derived` parameter packs
- collecting results
- writing summary JSON files
- generating Makie figures

## Generated Case Schema

Each generated case defines `const TCMT_CASE = (...)`.

Important top-level keys:

- `name`
- `source`
- `material`
- `pulse`
- `kappa_probe_meep`
- `legacy`
- `rates`
- `derived`
- `plots`

The examples rely on this schema. If you generate a new case file from Python, keep the same field names unless you also update the Julia helpers.

## Units and Normalization

The current implementation mixes physical inputs and normalized TCMT variables in a deliberate way:

- user-facing pulse intensities are specified in `W/cm^2`
- `intensity_to_meep_amplitude` converts them to Meep electric-field amplitude
- amplitudes are then normalized by `sqrt(kappa_probe_meep)`
- simulation time in the ODE uses probe-linewidth normalization
- helper code converts back to femtoseconds for plotting

Important constants used in the helpers:

- `EPS0`
- `C0`
- `SCALE_E`
- `FS_PER_MEEP`

The generated cases store the probe normalization factor as `kappa_probe_meep`.

## Counter-Rotating vs Co-Rotating Models

### Counter-rotating

State vector:

- `p1`
- `p2`
- `a_plus`
- `a_minus`
- `b_plus`
- `b_minus`

This is the simpler and physically relevant configuration for the main intended Faraday-rotation process in the current project.

### Co-rotating

State vector:

- `p1`
- `p2`
- `a_plus`
- `a_minus`
- `b_p_p`
- `b_p_m`
- `b_m_p`
- `b_m_m`

The derivation-consistent version also allows:

- `ΛΩp`
- `ΛΩm`

to represent sideband linear mixing inside the `Ω+` and `Ω-` blocks.

## Legacy vs Derived Model

### Legacy

Pros:

- simple
- compact
- backward-compatible with the older Julia implementation

Limitations:

- uses scalar overlap proxies instead of the derivation’s mode-overlap coefficients
- uses isotropic `χiso` everywhere
- can overestimate rotation substantially for some geometries

### Derived

Pros:

- uses extracted coefficients matched to the derivation
- supports physical output observables
- supports optional sideband circular mixing in the co-rotating case

Limitations:

- still inherits simplifications from the current extracted-case workflow
- still uses the reduced one-port-like drive normalization in the current Julia examples

## Current Assumptions and Simplifications

The current package intentionally keeps several simplifications because they are useful for comparison to older runs:

- weak-probe assumption
- loaded linewidth drives `sqrt(κ)` in the example workflow
- no explicit `κ_int` vs `κ_ext` decomposition in the ODE driving terms
- simple output map with `κ_out,+ = κ_out,- = 1` and zero direct term in current generated cases
- equal-time Gaussian pump and probe pulse envelopes in the example helpers

These are project choices, not general TCMT requirements.

## Working With New Extracted Cases

The typical workflow is:

1. Run the Meep extraction pipeline to produce a new `tcmt_extracted_params_derivation.json`
2. Generate a Julia case file `tcmt_case_<name>.jl`
3. Place it under `FaradayJL/examples/generated/`
4. Load it with `demo.jl` or `interactive_tcmt_explorer.jl`

If you add new fields to the generated case, update:

- [tcmt_example_utils.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/tcmt_example_utils.jl)
- [interactive_tcmt_explorer.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/interactive_tcmt_explorer.jl)
- [demo.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/demo.jl), if needed

## Outputs

The example scripts write:

- `legacy_counter.png`
- `legacy_coro.png`
- `derived_counter.png`
- `derived_coro.png`
- `legacy_counter_summary.json`
- `legacy_coro_summary.json`
- `derived_counter_summary.json`
- `derived_coro_summary.json`
- `counter_rotation_compare.png`
- `coro_rotation_compare.png`

Summary JSON files currently include:

- final legacy rotation in degrees
- final physical rotation in degrees
- peak absolute legacy rotation in degrees
- peak absolute physical rotation in degrees

## Limitations

- `docs/` and `tests/` are currently placeholders; there is no separate Julia doc build or automated Julia test suite in this checkout.
- The package is tightly coupled to the specific extracted-case schema used by this project.
- The output-field model is present but still intentionally simplified in the current generated cases.
- The package does not perform mode solving or overlap extraction on its own.

## Troubleshooting

### Julia does not start correctly

Use the direct Julia binary if `julia` routes through a broken `juliaup` installation:

```bash
'/Users/nikita/.julia/juliaup/julia-1.12.4+0.aarch64.apple.darwin14/bin/julia' \
  --project=FaradayJL \
  FaradayJL/examples/demo.jl
```

### A generated case file does not load

Check:

- the path exists
- it defines `const TCMT_CASE = (...)`
- required keys such as `pulse`, `rates`, `legacy`, and `derived` are present

### I want to flip the pump ordering

In [interactive_tcmt_explorer.jl](/Users/nikita/Google%20Drive/Work/In%20process/Projects/!Miniprojects/Faraday_rotation_Optica_2019/scripts/FaradayJL/examples/interactive_tcmt_explorer.jl), set:

```julia
swap_pump_order = true
```

This swaps pump-associated rates and coefficients consistently. It is better than manually changing detuning signs only.

## Status

`FaradayJL` is currently best viewed as a focused project package for:

- reproducing extracted TCMT runs
- comparing the old and derivation-consistent models
- interactively inspecting how linewidths, detunings, overlaps, and pulse settings affect predicted rotation