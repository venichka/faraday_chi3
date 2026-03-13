using CairoMakie
using FaradayJL

include(joinpath(@__DIR__, "tcmt_example_utils.jl"))

## Available cases
generated_cases = Dict(
    "sin_090326_mf" => joinpath(@__DIR__, "generated", "tcmt_case_sin_090326_mf.jl"),
    "sin_090326_new" => joinpath(@__DIR__, "generated", "tcmt_case_sin_090326_new.jl"),
)

selected_case_name = "sin_090326_mf"
case_file = abspath(generated_cases[selected_case_name])
isfile(case_file) || error("Case file not found: $case_file")

base_case = load_tcmt_case(case_file)

## Pulse controls
pump_intensity_w_cm2 = base_case.pulse.pump_intensity_w_cm2
probe_intensity_w_cm2 = base_case.pulse.probe_intensity_w_cm2
pulse_fwhm_intensity_fs = base_case.pulse.pulse_fwhm_intensity_fs

## Simulation controls
t_window_fs = (0.0, 3000.0)
saveat_fs = 2.5
swap_pump_order = false

## Detail figure controls
detail_variant = :derived
detail_mode = :counter

## Load the selected case with pulse overrides
case_data = case_with_pulse(
    base_case;
    pump_intensity_w_cm2 = pump_intensity_w_cm2,
    probe_intensity_w_cm2 = probe_intensity_w_cm2,
    pulse_fwhm_intensity_fs = pulse_fwhm_intensity_fs,
)
case_data = swap_pump_order ? case_with_swapped_pumps(case_data) : case_data

case_overview = (
    name = case_data.name,
    case_file = case_file,
    source_json = case_data.source.extracted_json,
    swap_pump_order = swap_pump_order,
    pulse = case_data.pulse,
    rates = [
        (
            mode = String(name),
            kappa_loaded = data.kappa_loaded,
            detune = data.detune,
        )
        for (name, data) in pairs(case_data.rates)
    ],
)

case_overview

## Run all four comparisons
variants = (:legacy, :derived)
modes = (:counter, :coro)

results = Dict{Symbol, Dict{Symbol, Any}}()
for variant in variants
    per_variant = Dict{Symbol, Any}()
    for mode in modes
        per_variant[mode] = simulate_variant(
            case_data,
            variant,
            mode;
            t_fs = t_window_fs,
            saveat_fs = saveat_fs,
        )
    end
    results[variant] = per_variant
end

results

## Summary table
summary_rows = [
    (
        variant = String(variant),
        mode = String(mode),
        final_deg = results[variant][mode].summary.physical_final_deg,
        peak_deg = results[variant][mode].summary.physical_peak_abs_deg,
    )
    for variant in variants for mode in modes
]

summary_rows

## Counter-rotating comparison
counter_compare_fig = build_rotation_comparison_figure(
    [results[:legacy][:counter], results[:derived][:counter]],
    :counter,
)

counter_compare_fig

## Co-rotating comparison
coro_compare_fig = build_rotation_comparison_figure(
    [results[:legacy][:coro], results[:derived][:coro]],
    :coro,
)

coro_compare_fig

## Detailed figure for one selected result
detail_result = results[detail_variant][detail_mode]
detail_fig = build_result_figure(detail_result)

detail_fig

## Optional save block
write_outputs = false
output_tag = swap_pump_order ? "$(case_data.name)_swapdelta_interactive" : "$(case_data.name)_interactive"
output_dir = joinpath(@__DIR__, "runs", output_tag)

if write_outputs
    mkpath(output_dir)
    for variant in variants
        for mode in modes
            result = results[variant][mode]
            save_result_figure(joinpath(output_dir, "$(variant)_$(mode).png"), result)
            write_summary_json(joinpath(output_dir, "$(variant)_$(mode)_summary.json"), result)
        end
    end
    save_rotation_comparison(
        joinpath(output_dir, "counter_rotation_compare.png"),
        [results[:legacy][:counter], results[:derived][:counter]],
        :counter,
    )
    save_rotation_comparison(
        joinpath(output_dir, "coro_rotation_compare.png"),
        [results[:legacy][:coro], results[:derived][:coro]],
        :coro,
    )
end

output_dir
