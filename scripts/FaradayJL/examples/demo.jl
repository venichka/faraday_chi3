using CairoMakie
using FaradayJL

include(joinpath(@__DIR__, "tcmt_example_utils.jl"))

function case_file_from_args()
    if !isempty(ARGS)
        return abspath(ARGS[1])
    end
    return get(
        ENV,
        "FARADAYJL_CASE_FILE",
        abspath(joinpath(@__DIR__, "generated", "tcmt_case_sin_090326_new.jl")),
    )
end

function output_dir_from_args(case_data)
    if length(ARGS) >= 2
        return abspath(ARGS[2])
    end
    return abspath(joinpath(@__DIR__, "runs", String(case_data.name)))
end

case_file = case_file_from_args()
isfile(case_file) || error("Case file not found: $case_file")
include(case_file)

output_dir = output_dir_from_args(TCMT_CASE)
results = run_demo(TCMT_CASE; output_dir = output_dir)

println("Saved figures to $output_dir")
for variant in (:legacy, :derived)
    for mode in (:counter, :coro)
        result = results[variant][mode]
        println(
            string(
                variant,
                " ",
                mode,
                ": final rotation legacy=",
                round(result.summary.legacy_final_deg; digits = 6),
                " deg, physical=",
                round(result.summary.physical_final_deg; digits = 6),
                " deg",
            ),
        )
    end
end
