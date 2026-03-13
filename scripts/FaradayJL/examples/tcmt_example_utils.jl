using CairoMakie
using FaradayJL

const EPS0 = 8.854187817e-12
const C0 = 299792458.0
const SCALE_E = 1.0 / (1e-6 * EPS0 * C0)
const FS_PER_MEEP = (1e-6 / C0) * 1e15

intensity_to_meep_amplitude(intensity_w_cm2; n_lin=1.0) = begin
    intensity_si = intensity_w_cm2 * 1e4
    e_si = sqrt(2.0 * intensity_si / (n_lin * EPS0 * C0))
    e_si / SCALE_E
end

function load_tcmt_case(case_file::AbstractString)
    module_name = gensym(:TCMTCaseModule)
    mod = Module(module_name)
    Base.include(mod, abspath(case_file))
    return Base.invokelatest(getproperty, mod, :TCMT_CASE)
end

function case_with_pulse(
    case_data;
    pump_intensity_w_cm2=case_data.pulse.pump_intensity_w_cm2,
    probe_intensity_w_cm2=case_data.pulse.probe_intensity_w_cm2,
    pulse_fwhm_intensity_fs=case_data.pulse.pulse_fwhm_intensity_fs,
)
    pulse = merge(
        case_data.pulse,
        (
            pump_intensity_w_cm2 = float(pump_intensity_w_cm2),
            probe_intensity_w_cm2 = float(probe_intensity_w_cm2),
            pulse_fwhm_intensity_fs = float(pulse_fwhm_intensity_fs),
        ),
    )
    return merge(case_data, (pulse = pulse,))
end

function case_with_swapped_pumps(case_data)
    rates = merge(
        case_data.rates,
        (
            pump1 = case_data.rates.pump2,
            pump2 = case_data.rates.pump1,
            sb_plus = case_data.rates.sb_minus,
            sb_minus = case_data.rates.sb_plus,
        ),
    )

    legacy_norms = merge(
        case_data.legacy.norms,
        (
            eta_s_u1_us = case_data.legacy.norms.eta_s_u2_us,
            eta_s_u2_us = case_data.legacy.norms.eta_s_u1_us,
            eta_Omega_p = case_data.legacy.norms.eta_Omega_m,
            eta_Omega_m = case_data.legacy.norms.eta_Omega_p,
        ),
    )
    legacy = merge(case_data.legacy, (norms = legacy_norms,))

    counter = merge(
        case_data.derived.counter,
        (
            alpha1_plus = case_data.derived.counter.alpha2_plus,
            alpha2_plus = case_data.derived.counter.alpha1_plus,
            alpha1_minus = case_data.derived.counter.alpha2_minus,
            alpha2_minus = case_data.derived.counter.alpha1_minus,
            zeta_plus = case_data.derived.counter.zeta_minus,
            zeta_minus = case_data.derived.counter.zeta_plus,
            eta_plus = case_data.derived.counter.eta_minus,
            eta_minus = case_data.derived.counter.eta_plus,
        ),
    )

    coro = merge(
        case_data.derived.coro,
        (
            alpha1_plus = case_data.derived.coro.alpha2_plus,
            alpha2_plus = case_data.derived.coro.alpha1_plus,
            alpha1_minus = case_data.derived.coro.alpha2_minus,
            alpha2_minus = case_data.derived.coro.alpha1_minus,
            zeta_pp = case_data.derived.coro.zeta_mp,
            zeta_pm = case_data.derived.coro.zeta_mm,
            zeta_mp = case_data.derived.coro.zeta_pp,
            zeta_mm = case_data.derived.coro.zeta_pm,
            eta_pp = case_data.derived.coro.eta_mp,
            eta_pm = case_data.derived.coro.eta_mm,
            eta_mp = case_data.derived.coro.eta_pp,
            eta_mm = case_data.derived.coro.eta_pm,
            Lambda_Omegap = case_data.derived.coro.Lambda_Omegam,
            Lambda_Omegam = case_data.derived.coro.Lambda_Omegap,
        ),
    )

    derived = merge(case_data.derived, (counter = counter, coro = coro))
    return merge(case_data, (rates = rates, legacy = legacy, derived = derived))
end

function rates_counter(case_data)
    p1 = case_data.rates.pump1
    p2 = case_data.rates.pump2
    ps = case_data.rates.probe
    sbp = case_data.rates.sb_plus
    sbm = case_data.rates.sb_minus
    return FaradayJL.Rates(
        κ1 = p1.kappa_loaded,
        Δ1 = p1.detune,
        κ2 = p2.kappa_loaded,
        Δ2 = p2.detune,
        κs = ps.kappa_loaded,
        Δs = ps.detune,
        κΩp = sbp.kappa_loaded,
        ΔΩp = sbp.detune,
        κΩm = sbm.kappa_loaded,
        ΔΩm = sbm.detune,
    )
end

function rates_coro(case_data)
    p1 = case_data.rates.pump1
    p2 = case_data.rates.pump2
    ps = case_data.rates.probe
    sbp = case_data.rates.sb_plus
    sbm = case_data.rates.sb_minus
    return FaradayJL.Rates(
        κ1 = p1.kappa_loaded,
        Δ1 = p1.detune,
        κ2 = p2.kappa_loaded,
        Δ2 = p2.detune,
        κs = ps.kappa_loaded,
        Δs = ps.detune,
        κΩp = sbp.kappa_loaded,
        ΔΩp = sbp.detune,
        κΩm = sbm.kappa_loaded,
        ΔΩm = sbm.detune,
        κΩp_p = sbp.kappa_loaded,
        ΔΩp_p = sbp.detune,
        κΩp_m = sbp.kappa_loaded,
        ΔΩp_m = sbp.detune,
        κΩm_p = sbm.kappa_loaded,
        ΔΩm_p = sbm.detune,
        κΩm_m = sbm.kappa_loaded,
        ΔΩm_m = sbm.detune,
    )
end

function build_drives(case_data)
    kappa_s_meep = case_data.kappa_probe_meep
    fs_per_tcmt = FS_PER_MEEP / kappa_s_meep

    pump_amp_meep = intensity_to_meep_amplitude(case_data.pulse.pump_intensity_w_cm2; n_lin=1.0)
    probe_amp_linear_meep = intensity_to_meep_amplitude(case_data.pulse.probe_intensity_w_cm2; n_lin=1.0)
    probe_amp_circ_meep = probe_amp_linear_meep / sqrt(2.0)

    pump_amp = pump_amp_meep / sqrt(kappa_s_meep)
    probe_amp_circ = probe_amp_circ_meep / sqrt(kappa_s_meep)

    tau_fs = case_data.pulse.pulse_fwhm_intensity_fs / (2.0 * log(2.0))
    tau_tcmt = tau_fs / fs_per_tcmt
    t0_fs = 4.0 * tau_fs
    t0_tcmt = t0_fs / fs_per_tcmt

    s1!(t) = complex(FaradayJL.gauss(t; A = pump_amp, t0 = t0_tcmt, τ = tau_tcmt))
    s2!(t) = complex(FaradayJL.gauss(t; A = pump_amp, t0 = t0_tcmt, τ = tau_tcmt))
    splus!(t) = complex(FaradayJL.gauss(t; A = probe_amp_circ, t0 = t0_tcmt, τ = tau_tcmt))
    sminus!(t) = complex(FaradayJL.gauss(t; A = probe_amp_circ, t0 = t0_tcmt, τ = tau_tcmt))

    return (
        drives = FaradayJL.Drives(s1!, s2!, splus!, sminus!),
        fs_per_tcmt = fs_per_tcmt,
        tau_fs = tau_fs,
        tau_tcmt = tau_tcmt,
        t0_fs = t0_fs,
        t0_tcmt = t0_tcmt,
        pump_amp = pump_amp,
        probe_amp_circ = probe_amp_circ,
    )
end

function legacy_parameters(case_data, mode::Symbol, drives)
    χiso = complex(case_data.material.chi_iso_meep)
    norms = FaradayJL.Norms(
        ηs_u1_us = case_data.legacy.norms.eta_s_u1_us,
        ηs_u2_us = case_data.legacy.norms.eta_s_u2_us,
        ηΩp = case_data.legacy.norms.eta_Omega_p,
        ηΩm = case_data.legacy.norms.eta_Omega_m,
        ηp1 = 0.0 + 0.0im,
        ηp2 = 0.0 + 0.0im,
    )
    chidir = FaradayJL.ChiDirect(
        A1 = χiso,
        B1 = χiso,
        C1 = χiso,
        A2 = χiso,
        B2 = χiso,
        C2 = χiso,
    )
    chisb = FaradayJL.ChiSideband(
        A_sb_p = χiso,
        B_sb_p = χiso,
        C_sb_p = χiso,
        A_sb_m = χiso,
        B_sb_m = χiso,
        C_sb_m = χiso,
        A_mx_p = χiso,
        B_mx_p = χiso,
        C_mx_p = χiso,
        A_mx_m = χiso,
        B_mx_m = χiso,
        C_mx_m = χiso,
    )
    rates = mode == :counter ? rates_counter(case_data) : rates_coro(case_data)
    return FaradayJL.make_parameters(
        case = mode,
        norms = norms,
        rates = rates,
        chidir = chidir,
        chisb = chisb,
        drives = drives,
    )
end

function derived_parameters(case_data, mode::Symbol, drives)
    rates = mode == :counter ? rates_counter(case_data) : rates_coro(case_data)
    output = FaradayJL.ProbeOutput(
        κ_out_plus = case_data.derived.output.kappa_out_plus,
        κ_out_minus = case_data.derived.output.kappa_out_minus,
        c_plus = case_data.derived.output.c_plus,
        c_minus = case_data.derived.output.c_minus,
    )
    if mode == :counter
        derived = FaradayJL.CounterDerived(
            α1_plus = case_data.derived.counter.alpha1_plus,
            α2_plus = case_data.derived.counter.alpha2_plus,
            α1_minus = case_data.derived.counter.alpha1_minus,
            α2_minus = case_data.derived.counter.alpha2_minus,
            ζ_plus = case_data.derived.counter.zeta_plus,
            ζ_minus = case_data.derived.counter.zeta_minus,
            η_plus = case_data.derived.counter.eta_plus,
            η_minus = case_data.derived.counter.eta_minus,
        )
    else
        derived = FaradayJL.CoroDerived(
            α1_plus = case_data.derived.coro.alpha1_plus,
            α2_plus = case_data.derived.coro.alpha2_plus,
            α1_minus = case_data.derived.coro.alpha1_minus,
            α2_minus = case_data.derived.coro.alpha2_minus,
            ζ_pp = case_data.derived.coro.zeta_pp,
            ζ_pm = case_data.derived.coro.zeta_pm,
            ζ_mp = case_data.derived.coro.zeta_mp,
            ζ_mm = case_data.derived.coro.zeta_mm,
            η_pp = case_data.derived.coro.eta_pp,
            η_pm = case_data.derived.coro.eta_pm,
            η_mp = case_data.derived.coro.eta_mp,
            η_mm = case_data.derived.coro.eta_mm,
            ΛΩp = case_data.derived.coro.Lambda_Omegap,
            ΛΩm = case_data.derived.coro.Lambda_Omegam,
        )
    end
    return FaradayJL.make_parameters_derived(
        case = mode,
        rates = rates,
        derived = derived,
        drives = drives,
        output = output,
    )
end

function sample_inputs(drives, t)
    splus = ComplexF64[drives.splus!(tt) for tt in t]
    sminus = ComplexF64[drives.sminus!(tt) for tt in t]
    return splus, sminus
end

function build_result(case_data, variant::Symbol, mode::Symbol, t, sol, drive_data, output)
    t_fs = t .* drive_data.fs_per_tcmt
    splus, sminus = sample_inputs(drive_data.drives, t)

    if mode == :counter
        p1, p2, a_plus, a_minus, b1, b2 = sol[1, :], sol[2, :], sol[3, :], sol[4, :], sol[5, :], sol[6, :]
        sidebands = (
            labels = ("|b(Ω+, +)|", "|b(Ω−, −)|"),
            series = (b1, b2),
        )
    else
        p1, p2, a_plus, a_minus = sol[1, :], sol[2, :], sol[3, :], sol[4, :]
        sidebands = (
            labels = ("|b(Ω+, +)|", "|b(Ω+, −)|", "|b(Ω−, +)|", "|b(Ω−, −)|"),
            series = (sol[5, :], sol[6, :], sol[7, :], sol[8, :]),
        )
    end

    theta_legacy, eps_legacy = FaradayJL.rotation_ellipticity(a_plus, a_minus)
    eout_plus, eout_minus = FaradayJL.output_probe_fields(
        a_plus,
        a_minus;
        output = output,
        splus_in = splus,
        sminus_in = sminus,
    )
    theta_phys, chi_phys = FaradayJL.rotation_ellipticity_physical(eout_plus, eout_minus)

    return (
        variant = variant,
        mode = mode,
        t = t,
        t_fs = t_fs,
        pump1 = p1,
        pump2 = p2,
        a_plus = a_plus,
        a_minus = a_minus,
        sidebands = sidebands,
        theta_legacy = theta_legacy,
        eps_legacy = eps_legacy,
        eout_plus = eout_plus,
        eout_minus = eout_minus,
        theta_phys = theta_phys,
        chi_phys = chi_phys,
        summary = (
            legacy_final_deg = rad2deg(theta_legacy[end]),
            physical_final_deg = rad2deg(theta_phys[end]),
            legacy_peak_abs_deg = maximum(abs.(rad2deg.(theta_legacy))),
            physical_peak_abs_deg = maximum(abs.(rad2deg.(theta_phys))),
        ),
    )
end

function simulate_variant(case_data, variant::Symbol, mode::Symbol; t_fs=(0.0, 3000.0), saveat_fs=2.5)
    drive_data = build_drives(case_data)
    params = if variant == :legacy
        legacy_parameters(case_data, mode, drive_data.drives)
    elseif variant == :derived
        derived_parameters(case_data, mode, drive_data.drives)
    else
        error("unknown variant")
    end

    t_tcmt = (t_fs[1] / drive_data.fs_per_tcmt, t_fs[2] / drive_data.fs_per_tcmt)
    saveat_tcmt = saveat_fs / drive_data.fs_per_tcmt
    _, sol = FaradayJL.run_sim(mode, params; T = t_tcmt, saveat = saveat_tcmt)
    output = haskey(params, :output) ? params.output : FaradayJL.ProbeOutput()
    return build_result(case_data, variant, mode, sol.t, sol, drive_data, output)
end

function write_summary_json(path, result)
    content = string(
        "{\n",
        "  \"variant\": \"", String(result.variant), "\",\n",
        "  \"mode\": \"", String(result.mode), "\",\n",
        "  \"legacy_final_deg\": ", repr(result.summary.legacy_final_deg), ",\n",
        "  \"physical_final_deg\": ", repr(result.summary.physical_final_deg), ",\n",
        "  \"legacy_peak_abs_deg\": ", repr(result.summary.legacy_peak_abs_deg), ",\n",
        "  \"physical_peak_abs_deg\": ", repr(result.summary.physical_peak_abs_deg), "\n",
        "}\n",
    )
    open(path, "w") do io
        write(io, content)
    end
end

function build_result_figure(result)
    fig = Figure(size = (1200, result.mode == :counter ? 1100 : 1300), fontsize = 13)

    ax11 = Axis(fig[1, 1], xlabel = "t (fs)", ylabel = "|pump|", title = "$(result.variant): pumps")
    lines!(ax11, result.t_fs, abs.(result.pump1), label = "|p1|")
    lines!(ax11, result.t_fs, abs.(result.pump2), label = "|p2|")
    axislegend(ax11, position = :rb)

    ax12 = Axis(fig[1, 2], xlabel = "t (fs)", ylabel = "|probe|", title = "Probe intracavity")
    lines!(ax12, result.t_fs, abs.(result.a_plus), label = "|a+|")
    lines!(ax12, result.t_fs, abs.(result.a_minus), label = "|a-|")
    axislegend(ax12, position = :rb)

    ax21 = Axis(fig[2, 1], xlabel = "t (fs)", ylabel = "|sb|", title = "Sidebands")
    for (label, series) in zip(result.sidebands.labels, result.sidebands.series)
        lines!(ax21, result.t_fs, abs.(series), label = label)
    end
    axislegend(ax21, position = :rb)

    ax22 = Axis(fig[2, 2], xlabel = "t (fs)", ylabel = "angle (deg)", title = "Rotation")
    lines!(ax22, result.t_fs, rad2deg.(result.theta_legacy), label = "legacy a+/a-")
    lines!(ax22, result.t_fs, rad2deg.(result.theta_phys), label = "physical output")
    axislegend(ax22, position = :rb)

    ax31 = Axis(fig[3, 1], xlabel = "t (fs)", ylabel = "ellipticity", title = "Ellipticity")
    lines!(ax31, result.t_fs, result.eps_legacy, label = "legacy log-ratio")
    lines!(ax31, result.t_fs, rad2deg.(result.chi_phys), label = "physical chi (deg)")
    axislegend(ax31, position = :rb)

    ax32 = Axis(fig[3, 2], xlabel = "t (fs)", ylabel = "|E_out|", title = "Probe output")
    lines!(ax32, result.t_fs, abs.(result.eout_plus), label = "|E+ out|")
    lines!(ax32, result.t_fs, abs.(result.eout_minus), label = "|E- out|")
    axislegend(ax32, position = :rb)

    return fig
end

function save_result_figure(path, result)
    fig = build_result_figure(result)
    save(path, fig)
    return path
end

function build_rotation_comparison_figure(results, mode::Symbol)
    fig = Figure(size = (1000, 520), fontsize = 13)
    ax1 = Axis(fig[1, 1], xlabel = "t (fs)", ylabel = "rotation (deg)", title = "$(mode): legacy vs derived")
    for result in results
        lines!(ax1, result.t_fs, rad2deg.(result.theta_phys), label = "$(result.variant) physical")
        lines!(ax1, result.t_fs, rad2deg.(result.theta_legacy), linestyle = :dash, label = "$(result.variant) legacy")
    end
    axislegend(ax1, position = :rb)
    return fig
end

function save_rotation_comparison(path, results, mode::Symbol)
    fig = build_rotation_comparison_figure(results, mode)
    save(path, fig)
    return path
end

function run_demo(case_data; output_dir, t_fs=(0.0, 3000.0), saveat_fs=2.5)
    mkpath(output_dir)
    results = Dict{Symbol, Dict{Symbol, Any}}()
    for variant in (:legacy, :derived)
        results[variant] = Dict{Symbol, Any}()
        for mode in (:counter, :coro)
            result = simulate_variant(case_data, variant, mode; t_fs = t_fs, saveat_fs = saveat_fs)
            results[variant][mode] = result
            fig_path = joinpath(output_dir, "$(variant)_$(mode).png")
            save_result_figure(fig_path, result)
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
    return results
end
