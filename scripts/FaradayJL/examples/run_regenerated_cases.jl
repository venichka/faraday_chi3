# Run the regenerated SiN/SiC TCMT cases through FaradayJL with the FIXED
# counter-rotating ODE (rhs_counter_derived!). No CairoMakie (fast).
# Run:  julia --project=. examples/run_regenerated_cases.jl
using FaradayJL
using DifferentialEquations
using Printf

const EPS0, C0 = 8.854187817e-12, 299792458.0
const SCALE_E = 1.0 / (1e-6 * EPS0 * C0)
const FS_PER_MEEP = (1e-6 / C0) * 1e15
i2a(I; n_lin=1.0) = sqrt(2.0 * I * 1e4 / (n_lin * EPS0 * C0)) / SCALE_E

function load_case(path)
    m = Module(gensym(:Case)); Base.include(m, abspath(path))
    return Base.invokelatest(getproperty, m, :TCMT_CASE)
end

rates_counter(c) = FaradayJL.Rates(
    κ1=c.rates.pump1.kappa_loaded, Δ1=c.rates.pump1.detune,
    κ2=c.rates.pump2.kappa_loaded, Δ2=c.rates.pump2.detune,
    κs=c.rates.probe.kappa_loaded, Δs=c.rates.probe.detune,
    κΩp=c.rates.sb_plus.kappa_loaded, ΔΩp=c.rates.sb_plus.detune,
    κΩm=c.rates.sb_minus.kappa_loaded, ΔΩm=c.rates.sb_minus.detune)

function build_drives(c)
    κs = c.kappa_probe_meep; fs_per = FS_PER_MEEP / κs
    pamp = i2a(c.pulse.pump_intensity_w_cm2) / sqrt(κs)
    samp = (i2a(c.pulse.probe_intensity_w_cm2) / sqrt(2.0)) / sqrt(κs)
    τfs = c.pulse.pulse_fwhm_intensity_fs / (2.0*log(2.0)); τ = τfs/fs_per; t0 = 4*τfs/fs_per
    s1(t)=complex(FaradayJL.gauss(t;A=pamp,t0=t0,τ=τ)); s2(t)=complex(FaradayJL.gauss(t;A=pamp,t0=t0,τ=τ))
    sp(t)=complex(FaradayJL.gauss(t;A=samp,t0=t0,τ=τ)); sm(t)=complex(FaradayJL.gauss(t;A=samp,t0=t0,τ=τ))
    return FaradayJL.Drives(s1,s2,sp,sm), fs_per
end

counter_derived(c) = FaradayJL.CounterDerived(
    α1_plus=c.derived.counter.alpha1_plus, α2_plus=c.derived.counter.alpha2_plus,
    α1_minus=c.derived.counter.alpha1_minus, α2_minus=c.derived.counter.alpha2_minus,
    ζ_plus=c.derived.counter.zeta_plus, ζ_minus=c.derived.counter.zeta_minus,
    η_plus=c.derived.counter.eta_plus, η_minus=c.derived.counter.eta_minus)

function run_case(path)
    c = load_case(path)
    drives, fs_per = build_drives(c)
    params = FaradayJL.make_parameters_derived(case=:counter, rates=rates_counter(c),
                derived=counter_derived(c), drives=drives, output=FaradayJL.ProbeOutput())
    T = (0.0, 3000.0/fs_per); saveat = 2.5/fs_per
    _, sol = FaradayJL.run_sim(:counter, params; T=T, saveat=saveat)
    ap, am = sol[3,:], sol[4,:]
    θ = rad2deg.(0.5 .* angle.(ap ./ (am .+ 1e-30)))
    s0 = abs2.(ap) .+ abs2.(am)
    χ = rad2deg.(0.5 .* asin.(clamp.((abs2.(ap).-abs2.(am))./(s0 .+ 1e-30), -1, 1)))
    return (name=c.name, θf=θ[end], θpk=maximum(abs.(θ)), χpk=maximum(abs.(χ)),
            I=c.pulse.pump_intensity_w_cm2, n2=c.material.n2_m2_per_w,
            kappa_s=c.kappa_probe_meep)
end

cases = [
    joinpath(@__DIR__, "generated", "tcmt_case_sin_best_absolute.jl"),
    joinpath(@__DIR__, "generated", "tcmt_case_sic_L3p2um.jl"),
]
println("="^78)
println("Regenerated cases through FIXED counter-rotating ODE (rhs_counter_derived!)")
println("="^78)
@printf("%-22s %10s %10s %10s %10s %10s\n", "case", "θ_final°", "θ_peak°", "χ_peak°", "pumpI", "n2")
for p in cases
    if !isfile(p); @printf("%-22s   (missing: %s)\n", basename(p), basename(p)); continue; end
    r = run_case(p)
    @printf("%-22s %10.4f %10.4f %10.4f %10.1e %10.1e\n", r.name, r.θf, r.θpk, r.χpk, r.I, r.n2)
end
println("="^78)
