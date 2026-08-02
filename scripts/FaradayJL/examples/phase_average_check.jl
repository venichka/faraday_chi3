# Does the TCMT rotation survive carrier-phase averaging?
#
# The FDTD study found that the headline rotations (1D 0.137 deg, 3D 1.991 deg) are the
# MAXIMUM of an interference fringe in the pump1 carrier phase, and that the rectified,
# phase-averaged chi5 Faraday rotation is ~40x smaller. The TCMT runner drives both pumps
# with REAL Gaussians, i.e. one fixed relative carrier phase, and reads theta at a single
# instant -- so its 1.17 deg may be the same kind of number.
#
# Here pump1's drive is multiplied by exp(i*phi) (in the rotating frame a carrier delay tau
# is exactly a constant phase phi = w1*tau on the slowly-varying drive) and phi is swept over
# one full cycle. Averaging is done on the STOKES vector, as in the FDTD analysis.
using FaradayJL, DifferentialEquations, Printf

const EPS0, C0 = 8.854187817e-12, 299792458.0
const SCALE_E = 1.0 / (1e-6 * EPS0 * C0)
const FS_PER_MEEP = (1e-6 / C0) * 1e15
i2a(I; n_lin=1.0) = sqrt(2.0 * I * 1e4 / (n_lin * EPS0 * C0)) / SCALE_E

load_case(p) = (m = Module(gensym(:Case)); Base.include(m, abspath(p));
                Base.invokelatest(getproperty, m, :TCMT_CASE))

rates_counter(c) = FaradayJL.Rates(
    κ1=c.rates.pump1.kappa_loaded, Δ1=c.rates.pump1.detune,
    κ2=c.rates.pump2.kappa_loaded, Δ2=c.rates.pump2.detune,
    κs=c.rates.probe.kappa_loaded, Δs=c.rates.probe.detune,
    κΩp=c.rates.sb_plus.kappa_loaded, ΔΩp=c.rates.sb_plus.detune,
    κΩm=c.rates.sb_minus.kappa_loaded, ΔΩm=c.rates.sb_minus.detune)

counter_derived(c) = FaradayJL.CounterDerived(
    α1_plus=c.derived.counter.alpha1_plus, α2_plus=c.derived.counter.alpha2_plus,
    α1_minus=c.derived.counter.alpha1_minus, α2_minus=c.derived.counter.alpha2_minus,
    ζ_plus=c.derived.counter.zeta_plus, ζ_minus=c.derived.counter.zeta_minus,
    η_plus=c.derived.counter.eta_plus, η_minus=c.derived.counter.eta_minus)

function run_phase(c, φ)
    κs = c.kappa_probe_meep; fs_per = FS_PER_MEEP / κs
    pamp = i2a(c.pulse.pump_intensity_w_cm2) / sqrt(κs)
    samp = (i2a(c.pulse.probe_intensity_w_cm2) / sqrt(2.0)) / sqrt(κs)
    τfs = c.pulse.pulse_fwhm_intensity_fs / (2.0*log(2.0)); τ = τfs/fs_per; t0 = 4*τfs/fs_per
    s1(t) = cis(φ) * complex(FaradayJL.gauss(t; A=pamp, t0=t0, τ=τ))   # <-- carrier phase on pump1
    s2(t) = complex(FaradayJL.gauss(t; A=pamp, t0=t0, τ=τ))
    sp(t) = complex(FaradayJL.gauss(t; A=samp, t0=t0, τ=τ))
    sm(t) = complex(FaradayJL.gauss(t; A=samp, t0=t0, τ=τ))
    params = FaradayJL.make_parameters_derived(case=:counter, rates=rates_counter(c),
                derived=counter_derived(c), drives=FaradayJL.Drives(s1,s2,sp,sm),
                output=FaradayJL.ProbeOutput())
    T = (0.0, 3000.0/fs_per); saveat = 2.5/fs_per
    _, sol = FaradayJL.run_sim(:counter, params; T=T, saveat=saveat)
    ap, am = sol[3,:], sol[4,:]
    θ = rad2deg.(0.5 .* angle.(ap ./ (am .+ 1e-30)))
    # pulse-energy-integrated Stokes of the intracavity probe (analogue of the FDTD readout)
    S1 = sum(2 .* real.(ap .* conj.(am))); S2 = sum(2 .* imag.(ap .* conj.(am)))
    Sp = sum(abs2.(ap)); Sm = sum(abs2.(am))
    return (θf=θ[end], θpk=maximum(abs.(θ)), S1=S1, S2=S2, Sp=Sp, Sm=Sm)
end

c = load_case(joinpath(@__DIR__, "generated", "tcmt_case_sin_best_absolute.jl"))
N = 8
println("="^76)
println("TCMT rotation vs pump1 carrier phase  (SiN best_absolute, I=1e12 W/cm^2)")
println("="^76)
@printf("%8s %12s %12s\n", "phi/2pi", "θ_final°", "θ_peak°")
res = []
for k in 0:N-1
    φ = 2π*k/N
    r = run_phase(c, φ); push!(res, r)
    @printf("%8.3f %12.5f %12.5f\n", k/N, r.θf, r.θpk)
end
θf = [r.θf for r in res]; θpk = [r.θpk for r in res]
S1 = sum(r.S1 for r in res)/N; S2 = sum(r.S2 for r in res)/N
θ_avg = rad2deg(0.5*atan(S2, S1))
println("-"^76)
@printf("θ_final : single-phase(φ=0) %+8.4f°   |  spread over φ: %.4f°  |  mean %+8.4f°\n",
        θf[1], maximum(θf)-minimum(θf), sum(θf)/N)
@printf("θ_peak  : single-phase(φ=0) %+8.4f°   |  spread over φ: %.4f°  |  mean %+8.4f°\n",
        θpk[1], maximum(θpk)-minimum(θpk), sum(θpk)/N)
@printf("CARRIER-AVERAGED (Stokes-averaged over φ): θ = %+8.5f°\n", θ_avg)
println("="^76)
println("FDTD for the same design/intensity:")
println("  1D: fringe max (legacy) 0.1383°   carrier-averaged chi5 envelope 0.00344°")
println("  3D: fringe max (legacy) 1.9913°   carrier-averaged chi5 envelope 0.05563°")
