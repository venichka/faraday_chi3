# Numerical demonstration of the counter-rotating back-mixing pump-dyad bug.
# Runs the package rhs_counter_derived! (buggy: back-mix reuses the generation dyad
# -> loop ∝ (p2 p1*)^2) vs an inline FIXED copy (back-mix = conjugate dyad
# -> loop ∝ |p1|^2|p2|^2), on a real extracted counter-rotating case.
# No CairoMakie (fast). Run:  julia --project=. examples/bug_demo_counter_backmix.jl
using FaradayJL
using DifferentialEquations
using Printf

# ---- minimal helpers (copied from tcmt_example_utils.jl, no plotting deps) ----
const EPS0 = 8.854187817e-12
const C0 = 299792458.0
const SCALE_E = 1.0 / (1e-6 * EPS0 * C0)
const FS_PER_MEEP = (1e-6 / C0) * 1e15

intensity_to_meep_amplitude(I; n_lin=1.0) = sqrt(2.0 * I * 1e4 / (n_lin * EPS0 * C0)) / SCALE_E

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
    κs = c.kappa_probe_meep
    fs_per = FS_PER_MEEP / κs
    pamp = intensity_to_meep_amplitude(c.pulse.pump_intensity_w_cm2) / sqrt(κs)
    samp = (intensity_to_meep_amplitude(c.pulse.probe_intensity_w_cm2) / sqrt(2.0)) / sqrt(κs)
    τfs = c.pulse.pulse_fwhm_intensity_fs / (2.0 * log(2.0)); τ = τfs / fs_per
    t0 = (4.0 * τfs) / fs_per
    s1(t)=complex(FaradayJL.gauss(t;A=pamp,t0=t0,τ=τ)); s2(t)=complex(FaradayJL.gauss(t;A=pamp,t0=t0,τ=τ))
    sp(t)=complex(FaradayJL.gauss(t;A=samp,t0=t0,τ=τ)); sm(t)=complex(FaradayJL.gauss(t;A=samp,t0=t0,τ=τ))
    return FaradayJL.Drives(s1,s2,sp,sm), fs_per
end

counter_derived(c) = FaradayJL.CounterDerived(
    α1_plus=c.derived.counter.alpha1_plus, α2_plus=c.derived.counter.alpha2_plus,
    α1_minus=c.derived.counter.alpha1_minus, α2_minus=c.derived.counter.alpha2_minus,
    ζ_plus=c.derived.counter.zeta_plus, ζ_minus=c.derived.counter.zeta_minus,
    η_plus=c.derived.counter.eta_plus, η_minus=c.derived.counter.eta_minus)

# ---- FIXED rhs: identical to package rhs_counter_derived! except the two
#      back-mixing dynamical pump dyads are the CONJUGATE of the generation dyad ----
function rhs_counter_derived_fixed!(dy, y, p, t)
    rates = p.rates; coeff = p.derived; drives = p.drives
    p1, p2, a₊, a₋, b₊, b₋ = y
    S1=drives.S1!(t); S2=drives.S2!(t); splus=drives.splus!(t); sminus=drives.sminus!(t)
    I1=abs2(p1); I2=abs2(p2)
    Φ₊=coeff.α1_plus*I1+coeff.α2_plus*I2; Φ₋=coeff.α1_minus*I1+coeff.α2_minus*I2
    dy[1]=(im*rates.Δ1-rates.κ1/2)*p1+sqrt(rates.κ1)*S1
    dy[2]=(im*rates.Δ2-rates.κ2/2)*p2+sqrt(rates.κ2)*S2
    # FIX: a₊ back-mix uses p1*conj(p2) (was p2*conj(p1)); a₋ uses p2*conj(p1) (was p1*conj(p2))
    dy[3]=(im*rates.Δs-rates.κs/2)*a₊+im*Φ₊*a₊+im*coeff.η_minus*(p1*conj(p2))*b₋+sqrt(rates.κs)*splus
    dy[4]=(im*rates.Δs-rates.κs/2)*a₋+im*Φ₋*a₋+im*coeff.η_plus *(p2*conj(p1))*b₊+sqrt(rates.κs)*sminus
    dy[5]=(im*rates.ΔΩp-rates.κΩp/2)*b₊+im*coeff.ζ_plus *(p1*conj(p2))*a₋
    dy[6]=(im*rates.ΔΩm-rates.κΩm/2)*b₋+im*coeff.ζ_minus*(p2*conj(p1))*a₊
    nothing
end

function solve_fixed(params, T, saveat)
    prob = ODEProblem{true}(rhs_counter_derived_fixed!, zeros(ComplexF64, 6), T, params)
    solve(prob, Tsit5(); reltol=1e-7, abstol=1e-9, saveat)
end

# ----------------------------------- run -------------------------------------- #
casefile = joinpath(@__DIR__, "generated", "tcmt_case_sin_090326_new.jl")
c = load_case(casefile)
drives, fs_per = build_drives(c)
rates = rates_counter(c)
derived = counter_derived(c)
params = FaradayJL.make_parameters_derived(case=:counter, rates=rates, derived=derived,
                                           drives=drives, output=FaradayJL.ProbeOutput())

t_fs = (0.0, 3000.0); saveat_fs = 2.5
T = (t_fs[1]/fs_per, t_fs[2]/fs_per); saveat = saveat_fs/fs_per

# buggy = the package function (rhs_counter_derived!), via run_sim
_, sol_bug = FaradayJL.run_sim(:counter, params; T=T, saveat=saveat)
sol_fix = solve_fixed(params, T, saveat)

θ(sol) = 0.5 .* angle.(sol[3,:] ./ (sol[4,:] .+ 1e-30))    # rotation (rad)
χ(sol) = 0.5 .* asin.(clamp.((abs2.(sol[3,:]).-abs2.(sol[4,:]))./(abs2.(sol[3,:]).+abs2.(sol[4,:]).+1e-30),-1,1))
deg = rad2deg

θb=deg.(θ(sol_bug)); θf=deg.(θ(sol_fix)); χb=deg.(χ(sol_bug)); χf=deg.(χ(sol_fix))

println("="^70)
println("Counter-rotating cascade — BUGGY (package) vs FIXED (conjugate dyad)")
println("case: tcmt_case_sin_090326_new   pumps I=$(c.pulse.pump_intensity_w_cm2) W/cm²")
println("="^70)
@printf("                              %12s %12s\n", "BUGGY", "FIXED")
@printf("rotation  θ  final  (deg)     %12.5f %12.5f\n", θb[end], θf[end])
@printf("rotation  θ  peak|.| (deg)    %12.5f %12.5f\n", maximum(abs.(θb)), maximum(abs.(θf)))
@printf("ellipticity χ final (deg)     %12.5f %12.5f\n", χb[end], χf[end])
@printf("ellipticity χ peak|.| (deg)   %12.5f %12.5f\n", maximum(abs.(χb)), maximum(abs.(χf)))

# ---- direct loop-factor diagnostic: phase of the a₊ self-loop pump factor ----
p1=sol_fix[1,:]; p2=sol_fix[2,:]
loop_buggy = (p2 .* conj.(p1)).^2                       # what the package multiplies in (gen×backmix)
loop_fixed = (p2 .* conj.(p1)) .* (p1 .* conj.(p2))     # = |p1|^2|p2|^2  (gen × conj-backmix)
# restrict to the window where pumps are appreciable
w = abs.(p1).*abs.(p2) .> 0.05*maximum(abs.(p1).*abs.(p2))
phb = rad2deg.(angle.(loop_buggy[w])); phf = rad2deg.(angle.(loop_fixed[w]))
println("-"^70)
println("a₊ self-loop pump factor (over the pumped window):")
@printf("  |loop|  identical?  max|Δ|/|loop| = %.2e  (magnitudes equal by construction)\n",
        maximum(abs.(abs.(loop_buggy[w]).-abs.(loop_fixed[w]))) / maximum(abs.(loop_fixed[w])))
@printf("  arg(loop) BUGGY : range [%.1f, %.1f] deg, std %.1f deg  <- spurious, time-varying\n",
        minimum(phb), maximum(phb), sqrt(sum((phb.-sum(phb)/length(phb)).^2)/length(phb)))
@printf("  arg(loop) FIXED : range [%.1f, %.1f] deg, std %.2f deg  <- DC (real, phase-stable)\n",
        minimum(phf), maximum(phf), sqrt(sum((phf.-sum(phf)/length(phf)).^2)/length(phf)))
println("="^70)
