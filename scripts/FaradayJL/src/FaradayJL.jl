module FaradayJL
# FaradayJL — TCMT simulator for probe Faraday rotation with χ(3) + cascaded sidebands
# Julia ≥ 1.9

using LinearAlgebra
using DifferentialEquations

# ----------------------------- Utilities & types ------------------------------ #

"""
Normalization factors that map susceptibilities to frequency pulls:
η_x = mu_0 ω_x / (2 U_x) * overlap_integral. Use complex if you include absorption.
"""
Base.@kwdef mutable struct Norms
    ηs_u1_us::ComplexF64     # probe @ ωs
    ηs_u2_us::ComplexF64     # probe @ ωs
    ηp1::ComplexF64    # (not used unless you add pump Kerr)
    ηp2::ComplexF64
    ηΩp::ComplexF64    # sideband @ Ω+: u_s^* u_1 u_2^* u_s
    ηΩm::ComplexF64    # sideband @ Ω−: u_s^* u_2 u_1^* u_s
end

"""
Direct χ^(3) coefficients at ωs for tuples (-ωs; ωs, ωm, -ωm).
Keep A,B,C separately for each pump m ∈ {1,2}.
"""
Base.@kwdef mutable struct ChiDirect
    A1::ComplexF64; B1::ComplexF64; C1::ComplexF64
    A2::ComplexF64; B2::ComplexF64; C2::ComplexF64
end

"""
Sideband generation and mixing coefficients at the proper frequency tuples.
- sb:  (-Ω±; ωs, ω(arm in), -ω(arm out))
- mx:  (-ωs; Ω±, ω(arm out), -ω(arm in))
"""
Base.@kwdef mutable struct ChiSideband
    # sideband generation
    A_sb_p::ComplexF64; B_sb_p::ComplexF64; C_sb_p::ComplexF64  # Ω+
    A_sb_m::ComplexF64; B_sb_m::ComplexF64; C_sb_m::ComplexF64  # Ω−
    # back-mixing to ωs
    A_mx_p::ComplexF64; B_mx_p::ComplexF64; C_mx_p::ComplexF64  # from Ω+
    A_mx_m::ComplexF64; B_mx_m::ComplexF64; C_mx_m::ComplexF64  # from Ω−
end

"""
Cavity linewidths κ and detunings Δ (single-pole per mode).
For the co-rotating model we allow separate circulars at Ω±.
"""
Base.@kwdef mutable struct Rates
    # pumps
    κ1::Float64; Δ1::Float64
    κ2::Float64; Δ2::Float64
    # probe (same κ, Δ for + and − here)
    κs::Float64; Δs::Float64
    # sidebands (aggregate; used by counter-rotating model)
    κΩp::Float64; ΔΩp::Float64     # Ω+
    κΩm::Float64; ΔΩm::Float64     # Ω−
    # sidebands split by circular (for co-rotating model; default to aggregate)
    κΩp_p::Float64 = κΩp; ΔΩp_p::Float64 = ΔΩp  # Ω+, +
    κΩp_m::Float64 = κΩp; ΔΩp_m::Float64 = ΔΩp  # Ω+, −
    κΩm_p::Float64 = κΩm; ΔΩm_p::Float64 = ΔΩm  # Ω−, +
    κΩm_m::Float64 = κΩm; ΔΩm_m::Float64 = ΔΩm  # Ω−, −
end

"""
Directly extracted counter-rotating coefficients normalized to probe kappa.
These correspond to the derivation coefficients α, ζ, η after mode-overlap
projection and time normalization τ = κs * t.
"""
Base.@kwdef mutable struct CounterDerived
    α1_plus::ComplexF64
    α2_plus::ComplexF64
    α1_minus::ComplexF64
    α2_minus::ComplexF64
    ζ_plus::ComplexF64
    ζ_minus::ComplexF64
    η_plus::ComplexF64
    η_minus::ComplexF64
end

"""
Directly extracted co-rotating coefficients normalized to probe kappa.
Diagonal generation/back-mixing terms follow the derivation; optional Λ terms
capture linear circular mixing inside each sideband arm.
"""
Base.@kwdef mutable struct CoroDerived
    α1_plus::ComplexF64
    α2_plus::ComplexF64
    α1_minus::ComplexF64
    α2_minus::ComplexF64
    ζ_pp::ComplexF64
    ζ_pm::ComplexF64
    ζ_mp::ComplexF64
    ζ_mm::ComplexF64
    η_pp::ComplexF64
    η_pm::ComplexF64
    η_mp::ComplexF64
    η_mm::ComplexF64
    ΛΩp::ComplexF64 = 0.0 + 0.0im
    ΛΩm::ComplexF64 = 0.0 + 0.0im
end

"""
Simple probe-output mapping. With κ_out the same for both circulars and zero
direct term, output rotation matches the intracavity a+/a− ratio.
"""
Base.@kwdef mutable struct ProbeOutput
    κ_out_plus::Float64 = 1.0
    κ_out_minus::Float64 = 1.0
    c_plus::ComplexF64 = 0.0 + 0.0im
    c_minus::ComplexF64 = 0.0 + 0.0im
end

# ------------------------ χ^(3)_eff helper functions ------------------------- #

"""
χ_eff^(3) at ωs for COUNTER-rotating pumps.
Returns (χ_++, χ_--) for the probe circulars given intensities I1=|p1|^2, I2=|p2|^2.
"""
function chi_eff3_counter(cd::ChiDirect, I1::Real, I2::Real, norm::Norms)
    χpp = (3/4) * ( (cd.A1 + cd.B1)*I1*norm.ηs_u1_us +
        (cd.A2 + cd.C2)*I2*norm.ηs_u2_us )
    χmm = (3/4) * ( (cd.A1 + cd.C1)*I1*norm.ηs_u1_us +
        (cd.A2 + cd.B2)*I2*norm.ηs_u2_us )
    return χpp, χmm
end

"""
χ_eff^(3) at ωs for CO-rotating pumps (direct part has same algebra in isotropic media).
"""
function chi_eff3_coro(cd::ChiDirect, I1::Real, I2::Real, norm::Norms)
    χpp = (3/4) * ( (cd.A1 + cd.B1)*I1*norm.ηs_u1_us +
        (cd.A2 + cd.B2)*I2*norm.ηs_u2_us )
    χmm = (3/4) * ( (cd.A1 + cd.C1)*I1*norm.ηs_u1_us +
        (cd.A2 + cd.C2)*I2*norm.ηs_u2_us )
    return χpp, χmm
end

# ----------------------------- Drives (parametric) --------------------------- #

"Gaussian complex-envelope (always returns Complex)."
gauss(t; A=1.0, t0=0.0, τ=2.0, φ=0.0) = A * exp(-((t - t0)/τ)^2) * cis(φ)  # cis gives complex. :contentReference[oaicite:3]{index=3}

"Drive closures (parametric for type stability)."
struct Drives{F1,F2,F3,F4}
    S1!::F1     # pump 1 input
    S2!::F2     # pump 2 input
    splus!::F3  # probe +
    sminus!::F4 # probe −
end

# ================= COUNTER-rotating ODE (p1 ∥ e+, p2 ∥ e−) ================== #

"Precompute scalar couplings for counter-rotating case."
function couplings_counter(norm::Norms, csb::ChiSideband)
    g_plus  = norm.ηΩp * (csb.B_sb_p + csb.C_sb_p)   # a_- → b_+ @ Ω+
    g_minus = norm.ηΩm * (csb.B_sb_m + csb.C_sb_m)   # a_+ → b_- @ Ω−
    m_plus  = norm.ηΩm  * (csb.B_mx_p + csb.C_mx_p)   # b_+ → a_-
    m_minus = norm.ηΩp  * (csb.B_mx_m + csb.C_mx_m)   # b_- → a_+
    return (g_plus=g_plus, g_minus=g_minus, m_plus=m_plus, m_minus=m_minus)
end

"""
RHS for counter-rotating pumps.
State y = [p1, p2, a_plus, a_minus, b_plus(Ω+,+), b_minus(Ω−,−)] (ComplexF64).
"""
function rhs_counter!(dy, y, p, t)
    rates::Rates = p.rates
    norms::Norms = p.norms
    cd::ChiDirect = p.chidir
    coup = p.coup
    drives = p.drives

    p1, p2, a₊, a₋, b₊, b₋ = y

    S1 = drives.S1!(t)
    S2 = drives.S2!(t)
    splus  = drives.splus!(t)
    sminus = drives.sminus!(t)

    # Direct Kerr (instantaneous)
    χpp, χmm = chi_eff3_counter(cd, abs2(p1), abs2(p2), norms)
    σ₊ = χpp
    σ₋ = χmm

    # Pumps
    dy[1] = (im*rates.Δ1 - rates.κ1/2)*p1 + sqrt(rates.κ1)*S1
    dy[2] = (im*rates.Δ2 - rates.κ2/2)*p2 + sqrt(rates.κ2)*S2

    # Probe — back-mixing carries the CONJUGATE of the generation pump dyad,
    # so the cascade loop ∝ |p1|²|p2|² (DC), not (p2 p1*)² (isotropic_derivation §1).
    dy[3] = (im*rates.Δs - rates.κs/2)*a₊ + im*σ₊*a₊ + im*coup.m_minus*(p1*conj(p2))*b₋ + sqrt(rates.κs)*splus
    dy[4] = (im*rates.Δs - rates.κs/2)*a₋ + im*σ₋*a₋ + im*coup.m_plus*(p2*conj(p1))*b₊ + sqrt(rates.κs)*sminus

    # Sidebands
    dy[5] = (im*rates.ΔΩp - rates.κΩp/2)*b₊ + im*coup.g_plus*(p1*conj(p2))*a₋
    dy[6] = (im*rates.ΔΩm - rates.κΩm/2)*b₋ + im*coup.g_minus*(p2*conj(p1))*a₊

    return nothing
end

"""
RHS for counter-rotating pumps using derivation-consistent extracted
coefficients instead of the legacy Norms/Chi proxy decomposition.
"""
function rhs_counter_derived!(dy, y, p, t)
    rates::Rates = p.rates
    coeff::CounterDerived = p.derived
    drives = p.drives

    p1, p2, a₊, a₋, b₊, b₋ = y

    S1 = drives.S1!(t)
    S2 = drives.S2!(t)
    splus  = drives.splus!(t)
    sminus = drives.sminus!(t)

    I1 = abs2(p1)
    I2 = abs2(p2)
    Φ₊ = coeff.α1_plus * I1 + coeff.α2_plus * I2
    Φ₋ = coeff.α1_minus * I1 + coeff.α2_minus * I2

    dy[1] = (im * rates.Δ1 - rates.κ1 / 2) * p1 + sqrt(rates.κ1) * S1
    dy[2] = (im * rates.Δ2 - rates.κ2 / 2) * p2 + sqrt(rates.κ2) * S2

    # back-mixing carries the CONJUGATE of the generation pump dyad, so the
    # cascade loop ∝ |p1|²|p2|² (DC), not (p2 p1*)² (isotropic_derivation §1).
    dy[3] = (im * rates.Δs - rates.κs / 2) * a₊ + im * Φ₊ * a₊ +
        im * coeff.η_minus * (p1 * conj(p2)) * b₋ + sqrt(rates.κs) * splus
    dy[4] = (im * rates.Δs - rates.κs / 2) * a₋ + im * Φ₋ * a₋ +
        im * coeff.η_plus * (p2 * conj(p1)) * b₊ + sqrt(rates.κs) * sminus

    dy[5] = (im * rates.ΔΩp - rates.κΩp / 2) * b₊ +
        im * coeff.ζ_plus * (p1 * conj(p2)) * a₋
    dy[6] = (im * rates.ΔΩm - rates.κΩm / 2) * b₋ +
        im * coeff.ζ_minus * (p2 * conj(p1)) * a₊

    return nothing
end

# =================== CO-rotating ODE (p1, p2 ∥ e+) ========================== #

"Precompute couplings for co-rotating case."
function couplings_coro(norm::Norms, csb::ChiSideband)
    # generation a→b @ Ω+
    g_p_p = norm.ηΩp * (csb.A_sb_p + csb.B_sb_p)  # (+) circular sb from a_+
    g_p_m = norm.ηΩp * (csb.A_sb_p + csb.C_sb_p)  # (−) circular sb from a_-
    # generation a→b @ Ω−
    g_m_p = norm.ηΩm * (csb.A_sb_m + csb.B_sb_m)  # (+) from a_+
    g_m_m = norm.ηΩm * (csb.A_sb_m + csb.C_sb_m)  # (−) from a_-
    # back-mixing b→a
    m_p_A = norm.ηΩm  * (csb.A_mx_p)
    m_p_B = norm.ηΩm  * (csb.B_mx_p)
    m_p_C = norm.ηΩm  * (csb.C_mx_p)
    m_m_A = norm.ηΩp  * (csb.A_mx_m)
    m_m_B = norm.ηΩp  * (csb.B_mx_m)
    m_m_C = norm.ηΩp  * (csb.C_mx_m)
    return (g_p_p=g_p_p, g_p_m=g_p_m, g_m_p=g_m_p, g_m_m=g_m_m,
            m_p_A=m_p_A, m_p_B=m_p_B, m_p_C=m_p_C, m_m_A=m_m_A,
            m_m_B=m_m_B, m_m_C=m_m_C)
end

"""
RHS for co-rotating pumps.
State y = [p1, p2, a_plus, a_minus, b_p_p, b_p_m, b_m_p, b_m_m] (ComplexF64).
"""
function rhs_coro!(dy, y, p, t)
    rates::Rates = p.rates
    norms::Norms = p.norms
    cd::ChiDirect = p.chidir
    coup = p.coup
    drives = p.drives

    p1, p2, a₊, a₋, b_p_p, b_p_m, b_m_p, b_m_m = y

    S1 = drives.S1!(t)
    S2 = drives.S2!(t)
    splus  = drives.splus!(t)
    sminus = drives.sminus!(t)

    χpp, χmm = chi_eff3_coro(cd, abs2(p1), abs2(p2), norms)
    σ₊ = χpp
    σ₋ = χmm

    # Pumps
    dy[1] = (im*rates.Δ1 - rates.κ1/2)*p1 + sqrt(rates.κ1)*S1
    dy[2] = (im*rates.Δ2 - rates.κ2/2)*p2 + sqrt(rates.κ2)*S2

    # Probe (Ω+ arm uses E2 & E1*)
    dy[3] = (im*rates.Δs - rates.κs/2)*a₊ + im*σ₊*a₊ +
        im*( (coup.m_p_A + coup.m_p_B)*(p2*conj(p1))*b_p_p +
             (coup.m_m_A + coup.m_m_B)*(p1*conj(p2))*b_m_p ) +
        sqrt(rates.κs)*splus
    dy[4] = (im*rates.Δs - rates.κs/2)*a₋ + im*σ₋*a₋ +
        im*( (coup.m_p_A + coup.m_p_C)*(p2*conj(p1))*b_p_m +
             (coup.m_m_A + coup.m_m_C)*(p1*conj(p2))*b_m_m ) +
        sqrt(rates.κs)*sminus

    # Sidebands @ Ω+ and Ω−
    dy[5] = (im*rates.ΔΩp_p - rates.κΩp_p/2)*b_p_p + im*coup.g_p_p*(p1*conj(p2))*a₊
    dy[6] = (im*rates.ΔΩp_m - rates.κΩp_m/2)*b_p_m + im*coup.g_p_m*(p1*conj(p2))*a₋
    dy[7] = (im*rates.ΔΩm_p - rates.κΩm_p/2)*b_m_p + im*coup.g_m_p*(p2*conj(p1))*a₊
    dy[8] = (im*rates.ΔΩm_m - rates.κΩm_m/2)*b_m_m + im*coup.g_m_m*(p2*conj(p1))*a₋

    return nothing
end

"""
RHS for co-rotating pumps using derivation-consistent extracted coefficients.
"""
function rhs_coro_derived!(dy, y, p, t)
    rates::Rates = p.rates
    coeff::CoroDerived = p.derived
    drives = p.drives

    p1, p2, a₊, a₋, b_p_p, b_p_m, b_m_p, b_m_m = y

    S1 = drives.S1!(t)
    S2 = drives.S2!(t)
    splus  = drives.splus!(t)
    sminus = drives.sminus!(t)

    I1 = abs2(p1)
    I2 = abs2(p2)
    Φ₊ = coeff.α1_plus * I1 + coeff.α2_plus * I2
    Φ₋ = coeff.α1_minus * I1 + coeff.α2_minus * I2

    dy[1] = (im * rates.Δ1 - rates.κ1 / 2) * p1 + sqrt(rates.κ1) * S1
    dy[2] = (im * rates.Δ2 - rates.κ2 / 2) * p2 + sqrt(rates.κ2) * S2

    dy[3] = (im * rates.Δs - rates.κs / 2) * a₊ + im * Φ₊ * a₊ +
        im * (
            coeff.η_pp * (p2 * conj(p1)) * b_p_p +
            coeff.η_mp * (p1 * conj(p2)) * b_m_p
        ) + sqrt(rates.κs) * splus

    dy[4] = (im * rates.Δs - rates.κs / 2) * a₋ + im * Φ₋ * a₋ +
        im * (
            coeff.η_pm * (p2 * conj(p1)) * b_p_m +
            coeff.η_mm * (p1 * conj(p2)) * b_m_m
        ) + sqrt(rates.κs) * sminus

    dy[5] = (im * rates.ΔΩp_p - rates.κΩp_p / 2) * b_p_p +
        im * coeff.ΛΩp * b_p_m + im * coeff.ζ_pp * (p1 * conj(p2)) * a₊
    dy[6] = (im * rates.ΔΩp_m - rates.κΩp_m / 2) * b_p_m +
        im * conj(coeff.ΛΩp) * b_p_p + im * coeff.ζ_pm * (p1 * conj(p2)) * a₋
    dy[7] = (im * rates.ΔΩm_p - rates.κΩm_p / 2) * b_m_p +
        im * coeff.ΛΩm * b_m_m + im * coeff.ζ_mp * (p2 * conj(p1)) * a₊
    dy[8] = (im * rates.ΔΩm_m - rates.κΩm_m / 2) * b_m_m +
        im * conj(coeff.ΛΩm) * b_m_p + im * coeff.ζ_mm * (p2 * conj(p1)) * a₋

    return nothing
end

# ============================= Helpers / API ================================ #

"Pack parameters for either :counter or :coro case."
function make_parameters(; case::Symbol,
    norms::Norms, rates::Rates, chidir::ChiDirect, chisb::ChiSideband, drives::Drives)
    coup = case == :counter ? couplings_counter(norms, chisb) : couplings_coro(norms, chisb)
    return (; norms, rates, chidir, chisb, drives, coup)
end

"Pack derivation-consistent parameters."
function make_parameters_derived(; case::Symbol, rates::Rates, derived, drives::Drives,
    output::ProbeOutput=ProbeOutput())
    if case == :counter
        derived isa CounterDerived || error("counter case requires CounterDerived")
    elseif case == :coro
        derived isa CoroDerived || error("coro case requires CoroDerived")
    else
        error("case must be :counter or :coro")
    end
    return (; rates, derived, drives, output)
end

"Run simulation. Returns (t, sol)."
function run_sim(case, params; T=(0.0, 30.0), saveat=0.02)
    if case == :counter
        y0 = zeros(ComplexF64, 6)
        rhs! = haskey(params, :derived) ? rhs_counter_derived! : rhs_counter!
        prob = ODEProblem{true}(rhs!, y0, T, params)
    elseif case == :coro
        y0 = zeros(ComplexF64, 8)
        rhs! = haskey(params, :derived) ? rhs_coro_derived! : rhs_coro!
        prob = ODEProblem{true}(rhs!, y0, T, params)
    else
        error("case must be :counter or :coro")
    end
    sol = solve(prob, Tsit5(); reltol=1e-7, abstol=1e-9, saveat)
    return sol.t, sol
end


"Compute instantaneous rotation θ(t) and ellipticity ε(t) from (a_plus, a_minus)."
function rotation_ellipticity(a_plus::AbstractVector, a_minus::AbstractVector)
    r = a_plus ./ (a_minus .+ 1e-30)  # guard divide-by-zero
    θ = 0.5 .* angle.(r)
    ε = 0.5 .* log.(abs.(r) .+ 1e-30)
    ε[ε .< 1e-10] .= 0.0
    return θ, ε
end

"Return Stokes parameters from circular-basis fields."
function stokes_from_circular(e_plus::AbstractVector, e_minus::AbstractVector)
    s0 = abs2.(e_plus) .+ abs2.(e_minus)
    s1 = 2 .* real.(e_plus .* conj.(e_minus))
    s2 = 2 .* imag.(e_plus .* conj.(e_minus))
    s3 = abs2.(e_plus) .- abs2.(e_minus)
    return s0, s1, s2, s3
end

"Map intracavity probe amplitudes to output fields under a simple port model."
function output_probe_fields(a_plus::AbstractVector, a_minus::AbstractVector;
    output::ProbeOutput=ProbeOutput(),
    splus_in::Union{Nothing,AbstractVector}=nothing,
    sminus_in::Union{Nothing,AbstractVector}=nothing)
    e_plus = sqrt(output.κ_out_plus) .* a_plus
    e_minus = sqrt(output.κ_out_minus) .* a_minus
    if splus_in !== nothing
        e_plus .+= output.c_plus .* splus_in
    end
    if sminus_in !== nothing
        e_minus .+= output.c_minus .* sminus_in
    end
    return e_plus, e_minus
end

"Compute physical rotation ψ and ellipticity angle χ from circular fields."
function rotation_ellipticity_physical(e_plus::AbstractVector, e_minus::AbstractVector)
    ρ = e_plus ./ (e_minus .+ 1e-30)
    ψ = 0.5 .* angle.(ρ)
    ratio = (abs2.(e_plus) .- abs2.(e_minus)) ./ (abs2.(e_plus) .+ abs2.(e_minus) .+ 1e-30)
    χ = 0.5 .* asin.(clamp.(real.(ratio), -1.0, 1.0))
    return ψ, χ
end

# ------------------------------ explicit exports ----------------------------- #

export Norms, ChiDirect, ChiSideband, Rates, Drives,
       CounterDerived, CoroDerived, ProbeOutput,
       chi_eff3_counter, chi_eff3_coro,
       couplings_counter, couplings_coro,
       gauss, make_parameters, make_parameters_derived, run_sim,
       rotation_ellipticity, stokes_from_circular,
       output_probe_fields, rotation_ellipticity_physical

end # module
