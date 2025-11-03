# demo.jl — runs both counter- and co-rotating pump cases and plots with CairoMakie

# Activate the project at the repo root (demo sits in examples/)
using Pkg
# Pkg.activate(normpath(joinpath(@__DIR__, ".."))); Pkg.instantiate()

# Optional: ensure these deps exist (or manage via Project.toml)
for pkg in ("DifferentialEquations", "CairoMakie", "Revise")
    try
        Base.require(pkg)
    catch
        Pkg.add(pkg)
    end
end

using Revise
using CairoMakie
using DifferentialEquations

# If FaradayJL is not installed as a package, include from src:
# include(normpath(joinpath(@__DIR__, "..", "src", "FaradayJL.jl")))
using FaradayJL

# ---------- Common parameters (EDIT with your physical values) ---------- #

# Normalizations (example placeholders)
norms = FaradayJL.Norms(ηs_u1_us = -1.0 + 0im, ηs_u2_us = -1.0 + 0im,
                        ηΩp = -0.6 + 0im, ηΩm = -0.6 + 0im,
                        ηp1 = 0.0 + 0im, ηp2 = 0.0 + 0im)

# χ^(3) coefficients (example; replace with your tuple-evaluated A,B,C)
chidir = FaradayJL.ChiDirect(
    A1=1.0+0.1im, B1=1.0+0.1im, C1=1.0+0.1im,
    A2=1.0+0.1im, B2=1.0+0.1im, C2=1.0+0.1im
)
chisb = FaradayJL.ChiSideband(
    A_sb_p=0.5+0im, B_sb_p=0.5+0.02im, C_sb_p=0.5+0.01im,
    A_sb_m=0.5+0im, B_sb_m=0.5+0.02im, C_sb_m=0.5+0.01im,
    A_mx_p=0.5+0im, B_mx_p=0.5+0.01im, C_mx_p=0.5+0.01im,
    A_mx_m=0.5+0im, B_mx_m=0.5+0.01im, C_mx_m=0.5+0.01im
)

# Cavity rates (example)
rates_counter = FaradayJL.Rates(
    κ1=1.0, Δ1=0.0, κ2=1.0, Δ2=0.0,
    κs=1.0, Δs=0.0,
    κΩp=1.0, ΔΩp=0.2,
    κΩm=1.0, ΔΩm=-0.2
)
# Allow different circulars at sidebands in co-rotating case (or keep same)
rates_coro = FaradayJL.Rates(
    κ1=1.0, Δ1=0.0, κ2=1.0, Δ2=0.0,
    κs=1.0, Δs=0.0,
    κΩp=1.0, ΔΩp=0.2,
    κΩm=1.0, ΔΩm=-0.2,
    κΩp_p=1.0, ΔΩp_p=0.2, κΩp_m=1.0, ΔΩp_m=0.2,
    κΩm_p=1.0, ΔΩm_p=-0.2, κΩm_m=1.0, ΔΩm_m=-0.2
)

# Drives — Gaussian pulses (explicitly complex for robustness)
S1!(t)    = complex(FaradayJL.gauss(t; A=2.0,  t0=10.0, τ=3.0))
S2!(t)    = complex(FaradayJL.gauss(t; A=2.0,  t0=10.0, τ=3.0))
splus!(t) = complex(FaradayJL.gauss(t; A=0.001, t0=10.0, τ=3.0))
sminus!(t)= complex(FaradayJL.gauss(t; A=0.001, t0=10.0, τ=3.0))
drives = FaradayJL.Drives(S1!, S2!, splus!, sminus!)

# ---------------------- Sim 1: Counter-rotating pumps ---------------------- #

params_counter = FaradayJL.make_parameters(
    case=:counter, norms=norms, rates=rates_counter,
    chidir=chidir, chisb=chisb, drives=drives
)


t_c, sol_c = FaradayJL.run_sim(:counter, params_counter; T=(0.0, 30.0), saveat=0.05)

p1_c, p2_c, aP_c, aM_c, bP_c, bM_c = sol_c[1,:], sol_c[2,:], sol_c[3,:], sol_c[4,:], sol_c[5,:], sol_c[6,:]
θc, εc = FaradayJL.rotation_ellipticity(aP_c, aM_c)

let
    fig1 = Figure(size=(1100, 900), fontsize=13)
    ax11 = Axis(fig1[1,1], xlabel="t", ylabel="|pump|", title="Counter-rotating: pumps")
    lines!(ax11, t_c, abs.(p1_c), label="|p1|")
    lines!(ax11, t_c, abs.(p2_c), label="|p2|")
    axislegend(ax11, position=:rb)

    ax12 = Axis(fig1[1,2], xlabel="t", ylabel="|probe|", title="Probe intracavity")
    lines!(ax12, t_c, abs.(aP_c), label="|a+|")
    lines!(ax12, t_c, abs.(aM_c), label="|a-|")
    axislegend(ax12, position=:rb)

    ax21 = Axis(fig1[2,1], xlabel="t", ylabel="|sb|", title="Sidebands")
    lines!(ax21, t_c, abs.(bP_c), label="|b(Ω+, +)|")
    lines!(ax21, t_c, abs.(bM_c), label="|b(Ω−, −)|")
    axislegend(ax21, position=:rb)

    ax22 = Axis(fig1[2,2], xlabel="t", ylabel="θ, ε", 
    #limits = (nothing, nothing, -pi, pi),
    title="Rotation θ(t) & Ellipticity ε(t)")
    lines!(ax22, t_c, θc, label="θ(t)")
    lines!(ax22, t_c, εc, label="ε(t)")
    axislegend(ax22, position=:rb)

    fig1

    # save("fig_counter.png", fig1)  # CairoMakie convention; see docs. :contentReference[oaicite:4]{index=4}
    # println("Saved fig_counter.png")
end

# ----------------------- Sim 2: Co-rotating pumps -------------------------- #

params_coro = FaradayJL.make_parameters(
    case=:coro, norms=norms, rates=rates_coro,
    chidir=chidir, chisb=chisb, drives=drives
)
t_r, sol_r = FaradayJL.run_sim(:coro, params_coro; T=(0.0, 30.0), saveat=0.05)

p1_r, p2_r, aP_r, aM_r = sol_r[1,:], sol_r[2,:], sol_r[3,:], sol_r[4,:]
bpp_r, bpm_r, bmp_r, bmm_r = sol_r[5,:], sol_r[6,:], sol_r[7,:], sol_r[8,:]
θr, εr = FaradayJL.rotation_ellipticity(aP_r, aM_r)

let
    fig2 = Figure(resolution=(1200, 1000), fontsize=13)
    ax31 = Axis(fig2[1,1], xlabel="t", ylabel="|pump|", title="Co-rotating: pumps")
    lines!(ax31, t_r, abs.(p1_r), label="|p1|")
    lines!(ax31, t_r, abs.(p2_r), label="|p2|")
    axislegend(ax31, position=:rb)

    ax32 = Axis(fig2[1,2], xlabel="t", ylabel="|probe|", title="Probe intracavity")
    lines!(ax32, t_r, abs.(aP_r), label="|a+|")
    lines!(ax32, t_r, abs.(aM_r), label="|a-|")
    axislegend(ax32, position=:rb)

    ax33 = Axis(fig2[2,1], xlabel="t", ylabel="|sb|", title="Ω+ sidebands")
    lines!(ax33, t_r, abs.(bpp_r), label="|b(Ω+, +)|")
    lines!(ax33, t_r, abs.(bpm_r), label="|b(Ω+, −)|")
    axislegend(ax33, position=:rb)

    ax34 = Axis(fig2[2,2], xlabel="t", ylabel="|sb|", title="Ω− sidebands")
    lines!(ax34, t_r, abs.(bmp_r), label="|b(Ω−, +)|")   # <- abs! is fine; keeps allocation down
    lines!(ax34, t_r, abs.(bmm_r), label="|b(Ω−, −)|")
    axislegend(ax34, position=:rb)

    ax35 = Axis(fig2[3,1:2], xlabel="t", ylabel="θ, ε", title="Rotation θ(t) & Ellipticity ε(t)")
    lines!(ax35, t_r, θr, label="θ(t)")
    lines!(ax35, t_r, εr, label="ε(t)")
    axislegend(ax35, position=:rb)

    fig2

    # save("fig_coro.png", fig2)
    # println("Saved fig_coro.png")
end

println("Done. Open fig_counter.png and fig_coro.png.")
