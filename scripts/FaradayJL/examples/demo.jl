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

# Meep/SI conversion constants (same conventions as meep_project/faraday_meep_fp_circ.py).
const EPS0 = 8.854187817e-12
const C0 = 299792458.0
const SCALE_E = 1.0 / (1e-6 * EPS0 * C0)
const FS_PER_MEEP = (1e-6 / C0) * 1e15
# Probe loaded linewidth from rates_harminv_primary.probe.kappa_loaded (in Meep units).
const KAPPA_S_MEEP = 0.01860073865487302
# Conversion for this normalized TCMT model (κs = 1): t_fs = t_tcmt * FS_PER_TCMT.
const FS_PER_TCMT = FS_PER_MEEP / KAPPA_S_MEEP

intensity_to_meep_amplitude(I_w_cm2; n_lin=1.0) = begin
    I_si = I_w_cm2 * 1e4
    E_si = sqrt(2.0 * I_si / (n_lin * EPS0 * C0))
    E_si / SCALE_E
end

# ---------- Common parameters (from latest TCMT extraction) ---------- #
# Source:
# meep_project/pipeline_tio2_20260302_162215/optimizers/mf/tcmt_extracted_params.json
# - rates_harminv_primary + rates_normalized_to_probe_kappa
# - eta_proxies_from_mode_profiles
# - A_equals_B_equals_C (requested isotropic assumption)

norms = FaradayJL.Norms(
    ηs_u1_us = 0.38993669951807 + 0im,
    ηs_u2_us = 0.3942690596747506 + 0im,
    ηΩp = 0.297731010816777 + 0im,
    ηΩm = 0.297731010816777 + 0im,
    ηp1 = 0.0 + 0im,
    ηp2 = 0.0 + 0im,
)

# IMPORTANT:
# chi3_si in the JSON is SI (m^2/V^2), not the dimensionless coefficient used in this model.
# Convert SI -> Meep-scaled nonlinear coefficient first, then apply isotropic A=B=C=chi/3.
chi3_si = 4.34370142617093e-20
chi3_meep = chi3_si * SCALE_E^2
χiso = (chi3_meep / 3.0) + 0im
chidir = FaradayJL.ChiDirect(
    A1=χiso, B1=χiso, C1=χiso,
    A2=χiso, B2=χiso, C2=χiso,
)
chisb = FaradayJL.ChiSideband(
    A_sb_p=χiso, B_sb_p=χiso, C_sb_p=χiso,
    A_sb_m=χiso, B_sb_m=χiso, C_sb_m=χiso,
    A_mx_p=χiso, B_mx_p=χiso, C_mx_p=χiso,
    A_mx_m=χiso, B_mx_m=χiso, C_mx_m=χiso,
)

# Dimensionless rates normalized to probe kappa (κs = 1).
rates_counter = FaradayJL.Rates(
    κ1=3.716757746306964, Δ1=0.5042963266753522,
    κ2=1.6971394712054382, Δ2=0.25809654282099687,
    κs=1.0, Δs=3.454584252059926,
    κΩp=1.9268120182078226, ΔΩp=8.32971694031558,
    κΩm=1.8393419845198256, ΔΩm=-0.23602250934287045,
)
# Co-rotating sideband circular channels use the same extracted Ω± rates.
rates_coro = FaradayJL.Rates(
    κ1=3.716757746306964, Δ1=0.5042963266753522,
    κ2=1.6971394712054382, Δ2=0.25809654282099687,
    κs=1.0, Δs=3.454584252059926,
    κΩp=1.9268120182078226, ΔΩp=8.32971694031558,
    κΩm=1.8393419845198256, ΔΩm=-0.23602250934287045,
    κΩp_p=1.9268120182078226, ΔΩp_p=8.32971694031558,
    κΩp_m=1.9268120182078226, ΔΩp_m=8.32971694031558,
    κΩm_p=1.8393419845198256, ΔΩm_p=-0.23602250934287045,
    κΩm_m=1.8393419845198256, ΔΩm_m=-0.23602250934287045,
)

# Drives mapped to Meep run settings:
# pumps = 1e12 W/cm^2, probe = 1e8 W/cm^2, pulse duration = 100 fs.
pump_intensity_w_cm2 = 2.0e12
probe_intensity_w_cm2 = 1.0e2
pulse_fwhm_intensity_fs = 100.0

pump_amp_meep = intensity_to_meep_amplitude(pump_intensity_w_cm2; n_lin=1.0)
probe_amp_linear_meep = intensity_to_meep_amplitude(probe_intensity_w_cm2; n_lin=1.0)
probe_amp_circ_meep = probe_amp_linear_meep / sqrt(2.0) # 45-degree linear probe => equal circular components

# TCMT normalization with τ = κs t implies s_hat = s / sqrt(κs).
pump_amp = pump_amp_meep / sqrt(KAPPA_S_MEEP)
probe_amp_circ = probe_amp_circ_meep / sqrt(KAPPA_S_MEEP)

# Match Meep helper df_from_pulse_duration():
# width_fs = pulse_duration_fs / (2*log(2)), with source cutoff=4 placing the pulse peak near 4*width_fs.
tau_fs = pulse_fwhm_intensity_fs / (2.0 * log(2.0))
tau_tcmt = tau_fs / FS_PER_TCMT
t0_fs = 4.0 * tau_fs
t0_tcmt = t0_fs / FS_PER_TCMT

S1!(t)    = complex(FaradayJL.gauss(t; A=pump_amp,       t0=t0_tcmt, τ=tau_tcmt))
S2!(t)    = complex(FaradayJL.gauss(t; A=pump_amp,       t0=t0_tcmt, τ=tau_tcmt))
splus!(t) = complex(FaradayJL.gauss(t; A=probe_amp_circ, t0=t0_tcmt, τ=tau_tcmt))
sminus!(t)= complex(FaradayJL.gauss(t; A=probe_amp_circ, t0=t0_tcmt, τ=tau_tcmt))
drives = FaradayJL.Drives(S1!, S2!, splus!, sminus!)

T_fs = (0.0, 3000.0)
saveat_fs = 2.5
T_tcmt = (T_fs[1] / FS_PER_TCMT, T_fs[2] / FS_PER_TCMT)
saveat_tcmt = saveat_fs / FS_PER_TCMT

# ---------------------- Sim 1: Counter-rotating pumps ---------------------- #

params_counter = FaradayJL.make_parameters(
    case=:counter, norms=norms, rates=rates_counter,
    chidir=chidir, chisb=chisb, drives=drives
)

t_c, sol_c = FaradayJL.run_sim(:counter, params_counter; T=T_tcmt, saveat=saveat_tcmt)
t_c_fs = t_c .* FS_PER_TCMT

p1_c, p2_c, aP_c, aM_c, bP_c, bM_c = sol_c[1,:], sol_c[2,:], sol_c[3,:], sol_c[4,:], sol_c[5,:], sol_c[6,:]
θc, εc = FaradayJL.rotation_ellipticity(aP_c, aM_c)

let
    fig1 = Figure(size=(1100, 900), fontsize=13)
    ax11 = Axis(fig1[1,1], xlabel="t (fs)", ylabel="|pump|", title="Counter-rotating: pumps")
    lines!(ax11, t_c_fs, abs.(p1_c), label="|p1|")
    lines!(ax11, t_c_fs, abs.(p2_c), label="|p2|")
    axislegend(ax11, position=:rb)

    ax12 = Axis(fig1[1,2], xlabel="t (fs)", ylabel="|probe|", title="Probe intracavity")
    lines!(ax12, t_c_fs, abs.(aP_c), label="|a+|")
    lines!(ax12, t_c_fs, abs.(aM_c), label="|a-|")
    axislegend(ax12, position=:rb)

    ax21 = Axis(fig1[2,1], xlabel="t (fs)", ylabel="|sb|", title="Sidebands")
    lines!(ax21, t_c_fs, abs.(bP_c), label="|b(Ω+, +)|")
    lines!(ax21, t_c_fs, abs.(bM_c), label="|b(Ω−, −)|")
    axislegend(ax21, position=:rb)

    ax22 = Axis(fig1[2,2], xlabel="t (fs)", ylabel="θ, ε",
    #limits = (nothing, nothing, -pi, pi),
    title="Rotation θ(t) & Ellipticity ε(t)")
    lines!(ax22, t_c_fs, θc, label="θ(t)")
    lines!(ax22, t_c_fs, εc, label="ε(t)")
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
t_r, sol_r = FaradayJL.run_sim(:coro, params_coro; T=T_tcmt, saveat=saveat_tcmt)
t_r_fs = t_r .* FS_PER_TCMT

p1_r, p2_r, aP_r, aM_r = sol_r[1,:], sol_r[2,:], sol_r[3,:], sol_r[4,:]
bpp_r, bpm_r, bmp_r, bmm_r = sol_r[5,:], sol_r[6,:], sol_r[7,:], sol_r[8,:]
θr, εr = FaradayJL.rotation_ellipticity(aP_r, aM_r)

let
    fig2 = Figure(size=(1200, 1000), fontsize=13)
    ax31 = Axis(fig2[1,1], xlabel="t (fs)", ylabel="|pump|", title="Co-rotating: pumps")
    lines!(ax31, t_r_fs, abs.(p1_r), label="|p1|")
    lines!(ax31, t_r_fs, abs.(p2_r), label="|p2|")
    axislegend(ax31, position=:rb)

    ax32 = Axis(fig2[1,2], xlabel="t (fs)", ylabel="|probe|", title="Probe intracavity")
    lines!(ax32, t_r_fs, abs.(aP_r), label="|a+|")
    lines!(ax32, t_r_fs, abs.(aM_r), label="|a-|")
    axislegend(ax32, position=:rb)

    ax33 = Axis(fig2[2,1], xlabel="t (fs)", ylabel="|sb|", title="Ω+ sidebands")
    lines!(ax33, t_r_fs, abs.(bpp_r), label="|b(Ω+, +)|")
    lines!(ax33, t_r_fs, abs.(bpm_r), label="|b(Ω+, −)|")
    axislegend(ax33, position=:rb)

    ax34 = Axis(fig2[2,2], xlabel="t (fs)", ylabel="|sb|", title="Ω− sidebands")
    lines!(ax34, t_r_fs, abs.(bmp_r), label="|b(Ω−, +)|")   # <- abs! is fine; keeps allocation down
    lines!(ax34, t_r_fs, abs.(bmm_r), label="|b(Ω−, −)|")
    axislegend(ax34, position=:rb)

    ax35 = Axis(fig2[3,1:2], xlabel="t (fs)", ylabel="θ, ε", title="Rotation θ(t) & Ellipticity ε(t)")
    lines!(ax35, t_r_fs, θr, label="θ(t)")
    lines!(ax35, t_r_fs, εr, label="ε(t)")
    axislegend(ax35, position=:rb)

    fig2

    # save("fig_coro.png", fig2)
    # println("Saved fig_coro.png")
end

println("Done. Open fig_counter.png and fig_coro.png.")
