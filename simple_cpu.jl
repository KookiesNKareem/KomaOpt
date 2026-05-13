# 1-D slice-selective RF design via KomaMRICore + Enzyme reverse-mode AD.

using Pkg
Pkg.activate(".")
Pkg.instantiate()

using KomaMRICore, KomaMRIBase, Suppressor
import KernelAbstractions as KA
import Enzyme
using Random: seed!
using CairoMakie
using JLD2: jldsave
include("utils.jl")

# Pre-built structs in args — avoids Const→Duplicated stores inside the AD region.
function loss!(M_xy::AbstractVector{ComplexF64}, M_z::AbstractVector{Float64},
    seqd::DiscreteSequence{Float64},
    phantom::Phantom{Float64},
    target_mag::Vector{Float64},
    invN::Float64)

    Nsp = length(M_xy)
    @inbounds for i in 1:Nsp
        M_xy[i] = ComplexF64(0); M_z[i] = 1.0
    end

    M = KomaMRICore.Mag(M_xy, M_z)

    KomaMRICore.run_spin_excitation!(
        phantom, seqd, ComplexF64[], M,
        BlochSimple(), 1, KA.CPU(), KomaMRICore.DefaultPrealloc{Float64}(),
    )

    L = 0.0
    # Loop, not broadcast: Enzyme + ComplexF64.
    @inbounds for i in 1:Nsp
        dr = real(M_xy[i])
        di = imag(M_xy[i]) - target_mag[i]
        L += 0.5 * (dr*dr + di*di) * invN
    end
    return L
end

# Locals (not globals) so Julia/Enzyme see concrete types.
function main()
    seed!(42)

    γ64_Hz = 42.57747892e6
    B1_amp = 4.9e-6
    Trf = 3.2e-3
    TBP = 8.0
    Δz = 6e-3
    zmax = 8e-3
    Nspins = 100

    sys = Scanner()
    Gz = TBP / Trf / (γ64_Hz * Δz)
    seq = PulseDesigner.RF_sinc(B1_amp, Trf, sys; G=[Gz; 0; 0], TBP=TBP)
    obj = Phantom(; x=collect(range(-zmax, zmax, length=Nspins)))

    Nrf = length(seq.RF[1].A)
    n_ctrl = Nrf

    sim_params = KomaMRICore.default_sim_params()
    sim_params["Δt_rf"] = Trf / (2*(Nrf - 1))
    sim_params["Δt"] = Inf
    sim_params["return_type"] = "state"
    sim_params["precision"] = "f32"
    sim_params["sim_method"] = KomaMRICore.Bloch()
    sim_params["Nthreads"] = 1
    sim_params["gpu"] = false

    mag_ref = @suppress simulate(obj, seq, sys; sim_params)

    seqd = KomaMRICore.discretize(seq; sampling_params=sim_params)
    Nt = length(seqd.Δt)

    rf_active_idx = findall(b1 -> abs(b1) > 1e-10, seqd.B1)

    target_mag = Float64.(abs.(0.5im ./ (1 .+ (obj.x ./ (Δz / 2)).^(2 * 5))))
    INVN = inv(Float64(Nspins))

    @info "Setup" Nt Nspins n_ctrl n_rf_active=length(rf_active_idx)

    interp = build_interpolation_tables(n_ctrl, rf_active_idx)
    csr = build_csr_gather_tables(n_ctrl, interp.j_lo, interp.j_hi, interp.w0, interp.w1)

    B1_timeline = zeros(ComplexF64, Nt)
    ∇B1 = zeros(ComplexF64, Nt)
    gx = zeros(Float64, n_ctrl)
    gi = zeros(Float64, n_ctrl)

    # Enzyme shadows: ∇B1 in B1 slot; zero arrays elsewhere keep activity consistent.
    seqd_primal = DiscreteSequence(seqd.Gx, seqd.Gy, seqd.Gz, B1_timeline,
                                   seqd.Δf, seqd.ψ, seqd.ADC, seqd.t, seqd.Δt)
    seqd_shadow = DiscreteSequence(zero(seqd.Gx), zero(seqd.Gy), zero(seqd.Gz),
                                   ∇B1,
                                   zero(seqd.Δf), zero(seqd.ψ), zero(seqd.ADC),
                                   zero(seqd.t), zero(seqd.Δt))
    obj_shadow = Phantom(;
        name = obj.name,
        x = zero(obj.x), y = zero(obj.y), z = zero(obj.z),
        ρ = zero(obj.ρ), T1 = zero(obj.T1), T2 = zero(obj.T2), T2s = zero(obj.T2s),
        Δw = zero(obj.Δw),
        Dλ1 = zero(obj.Dλ1), Dλ2 = zero(obj.Dλ2), Dθ = zero(obj.Dθ),
        motion = NoMotion(),
    )

    function gradient!(x_r::Vector{Float64}, x_i::Vector{Float64})
        cpu_map_ctrl_to_B1!(B1_timeline, x_r, x_i, interp.idx_rf, interp.j_lo, interp.j_hi, interp.w0, interp.w1)
        fill!(∇B1, 0)

        M_xy = zeros(ComplexF64, Nspins)
        M_z = ones(Float64, Nspins)
        dM_xy = zeros(ComplexF64, Nspins)
        dM_z = zeros(Float64, Nspins)

        result = Enzyme.autodiff(
            Enzyme.ReverseWithPrimal,
            loss!,
            Enzyme.Active,
            Enzyme.Duplicated(M_xy, dM_xy),
            Enzyme.Duplicated(M_z, dM_z),
            Enzyme.Duplicated(seqd_primal, seqd_shadow),
            Enzyme.Duplicated(obj, obj_shadow),
            Enzyme.Const(target_mag),
            Enzyme.Const(INVN))

        cpu_gather_grads!(gx, gi, ∇B1, interp.idx_rf, csr.ptr, csr.idx, csr.w)
        return result[2]
    end

    # GD with step capped by gradient RMS.
    Niters = 20
    η_base = 3e-9
    LOG_EVERY = 10

    losses = zeros(Float64, Niters)
    x_r = zeros(Float64, n_ctrl)
    x_i = zeros(Float64, n_ctrl)
    n_performed = 0

    @info "Running GD ($Niters iters, η_base=$η_base)..."
    for k in 1:Niters
        n_performed = k
        losses[k] = gradient!(x_r, x_i)

        g_rms = sqrt((sum(gx.^2) + sum(gi.^2)) / (2*n_ctrl)) + 1e-20

        η = min(η_base, 1e-6 / (10.0 * g_rms))
        @inbounds for j in 1:n_ctrl
            x_r[j] -= η * gx[j]
            x_i[j] -= η * gi[j]
        end

        (k % LOG_EVERY == 0 || k == 1) && @info "Iter $k: loss=$(round(losses[k]; sigdigits=4))"
    end

    @info "Final loss after $n_performed iterations: $(losses[n_performed])"

    cpu_map_ctrl_to_B1!(B1_timeline, x_r, x_i, interp.idx_rf, interp.j_lo, interp.j_hi, interp.w0, interp.w1)
    seqd_final = DiscreteSequence(seqd.Gx, seqd.Gy, seqd.Gz, copy(B1_timeline), seqd.Δf, seqd.ψ, seqd.ADC, seqd.t, seqd.Δt)
    M_final = KomaMRICore.Mag(zeros(ComplexF64, Nspins), ones(Float64, Nspins))
    KomaMRICore.run_spin_excitation!(obj, seqd_final, ComplexF64[], M_final, BlochSimple(), 1, KA.CPU(), KomaMRICore.DefaultPrealloc{Float64}())
    mag_achieved = abs.(M_final.xy)

    z_mm = obj.x .* 1e3
    t_ms = collect(range(0, Trf, n_ctrl)) .* 1e3

    fig = Figure(size=(800, 600))
    ax1 = Axis(fig[1, 1], xlabel="t (ms)", ylabel="B1 (µT)", title="Optimized RF Pulse")
    lines!(ax1, t_ms, x_r .* 1e6, color=:blue, label="Real")
    lines!(ax1, t_ms, x_i .* 1e6, color=:red,  label="Imag")
    axislegend(ax1, position=:rt)
    ax2 = Axis(fig[2, 1], xlabel="z (mm)", ylabel="|Mxy|", title="Slice Profile")
    lines!(ax2, z_mm, mag_achieved, color=:blue, label="Achieved")
    lines!(ax2, z_mm, target_mag, color=:black, linestyle=:dash, label="Target")
    axislegend(ax2, position=:rt)

    outdir = "pulses/simple_cpu"
    mkpath(outdir)
    save(joinpath(outdir, "simple_cpu.png"), fig, px_per_unit=4)
    display(fig)

    Δt_rf = 10e-6
    seq.RF[1].A .= ComplexF32.(x_r .+ im .* x_i)
    T_seq = (dur(seq) ÷ Δt_rf) * Δt_rf
    t_grid = collect(range(0, T_seq, step=Δt_rf))
    B1 = KomaMRIBase.get_rfs(seq, t_grid)[1]
    Grads = KomaMRIBase.get_grads(seq, t_grid)
    jldsave(joinpath(outdir, "simple_cpu.jld2");
            B1=B1, Gx=Grads[1], Gy=Grads[2], T=T_seq, seq)
    @info "Saved pulse to $outdir"
end

main()
