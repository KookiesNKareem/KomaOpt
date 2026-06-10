# 2-D selective excitation: spiral k-space + complex RF optimization.
# backend ∈ (:cpu, :reactant_gpu, :cuda). :cuda lazily installs CUDA.jl (not in [deps]).

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

include(joinpath(@__DIR__, "utils.jl"))

using KomaMRICore
using CairoMakie, LaTeXStrings
using JLD2: jldsave
using Statistics: mean, std
using LinearAlgebra: norm

const γ64 = γ64_rad
const γ32 = γ_rad

function ensure_cuda_loaded()
    if !haskey(Pkg.project().dependencies, "CUDA")
        @info "CUDA.jl is not in the project. Adding it (heavy one-time install)..."
        Pkg.add("CUDA")
    end
    @eval Main using CUDA
    return nothing
end

function _run_reactant_pipeline(; Nrf, Nspins, n_ctrl, TL, rf_idx, spin_params,
                                target_profile, mask_f32, INVN, interp_h, csr_h,
                                Niters, λ0, img_path)
    @info "[$(basename(img_path))] Setting up Reactant backend..."
    reactant_ctx = setup_reactant_backend(;
        Nspins, n_ctrl, TL, rf_idx,
        spin=spin_params, target=target_profile, mask=mask_f32, invN=INVN,
        interp=interp_h, csr=csr_h)

    grad_fn! = reactant_ctx.grad_fn!
    forward_fn = reactant_ctx.forward_fn
    gx_buf = reactant_ctx.gx_buf
    gi_buf = reactant_ctx.gi_buf

    x_r = zeros(Float32, Nrf)
    x_i = zeros(Float32, Nrf)

    @info "[$(basename(img_path))] Starting optimization..." λ0 Niters
    start_time = time()
    (; loss_history, performed_iterations) = bb_optimize!(
        x_r, x_i, gx_buf, gi_buf, grad_fn!, Niters, λ0;
        backend=KA.CPU(), group_size=1, log_every=5)
    @info "[$(basename(img_path))] Optimization complete in $(round(time() - start_time, digits=2))s"

    x_opt = ComplexF32.(x_r .+ im .* x_i)
    M_xy_achieved, _ = forward_fn(Float32.(real.(x_opt)), Float32.(imag.(x_opt)))
    return x_opt, M_xy_achieved, loss_history, performed_iterations
end

# `CUDA`/`CuArray` resolve at call time; call this via Base.invokelatest after CUDA is loaded.
function _run_cuda_pipeline(; Nrf, Nspins, n_ctrl, TL, rf_idx, spin_params,
                            target_profile, mask_f32, INVN, interp_h, csr_h,
                            Niters, λ0, img_path)
    adapt_dev(x) = CUDA.adapt(CuArray, x)
    ka_backend = CUDA.CUDABackend()
    group_size = 256

    # Enzyme reverse-mode tapes overflow the 8 MB default in-kernel malloc heap at this sim size
    # (surfaces as ERROR_ILLEGAL_ADDRESS in the reverse kernel). Bump to 5 GB.
    if !isdefined(Main, :_CUDA_HEAP_INITIALIZED)
        CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 5*1024^3)
        @eval Main const _CUDA_HEAP_INITIALIZED = true
    end

    @info "[$(basename(img_path))] Setting up CUDA backend (native KA + Enzyme)..."
    bs = init_bloch_setup(;
        Nspins, n_ctrl, TL, rf_idx, spin=spin_params,
        interp=interp_h, csr=csr_h, to_device=adapt_dev, backend=ka_backend, group_size)

    target_d = adapt_dev(ComplexF32.(target_profile))
    mask_d = adapt_dev(mask_f32)

    seed_mxy! = (b) -> seed_and_loss_kernel!(b.backend, b.group_size)(
        b.acc_loss_d, b.dM_xy, b.M_xy, target_d, mask_d, INVN;
        ndrange = Int(b.Nspins))

    grad_fn! = (xr, xi) -> grad_and_loss!(bs, xr, xi, seed_mxy!)
    forward_fn = (xr, xi) -> begin
        Mr, _ = forward_sim!(bs, adapt_dev(xr), adapt_dev(xi))
        return ComplexF32.(Mr), nothing
    end

    x_r = adapt_dev(zeros(Float32, Nrf))
    x_i = adapt_dev(zeros(Float32, Nrf))

    grad_fn!(x_r, x_i)
    CUDA.synchronize()

    @info "[$(basename(img_path))] Starting optimization..." λ0 Niters
    start_time = time()
    (; loss_history, performed_iterations) = bb_optimize!(
        x_r, x_i, bs.gx_d, bs.gi_d, grad_fn!, Niters, λ0;
        backend=ka_backend, group_size, log_every=5)
    @info "[$(basename(img_path))] Optimization complete in $(round(time() - start_time, digits=2))s"

    x_opt = ComplexF32.(Array(x_r) .+ im .* Array(x_i))
    M_xy_achieved, _ = forward_fn(Float32.(real.(x_opt)), Float32.(imag.(x_opt)))
    return x_opt, M_xy_achieved, loss_history, performed_iterations
end

function main(;
    backend::Symbol = :cpu,
    img_path::String = joinpath(@__DIR__, "targets", "stanford_logo.png"),
    show_figures::Bool = isinteractive(),
)
    if backend in (:cpu, :reactant_gpu)
        Reactant.set_default_backend(backend == :cpu ? "cpu" : "cuda")
    elseif backend != :cuda
        error("Unknown backend $backend (expected :cpu, :reactant_gpu, or :cuda)")
    end

    Nrf = 350
    seq, Trf = build_spiral_sequence(; Nrf)

    seq.ADC[1].N = 0

    Nspins_y = 80
    Nspins_x = 80
    FOV_sim = 120e-3
    xs = range(-FOV_sim/2, FOV_sim/2, Nspins_x)
    ys = range(-FOV_sim/2, FOV_sim/2, Nspins_y)
    xgrid = Float32.([xx for (xx, _) in Iterators.product(Float32.(xs), Float32.(ys))][:])
    ygrid = Float32.([yy for (_, yy) in Iterators.product(Float32.(xs), Float32.(ys))][:])
    Nspins = Nspins_x * Nspins_y

    sim_params = KomaMRICore.default_sim_params()
    sim_params["sim_method"] = KomaMRICore.Bloch()
    sim_params["Δt_rf"] = Trf / (Nrf - 1)
    sim_params["Δt"] = Inf
    sim_params["return_type"] = "state"

    mask = [sqrt(x^2 + y^2) .<= FOV_sim/2.2 for (x, y) in Iterators.product(xs, ys)][:]

    TL = discretize_timeline(seq, sim_params)
    rf_idx = TL.rf_active_idx
    n_ctrl = Nrf
    interp_h = build_rf_event_interpolation_tables(seq, TL, n_ctrl, rf_idx)
    csr_h = build_csr_gather_tables(n_ctrl, interp_h.j_lo, interp_h.j_hi, interp_h.w0, interp_h.w1)

    spin_params = (
        p_x = xgrid,
        p_y = ygrid,
        p_z = zeros(Float32, Nspins),
        p_ΔBz = zeros(Float32, Nspins),
        p_T1 = fill(Float32(1e9), Nspins),
        p_T2 = fill(Float32(1e9), Nspins),
        p_ρ = ones(Float32, Nspins),
    )

    target_profile, mag_target_2d = load_image_target(img_path, Nspins_x, Nspins_y)
    INVN = inv(Float32(Nspins))
    mask_f32 = Float32.(mask)

    Niters = 20
    λ0 = 2f-8

    if backend == :cuda
        ensure_cuda_loaded()
        # invokelatest: pipeline must run in the post-`using CUDA` world age.
        x_opt, M_xy_achieved, loss_history, performed_iterations = Base.invokelatest(
            _run_cuda_pipeline;
            Nrf, Nspins, n_ctrl, TL, rf_idx, spin_params,
            target_profile, mask_f32, INVN, interp_h, csr_h, Niters, λ0, img_path)
    else
        x_opt, M_xy_achieved, loss_history, performed_iterations = _run_reactant_pipeline(;
            Nrf, Nspins, n_ctrl, TL, rf_idx, spin_params,
            target_profile, mask_f32, INVN, interp_h, csr_h,
            Niters, λ0, img_path)
    end

    seq.RF[1].A .= x_opt

    stem = splitext(basename(img_path))[1]
    outdir = joinpath(@__DIR__, "pulses", stem)
    mkpath(outdir)

    t_ms = collect(range(0, Trf, Nrf)) .* 1e3
    fig_rf = Figure(size = (1000, 350))
    axrf = Axis(fig_rf[1, 1], xlabel = "Time (ms)", ylabel = "B1 (µT)", title = "Optimized RF (real & imag)")
    lines!(axrf, t_ms, real(x_opt) .* 1e6, color = :blue, label = "Real")
    lines!(axrf, t_ms, imag(x_opt) .* 1e6, color = :red, label = "Imag")
    axislegend(axrf, position = :rt)
    save(joinpath(outdir, "RF_2D_optimized_RF.png"), fig_rf, px_per_unit = 4)
    show_figures && display(fig_rf)

    mask_2d = reshape(mask_f32, Nspins_x, Nspins_y)

    fig_prof = Figure(size = (1200, 800))
    axp = Axis(fig_prof[1, 1], xlabel = L"$x$ (cm)", ylabel = L"$y$ (cm)", title = "Target |Mₓᵧ|", aspect = DataAspect())
    hm_t = heatmap!(axp, xs * 1e2, ys * 1e2, abs.(mag_target_2d) .* mask_2d; colormap = :grays)
    Colorbar(fig_prof[1, 2], hm_t, ticks = [0.0, 0.25, 0.5, 0.75, 1.0])

    achieved = abs.(reshape(M_xy_achieved, Nspins_x, Nspins_y)) .* mask_2d
    axp2 = Axis(fig_prof[1, 3], xlabel = L"$x$ (cm)", ylabel = L"$y$ (cm)", title = "Achieved |Mₓᵧ|", aspect = DataAspect())
    hm_a = heatmap!(axp2, xs * 1e2, ys * 1e2, achieved; colormap = :grays)
    Colorbar(fig_prof[1, 4], hm_a, ticks = [0.0, 0.25, 0.5, 1.0])

    colsize!(fig_prof.layout, 1, Auto(1))
    colsize!(fig_prof.layout, 2, Fixed(18))
    colsize!(fig_prof.layout, 3, Auto(1))
    colsize!(fig_prof.layout, 4, Fixed(18))
    colgap!(fig_prof.layout, 8)

    cl_hi = max(maximum(abs.(mag_target_2d)), maximum(achieved))
    hm_t.colorrange[] = (0.0, cl_hi)
    hm_a.colorrange[] = (0.0, cl_hi)

    save(joinpath(outdir, "RF_2D_image_profile.png"), fig_prof, px_per_unit=4)
    show_figures && display(fig_prof)

    Δt_rf = 10e-6
    T_seq = (dur(seq) ÷ Δt_rf) * Δt_rf
    t_grid = collect(range(0, T_seq, step=Δt_rf))
    B1 = KomaMRIBase.get_rfs(seq, t_grid)[1]
    Grads = KomaMRIBase.get_grads(seq, t_grid)
    jldsave(joinpath(outdir, "$(stem).jld2");
        B1=B1, Gx=Grads[1], Gy=Grads[2], T=T_seq, seq)

    @info "Optimization complete."
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
