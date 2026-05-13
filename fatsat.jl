# Fat saturation RF pulse optimization.
# backend ∈ (:cpu, :cuda, :reactant_gpu).

const BACKEND = :cpu

include("utils.jl")

if BACKEND == :cuda
    using CUDA
    CUDA.device!(0)
    # increase malloc heap so reverse-mode tapes fit
    if !isdefined(Main, :_CUDA_HEAP_INITIALIZED)
        CUDA.limit!(CUDA.CU_LIMIT_MALLOC_HEAP_SIZE, 5*1024^3)
        @eval Main const _CUDA_HEAP_INITIALIZED = true
    end
    adapt_dev(x) = CUDA.adapt(CuArray, x)
elseif BACKEND in (:cpu, :reactant_gpu)
    Reactant.set_default_backend(BACKEND == :cpu ? "cpu" : "cuda")
end

using Suppressor
using KomaMRICore: get_flip_angles
using CairoMakie, LaTeXStrings
using JLD2
using Statistics: mean

const γ = γ_Hz

# Pulse
const B1_INIT = 2e-6
const Trf = 10e-3
const TBP = 4.0
const Nspins = 5000
const GROUP_SIZE = 256

# Physics
const B0 = 3.0
const FAT_PPM = -3.5
const FAT_FREQ = FAT_PPM * γ64_Hz * B0 * 1e-6
const FAT_BW = abs(FAT_FREQ)
const WATER_PROTECT = 100.0
const FREQ_MAX = 800.0

const T1_FAT = 350f-3
const T1_WATER = 1400f-3
const T_DELAY = 55f-3 # post-pulse delay before readout

# Optimization
const MAX_ITERS = 40
const GRAD_TOL = 1e-10
const LOG_EVERY = 10
const LAMBDA_MAX = 5f-9

# Target
const TARGET_MODE = :butterworth # :butterworth or :gaussian
const BUTTERWORTH_DEGREE = 4
const GAUSSIAN_FLIP_DEG = 90.0
const GENERATE_COMPARISON_FIGURE = true

sys = Scanner()
freq_offsets = Float32.(collect(range(-FREQ_MAX, FREQ_MAX, length=Nspins)))
seq = PulseDesigner.RF_sinc(B1_INIT, Trf, sys; G=[0; 0; 0], TBP=TBP)
const Nrf = length(seq.RF[1].A)
const n_ctrl = Nrf

sim_params = KomaMRICore.default_sim_params()
sim_params["Δt_rf"] = Trf / (4 * (Nrf - 1))
sim_params["Δt"] = Inf
sim_params["return_type"] = "state"
sim_params["precision"] = "f32"
sim_params["sim_method"] = KomaMRICore.Bloch()

TL = discretize_timeline(seq, sim_params)
rf_idx = TL.rf_active_idx

# Fat T1 inside the fat band, water T1 elsewhere.
const ΔBz_h = Float32.(freq_offsets ./ γ_Hz)
const spin_T1 = Float32[abs(f - FAT_FREQ) < FAT_BW/2 ? T1_FAT : T1_WATER for f in freq_offsets]
const spin_params = (
    p_x = zeros(Float32, Nspins),
    p_y = zeros(Float32, Nspins),
    p_z = zeros(Float32, Nspins),
    p_ΔBz = ΔBz_h,
    p_T1 = spin_T1,
    p_T2 = fill(Float32(1e9), Nspins),
    p_ρ = ones(Float32, Nspins),
)
const E1_delay_h = Float32.(exp.(-T_DELAY ./ spin_params.p_T1))
const interp_h = build_interpolation_tables(n_ctrl, rf_idx)
const csr_h = build_csr_gather_tables(n_ctrl, interp_h.j_lo, interp_h.j_hi, interp_h.w0, interp_h.w1)

const fat_center = Float32(FAT_FREQ)
const fat_half_bw = Float32(FAT_BW) / 2

butterworth_target = Float32[
    1f0 - 1f0 / (1f0 + (abs(f - fat_center) / fat_half_bw)^(2 * BUTTERWORTH_DEGREE))
    for f in freq_offsets
]

function gaussian_pulse_target(α_deg, Δf_fat, Tpulse, freq_offs, scanner; with_delay=true)
    cutoff = abs(Δf_fat) / π
    a = sqrt(log(2) / 2) / cutoff
    τ = range(-Tpulse/2, Tpulse/2, 64)
    gauss = exp.(-(π .* τ ./ a) .^ 2)

    seq_g = Sequence()
    seq_g += RF(gauss, Tpulse, Δf_fat)
    α_ref = get_flip_angles(seq_g)[end]
    seq_g = (α_deg / α_ref + 0im) * seq_g

    T1_g = Float64[abs(f - Δf_fat) < abs(Δf_fat)/2 ? T1_FAT : T1_WATER for f in freq_offs]
    obj = Phantom(
        x = zeros(length(freq_offs)),
        y = zeros(length(freq_offs)),
        z = zeros(length(freq_offs)),
        ρ = ones(length(freq_offs)),
        T1 = T1_g,
        T2 = fill(1e9, length(freq_offs)),
        Δw = 2π .* freq_offs,
    )
    sp = KomaMRICore.default_sim_params()
    sp["Δt_rf"] = Tpulse / 256
    sp["Δt"] = Inf
    sp["return_type"] = "state"
    sp["sim_method"] = KomaMRICore.Bloch()

    state = @suppress simulate(obj, seq_g, scanner; sim_params=sp)
    Mz_raw = Float32.(real.(state.z))
    if with_delay
        E1 = Float32.(exp.(-T_DELAY ./ T1_g))
        return Mz_raw .* E1 .+ (1f0 .- E1)
    else
        return Mz_raw
    end
end

target_mz = if TARGET_MODE == :gaussian
    @info "Computing Gaussian FatSat target (flip=$(GAUSSIAN_FLIP_DEG)°)..."
    gaussian_pulse_target(GAUSSIAN_FLIP_DEG, FAT_FREQ, Trf, freq_offsets, sys)
else
    butterworth_target
end
@info "Target mode: $TARGET_MODE"

# Don't fit positive frequencies above the water-protect band.
mask_profile = Float32[(f > WATER_PROTECT) ? 0f0 : 1f0 for f in freq_offsets]
INVN = inv(sum(mask_profile))

function setup_backend(; spin, target_mz, E1_delay, with_delay::Bool)
    if BACKEND == :cuda
        ka_backend = CUDA.CUDABackend()

        bs = init_bloch_setup(;
            Nspins, n_ctrl, TL, rf_idx, spin,
            to_device=adapt_dev, backend=ka_backend, group_size=GROUP_SIZE)

        target_d = adapt_dev(target_mz)
        mask_d = adapt_dev(mask_profile)
        E1_d = adapt_dev(E1_delay)
        ρ_d = adapt_dev(spin.p_ρ)

        seed! = if with_delay
            (b) -> begin
                b.M_z .= b.M_z .* E1_d .+ ρ_d .* (1f0 .- E1_d)
                seed_and_loss_z_kernel!(b.backend, b.group_size)(
                    b.acc_loss_d, b.dM_z, b.M_z, target_d, mask_d, INVN;
                    ndrange=Int(b.Nspins))
                b.dM_z .= b.dM_z .* E1_d
            end
        else
            (b) -> seed_and_loss_z_kernel!(b.backend, b.group_size)(
                b.acc_loss_d, b.dM_z, b.M_z, target_d, mask_d, INVN;
                ndrange=Int(b.Nspins))
        end
        grad_fn! = (xr, xi) -> grad_and_loss!(bs, xr, xi, seed!)

        forward_fn = (xr, xi) -> begin
            Mxy, Mz = forward_sim!(bs, adapt_dev(xr), adapt_dev(xi))
            if with_delay
                Mz .= Mz .* E1_delay .+ spin.p_ρ .* (1f0 .- E1_delay)
            end
            return Mxy, Mz
        end

        # Warmup so compilation lands outside the timed region.
        try
            grad_fn!(adapt_dev(zeros(Float32, n_ctrl)), adapt_dev(zeros(Float32, n_ctrl)))
        catch e
            @warn "Warmup error: $e"
        end
        CUDA.synchronize()

        return (; grad_fn!, gx_buf=bs.gx_d, gi_buf=bs.gi_d,
                  ka_backend, group_size=GROUP_SIZE, forward_fn, adapt_dev)
    else
        E1_arg = with_delay ? E1_delay : nothing
        ctx = setup_reactant_backend(;
            Nspins, n_ctrl, TL, rf_idx, spin,
            target=target_mz, mask=mask_profile, invN=INVN,
            loss_mode=:mz, interp=interp_h, csr=csr_h, E1_delay=E1_arg)

        forward_fn = (xr, xi) -> begin
            Mxy, Mz = ctx.forward_fn(xr, xi)
            if with_delay
                Mz .= Mz .* E1_delay .+ spin.p_ρ .* (1f0 .- E1_delay)
            end
            return Mxy, Mz
        end

        return (; grad_fn! = ctx.grad_fn!, gx_buf=ctx.gx_buf, gi_buf=ctx.gi_buf,
                  ka_backend=ctx.ka_backend, group_size=ctx.group_size,
                  forward_fn, adapt_dev=identity)
    end
end

function optimize_pulse(ctx; label::String)
    x_r_init = 1f-10 .* randn(Float32, n_ctrl)
    x_i_init = 1f-10 .* randn(Float32, n_ctrl)
    x_r = BACKEND == :cuda ? ctx.adapt_dev(x_r_init) : copy(x_r_init)
    x_i = BACKEND == :cuda ? ctx.adapt_dev(x_i_init) : copy(x_i_init)

    t0 = time()
    (; loss_history, performed_iterations) = bb_optimize!(
        x_r, x_i, ctx.gx_buf, ctx.gi_buf, ctx.grad_fn!, MAX_ITERS, LAMBDA_MAX;
        backend=ctx.ka_backend, group_size=ctx.group_size,
        log_every=LOG_EVERY, grad_tol=Float64(GRAD_TOL))
    best = minimum(loss_history[1:performed_iterations])
    @info "$label optimization done in $(round(time()-t0, digits=2))s, best loss = $best"

    return ComplexF32.(Array(x_r) .+ im .* Array(x_i)), loss_history, performed_iterations
end

@info "Fat Saturation Pulse Optimization" backend=BACKEND fat_Hz=round(FAT_FREQ, digits=1) T1_fat_ms=T1_FAT*1e3 T1_water_ms=T1_WATER*1e3 delay_ms=T_DELAY*1e3 Trf_ms=Trf*1e3 λ=LAMBDA_MAX

ctx_t1 = setup_backend(; spin=spin_params, target_mz=target_mz,
                        E1_delay=E1_delay_h, with_delay=true)
x_opt, loss_history, n_iters = optimize_pulse(ctx_t1; label="T1-aware")

M_xy_opt, M_z_opt = ctx_t1.forward_fn(Float32.(real.(x_opt)), Float32.(imag.(x_opt)))
@info "Max |Mxy| after FatSat: $(maximum(abs.(M_xy_opt)))"

@info "Naive optimization (T1=∞, no post-pulse delay)"
naive_spin_params = merge(spin_params, (p_T1 = fill(Float32(1e9), Nspins),))
ctx_naive = setup_backend(; spin=naive_spin_params, target_mz=target_mz,
                           E1_delay=ones(Float32, Nspins), with_delay=false)
x_opt_naive, naive_loss_history, naive_n_iters = optimize_pulse(ctx_naive; label="Naive")

freq_hz = freq_offsets
t_ms = collect(range(0, Trf, n_ctrl)) .* 1e3

outdir = "pulses/fatsat"
mkpath(outdir)

fig = Figure(size=(800, 600))
ax1 = Axis(fig[1, 1], xlabel="Δf (Hz)", ylabel="Mz",
           title="Fat Saturation Profile ($(round(Int, T_DELAY*1e3)) ms post-pulse)")
lines!(ax1, freq_hz, target_mz, color=:black, linestyle=:dash, label="Target")
lines!(ax1, freq_hz, M_z_opt,   color=:blue,  label="Optimized")
vlines!(ax1, [fat_center - fat_half_bw, fat_center + fat_half_bw],
        color=:red, linestyle=:dot)
axislegend(ax1, position=:rt)

ax2 = Axis(fig[2, 1], xlabel="t (ms)", ylabel="B₁ (μT)", title="Optimized RF Pulse")
lines!(ax2, t_ms, real.(x_opt) .* 1e6, color=:blue, label="Real")
lines!(ax2, t_ms, imag.(x_opt) .* 1e6, color=:red,  label="Imag")
axislegend(ax2, position=:rt)

save(joinpath(outdir, "fatsat_figure.png"), fig, px_per_unit=4)
display(fig)
@info "Saved $(joinpath(outdir, "fatsat_figure.png"))"

# Comparison: Optimized (T1-aware) vs Naive vs Gaussian reference.
if GENERATE_COMPARISON_FIGURE
    @info "Generating comparison figure..."
    Nspins_comp = 1000

    gaussian_fatsat_ref(α_deg, Δf, Tpulse) = begin
        cutoff = abs(Δf) / π
        a = sqrt(log(2) / 2) / cutoff
        τ = range(-Tpulse/2, Tpulse/2, 64)
        gauss = exp.(-(π .* τ ./ a) .^ 2)
        s = Sequence()
        s += Grad(-8e-3, 3000e-6, 500e-6)
        s += RF(gauss, Tpulse, Δf)
        α_ref = get_flip_angles(s)[2]
        s = (α_deg / α_ref + 0im) * s
        s += Grad(8e-3, 3000e-6, 500e-6)
        return s
    end

    fatsat_with_spoilers(B1_opt, Tpulse) = begin
        s = Sequence()
        s += Grad(-8e-3, 3000e-6, 500e-6)
        s += RF(B1_opt, Tpulse, 0.0)
        s += Grad(8e-3, 3000e-6, 500e-6)
        return s
    end

    seq_gaussian = gaussian_fatsat_ref(GAUSSIAN_FLIP_DEG, FAT_FREQ, Trf)
    seq_optimized = fatsat_with_spoilers(x_opt, Trf)
    seq_naive = fatsat_with_spoilers(x_opt_naive, Trf)

    freq_comp = collect(range(-FREQ_MAX, FREQ_MAX, length=Nspins_comp))
    T1_comp = Float64[abs(f - FAT_FREQ) < FAT_BW/2 ? T1_FAT : T1_WATER for f in freq_comp]
    obj_comp = Phantom(
        x = zeros(Nspins_comp),
        y = zeros(Nspins_comp),
        z = zeros(Nspins_comp),
        ρ = ones(Nspins_comp),
        T1 = T1_comp,
        T2 = fill(1e9, Nspins_comp),
        Δw = 2π .* freq_comp,
    )
    E1_comp = Float32.(exp.(-T_DELAY ./ T1_comp))

    sp_comp = KomaMRICore.default_sim_params()
    sp_comp["Δt_rf"] = Trf / (4 * (n_ctrl - 1))
    sp_comp["Δt"] = 1e-3
    sp_comp["return_type"] = "state"
    sp_comp["sim_method"] = KomaMRICore.Bloch()

    apply_delay(Mz_raw) = Float32.(Mz_raw .* E1_comp .+ (1f0 .- E1_comp))

    @info "Simulating Gaussian FatSat..."
    Mz_gauss = apply_delay(real.(@suppress(simulate(obj_comp, seq_gaussian, sys; sim_params=sp_comp)).z))
    @info "Simulating Optimized FatSat (T1-aware)..."
    Mz_t1a = apply_delay(real.(@suppress(simulate(obj_comp, seq_optimized, sys; sim_params=sp_comp)).z))
    @info "Simulating Naive FatSat..."
    Mz_naive = apply_delay(real.(@suppress(simulate(obj_comp, seq_naive, sys; sim_params=sp_comp)).z))

    # Resample target_mz onto the (shorter, same-span) freq_comp grid via linear interp.
    n_src = length(freq_offsets)
    f0, f1 = Float64(first(freq_offsets)), Float64(last(freq_offsets))
    target_mz_comp = Float32.(map(freq_comp) do f
        t = clamp((Float64(f) - f0) / (f1 - f0), 0.0, 1.0) * (n_src - 1) + 1
        j = clamp(floor(Int, t), 1, n_src - 1)
        α = t - j
        (1 - α) * target_mz[j] + α * target_mz[j+1]
    end)

    fat_mask = (freq_comp .>= (FAT_FREQ - fat_half_bw)) .& (freq_comp .<= (FAT_FREQ + fat_half_bw))
    water_mask = (freq_comp .>= -100) .& (freq_comp .<= 100)

    @info "Mz at $(round(Int, T_DELAY*1e3)) ms post-pulse (T1 fat=$(T1_FAT*1e3)ms, water=$(T1_WATER*1e3)ms):"
    @info "Fat Mz (lower = better):"  Gaussian=round(mean(Mz_gauss[fat_mask]), sigdigits=3) Naive=round(mean(Mz_naive[fat_mask]), sigdigits=3) T1_Aware=round(mean(Mz_t1a[fat_mask]), sigdigits=3)
    @info "Water Mz (closer to 1 = better):" Gaussian=round(mean(Mz_gauss[water_mask]), sigdigits=3) Naive=round(mean(Mz_naive[water_mask]), sigdigits=3) T1_Aware=round(mean(Mz_t1a[water_mask]), sigdigits=3)

    B1_gaussian_env = seq_gaussian.RF[2].A
    t_gauss_s = collect(range(0, Trf, length(B1_gaussian_env)))
    B1_gauss = B1_gaussian_env .* exp.(im .* 2π .* FAT_FREQ .* t_gauss_s)
    t_gauss_ms = t_gauss_s .* 1e3

    fig_comp = Figure(size=(1200, 600))

    function mz_axis(pos, title, color, achieved; show_target=true)
        ax = Axis(fig_comp[1, pos], xlabel="Δf (Hz)", ylabel="Mz", title=title)
        if show_target
            lines!(ax, freq_comp, target_mz_comp, color=:black, linestyle=:dash, label="Target")
        end
        lines!(ax, freq_comp, achieved, color=color, label="Achieved")
        vlines!(ax, [fat_center - fat_half_bw, fat_center + fat_half_bw],
                color=:red, linestyle=:dot)
        ylims!(ax, -0.1, 1.1)
        return ax
    end

    mz_axis(1, "Gaussian Pulse", :gray, Mz_gauss; show_target=false)
    mz_axis(2, "Naively Optimized Pulse",   :orange, Mz_naive)
    ax_o1 = mz_axis(3, "T1-Aware Optimized Pulse", :blue, Mz_t1a)
    axislegend(ax_o1, position=:rt)

    max_amp = max(maximum(abs.(B1_gauss)),
                  maximum(abs.(x_opt)),
                  maximum(abs.(x_opt_naive))) * 1e6 * 1.1

    function rf_axis(pos, ts, B1)
        ax = Axis(fig_comp[2, pos], xlabel="t (ms)", ylabel="B₁ (μT)")
        lines!(ax, ts,  abs.(B1) .* 1e6, color=:gray, label="±|B₁|")
        lines!(ax, ts, -abs.(B1) .* 1e6, color=:gray)
        lines!(ax, ts, real.(B1) .* 1e6, color=:blue, label="Real")
        lines!(ax, ts, imag.(B1) .* 1e6, color=:red,  label="Imag")
        ylims!(ax, -max_amp, max_amp)
        return ax
    end

    rf_axis(1, t_gauss_ms, B1_gauss)
    rf_axis(2, t_ms,       x_opt_naive)
    ax_o2 = rf_axis(3, t_ms, x_opt)
    axislegend(ax_o2, position=:rt)

    save(joinpath(outdir, "fatsat_comparison.png"), fig_comp, px_per_unit=4)
    display(fig_comp)
    @info "Saved $(joinpath(outdir, "fatsat_comparison.png"))"
end

Δt_rf = 10e-6
function save_pulse_jld2(path, x_opt_to_save; rf_block_idx=1)
    seq_block = seq[rf_block_idx]
    T = (dur(seq_block) ÷ Δt_rf) * Δt_rf
    seq_block.RF[1].A = x_opt_to_save
    t_grid = collect(range(0, T, step=Δt_rf))
    B1 = KomaMRIBase.get_rfs(seq_block, t_grid)[1]
    Grads = KomaMRIBase.get_grads(seq_block, t_grid)
    jldsave(path; B1=B1, Gx=Grads[1], Gy=Grads[2], T=T, seq=seq_block)
    @info "Saved $path"
end

save_pulse_jld2(joinpath(outdir, "fatsat_opt.jld2"), x_opt)
save_pulse_jld2(joinpath(outdir, "no_fatsat.jld2"),  x_opt_naive)

if GENERATE_COMPARISON_FIGURE
    # Resample the Gaussian reference pulse onto the standard 10 µs grid.
    seq_g_rf = seq_gaussian[2] # block 1 = spoiler, 2 = RF, 3 = spoiler
    T_g = (dur(seq_g_rf) ÷ Δt_rf) * Δt_rf
    t_grid_g = collect(range(0, T_g, step=Δt_rf))
    B1_g = KomaMRIBase.get_rfs(seq_g_rf, t_grid_g)[1]
    B1_g .*= exp.(-im .* 2π .* FAT_FREQ .* t_grid_g)
    Grads_g = KomaMRIBase.get_grads(seq_g_rf, t_grid_g)
    jldsave(joinpath(outdir, "fatsat_gaussian.jld2");
            B1=B1_g, Gx=Grads_g[1], Gy=Grads_g[2], T=T_g, seq=seq_g_rf)
    @info "Saved $(joinpath(outdir, "fatsat_gaussian.jld2"))"
end

include("pulseq_utils/preppulse_bSSFP.jl")
generate_bssfp_seq(joinpath(outdir, "fatsat_opt.jld2"), joinpath(outdir, "fatsat_opt.seq"))
generate_bssfp_seq(joinpath(outdir, "fatsat_opt.jld2"), joinpath(outdir, "no_fatsat.seq"); disable_prep=true)
if GENERATE_COMPARISON_FIGURE
    generate_bssfp_seq(joinpath(outdir, "fatsat_gaussian.jld2"), joinpath(outdir, "fatsat_gaussian.seq"))
end
