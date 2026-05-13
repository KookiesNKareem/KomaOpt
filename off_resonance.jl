# 2-D RF excitation with B0 inhomogeneity.
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

using CairoMakie, LaTeXStrings
using JLD2, FFTW, DICOM
import Images
using Statistics: mean
using Interpolations

const γ64 = γ64_rad
const γ = γ_Hz

# B0 source
const B0_SOURCE = :gaussian # :gaussian, :dicom, or :hz_file
const DICOM_B0_PATH = "data/b0maps/in_vivo_phase.dcm"

const B0_PEAK_HZ = 80.0
const B0_SIGMA_M = nothing

const HZ_FILE_PATH = "data/b0maps/b0_hz.jld2"
const HZ_FILE_KEY = "Δf_hz"
const HZ_FILE_FOV = 250e-3
const HZ_FILE_SCALE = 1.0

const DICOM_DELTA_TE = 5e-3
const DICOM_B0_TESLA = 3.0
const DICOM_SCALE_FACTOR = 1.0
const DICOM_SMOOTH_SIGMA = 0.0
const DICOM_MAG_PATH = "data/b0maps/in_vivo_mag.dcm"
const DICOM_MAG_THRESHOLD = 0.1
const DICOM_UNWRAP_PHASE = :auto

# Optimization
const N_ITERS = 20
const LAMBDA_0 = 2f-8
const LOG_EVERY = 10
const B1_MAX = 12.5f-6 # Tesla
const LAMBDA_AMP = 1f8 # B1 amplitude penalty

function read_dicom_fov_meters(path::String)
    dcm = dcm_parse(path)
    rows = Int(dcm[tag"Rows"])
    cols = Int(dcm[tag"Columns"])
    pixel_spacing = nothing
    if haskey(dcm, (0x5200, 0x9230))
        seq_pf = dcm[(0x5200, 0x9230)]
        if !isempty(seq_pf) && haskey(seq_pf[1], (0x0028, 0x9110))
            pm = seq_pf[1][(0x0028, 0x9110)][1]
            pixel_spacing = get(pm, (0x0028, 0x0030), nothing)
        end
    end
    pixel_spacing = something(pixel_spacing, get(dcm, tag"PixelSpacing", [1.0, 1.0]))
    fov_x = cols * pixel_spacing[2] * 1e-3
    fov_y = rows * pixel_spacing[1] * 1e-3
    return max(fov_x, fov_y)
end

# 2D phase unwrap via DCT Poisson solver (Neumann BC).
function unwrap_phase_2d(wrapped::AbstractMatrix{<:AbstractFloat})
    M, N = size(wrapped)
    T = eltype(wrapped)
    wrap_diff(a, b) = mod(a - b + T(π), T(2π)) - T(π)
    dx = zeros(T, M, N); dy = zeros(T, M, N)
    @inbounds for i in 1:M, j in 1:N-1
        dx[i,j] = wrap_diff(wrapped[i,j+1], wrapped[i,j])
    end
    @inbounds for i in 1:M-1, j in 1:N
        dy[i,j] = wrap_diff(wrapped[i+1,j], wrapped[i,j])
    end
    ρ = zeros(T, M, N)
    @inbounds for i in 1:M, j in 1:N
        ρ[i,j] = (j < N ? dx[i,j] : zero(T)) - (j > 1 ? dx[i,j-1] : zero(T)) +
                 (i < M ? dy[i,j] : zero(T)) - (i > 1 ? dy[i-1,j] : zero(T))
    end
    ρ_dct = FFTW.r2r(ρ, FFTW.REDFT10)
    denom = [2.0 * (cos(π * i / M) + cos(π * j / N) - 2.0)
             for i in 0:M-1, j in 0:N-1]
    denom[1, 1] = 1.0
    φ_dct = ρ_dct ./ denom
    φ_dct[1, 1] = 0.0
    φ = FFTW.r2r(φ_dct, FFTW.REDFT01) ./ (4 * M * N)
    φ .+= wrapped[1, 1] - φ[1, 1]
    return φ
end

function phase_has_wraps(phase_2d::AbstractMatrix; threshold=0.9π)
    dx = abs.(diff(phase_2d, dims=2))
    dy = abs.(diff(phase_2d, dims=1))
    n_wraps = count(dx .> threshold) + count(dy .> threshold)
    return n_wraps > 0, n_wraps
end

function load_b0_from_dicom(dicom_path::String, sim_xs, sim_ys;
                            delta_te::Float64=5e-3,
                            B0_tesla::Float64=3.0,
                            scale_factor::Float64=1.0,
                            smooth_sigma::Float64=0.0,
                            mag_path::Union{String,Nothing}=nothing,
                            mag_threshold::Float64=0.1,
                            dicom_fov::Union{Float64,Nothing}=nothing,
                            unwrap_phase::Union{Symbol,Bool}=false)
    dcm = dcm_parse(dicom_path)
    b0_raw = vec(Float64.(dcm[tag"PixelData"]))
    rows = Int(dcm[tag"Rows"]); cols = Int(dcm[tag"Columns"])

    phase_rad = (Float64.(b0_raw) ./ 4095.0) .* 2π .- π
    phase_2d = reshape(phase_rad, rows, cols)

    mag_mask_2d = trues(rows, cols)
    if !isnothing(mag_path) && isfile(mag_path)
        mag_dcm = dcm_parse(mag_path)
        mag_rows = Int(mag_dcm[tag"Rows"]); mag_cols = Int(mag_dcm[tag"Columns"])
        mag_2d = reshape(Float64.(vec(mag_dcm[tag"PixelData"])), mag_rows, mag_cols)
        if (mag_rows, mag_cols) != (rows, cols)
            mag_2d = Float64.(Images.imresize(mag_2d, (rows, cols)))
        end
        mag_mask_2d = (mag_2d ./ maximum(mag_2d)) .>= mag_threshold
        phase_2d[.!mag_mask_2d] .= 0.0
    elseif !isnothing(mag_path)
        @warn "Magnitude file not found, skipping thresholding" path=mag_path
    end

    if unwrap_phase == :auto || unwrap_phase == true
        has_wraps, _ = phase_has_wraps(phase_2d)
        if unwrap_phase == true || has_wraps
            phase_2d = unwrap_phase_2d(phase_2d)
            phase_2d[.!mag_mask_2d] .= 0.0
        end
    end

    Δf_hz_dcm = vec(phase_2d) ./ (2π * delta_te)
    Δf_hz_dcm .*= scale_factor
    b0_2d = reshape(Δf_hz_dcm, rows, cols)

    if smooth_sigma > 0
        kx = fftfreq(cols); ky = fftfreq(rows)
        k2 = ky .^ 2 .+ kx' .^ 2
        gaussian_filter = exp.(-2π^2 * smooth_sigma^2 .* k2)
        b0_2d = real.(ifft(fft(b0_2d) .* gaussian_filter))
    end
    b0_2d = reverse(b0_2d, dims=1)

    if !isnothing(dicom_fov)
        dcm_xs = collect(range(-dicom_fov/2, dicom_fov/2, length=cols))
        dcm_ys = collect(range(-dicom_fov/2, dicom_fov/2, length=rows))
    else
        pixel_spacing = get(dcm, tag"PixelSpacing", [1.0, 1.0])
        dcm_xs = collect(range(0, step=pixel_spacing[2]*1e-3, length=cols))
        dcm_ys = collect(range(0, step=pixel_spacing[1]*1e-3, length=rows))
        dcm_xs .-= mean(dcm_xs); dcm_ys .-= mean(dcm_ys)
    end

    itp = extrapolate(interpolate((dcm_ys, dcm_xs), b0_2d, Gridded(Linear())), 0.0)
    Δf_hz = Float32.(itp.(sim_ys, sim_xs))
    return Δf_hz ./ Float32(γ_Hz), Δf_hz
end

function load_b0_from_hz_file(file_path::String, sim_xs, sim_ys;
                              hz_fov::Float64=200e-3,
                              scale_factor::Float64=1.0,
                              jld2_key::String="Δf_hz")
    ext = lowercase(splitext(file_path)[2])
    Δf_raw = if ext == ".jld2"
        data = JLD2.load(file_path)
        haskey(data, jld2_key) || error("Key '$jld2_key' not found in $file_path. Keys: $(collect(keys(data)))")
        Float64.(data[jld2_key])
    elseif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        img = Images.load(file_path)
        img_gray = eltype(img) <: Images.AbstractRGB ? Images.Gray.(img) : img
        Float64.(img_gray)
    else
        error("Unsupported file format: $ext. Use .jld2 or an image file")
    end
    Δf_raw .*= scale_factor
    rows, cols = size(Δf_raw)
    hz_xs = collect(range(-hz_fov/2, hz_fov/2, length=cols))
    hz_ys = collect(range(-hz_fov/2, hz_fov/2, length=rows))
    itp = extrapolate(interpolate((hz_ys, hz_xs), Δf_raw, Gridded(Linear())), 0.0)
    Δf_hz = Float32.(itp.(sim_ys, sim_xs))
    return Δf_hz ./ Float32(γ_Hz), Δf_hz
end

function create_b0_map(xgrid, ygrid; peak_hz=100.0, sigma=nothing, FOV_sim=0.2)
    σ = isnothing(sigma) ? FOV_sim / 6 : sigma
    Δf = @. peak_hz * exp(-(xgrid^2 + ygrid^2) / (2 * σ^2))
    return Float32.(Δf ./ γ_Hz), Float32.(Δf)
end

FOV_sim = if B0_SOURCE == :dicom
    fov = read_dicom_fov_meters(DICOM_B0_PATH)
    @info "DICOM FOV" path=DICOM_B0_PATH FOV_m=fov FOV_mm=round(fov*1e3, digits=2)
    fov
elseif B0_SOURCE == :hz_file
    HZ_FILE_FOV
elseif B0_SOURCE == :gaussian
    200e-3
else
    error("Unknown B0_SOURCE: $B0_SOURCE")
end

Nrf = 350
seq, Trf = build_spiral_sequence(; Nrf)
seq.GR[1].A = seq.GR[1].A ./ 2
seq.GR[2].A = seq.GR[2].A ./ 2

Nspins_x = 80; Nspins_y = 80
xs = range(-FOV_sim/2, FOV_sim/2, Nspins_x)
ys = range(-FOV_sim/2, FOV_sim/2, Nspins_y)
xgrid = Float32.(vec(repeat(collect(xs),  1, Nspins_y)))
ygrid = Float32.(vec(repeat(collect(ys)', Nspins_x, 1)))
const Nspins = Nspins_x * Nspins_y

sim_params = KomaMRICore.default_sim_params()
sim_params["sim_method"] = KomaMRICore.Bloch()
sim_params["Δt_rf"] = (Trf / (Nrf - 1)) / 2
sim_params["Δt"] = Inf
sim_params["return_type"] = "state"

mask = [sqrt(x^2 + y^2) <= FOV_sim/2.2 for x in xs, y in ys][:]

const TL = discretize_timeline(seq, sim_params)
const rf_idx = TL.rf_active_idx
const n_ctrl = length(seq.RF[1].A)
const interp_h = build_interpolation_tables(n_ctrl, rf_idx)
const csr_h = build_csr_gather_tables(n_ctrl, interp_h.j_lo, interp_h.j_hi, interp_h.w0, interp_h.w1)

make_spin_params(ΔBz) = (
    p_x = xgrid, p_y = ygrid, p_z = zeros(Float32, Nspins),
    p_ΔBz = Float32.(ΔBz),
    p_T1 = fill(Float32(1e9), Nspins),
    p_T2 = fill(Float32(1e9), Nspins),
    p_ρ = ones(Float32, Nspins),
)

img_path = "target_images/stanford_logo.png"
target_profile, mag_target_2d = load_image_target(img_path, Nspins_x, Nspins_y)

INVN = inv(Float32(Nspins))
mask_f32 = Float32.(mask)

# Soft B1 penalty: λ * Σ max(|x| - B1max, 0)^2
function add_b1_amp_penalty!(gx, gi, x_r, x_i, B1_max::Float32, λ::Float32)
    amp = sqrt.(x_r .^ 2 .+ x_i .^ 2) .+ 1f-20
    excess = max.(amp .- B1_max, 0f0)
    coeff = 2f0 * λ
    gx .+= coeff .* excess .* x_r ./ amp
    gi .+= coeff .* excess .* x_i ./ amp
    return Float64(λ * sum(excess .^ 2))
end

function run_b0_optimization(ΔBz_vec, x0, Niters, λ0; log_every=0)
    spin = make_spin_params(ΔBz_vec)
    if BACKEND == :cuda
        ka_backend = CUDA.CUDABackend()
        group_size = 256

        bs = init_bloch_setup(;
            Nspins, n_ctrl, TL, rf_idx, spin,
            to_device=adapt_dev, backend=ka_backend, group_size)
        fill!(bs.s_Δf, 0.0f0)

        target_d = adapt_dev(ComplexF32.(target_profile))
        mask_d = adapt_dev(mask_f32)

        seed_fn! = (b) -> seed_and_loss_kernel!(b.backend, b.group_size)(
            b.acc_loss_d, b.dM_xy, b.M_xy, target_d, mask_d, INVN; ndrange=Int(b.Nspins))
        grad_fn! = (xr, xi) -> grad_and_loss!(bs, xr, xi, seed_fn!)
        gx_buf, gi_buf = bs.gx_d, bs.gi_d

        forward_fn = (xr, xi) -> begin
            Mxy, Mz = forward_sim!(bs, adapt_dev(xr), adapt_dev(xi))
            return ComplexF32.(Mxy), Mz
        end

        x_r = adapt_dev(Float32.(real.(x0)))
        x_i = adapt_dev(Float32.(imag.(x0)))
    else
        ctx = setup_reactant_backend(;
            Nspins, n_ctrl, TL, rf_idx, spin,
            target=target_profile, mask=mask_f32, invN=INVN,
            loss_mode=:mxy, interp=interp_h, csr=csr_h)

        ka_backend = ctx.ka_backend
        group_size = ctx.group_size
        grad_fn! = ctx.grad_fn!
        gx_buf = ctx.gx_buf
        gi_buf = ctx.gi_buf
        forward_fn = ctx.forward_fn

        x_r = Float32.(real.(x0))
        x_i = Float32.(imag.(x0))
    end

    grad_fn_with_penalty! = (xr, xi) -> begin
        loss = grad_fn!(xr, xi)
        loss += add_b1_amp_penalty!(gx_buf, gi_buf, xr, xi, B1_MAX, LAMBDA_AMP)
        return loss
    end

    (; loss_history, performed_iterations) = bb_optimize!(
        x_r, x_i, gx_buf, gi_buf, grad_fn_with_penalty!, Niters, λ0;
        backend=ka_backend, group_size, log_every)

    x_opt = ComplexF32.(Array(x_r) .+ im .* Array(x_i))
    return x_opt, loss_history, performed_iterations, forward_fn
end

function load_b0()
    if B0_SOURCE == :dicom
        return load_b0_from_dicom(DICOM_B0_PATH, xgrid, ygrid;
                                  delta_te=DICOM_DELTA_TE, B0_tesla=DICOM_B0_TESLA,
                                  scale_factor=DICOM_SCALE_FACTOR,
                                  smooth_sigma=DICOM_SMOOTH_SIGMA,
                                  mag_path=DICOM_MAG_PATH,
                                  mag_threshold=DICOM_MAG_THRESHOLD,
                                  dicom_fov=FOV_sim,
                                  unwrap_phase=DICOM_UNWRAP_PHASE)
    elseif B0_SOURCE == :gaussian
        σ = isnothing(B0_SIGMA_M) ? FOV_sim / 6 : B0_SIGMA_M
        return create_b0_map(xgrid, ygrid; peak_hz=B0_PEAK_HZ, sigma=σ, FOV_sim=FOV_sim)
    elseif B0_SOURCE == :hz_file
        return load_b0_from_hz_file(HZ_FILE_PATH, xgrid, ygrid;
                                    hz_fov=HZ_FILE_FOV,
                                    scale_factor=HZ_FILE_SCALE,
                                    jld2_key=HZ_FILE_KEY)
    else
        error("Unknown B0_SOURCE: $B0_SOURCE")
    end
end

outdir = "pulses/off_resonance"
mkpath(outdir)

x0 = 1f-7 .* randn(ComplexF32, Nrf)
Δt_rf = 10e-6
T_seq = (dur(seq) ÷ Δt_rf) * Δt_rf
t_grid = collect(range(0, T_seq, step=Δt_rf))
t_ms = collect(range(0, Trf, Nrf)) .* 1e3

ΔBz_h, Δf_hz_h = load_b0()
@info "B0 range" min_hz=round(minimum(Δf_hz_h), digits=1) max_hz=round(maximum(Δf_hz_h), digits=1)

@info "Optimization WITHOUT B0 (off-resonance = 0)"
t0 = time()
x_opt_no_b0, loss_history_no_b0, n_iters_no_b0, _ = run_b0_optimization(
    zeros(Float32, Nspins), x0, N_ITERS, LAMBDA_0; log_every=LOG_EVERY)
@info "Done in $(round(time()-t0, digits=2))s, loss: $(loss_history_no_b0[n_iters_no_b0])"

seq.RF[1].A = x_opt_no_b0
B1_no_b0 = KomaMRIBase.get_rfs(seq, t_grid)[1]
Grads_no_b0 = KomaMRIBase.get_grads(seq, t_grid)
jldsave(joinpath(outdir, "b0_ignorant.jld2");
        B1=B1_no_b0, Gx=Grads_no_b0[1], Gy=Grads_no_b0[2], T=T_seq, seq)

@info "Optimization WITH B0"
t0 = time()
x_opt, loss_history, n_iters, fwd_b0 = run_b0_optimization(
    ΔBz_h, copy(x0), N_ITERS, LAMBDA_0; log_every=LOG_EVERY)
@info "Done in $(round(time()-t0, digits=2))s, loss: $(loss_history[n_iters])"

# Forward-evaluate both pulses against the true B0 field.
result_no_b0_with_b0, _ = fwd_b0(Float32.(real.(x_opt_no_b0)), Float32.(imag.(x_opt_no_b0)))
achieved_no_b0_with_b0 = abs.(reshape(result_no_b0_with_b0, Nspins_x, Nspins_y))

result_b0, _ = fwd_b0(Float32.(real.(x_opt)), Float32.(imag.(x_opt)))
achieved_2d = abs.(reshape(result_b0, Nspins_x, Nspins_y))

seq.RF[1].A = x_opt
B1_with_b0 = KomaMRIBase.get_rfs(seq, t_grid)[1]
Grads_with_b0 = KomaMRIBase.get_grads(seq, t_grid)
jldsave(joinpath(outdir, "b0_aware.jld2");
        B1=B1_with_b0, Gx=Grads_with_b0[1], Gy=Grads_with_b0[2], T=T_seq, seq)

mask_2d = reshape(mask, Nspins_x, Nspins_y)
achieved_no_b0_with_b0[mask_2d .== 0] .= 0
achieved_2d[mask_2d .== 0] .= 0
b0_map_2d = reshape(Δf_hz_h, Nspins_x, Nspins_y)

fig = Figure(size=(1400, 900))

ax1 = Axis(fig[1, 1], xlabel=L"$x$ (cm)", ylabel=L"$y$ (cm)", title="Target |Mₓᵧ|", aspect=1)
hm1 = heatmap!(ax1, collect(xs) * 1e2, collect(ys) * 1e2, abs.(mag_target_2d); colormap=:grays)
Colorbar(fig[1, 2], hm1)

ax2 = Axis(fig[1, 3], xlabel=L"$x$ (cm)", ylabel=L"$y$ (cm)", title="B0 Map (Hz)", aspect=1)
hm2 = heatmap!(ax2, collect(xs) * 1e2, collect(ys) * 1e2, b0_map_2d; colormap=:RdBu)
Colorbar(fig[1, 4], hm2, label="Δf (Hz)")

ax3 = Axis(fig[2, 1], xlabel=L"$x$ (cm)", ylabel=L"$y$ (cm)", title="No B0 correction", aspect=1)
hm3 = heatmap!(ax3, collect(xs) * 1e2, collect(ys) * 1e2, achieved_no_b0_with_b0; colormap=:grays)
Colorbar(fig[2, 2], hm3)

ax4 = Axis(fig[2, 3], xlabel=L"$x$ (cm)", ylabel=L"$y$ (cm)", title="B0-aware optimization", aspect=1)
hm4 = heatmap!(ax4, collect(xs) * 1e2, collect(ys) * 1e2, achieved_2d; colormap=:grays)
Colorbar(fig[2, 4], hm4)

ax5 = Axis(fig[3, 1:2], xlabel="Time (ms)", ylabel="B1 (µT)", title="RF Pulse (no B0 correction)")
lines!(ax5, t_ms, real(x_opt_no_b0) .* 1e6, color=:blue, label="Real")
lines!(ax5, t_ms, imag(x_opt_no_b0) .* 1e6, color=:red,  label="Imag")
axislegend(ax5, position=:rt)

ax6 = Axis(fig[3, 3:4], xlabel="Time (ms)", ylabel="B1 (µT)", title="RF Pulse (B0-aware)")
lines!(ax6, t_ms, real(x_opt) .* 1e6, color=:blue, label="Real")
lines!(ax6, t_ms, imag(x_opt) .* 1e6, color=:red,  label="Imag")
axislegend(ax6, position=:rt)

cl_hi = max(maximum(abs.(mag_target_2d)),
            maximum(achieved_2d),
            maximum(achieved_no_b0_with_b0))
hm1.colorrange[] = (0.0, cl_hi)
hm3.colorrange[] = (0.0, cl_hi)
hm4.colorrange[] = (0.0, cl_hi)

fig_path = joinpath(outdir, "results.png")
save(fig_path, fig, px_per_unit=3)
display(fig)
@info "Saved $fig_path"

include("pulseq_utils/logo_gre.jl")
generate_logo_gre_seq(joinpath(outdir, "b0_ignorant.jld2"), joinpath(outdir, "b0_ignorant.seq"))
generate_logo_gre_seq(joinpath(outdir, "b0_aware.jld2"),    joinpath(outdir, "b0_aware.seq"))
