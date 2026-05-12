# utils.jl — RF optimization utilities (Reactant CPU / GPU backend)

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const, @uniform, @groupsize
using KomaMRICore, KomaMRIBase
using Reactant
using ReactantCore: @trace
import Enzyme
using Enzyme: gradient, ReverseWithPrimal, Const
using FFTW, FileIO
import ImageTransformations

Reactant.allowscalar(true)

# Physical constants
const γ_Hz = Float32(42.57747892e6) # Hz/T
const γ64_Hz = 42.57747892e6 # Hz/T (Float64)
const γ_rad = Float32(2π * 42.57747892e6) # rad/(s·T)
const γ64_rad = 2π * 42.57747892e6 # rad/(s·T) (Float64)

# Interpolation between control points and the RF timeline

# n_ctrl control points → rf_active timeline points
function build_interpolation_tables(n_ctrl::Int, rf_idx::Vector{Int})
    n_taps = length(rf_idx)
    ctrl_pos = collect(range(0.0, 1.0, length=n_ctrl))
    tap_pos = collect(range(0.0, 1.0, length=n_taps))
    j_lo = Vector{Int}(undef, n_taps)
    j_hi = Vector{Int}(undef, n_taps)
    w0 = Vector{Float64}(undef, n_taps)
    w1 = Vector{Float64}(undef, n_taps)
    idx_rf = Vector{Int}(undef, n_taps)
    j = 1
    @inbounds for (k, idx) in enumerate(rf_idx)
        t = tap_pos[k]
        while j < n_ctrl && ctrl_pos[j+1] < t; j += 1; end
        if j == n_ctrl
            j_lo[k] = n_ctrl; j_hi[k] = n_ctrl
            w0[k] = 1.0; w1[k] = 0.0
        else
            α = (t - ctrl_pos[j]) / (ctrl_pos[j+1] - ctrl_pos[j] + 1e-30)
            j_lo[k] = j; j_hi[k] = j + 1
            w0[k] = 1.0 - α; w1[k] = α
        end
        idx_rf[k] = idx
    end
    return (j_lo=j_lo, j_hi=j_hi, w0=w0, w1=w1, idx_rf=idx_rf)
end

# CSR gather: adjoint of interpolation (timeline grads → control grads)
function build_csr_gather_tables(n_ctrl::Int, j_lo, j_hi, w0, w1)
    counts = zeros(Int, n_ctrl)
    @inbounds for k in eachindex(j_lo)
        counts[j_lo[k]] += 1; counts[j_hi[k]] += 1
    end
    csr_ptr = Vector{Int}(undef, n_ctrl + 1)
    csr_ptr[1] = 1
    @inbounds for j in 1:n_ctrl; csr_ptr[j+1] = csr_ptr[j] + counts[j]; end
    nnz = csr_ptr[end] - 1
    csr_idx = Vector{Int}(undef, nnz)
    csr_w = Vector{Float64}(undef, nnz)
    fill!(counts, 0)
    @inbounds for k in eachindex(j_lo)
        j0 = j_lo[k]; j1 = j_hi[k]
        p0 = csr_ptr[j0] + counts[j0]; counts[j0] += 1
        csr_idx[p0] = k; csr_w[p0] = w0[k]
        p1 = csr_ptr[j1] + counts[j1]; counts[j1] += 1
        csr_idx[p1] = k; csr_w[p1] = w1[k]
    end
    return (ptr=csr_ptr, idx=csr_idx, w=csr_w)
end

# Control → timeline B1 mapping (outside AD), complex output. Used by simple_cpu.jl
function cpu_map_ctrl_to_B1!(B1, x_r, x_i, idx_rf, jlo, jhi, w0, w1)
    fill!(B1, zero(eltype(B1)))
    @inbounds for k in eachindex(idx_rf)
        idx = idx_rf[k]; j0 = jlo[k]; j1 = jhi[k]
        vr = w0[k]*x_r[j0] + w1[k]*x_r[j1]
        vi = w0[k]*x_i[j0] + w1[k]*x_i[j1]
        B1[idx] = complex(vr, vi)
    end
end

# Timeline grads → control grads (outside AD), complex input. Used by simple_cpu.jl
function cpu_gather_grads!(gx, gi, dB1, idx_rf, ptr, csr_idx, csr_w)
    fill!(gx, 0.0); fill!(gi, 0.0)
    @inbounds for j in eachindex(gx)
        s_r = 0.0; s_i = 0.0
        for p in ptr[j]:(ptr[j+1]-1)
            k = csr_idx[p]; tid = idx_rf[k]; w = csr_w[p]
            s_r += w * real(dB1[tid]); s_i += w * imag(dB1[tid])
        end
        gx[j] = s_r; gi[j] = s_i
    end
end

# Split real/imag versions used by the Reactant backend (Float32 arrays + interp NamedTuple)
function map_ctrl_to_timeline_cpu!(B1_r_tl, B1_i_tl, x_r, x_i, interp)
    fill!(B1_r_tl, 0f0)
    fill!(B1_i_tl, 0f0)
    @inbounds for k in eachindex(interp.idx_rf)
        idx = Int(interp.idx_rf[k])
        j0 = Int(interp.j_lo[k])
        j1 = Int(interp.j_hi[k])
        w0 = Float32(interp.w0[k])
        w1 = Float32(interp.w1[k])
        B1_r_tl[idx] = w0 * x_r[j0] + w1 * x_r[j1]
        B1_i_tl[idx] = w0 * x_i[j0] + w1 * x_i[j1]
    end
    return nothing
end

function gather_grads_cpu!(gx, gi, dB1_r, dB1_i, interp, csr)
    fill!(gx, 0f0)
    fill!(gi, 0f0)
    @inbounds for j in eachindex(gx)
        s_r = 0f0
        s_i = 0f0
        for p in Int(csr.ptr[j]):(Int(csr.ptr[j+1]) - 1)
            k = Int(csr.idx[p])
            tid = Int(interp.idx_rf[k])
            w = Float32(csr.w[p])
            s_r += w * dB1_r[tid]
            s_i += w * dB1_i[tid]
        end
        gx[j] = s_r
        gi[j] = s_i
    end
    return nothing
end

# ============================================================================
# Sequence + target image utilities
# ============================================================================

function build_spiral_sequence(; FOV=1000e-3, N=60, Nrf=350, Smax=150.0, Gmax=100e-3)
    sys = Scanner()
    sys.Smax = Smax
    sys.Gmax = Gmax
    seq = PulseDesigner.spiral_base(FOV, N, sys; Nint=1)(0)

    # Reverse gradient waveforms (excitation k-space traversal)
    x, y = 1, 2
    seq.GR[x].A = reverse(seq.GR[x].A)
    seq.GR[y].A = reverse(seq.GR[y].A)
    seq.GR[x].rise, seq.GR[x].fall = seq.GR[x].fall, seq.GR[x].rise
    seq.GR[y].rise, seq.GR[y].fall = seq.GR[y].fall, seq.GR[y].rise
    seq.GR[x].delay = max(dur(seq.GR[x]), dur(seq.GR[y])) - dur(seq.GR[x])
    seq.GR[y].delay = max(dur(seq.GR[x]), dur(seq.GR[y])) - dur(seq.GR[y])

    # Rebuild the block with a vector RF pulse (placeholder, overwritten by optimization)
    B1 = ComplexF32.(1f-6 .* ones(Float32, Nrf))
    rf_delay = max(dur(seq.GR[x]), dur(seq.GR[y])) - dur(seq.ADC[1])
    Trf = dur(seq.ADC[1]) - seq.ADC[1].delay
    new_rf = RF(B1, Trf, 0.0, rf_delay)

    new_seq = Sequence()
    @addblock new_seq += (new_rf; x=seq.GR[x], y=seq.GR[y], z=seq.GR[3])

    return new_seq, Trf
end

function load_image_target(img_path, Nspins_x, Nspins_y; radius_px=20, scale=0.5f0)
    img = load(img_path)
    img_bw = reverse(getproperty.(img', :b) .* 1.0, dims=2)
    img_bw .= img_bw ./ maximum(img_bw)

    # Gaussian-windowed low-pass filter in frequency domain
    cx, cy = size(img_bw) .÷ 2 .+ 1
    ft_mask = [(sqrt((i - cx)^2 + (j - cy)^2) <= radius_px) *
               exp(-π * ((i - cx)^2 + (j - cy)^2) / (2 * radius_px^2))
              for i in 1:size(img_bw, 1), j in 1:size(img_bw, 2)]
    img_bw_lowpass = abs.(fftshift(ifft(fftshift(fft(fftshift(img_bw))) .* ft_mask)))

    img_resized = ImageTransformations.imresize(img_bw_lowpass, (Nspins_x, Nspins_y))
    target_profile = complex.(zeros(Float32, length(img_resized)),
                              scale .* Float32.(img_resized[:]))
    mag_target_2d = reshape(target_profile, Nspins_x, Nspins_y)

    return target_profile, mag_target_2d
end

# ============================================================================
# Timeline discretization
# ============================================================================

function discretize_timeline(seq, sim_params; rf_threshold=1f-10)
    seqd = KomaMRICore.discretize(seq; sampling_params=sim_params)
    Nt = length(seqd.Δt)

    Δt_raw = Float32.(seqd.Δt[1:Nt])
    @inbounds for i in eachindex(Δt_raw)
        if !isfinite(Δt_raw[i])
            Δt_raw[i] = 0.0f0
        end
    end

    B1 = ComplexF32.(seqd.B1[1:Nt])

    return (
        B1 = B1,
        Gx = Float32.(seqd.Gx[1:Nt]),
        Gy = Float32.(seqd.Gy[1:Nt]),
        Gz = Float32.(seqd.Gz[1:Nt]),
        Δt = Δt_raw,
        Δf = Float32.(seqd.Δf[1:Nt]),
        t  = Float32.(seqd.t[1:Nt]),
        Nt32 = Int32(Nt),
        rf_active_idx = findall(b1 -> abs(b1) > rf_threshold, B1),
    )
end

# ============================================================================
# Reactant-traced Bloch forward + Enzyme reverse for Mxy loss
# ============================================================================
#
# Uses split real/imag Float32 timelines for B1. The "natural" complex form
# (single ComplexF32 array) breaks through Reactant: Enzyme-via-XLA does NOT
# propagate complex gradients with the same convention as direct Enzyme, and
# the imaginary-part gradient comes back wrong (effectively zero or wrong-sign
# from a Wirtinger viewpoint). Symptom: Re(B1) blows up to nonsense magnitudes
# while Im(B1) stays near zero. Differentiating two real arrays avoids it.

function bloch_forward_reactant!(
    M_xy_r, M_xy_i, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf,
    s_B1_r, s_B1_i,
    ::Val{N_Δt}
) where {N_Δt}
    neg_pi_gamma_local = Float32(-π * γ_Hz)
    inv_γ_rad          = 1.0f0 / Float32(γ_rad)

    @trace for s_idx in 1:N_Δt
        Bz = @. p_x * s_Gx[s_idx] + p_y * s_Gy[s_idx] + p_z * s_Gz[s_idx] +
                p_ΔBz - s_Δf[s_idx] * inv_γ_rad
        B1_r = s_B1_r[s_idx]
        B1_i = s_B1_i[s_idx]
        Δt   = s_Δt[s_idx]

        B = @. sqrt(B1_r^2 + B1_i^2 + Bz^2) + 1f-20
        φ = @. neg_pi_gamma_local * B * Δt
        sin_φ = @. sin(φ)
        cos_φ = @. cos(φ)

        α_r = cos_φ
        α_i = @. -(Bz / B) * sin_φ
        β_r = @. (B1_i / B) * sin_φ
        β_i = @. -(B1_r / B) * sin_φ

        Mxy_new_r = @. 2f0 * (M_xy_i * (α_r * α_i - β_r * β_i) +
                    M_z * (α_i * β_i + α_r * β_r)) +
                    M_xy_r * (α_r^2 - α_i^2 - β_r^2 + β_i^2)

        Mxy_new_i = @. -2f0 * (M_xy_r * (α_r * α_i + β_r * β_i) -
                    M_z * (α_r * β_i - α_i * β_r)) +
                    M_xy_i * (α_r^2 - α_i^2 + β_r^2 - β_i^2)

        Mz_new = @. M_z * (α_r^2 + α_i^2 - β_r^2 - β_i^2) -
                 2f0 * (M_xy_r * (α_r * β_r - α_i * β_i) +
                 M_xy_i * (α_r * β_i + α_i * β_r))

        ΔT1 = @. exp(-Δt / p_T1)
        ΔT2 = @. exp(-Δt / p_T2)
        M_xy_r = @. Mxy_new_r * ΔT2
        M_xy_i = @. Mxy_new_i * ΔT2
        M_z    = @. Mz_new * ΔT1 + p_ρ * (1f0 - ΔT1)
    end

    return M_xy_r, M_xy_i, M_z
end

function reactant_loss_mxy(
    B1_r_tl, B1_i_tl,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf,
    target_r, target_i, mask, invN,
    ::Val{N_Δt}
) where {N_Δt}
    M_xy_r = zero(p_x)
    M_xy_i = zero(p_x)
    M_z    = one.(p_x)

    M_xy_r, M_xy_i, M_z = bloch_forward_reactant!(
        M_xy_r, M_xy_i, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf,
        B1_r_tl, B1_i_tl, Val(N_Δt))

    d_r = @. (M_xy_r - target_r) * mask
    d_i = @. (M_xy_i - target_i) * mask
    return sum(@. invN * (d_r^2 + d_i^2))
end

function reactant_grad_and_loss_mxy(
    B1_r_tl, B1_i_tl,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf,
    target_r, target_i, mask, invN,
    ::Val{N_Δt}
) where {N_Δt}
    (; val, derivs) = gradient(
        ReverseWithPrimal,
        reactant_loss_mxy,
        B1_r_tl, B1_i_tl,
        Const(p_x), Const(p_y), Const(p_z),
        Const(p_ΔBz), Const(p_T1), Const(p_T2),
        Const(p_ρ),
        Const(s_Gx), Const(s_Gy), Const(s_Gz),
        Const(s_Δt), Const(s_Δf),
        Const(target_r), Const(target_i),
        Const(mask), Const(invN),
        Const(Val(N_Δt)))
    return val, derivs[1], derivs[2]
end

function reactant_forward(
    B1_r_tl, B1_i_tl,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf,
    ::Val{N_Δt}
) where {N_Δt}
    M_xy_r = zero(p_x)
    M_xy_i = zero(p_x)
    M_z    = one.(p_x)

    M_xy_r, M_xy_i, M_z = bloch_forward_reactant!(
        M_xy_r, M_xy_i, M_z,
        p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        s_Gx, s_Gy, s_Gz, s_Δt, s_Δf,
        B1_r_tl, B1_i_tl, Val(N_Δt))

    return M_xy_r, M_xy_i, M_z
end

# Reactant setup: compiles forward + grad pipelines and returns CPU-facing closures.
function setup_reactant_backend(;
    Nspins::Int,
    n_ctrl::Int,
    TL,
    rf_idx::Vector{Int},
    spin,
    target,
    mask::Vector{Float32},
    invN::Float32,
    interp = nothing,
    csr = nothing,
)
    Nt = Int(TL.Nt32)
    N_Δt_val = Val(Nt)

    if isnothing(interp)
        interp = build_interpolation_tables(n_ctrl, rf_idx)
    end
    if isnothing(csr)
        csr = build_csr_gather_tables(n_ctrl, interp.j_lo, interp.j_hi, interp.w0, interp.w1)
    end

    p_x_ra  = Reactant.to_rarray(Float32.(spin.p_x))
    p_y_ra  = Reactant.to_rarray(Float32.(spin.p_y))
    p_z_ra  = Reactant.to_rarray(Float32.(spin.p_z))
    p_ΔBz_ra = Reactant.to_rarray(Float32.(spin.p_ΔBz))
    p_T1_ra = Reactant.to_rarray(Float32.(spin.p_T1))
    p_T2_ra = Reactant.to_rarray(Float32.(spin.p_T2))
    p_ρ_ra  = Reactant.to_rarray(Float32.(spin.p_ρ))

    Gx_ra = Reactant.to_rarray(Float32.(TL.Gx))
    Gy_ra = Reactant.to_rarray(Float32.(TL.Gy))
    Gz_ra = Reactant.to_rarray(Float32.(TL.Gz))
    Δt_ra = Reactant.to_rarray(Float32.(TL.Δt))
    Δf_ra = Reactant.to_rarray(Float32.(TL.Δf))

    B1_r_ra    = Reactant.to_rarray(zeros(Float32, Nt))
    B1_i_ra    = Reactant.to_rarray(zeros(Float32, Nt))
    mask_ra    = Reactant.to_rarray(Float32.(mask))
    target_r_ra = Reactant.to_rarray(Float32.(real.(target)))
    target_i_ra = Reactant.to_rarray(Float32.(imag.(target)))

    @info "Compiling Reactant Mxy grad_and_loss..."
    compiled_gl = Reactant.@compile sync=true reactant_grad_and_loss_mxy(
        B1_r_ra, B1_i_ra,
        p_x_ra, p_y_ra, p_z_ra, p_ΔBz_ra, p_T1_ra, p_T2_ra, p_ρ_ra,
        Gx_ra, Gy_ra, Gz_ra, Δt_ra, Δf_ra,
        target_r_ra, target_i_ra, mask_ra, invN,
        N_Δt_val)

    compiled_fwd = Reactant.@compile sync=true reactant_forward(
        B1_r_ra, B1_i_ra,
        p_x_ra, p_y_ra, p_z_ra, p_ΔBz_ra, p_T1_ra, p_T2_ra, p_ρ_ra,
        Gx_ra, Gy_ra, Gz_ra, Δt_ra, Δf_ra,
        N_Δt_val)
    @info "Reactant compilation complete."

    B1_r_tl = zeros(Float32, Nt)
    B1_i_tl = zeros(Float32, Nt)
    gx_buf  = zeros(Float32, n_ctrl)
    gi_buf  = zeros(Float32, n_ctrl)

    spin_ra      = (p_x_ra, p_y_ra, p_z_ra, p_ΔBz_ra, p_T1_ra, p_T2_ra, p_ρ_ra)
    seq_ra       = (Gx_ra, Gy_ra, Gz_ra, Δt_ra, Δf_ra)
    target_tuple = (target_r_ra, target_i_ra, mask_ra, invN)

    function grad_fn!(x_r::Vector{Float32}, x_i::Vector{Float32})
        map_ctrl_to_timeline_cpu!(B1_r_tl, B1_i_tl, x_r, x_i, interp)
        copyto!(B1_r_ra, B1_r_tl)
        copyto!(B1_i_ra, B1_i_tl)

        loss_val, dB1_r_ra, dB1_i_ra = compiled_gl(
            B1_r_ra, B1_i_ra, spin_ra..., seq_ra..., target_tuple..., N_Δt_val)

        gather_grads_cpu!(gx_buf, gi_buf,
                          Array(dB1_r_ra), Array(dB1_i_ra),
                          interp, csr)

        return Float64(loss_val)
    end

    function forward_fn(x_r::Vector{Float32}, x_i::Vector{Float32})
        map_ctrl_to_timeline_cpu!(B1_r_tl, B1_i_tl, x_r, x_i, interp)
        copyto!(B1_r_ra, B1_r_tl)
        copyto!(B1_i_ra, B1_i_tl)

        Mr, Mi, Mz = compiled_fwd(B1_r_ra, B1_i_ra, spin_ra..., seq_ra..., N_Δt_val)

        return ComplexF32.(Array(Mr) .+ im .* Array(Mi)), Array(Mz)
    end

    return (; grad_fn!, forward_fn, gx_buf, gi_buf,
              x_r=zeros(Float32, n_ctrl), x_i=zeros(Float32, n_ctrl),
              ka_backend=KA.CPU(), group_size=1)
end

# ============================================================================
# Barzilai-Borwein adaptive gradient descent (CPU broadcasts)
# ============================================================================

function bb_optimize!(
    x_r_d, x_i_d,
    gx_d, gi_d,
    grad_fn!,
    Niters::Integer, λ0;
    backend = KA.CPU(),
    group_size::Int = 1,
    log_every::Int = 0,
    grad_tol::Float64 = 0.0,
)
    T = eltype(x_r_d)
    loss_history = zeros(Float64, Niters)

    gx_prev_d  = similar(gx_d);  gi_prev_d  = similar(gi_d)
    fill!(gx_prev_d, zero(T));   fill!(gi_prev_d, zero(T))
    x_prev_r_d = similar(x_r_d); x_prev_i_d = similar(x_i_d)
    copyto!(x_prev_r_d, x_r_d);  copyto!(x_prev_i_d, x_i_d)

    λ_prev = T(λ0)
    θ_prev = T(Inf)
    performed_iterations = 0

    for k in 1:Niters
        performed_iterations = k

        f_k = grad_fn!(x_r_d, x_i_d)
        loss_history[k] = Float64(f_k)

        gnorm = sqrt(Float64(sum(gx_d .* gx_d .+ gi_d .* gi_d)))
        num = sqrt(Float64(sum((x_r_d .- x_prev_r_d) .* (x_r_d .- x_prev_r_d) .+
                                (x_i_d .- x_prev_i_d) .* (x_i_d .- x_prev_i_d))))
        den = sqrt(Float64(sum((gx_d .- gx_prev_d) .* (gx_d .- gx_prev_d) .+
                                (gi_d .- gi_prev_d) .* (gi_d .- gi_prev_d))))

        if !isfinite(f_k) || !isfinite(gnorm)
            @warn "Non-finite loss/gradient; stopping" iter=k loss=f_k gnorm=gnorm
            break
        end

        if log_every > 0 && (k % log_every == 0 || k == 1)
            @info "Iter $k: loss = $f_k, λ = $λ_prev, ||g|| = $gnorm"
        end

        if grad_tol > 0 && gnorm < grad_tol
            @info "Converged at iter $k (||g|| = $gnorm)"
            break
        end

        # Barzilai-Borwein step size
        ratio = den > 0 ? num / (2 * den) : Inf
        grow = sqrt(1 + Float64(θ_prev)) * Float64(λ_prev)
        λ_k = T(min(grow, ratio))
        if !isfinite(λ_k) || λ_k <= zero(T)
            λ_k = λ_prev
        end

        copyto!(gx_prev_d, gx_d);   copyto!(gi_prev_d, gi_d)
        copyto!(x_prev_r_d, x_r_d); copyto!(x_prev_i_d, x_i_d)

        @. x_r_d -= λ_k * gx_d
        @. x_i_d -= λ_k * gi_d

        θ_prev = λ_k / λ_prev
        λ_prev = λ_k
    end

    return (; loss_history, performed_iterations)
end

# ============================================================================
# Native KernelAbstractions + Enzyme CUDA path
# (alternative to the Reactant pipeline above; uses KA kernels differentiated
#  directly by Enzyme. Caller imports CUDA, builds a CUDABackend, and provides
#  `to_device = x -> CUDA.adapt(CuArray, x)`.)
# ============================================================================

# --- Single-coil Bloch SU(2) excitation kernel (Enzyme-differentiable) -------
@kernel unsafe_indices=true inbounds=true function excitation_kernel!(
    M_xy::AbstractVector{Complex{T}},
    M_z::AbstractVector{T},
    @Const(p_x), @Const(p_y), @Const(p_z),
    @Const(p_ΔBz), @Const(p_T1), @Const(p_T2), @Const(p_ρ),
    N_Spins,
    @Const(s_Gx), @Const(s_Gy), @Const(s_Gz),
    @Const(s_Δt), @Const(s_Δf),
    s_B1::AbstractVector{Complex{T}},
    N_Δt,
) where {T}
    @uniform N = @groupsize()[1]
    i_l = @index(Local, Linear)
    i_g = @index(Group, Linear)
    i = (i_g - Int32(1)) * Int32(N) + i_l

    if i <= N_Spins
        x = p_x[i]; y = p_y[i]; z = p_z[i]
        ΔBz = p_ΔBz[i]; T1 = p_T1[i]; T2 = p_T2[i]; ρ = p_ρ[i]

        Mxy_r, Mxy_i = reim(M_xy[i])
        Mz = M_z[i]

        s_idx = Int32(1)
        while s_idx <= N_Δt
            Bz = (x * s_Gx[s_idx] + y * s_Gy[s_idx] + z * s_Gz[s_idx]) +
                 ΔBz - s_Δf[s_idx] / T(γ_rad)
            B1_r, B1_i = reim(s_B1[s_idx])
            B = sqrt(B1_r^2 + B1_i^2 + Bz^2)
            Δt = s_Δt[s_idx]

            φ = T(-π * γ_Hz) * B * Δt
            @noinline sin_φ, cos_φ = sincos(φ)

            if iszero(B)
                α_r = cos_φ; α_i = zero(T)
                β_r = zero(T); β_i = zero(T)
            else
                α_r = cos_φ
                α_i = -(Bz / B) * sin_φ
                β_r =  (B1_i / B) * sin_φ
                β_i = -(B1_r / B) * sin_φ
            end

            Mxy_new_r = 2 * (Mxy_i * (α_r * α_i - β_r * β_i) +
                        Mz * (α_i * β_i + α_r * β_r)) +
                        Mxy_r * (α_r^2 - α_i^2 - β_r^2 + β_i^2)
            Mxy_new_i = -2 * (Mxy_r * (α_r * α_i + β_r * β_i) -
                        Mz * (α_r * β_i - α_i * β_r)) +
                        Mxy_i * (α_r^2 - α_i^2 + β_r^2 - β_i^2)
            Mz_new = Mz * (α_r^2 + α_i^2 - β_r^2 - β_i^2) -
                     2 * (Mxy_r * (α_r * β_r - α_i * β_i) +
                     Mxy_i * (α_r * β_i + α_i * β_r))

            ΔT1 = exp(-Δt / T1); ΔT2 = exp(-Δt / T2)
            Mxy_r = Mxy_new_r * ΔT2
            Mxy_i = Mxy_new_i * ΔT2
            Mz    = Mz_new * ΔT1 + ρ * (T(1) - ΔT1)

            s_idx += Int32(1)
        end
        M_xy[i] = complex(Mxy_r, Mxy_i)
        M_z[i]  = Mz
    end
end

# @noinline launcher — Enzyme.autodiff differentiates through this.
Base.@noinline function excite_only!(
    M_xy, M_z,
    p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
    N_Spins::Int32,
    s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1,
    N_Δt::Int32, backend, group_size::Int = 256,
)
    excitation_kernel!(backend, group_size)(
        M_xy, M_z, p_x, p_y, p_z, p_ΔBz, p_T1, p_T2, p_ρ,
        N_Spins, s_Gx, s_Gy, s_Gz, s_Δt, s_Δf, s_B1, N_Δt;
        ndrange = Int(N_Spins))
    return nothing
end

# --- Control <-> timeline kernels (GPU) --------------------------------------
@kernel inbounds=true function map_ctrl_to_timeline_kernel!(
    out_B1::AbstractVector{Complex{T}},
    @Const(x_r), @Const(x_i),
    @Const(idx_rf), @Const(jlo), @Const(jhi),
    @Const(w0), @Const(w1),
    sgn_r::T, sgn_i::T,
) where {T<:AbstractFloat}
    k = Int32(@index(Global, Linear))
    if k <= Int32(length(idx_rf))
        idx = Int32(idx_rf[k])
        j0  = Int32(jlo[k]); j1 = Int32(jhi[k])
        vr = sgn_r * (w0[k] * x_r[j0] + w1[k] * x_r[j1])
        vi = sgn_i * (w0[k] * x_i[j0] + w1[k] * x_i[j1])
        out_B1[idx] = Complex{T}(vr, vi)
    end
end

@kernel inbounds=true function gather_ctrl_grads_kernel!(
    gx::AbstractVector{T}, gi::AbstractVector{T},
    @Const(dB1),
    @Const(idx_rf), @Const(ptr), @Const(csr_idx), @Const(csr_w),
) where {T<:AbstractFloat}
    j = Int32(@index(Global, Linear))
    if j <= Int32(length(gx))
        s_r = zero(T); s_i = zero(T)
        p0 = Int32(ptr[j]); p1 = Int32(ptr[j+1]) - Int32(1)
        @inbounds for p in p0:p1
            k = Int32(csr_idx[p])
            tid = Int32(idx_rf[k])
            w = csr_w[p]
            db1 = dB1[tid]
            s_r += w * real(db1)
            s_i += w * imag(db1)
        end
        gx[j] = s_r; gi[j] = s_i
    end
end

# MSE-on-Mxy loss + adjoint seed (atomic accumulation of scalar loss).
@kernel inbounds=true function seed_and_loss_kernel!(
    acc::AbstractVector{T},
    dM_xy::AbstractVector{Complex{T}},
    M_xy::AbstractVector{Complex{T}},
    @Const(target), @Const(mask),
    invN::T,
) where {T<:AbstractFloat}
    i = Int32(@index(Global, Linear))
    N = Int32(length(mask))
    if i <= N
        m = M_xy[i]; t = target[i]
        d = m - t
        w = mask[i] * invN
        @KA.atomic acc[1] += w * (real(d)^2 + imag(d)^2)
        dM_xy[i] = Complex{T}(2 * w * real(d), 2 * w * imag(d))
    end
end

# --- BlochSetup: bundles device buffers for forward + Enzyme reverse --------
struct BlochSetup{T}
    M_xy::AbstractVector{Complex{T}}
    M_z::AbstractVector{T}
    dM_xy::AbstractVector{Complex{T}}
    dM_z::AbstractVector{T}
    p_x::AbstractVector{T}; p_y::AbstractVector{T}; p_z::AbstractVector{T}
    p_ΔBz::AbstractVector{T}
    p_T1::AbstractVector{T}; p_T2::AbstractVector{T}; p_ρ::AbstractVector{T}
    s_Gx::AbstractVector{T}; s_Gy::AbstractVector{T}; s_Gz::AbstractVector{T}
    s_Δt::AbstractVector{T}; s_Δf::AbstractVector{T}
    s_B1::AbstractVector{Complex{T}}
    ∇B1::AbstractVector{Complex{T}}
    acc_loss_d::AbstractVector{T}
    gx_d::AbstractVector{T}; gi_d::AbstractVector{T}
    j_lo_d::AbstractVector{Int32}; j_hi_d::AbstractVector{Int32}
    w0_d::AbstractVector{T};       w1_d::AbstractVector{T}
    idx_rf_d::AbstractVector{Int32}
    csr_ptr_d::AbstractVector{Int32}; csr_idx_d::AbstractVector{Int32}
    csr_w_d::AbstractVector{T}
    Nspins::Int32; N_Δt::Int32
    n_ctrl::Int; n_rf::Int
    backend::Any
    group_size::Int
end

# Upload host interp/csr tables (which are Int/Float64) as Int32/Float32.
function setup_control_mapping_device(n_ctrl::Int, rf_idx, to_device)
    interp = build_interpolation_tables(n_ctrl, rf_idx)
    csr    = build_csr_gather_tables(n_ctrl, interp.j_lo, interp.j_hi, interp.w0, interp.w1)
    return (
        interp_tables = interp,
        csr_tables    = csr,
        j_lo_d   = to_device(Int32.(interp.j_lo)),
        j_hi_d   = to_device(Int32.(interp.j_hi)),
        w0_d     = to_device(Float32.(interp.w0)),
        w1_d     = to_device(Float32.(interp.w1)),
        idx_rf_d = to_device(Int32.(interp.idx_rf)),
        csr_ptr_d = to_device(Int32.(csr.ptr)),
        csr_idx_d = to_device(Int32.(csr.idx)),
        csr_w_d   = to_device(Float32.(csr.w)),
    )
end

function init_bloch_setup(;
    Nspins::Int, n_ctrl::Int,
    TL, rf_idx::Vector{Int},
    spin,
    to_device, backend,
    group_size::Int = 256,
)
    cm = setup_control_mapping_device(n_ctrl, rf_idx, to_device)
    BlochSetup{Float32}(
        to_device(zeros(ComplexF32, Nspins)),
        to_device(ones(Float32, Nspins)),
        to_device(zeros(ComplexF32, Nspins)),
        to_device(zeros(Float32, Nspins)),
        to_device(Float32.(spin.p_x)),
        to_device(Float32.(spin.p_y)),
        to_device(Float32.(spin.p_z)),
        to_device(Float32.(spin.p_ΔBz)),
        to_device(Float32.(spin.p_T1)),
        to_device(Float32.(spin.p_T2)),
        to_device(Float32.(spin.p_ρ)),
        to_device(TL.Gx), to_device(TL.Gy), to_device(TL.Gz),
        to_device(TL.Δt), to_device(TL.Δf),
        to_device(TL.B1),
        to_device(zeros(ComplexF32, length(TL.Δt))),
        to_device(zeros(Float32, 1)),
        to_device(zeros(Float32, n_ctrl)),
        to_device(zeros(Float32, n_ctrl)),
        cm.j_lo_d, cm.j_hi_d, cm.w0_d, cm.w1_d, cm.idx_rf_d,
        cm.csr_ptr_d, cm.csr_idx_d, cm.csr_w_d,
        Int32(Nspins), TL.Nt32, n_ctrl, length(rf_idx),
        backend, group_size,
    )
end

# Forward + Enzyme reverse, returns scalar loss (Float64).
# `seed_fn!(bs)` fills bs.dM_xy and atomically accumulates loss into bs.acc_loss_d.
function grad_and_loss!(bs::BlochSetup, x_r_d, x_i_d, seed_fn!;
                        sign_r::Float32 = 1f0, sign_i::Float32 = 1f0)
    map_ctrl_to_timeline_kernel!(bs.backend, bs.group_size)(
        bs.s_B1, x_r_d, x_i_d,
        bs.idx_rf_d, bs.j_lo_d, bs.j_hi_d, bs.w0_d, bs.w1_d,
        sign_r, sign_i; ndrange = bs.n_rf)

    fill!(bs.M_xy, ComplexF32(0)); fill!(bs.M_z, 1.0f0)
    excite_only!(bs.M_xy, bs.M_z,
        bs.p_x, bs.p_y, bs.p_z, bs.p_ΔBz, bs.p_T1, bs.p_T2, bs.p_ρ,
        bs.Nspins, bs.s_Gx, bs.s_Gy, bs.s_Gz, bs.s_Δt, bs.s_Δf, bs.s_B1,
        bs.N_Δt, bs.backend, bs.group_size)

    fill!(bs.acc_loss_d, 0f0)
    fill!(bs.dM_xy, ComplexF32(0)); fill!(bs.dM_z, 0f0)
    fill!(bs.∇B1, ComplexF32(0))
    seed_fn!(bs)

    fill!(bs.M_xy, ComplexF32(0)); fill!(bs.M_z, 1.0f0)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        excite_only!,
        Enzyme.Duplicated(bs.M_xy, bs.dM_xy),
        Enzyme.Duplicated(bs.M_z,  bs.dM_z),
        Enzyme.Const(bs.p_x), Enzyme.Const(bs.p_y), Enzyme.Const(bs.p_z),
        Enzyme.Const(bs.p_ΔBz),
        Enzyme.Const(bs.p_T1), Enzyme.Const(bs.p_T2), Enzyme.Const(bs.p_ρ),
        Enzyme.Const(bs.Nspins),
        Enzyme.Const(bs.s_Gx), Enzyme.Const(bs.s_Gy), Enzyme.Const(bs.s_Gz),
        Enzyme.Const(bs.s_Δt), Enzyme.Const(bs.s_Δf),
        Enzyme.Duplicated(bs.s_B1, bs.∇B1),
        Enzyme.Const(bs.N_Δt),
        Enzyme.Const(bs.backend),
        Enzyme.Const(bs.group_size),
    )

    fill!(bs.gx_d, 0f0); fill!(bs.gi_d, 0f0)
    gather_ctrl_grads_kernel!(bs.backend, bs.group_size)(
        bs.gx_d, bs.gi_d, bs.∇B1,
        bs.idx_rf_d, bs.csr_ptr_d, bs.csr_idx_d, bs.csr_w_d;
        ndrange = bs.n_ctrl)

    return Float64(Array(bs.acc_loss_d)[1])
end

function forward_sim!(bs::BlochSetup, x_r_d, x_i_d;
                      sign_r::Float32 = 1f0, sign_i::Float32 = 1f0)
    map_ctrl_to_timeline_kernel!(bs.backend, bs.group_size)(
        bs.s_B1, x_r_d, x_i_d,
        bs.idx_rf_d, bs.j_lo_d, bs.j_hi_d, bs.w0_d, bs.w1_d,
        sign_r, sign_i; ndrange = bs.n_rf)
    fill!(bs.M_xy, ComplexF32(0)); fill!(bs.M_z, 1.0f0)
    excite_only!(bs.M_xy, bs.M_z,
        bs.p_x, bs.p_y, bs.p_z, bs.p_ΔBz, bs.p_T1, bs.p_T2, bs.p_ρ,
        bs.Nspins, bs.s_Gx, bs.s_Gy, bs.s_Gz, bs.s_Δt, bs.s_Δf, bs.s_B1,
        bs.N_Δt, bs.backend, bs.group_size)
    KA.synchronize(bs.backend)
    return Array(bs.M_xy), Array(bs.M_z)
end
