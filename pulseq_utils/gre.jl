function _lobe_timing(M_vec, sys)
    if M_vec ≤ sys.Gmax^2 / sys.Smax
        ζ_ideal = sqrt(M_vec / sys.Smax)
        ζ_ceil = ceil(ζ_ideal / sys.GR_Δt) * sys.GR_Δt
        ζ_floor = floor(ζ_ideal / sys.GR_Δt) * sys.GR_Δt

        if ζ_floor > 0 && ζ_floor < ζ_ceil        # strictly off-raster
            G_trap = M_vec / (ζ_floor + sys.GR_Δt)
            if G_trap / ζ_floor ≤ sys.Smax          # slew check on shorter ramp
                return sys.GR_Δt, ζ_floor
            end
        end
        return 0.0, ζ_ceil
    else
        ζ_p = ceil(sys.Gmax / sys.Smax / sys.GR_Δt) * sys.GR_Δt
        T_p = max(0.0, ceil((M_vec / sys.Gmax - ζ_p) / sys.GR_Δt) * sys.GR_Δt)
        return T_p, ζ_p
    end
end


function gre_base(FOV, matrix, sys, BWpp; fixed_area=(0.0, 0.0, 0.0))

    # ── Input validation ───────────────────────────────────────────────────────
    length(FOV) == length(matrix) || error(
        "FOV and matrix must have the same length " *
        "(got $(length(FOV)) and $(length(matrix)))")
    ndim = length(FOV)
    1 ≤ ndim ≤ 3 || error("FOV and matrix must have 1, 2, or 3 elements (got $ndim)")
    length(fixed_area) == 3 || error("fixed_area must have exactly 3 elements (x, y, z)")

    Ax, Ay, Az = fixed_area
    FOV_ro, N_ro = FOV[1], matrix[1]

    # ── ADC dwell time, snapped to hardware raster ─────────────────────────────
    dt_ideal = 1.0 / (N_ro * BWpp)
    N_dwell = max(round(Int, dt_ideal / sys.ADC_Δt), 1)
    dt = N_dwell * sys.ADC_Δt

    # ── Readout gradient (Gx only) ─────────────────────────────────────────────
    Ga = 1.0 / (γ * dt * FOV_ro)
    Ga > sys.Gmax && error(
        "Readout Ga=$(round(Ga*1e3,digits=2)) mT/m exceeds " *
        "Gmax=$(round(sys.Gmax*1e3,digits=2)) mT/m — reduce BWpp")

    Ta_adc = dt * (N_ro - 1)
    Ta = ceil(Ta_adc / sys.GR_Δt) * sys.GR_Δt
    ζ = ceil(Ga / sys.Smax / sys.GR_Δt) * sys.GR_Δt

    # ── Readout block (fixed for all lines) ────────────────────────────────────
    RO = Sequence()
    @addblock RO += (ADC(N_ro, Ta_adc, ζ), x=Grad(Ga, Ta, ζ))

    # ── Prephaser moments ──────────────────────────────────────────────────────
    n_echo = N_ro ÷ 2
    M_ro = Ga * (ζ / 2 + n_echo * dt)

    # ── Worst-case moment per axis ─────────────────────────────────────────────
    # Use symmetric ±(N÷2) endpoints: conservative enough to cover the even-N
    # rewinder error (one extra ADC dwell step) without separate case analysis.
    Mx_worst = abs(-M_ro + Ax)

    pe = if ndim ≥ 2
        N, fov = matrix[2], FOV[2]
        My_pe = (N ÷ 2) / (γ * fov)
        (N=N, fov=fov, worst=max(abs(Ay - My_pe), abs(Ay + My_pe)))
    else
        (N=0, fov=1.0, worst=abs(Ay))
    end
    N_pe, FOV_pe, My_worst = pe.N, pe.fov, pe.worst

    par = if ndim == 3
        N, fov = matrix[3], FOV[3]
        Mz_par = (N ÷ 2) / (γ * fov)
        (N=N, fov=fov, worst=max(abs(Az - Mz_par), abs(Az + Mz_par)))
    else
        (N=0, fov=1.0, worst=abs(Az))
    end
    N_par, FOV_par, Mz_worst = par.N, par.fov, par.worst

    M_vec = sqrt(Mx_worst^2 + My_worst^2 + Mz_worst^2)

    # ── Prephaser timing ───────────────────────────────────────────────────────
    T_p, ζ_p = _lobe_timing(M_vec, sys)
    inv_area = 1.0 / (T_p + ζ_p)
    G_x_pre = (-M_ro + Ax) * inv_area
    G_z_pre = Az * inv_area

    # ── Inner callables ────────────────────────────────────────────────────────
    function gre_1D()
        PRE = Sequence()
        @addblock PRE += (x=Grad(G_x_pre, T_p, ζ_p), y=Grad(Ay * inv_area, T_p, ζ_p), z=Grad(G_z_pre, T_p, ζ_p))
        seq = Sequence()
        @addblock seq += PRE + RO
        return seq
    end

    function gre_2D(i)
        i_lo, i_hi = -(N_pe ÷ 2), N_pe - 1 - N_pe ÷ 2
        i_lo ≤ i ≤ i_hi || error("PE index i=$i out of bounds [$i_lo, $i_hi]")
        G_pe = (i / (γ * FOV_pe) + Ay) * inv_area
        PRE = Sequence()
        @addblock PRE += (x=Grad(G_x_pre, T_p, ζ_p), y=Grad(G_pe, T_p, ζ_p), z=Grad(G_z_pre, T_p, ζ_p))
        seq = Sequence()
        @addblock seq += PRE + RO
        # TODO Replace this with custom label that is converted to LIN/PAR at write time
        seq.EXT[2] = [LabelSet(i - i_lo, "LIN")]
        return seq
    end

    function gre_3D(i, j)
        i_lo, i_hi = -(N_pe ÷ 2), N_pe - 1 - N_pe ÷ 2
        j_lo, j_hi = -(N_par ÷ 2), N_par - 1 - N_par ÷ 2
        i_lo ≤ i ≤ i_hi || error("PE index i=$i out of bounds [$i_lo, $i_hi]")
        j_lo ≤ j ≤ j_hi || error("Partition index j=$j out of bounds [$j_lo, $j_hi]")
        G_pe = (i / (γ * FOV_pe) + Ay) * inv_area
        G_par = (j / (γ * FOV_par) + Az) * inv_area
        PRE = Sequence()
        @addblock PRE += (x=Grad(G_x_pre, T_p, ζ_p), y=Grad(G_pe, T_p, ζ_p), z=Grad(G_par, T_p, ζ_p))
        seq = Sequence()
        @addblock seq += PRE + RO
        seq.EXT[2] = [LabelSet(i - i_lo, "LIN"), LabelSet(j - j_lo, "PAR")]
        return seq
    end

    return ndim == 1 ? gre_1D : ndim == 2 ? gre_2D : gre_3D
end


function bssfp_base(FOV, matrix, sys, BWpp; fixed_area=(0.0, 0.0, 0.0))

    gre_line = gre_base(FOV, matrix, sys, BWpp; fixed_area=fixed_area)
    ndim = length(FOV)

    Ax, Ay, Az = fixed_area
    FOV_ro, N_ro = FOV[1], matrix[1]
    dt_ideal = 1.0 / (N_ro * BWpp)
    N_dwell = max(round(Int, dt_ideal / sys.ADC_Δt), 1)
    dt = N_dwell * sys.ADC_Δt
    Ga = 1.0 / (γ * dt * FOV_ro)
    ζ = ceil(Ga / sys.Smax / sys.GR_Δt) * sys.GR_Δt
    n_echo = N_ro ÷ 2
    M_ro = Ga * (ζ / 2 + n_echo * dt)
    Mx_worst = abs(-M_ro + Ax)

    pe = if ndim ≥ 2
        N, fov = matrix[2], FOV[2]
        My_pe = (N ÷ 2) / (γ * fov)
        (N=N, fov=fov, worst=max(abs(Ay - My_pe), abs(Ay + My_pe)))
    else
        (N=0, fov=1.0, worst=abs(Ay))
    end
    N_pe, FOV_pe, My_worst = pe.N, pe.fov, pe.worst

    par = if ndim == 3
        N, fov = matrix[3], FOV[3]
        Mz_par = (N ÷ 2) / (γ * fov)
        (N=N, fov=fov, worst=max(abs(Az - Mz_par), abs(Az + Mz_par)))
    else
        (N=0, fov=1.0, worst=abs(Az))
    end
    N_par, FOV_par, Mz_worst = par.N, par.fov, par.worst

    T_p, ζ_p = _lobe_timing(sqrt(Mx_worst^2 + My_worst^2 + Mz_worst^2), sys)
    inv_area = 1.0 / (T_p + ζ_p)
    G_x_pre = (-M_ro + Ax) * inv_area
    G_z_pre = Az * inv_area
    flip_encode(G, fixed) = iszero(G) ? zero(G) : G * (2 * fixed / (G * (T_p + ζ_p)) - 1)

    function bSSFP_1D()
        seq = gre_line()
        @addblock seq += (x=Grad(G_x_pre, T_p, ζ_p), y=Grad(Ay * inv_area, T_p, ζ_p), z=Grad(G_z_pre, T_p, ζ_p))
        return seq
    end

    function bSSFP_2D(i)
        i_lo, i_hi = -(N_pe ÷ 2), N_pe - 1 - N_pe ÷ 2
        i_lo ≤ i ≤ i_hi || error("PE index i=$i out of bounds [$i_lo, $i_hi]")
        G_y_pre = (i / (γ * FOV_pe) + Ay) * inv_area
        seq = gre_line(i)
        @addblock seq += (x=Grad(G_x_pre, T_p, ζ_p), y=Grad(flip_encode(G_y_pre, Ay), T_p, ζ_p), z=Grad(G_z_pre, T_p, ζ_p))
        return seq
    end

    function bSSFP_3D(i, j)
        i_lo, i_hi = -(N_pe ÷ 2), N_pe - 1 - N_pe ÷ 2
        j_lo, j_hi = -(N_par ÷ 2), N_par - 1 - N_par ÷ 2
        i_lo ≤ i ≤ i_hi || error("PE index i=$i out of bounds [$i_lo, $i_hi]")
        j_lo ≤ j ≤ j_hi || error("Partition index j=$j out of bounds [$j_lo, $j_hi]")
        G_y_pre = (i / (γ * FOV_pe) + Ay) * inv_area
        G_z_pre_line = (j / (γ * FOV_par) + Az) * inv_area
        seq = gre_line(i, j)
        @addblock seq += (x=Grad(G_x_pre, T_p, ζ_p), y=Grad(flip_encode(G_y_pre, Ay), T_p, ζ_p), z=Grad(flip_encode(G_z_pre_line, Az), T_p, ζ_p))
        return seq
    end

    return ndim == 1 ? bSSFP_1D : ndim == 2 ? bSSFP_2D : bSSFP_3D

end


function sgre_base(FOV, matrix, sys, BWpp;
    spoil_area,
    fixed_area=(0.0, 0.0, 0.0))

    length(spoil_area) == 3 || error("spoil_area must have exactly 3 elements (x, y, z)")

    gre_line = gre_base(FOV, matrix, sys, BWpp; fixed_area=fixed_area)

    ndim = length(FOV)
    FOV_ro, N_ro = FOV[1], matrix[1]
    dt_ideal = 1.0 / (N_ro * BWpp)
    N_dwell = max(round(Int, dt_ideal / sys.ADC_Δt), 1)
    dt = N_dwell * sys.ADC_Δt
    Ga = 1.0 / (γ * dt * FOV_ro)
    Ta_adc = dt * (N_ro - 1)
    Ta = ceil(Ta_adc / sys.GR_Δt) * sys.GR_Δt
    ζ = ceil(Ga / sys.Smax / sys.GR_Δt) * sys.GR_Δt
    M_ro = Ga * (ζ / 2 + (N_ro ÷ 2) * dt)
    M_x_nom = -M_ro + Ga * (Ta + ζ)
    Sx, Sy, Sz = spoil_area
    Mx_spo = abs(Sx - M_x_nom)

    pe = if ndim ≥ 2
        N, fov = matrix[2], FOV[2]
        ΔM = 1.0 / (γ * fov)
        M = (N ÷ 2) * ΔM
        (ΔM=ΔM, worst=max(abs(Sy - M), abs(Sy + M)))
    else
        (ΔM=0.0, worst=abs(Sy))
    end
    ΔMy, My_spo = pe.ΔM, pe.worst

    par = if ndim == 3
        N, fov = matrix[3], FOV[3]
        ΔM = 1.0 / (γ * fov)
        M = (N ÷ 2) * ΔM
        (ΔM=ΔM, worst=max(abs(Sz - M), abs(Sz + M)))
    else
        (ΔM=0.0, worst=abs(Sz))
    end
    ΔMz, Mz_spo = par.ΔM, par.worst

    # ── Spoiler lobe timing ────────────────────────────────────────────────────
    M_vec_spo = sqrt(Mx_spo^2 + My_spo^2 + Mz_spo^2)
    T_s, ζ_s = M_vec_spo > 0 ? _lobe_timing(M_vec_spo, sys) : (0.0, sys.GR_Δt)
    inv_area_s = M_vec_spo > 0 ? 1.0 / (T_s + ζ_s) : 0.0
    G_x_spo = (Sx - M_x_nom) * inv_area_s      # constant across all lines

    # ── Inner callables ────────────────────────────────────────────────────────
    function sGRE_1D()
        G_y_spo = Sy * inv_area_s
        G_z_spo = Sz * inv_area_s
        seq = gre_line()
        @addblock seq += (x=Grad(G_x_spo, T_s, ζ_s), y=Grad(G_y_spo, T_s, ζ_s), z=Grad(G_z_spo, T_s, ζ_s))
        return seq
    end

    function sGRE_2D(i)
        G_y_spo = (Sy - i * ΔMy) * inv_area_s
        G_z_spo = Sz * inv_area_s
        seq = gre_line(i)
        @addblock seq += (x=Grad(G_x_spo, T_s, ζ_s), y=Grad(G_y_spo, T_s, ζ_s), z=Grad(G_z_spo, T_s, ζ_s))
        return seq
    end

    function sGRE_3D(i, j)
        G_y_spo = (Sy - i * ΔMy) * inv_area_s
        G_z_spo = (Sz - j * ΔMz) * inv_area_s
        seq = gre_line(i, j)
        @addblock seq += (x=Grad(G_x_spo, T_s, ζ_s), y=Grad(G_y_spo, T_s, ζ_s), z=Grad(G_z_spo, T_s, ζ_s))
        return seq
    end

    return ndim == 1 ? sGRE_1D : ndim == 2 ? sGRE_2D : sGRE_3D
end
