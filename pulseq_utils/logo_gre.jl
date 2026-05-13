using KomaMRI
using JLD2

include("gre.jl")
include("excitation.jl")

function generate_logo_gre_seq(pulse_fn::String, output_fn::String;
                                disable_prep::Bool=false)
    #### Settings ####
    fov = (0.256, 0.256, 0.01) # m
    matrix = (128, 128, 1)
    flip_angle = π / 4
    BW_ro = 500.0 # Hz
    BW_rf = nothing # Hz
    spoil_moment = 4π # slice direction

    intershot_delay = 1 # seconds

    slab_spoil_moment = 400π # m
    slab_gap = 2*fov[3] # m
    slab_thickness = 0.1

    deadtime = 100e-6
    ringdowntime = 20e-6

    sys = Scanner(
        B0=2.89,
        B1=17e-6,
        Gmax=40e-3,
        Smax=100.0,
        ADC_Δt=1e-6,
        DUR_Δt=10e-6,
        GR_Δt=10e-6,
        RF_Δt=1e-6,
        RF_ring_down_time=20e-6,
        RF_dead_time=100e-6,
        ADC_dead_time=10e-6,
    )

    Δz = fov[3]
    rf = slice_selective_sinc(flip_angle, Δz, sys, deadtime; BW=BW_rf)

    calc_trap_area(GR) = GR.A * (GR.T + GR.rise)
    ss_rewind_area = calc_trap_area(rf.GR[3, 2])

    M_spoiler = spoil_moment / (γ * 2π) / fov[3]

    if matrix[3] == 1
        sgre_kernel = sgre_base(fov[1:2], matrix[1:2], sys, BW_ro;
            spoil_area=(0.0, 0.0, M_spoiler), fixed_area=(0.0, 0.0, ss_rewind_area))
    else
        sgre_kernel = sre_base(fov, matrix, sys, BW_ro;
            fixed_area=(0.0, 0.0, ss_rewind_area))
    end

    rf_excitation(rf, scale) = begin
        rf_event = scale * rf[1]
        rf_event.RF[1].use = Excitation()
        return rf_event
    end

    rf_spoil_phase(i) = (117 * i * (i + 1) / 2) / 180 * π

    # Build sat slabs
    sat = slice_selective_sinc(π / 2, slab_thickness, sys, deadtime; TBP=10)
    sat_center_shift = slab_thickness / 2 + slab_gap / 2
    freq_shift = γ * sat.GR[3, 1].A * sat_center_shift
    sat.RF[1].use = Preparation()

    sat_rewind_area = calc_trap_area(sat.GR[3, 2])
    sat_spoil_area = slab_spoil_moment / (γ * 2π) / slab_thickness
    net_area = sat_rewind_area + sat_spoil_area

    T_sp, ζ_sp = _lobe_timing(abs(net_area), sys)
    sat_spoil = Grad(net_area / (T_sp + ζ_sp), T_sp, ζ_sp)

    sat_pos = Sequence()
    @addblock sat_pos += sat[1] + (z = sat_spoil)

    sat_neg = Sequence()
    @addblock sat_neg += sat[1] + (z = sat_spoil)

    sat_pos.RF[1].Δf = freq_shift
    sat_neg.RF[1].Δf = -freq_shift

    function grab_area(grad)
        a = ampls(grad)
        t = times(grad)
        area = (a[1] + a[2]) * (t[2] - t[1]) / 2
        for i in 2:length(a)-1
            area += (a[i] + a[i+1]) * (t[i+1] - t[i]) / 2
        end
        return area
    end

    # Load prep pulse from JLD2
    f = jldopen(pulse_fn, "r")
    B1 = read(f, "B1")
    Gx = read(f, "Gx")
    Gy = read(f, "Gy")
    T = read(f, "T")
    close(f)
    if disable_prep
        B1 .= 0.0
        Gx .= 0.0
        Gy .= 0.0
    end
    prep_RF = RF(B1, T, 0.0, deadtime, T, angle(B1[end]), Excitation())
    prep_Gx = Grad(Gx, T, 0.0, 0.0, deadtime)
    prep_Gy = Grad(Gy, T, 0.0, 0.0, deadtime)
    prep_rw_Gx = Grad(1.0, 1e-4, 1e-4)
    prep_rw_Gy = Grad(1.0, 1e-4, 1e-4)
    prep_rw_Gx.A = -grab_area(prep_Gx) / grab_area(prep_rw_Gx)
    prep_rw_Gy.A = -grab_area(prep_Gy) / grab_area(prep_rw_Gy)
    prep = Sequence()
    @addblock prep += (x=prep_rw_Gx, y=prep_rw_Gy) + (Duration(dur(prep_RF)+ringdowntime), prep_RF; x=prep_Gx, y=prep_Gy)

    delay = Delay(intershot_delay)

    seq = Sequence()

    @addblock check_timing = true for i in 1:matrix[2]
        seq += delay
        seq += sat_pos
        seq += sat_neg
        seq += prep
        readout = sgre_kernel(i - (matrix[2] ÷ 2 + 1))[1:2]
        remove_Gz = [
            1.0 0.0 0.0;
            0.0 1.0 0.0;
            0.0 0.0 0.0
        ]
        seq += remove_Gz * readout
    end

    seq.DEF["Nx"] = 128
    seq.DEF["Ny"] = 128
    seq.DEF["Nz"] = 1
    seq.DEF["FOV"] = fov

    write_seq(seq, output_fn)
    @info "Saved .seq to $output_fn"
end
