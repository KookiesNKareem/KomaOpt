using KomaMRI
using JLD2
using Random

include("gre.jl")
include("excitation.jl")

function generate_bssfp_seq(prep_fn::String, output_fn::String;
                             disable_prep::Bool=false)
    #### Settings ####
    fov = (0.256, 0.256, 0.01) # m
    matrix = (128, 128, 1)
    flip_angle = π / 6
    BW_ro = 2000.0 # Hz
    BW_rf = nothing # Hz

    trig_delay = 0.5
    lines_per_trigger = 5
    n_ramp = 13

    prep_spoil_moment = 50π # slice direction
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
    rf = slice_selective_sinc(flip_angle, Δz, sys, sys.RF_dead_time; BW=BW_rf)

    calc_trap_area(GR) = GR.A * (GR.T + GR.rise)
    ss_rewind_area = calc_trap_area(rf.GR[3, 2])

    if matrix[3] == 1
        bSSFP_kernel = bssfp_base(fov[1:2], matrix[1:2], sys, BW_ro;
            fixed_area=(0.0, 0.0, ss_rewind_area))
    else
        bSSFP_kernel = bssfp_base(fov, matrix, sys, BW_ro;
            fixed_area=(0.0, 0.0, ss_rewind_area))
    end

    rf_excitation(rf, scale) = begin
        rf_event = scale * rf[1]
        rf_event.RF[1].use = Excitation()
        return rf_event
    end

    function build_bSSFP(rf, bSSFP_kernel, matrix, fov, flip_angle, sys;
                          n_ramp_shots=10, lines=nothing)
        bSSFP = Sequence(sys)

        ramp_angle_start = π / 180 * 3
        ramp_factors = range(ramp_angle_start / flip_angle, stop=1, length=n_ramp_shots)
        first_line = isnothing(lines) ? -matrix[2] ÷ 2 : lines[1]
        ramp_readout = bSSFP_kernel(first_line)

        @addblocks for (i, f) in enumerate(ramp_factors)
            phase = cis(π * (i - 1))
            bSSFP += rf_excitation(rf, f * phase) + phase * ramp_readout
        end

        for i in findall(is_ADC_on.(bSSFP))
            bSSFP.ADC[i] = ADC(0, 0.0)
        end

        n_lines = isnothing(lines) ? matrix[2] : length(lines)

        @addblocks for i in 1:n_lines
            phase = cis(π * (n_ramp_shots + i - 1))
            j = isnothing(lines) ? i - 1 - matrix[2] ÷ 2 : lines[i]
            bSSFP += rf_excitation(rf, phase) + phase * bSSFP_kernel(j)
        end

        bSSFP.DEF["FOV"] = fov
        return bSSFP
    end

    # Load prep pulse from JLD2
    f = jldopen(prep_fn, "r")
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
    prep_rf_delay = sys.RF_dead_time
    prep_RF = RF(B1, T, 0.0, prep_rf_delay)
    prep_RF.use = Preparation()
    prep_Gx = Grad(Gx, T, 0.0, 0.0, sys.RF_dead_time)
    prep_Gy = Grad(Gy, T, 0.0, 0.0, sys.RF_dead_time)
    prep = Sequence(sys)
    prep_duration = ceil((sys.RF_dead_time + T + sys.RF_ring_down_time) / sys.DUR_Δt) * sys.DUR_Δt
    @addblock check_timing=true prep += (prep_RF, Duration(prep_duration), x=prep_Gx, y=prep_Gy)

    # Build Trigger
    function make_trigger(delay, duration)
        s = Sequence([Grad(0.0, duration + delay, 0.0); Grad(0.0, duration + delay, 0.0); Grad(0.0, duration + delay, 0.0);;])
        s.EXT = [[Trigger(2, 1, delay, duration)]]
        return s
    end
    trigger = make_trigger(0.0, trig_delay)

    # Build spoiler
    M_spoiler = prep_spoil_moment / (γ * 2π) / fov[3]
    T_sp, ζ_sp = _lobe_timing(M_spoiler, sys)
    spoiler = Sequence()
    @addblock spoiler += (z = Grad(M_spoiler / (T_sp + ζ_sp), T_sp, ζ_sp))

    seq = Sequence(sys)
    lines = 1:matrix[2]

    @addblocks check_timing = true for i in 1:lines_per_trigger:matrix[2]
        line_subset = i:min(i + lines_per_trigger - 1, matrix[2])
        line_subset = lines[line_subset] .- (matrix[2] ÷ 2 + 1)
        bSSFP_subset = build_bSSFP(rf, bSSFP_kernel, matrix, fov, flip_angle, sys;
            n_ramp_shots=n_ramp, lines=line_subset)
        seq += trigger
        seq += prep
        seq += spoiler
        seq += bSSFP_subset
    end

    write_seq(seq, output_fn)
    @info "Saved .seq to $output_fn"
end