function RF_sinc(B1, T, sys, G, Δf, a, TBP, deadtime)
    t0 = T / TBP
    ζ = ceil(maximum(abs.(G)) / sys.Smax / sys.GR_Δt) * sys.GR_Δt
    sinc_pulse(t) = B1 * sinc(t / t0) .* ((1 - a) + a * cos((2π * t) / (TBP * t0)))
    t_rf = 0:sys.RF_Δt:T
    A_rf = sinc_pulse.(t_rf .- T / 2)
    T_rew = ceil(max((T - ζ) / 2, 0.0) / sys.GR_Δt) * sys.GR_Δt
    G_rew_amp = iszero(T_rew + ζ) ? zero.(G) : G .* (-(T + ζ) / (2 * (T_rew + ζ)))
    rf_event = RF(collect(A_rf), diff(t_rf), Δf, ζ + deadtime)
    G_ss = (x=Grad(G[1], T, ζ, ζ, deadtime), y=Grad(G[2], T, ζ, ζ, deadtime), z=Grad(G[3], T, ζ, ζ, deadtime))
    G_rew = (x=Grad(G_rew_amp[1], T_rew, ζ), y=Grad(G_rew_amp[2], T_rew, ζ), z=Grad(G_rew_amp[3], T_rew, ζ))
    seq = Sequence(sys)
    @addblock seq += (Duration(T+2ζ+deadtime+sys.GR_Δt), rf_event; G_ss...) + (; G_rew...)
    return seq
end


function slice_selective_sinc(ϕ, Δz, sys, deadtime; BW=nothing, Δf=0.0, a=0.46, TBP=4)

    # ── Bandwidth: use supplied value or compute hardware maximum ──────────────
    BW_Gz = γ * sys.Gmax * Δz                        # gradient ceiling [Hz]
    BW_B1 = 2π * γ * sys.B1 / ϕ                     # B1 ceiling [Hz]

    BW = if isnothing(BW)
        min(BW_Gz, BW_B1)
    else
        BW > BW_Gz && error(
            "BW=$(round(BW))Hz requires Gz=$(round(BW/(γ*Δz)*1e3,digits=2)) mT/m " *
            "exceeding Gmax=$(round(sys.Gmax*1e3,digits=2)) mT/m")
        BW > BW_B1 && error(
            "BW=$(round(BW))Hz requires B1=$(round(ϕ*BW/(2π*γ)*1e6,digits=2)) μT " *
            "exceeding sys.B1=$(round(sys.B1*1e6,digits=2)) μT")
        BW
    end

    # ── Derived parameters ─────────────────────────────────────────────────────
    T = ceil(TBP / BW / sys.GR_Δt) * sys.GR_Δt       # pulse duration [s]
    BW = TBP / T
    Gz = BW / (γ * Δz)                               # slice-select gradient [T/m]
    B1 = ϕ * BW / (2π * γ)                           # peak B1 [T]

    return RF_sinc(B1, T, sys, (0.0, 0.0, Gz), Δf, a, TBP, deadtime)
end
