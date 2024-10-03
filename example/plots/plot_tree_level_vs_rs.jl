using Colors
using CompositeGrids
using ElectronGas
using ElectronLiquid
using GreenFunc
using JLD2
using Lehmann
using LQSGW
using Parameters
using PyCall
using PyPlot

@pyimport numpy as np   # for saving/loading numpy data
@pyimport scienceplots  # for style "science"
@pyimport scipy.interpolate as interp

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "orange" => "#EE7733",
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "magenta" => "#EE3377",
    "red" => "#CC3311",
    "teal" => "#009988",
    "grey" => "#BBBBBB",
]);
style = PyPlot.matplotlib."style"
style.use(["science", "std-colors"])
const color = [
    "black",
    cdict["orange"],
    cdict["blue"],
    cdict["cyan"],
    cdict["magenta"],
    cdict["red"],
    cdict["teal"],
]
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 16
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "Times New Roman"

function spline(x, y, e; xmin=0.0, xmax=x[end])
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(xmin, xmax, 1000))
    yfit = spl(__x)
    return __x, yfit
end

# exchange interaction (Ws + Wa \sigma\sigma)_ex to a direct interaction Ws'+Wa' \sigma\sigma 
# # exchange S/A interaction projected to the spin-symmetric and antisymmetric parts
function exchange_to_direct(Wse, Wae)
    Ws = (Wse + 3 * Wae) / 2
    Wa = (Wse - Wae) / 2
    return Ws, Wa
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via interpolation of the 
compressibility ratio data of Perdew & Wang (1992) [Phys. Rev. B 45, 13244].
"""
@inline function get_Fs_PW(rs)
    # if rs < 1.0 || rs > 5.0
    #     @warn "The Perdew-Wang interpolation for Fs may " *
    #           "be inaccurate outside the metallic regime!"
    # end
    kappa0_over_kappa = 1.0025 - 0.1721 * rs - 0.0036 * rs^2
    # NOTE: NEFT uses opposite sign convention for F!
    # |F⁰ₛ| ≈ 1 - κ₀/κ ⪆ 0
    return 1.0 - kappa0_over_kappa
end

"""
Get the antisymmetric l=0 Fermi-liquid parameter F⁰ₐ via interpolation of the 
susceptibility ratio data (c.f. Kukkonen & Chen, 2021)
"""
@inline function get_Fa_PW(rs)
    chi0_over_chi = 0.9821 - 0.1232 * rs + 0.0091 * rs^2
    # NOTE: NEFT uses opposite sign convention for F!
    # |F⁰ₐ| ≈ 1 - χ₀/χ ⪆ 0
    return 1.0 - chi0_over_chi
end

const alpha_ueg = (4 / 9π)^(1 / 3)

function lindhard(x)
    if x == 0
        return 1.0
    elseif x == 1
        return 0.5
    end
    return 0.5 + ((1 - x^2) / 4x) * log(abs((1 + x) / (1 - x)))
end

function integrand_F0p(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde) && Fs == 0.0
        return -x / lindhard(x)
    elseif isinf(rs_tilde)
        return Inf  # Fs ~ rs^2!
    end
    coeff = rs_tilde + Fs * x^2
    NF_times_Rp_ex = coeff / (x^2 + coeff * lindhard(x)) - Fs
    return -x * NF_times_Rp_ex
end

function integrand_F0m(x, Fa=0.0)
    NF_times_Rm_ex = Fa / (1 + Fa * lindhard(x)) - Fa
    return -x * NF_times_Rm_ex
end

function integrand_F0(x, rs_tilde, Fs=0.0, Fa=0.0)
    return integrand_F0p(x, rs_tilde, Fs) + 3 * integrand_F0m(x, Fa)
end

function integrand_F1p(x, rs_tilde, Fs=0.0)
    return (1 - 2 * x^2) * integrand_F0p(x, rs_tilde, Fs)
end

function integrand_F1m(x, Fa=0.0)
    return (1 - 2 * x^2) * integrand_F0m(x, Fa)
end

function integrand_F1(x, rs_tilde, Fs=0.0, Fa=0.0)
    return integrand_F1p(x, rs_tilde, Fs) + 3 * integrand_F1m(x, Fa)
end

function plot_integrand_F0p(; sign_Fsa=+1.0)
    rslist = [0, 1, 5, 10, Inf]
    colorlist = [cdict["grey"], cdict["orange"], cdict["blue"], cdict["magenta"], "black"]
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 1000)
    for (rs, color) in zip(rslist, colorlist)
        Fs_RPA = 0.0
        Fs_PW = sign_Fsa * get_Fs_PW(rs)
        # if rs > 0.25
        #     @assert Fs_PW < 0 "Incorrect sign for Fs"
        # end
        for (Fs, linestyle) in zip([Fs_RPA, Fs_PW], ["--", "-"])
            if linestyle == "--"
                label = nothing
            else
                rsstr = rs == Inf ? "\\infty" : string(Int(rs))
                label = "\$r_s = $rsstr\$"
            end
            y = [integrand_F0p(xi, rs * alpha_ueg / π, Fs) for xi in x]
            ax.plot(x, y; linestyle=linestyle, color=color, label=label)
        end
    end
    # ax.set_xlabel("\$x = \\left| \\mathbf{k} - \\mathbf{k}^\\prime \\right| / k_F\$")
    ax.set_xlabel("\$x\$")
    ax.set_ylabel("\$I_0(x)\$")
    ax.legend(; fontsize=10, loc="best")
    ylim(-2.4, nothing)
    # tight_layout()
    signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
    fig.savefig("integrand_F0p_$(signstr_Fsa).pdf")
    plt.close("all")
end

function plot_integrand_F1p(; sign_Fsa=+1.0)
    rslist = [0, 1, 5, 10, Inf]
    colorlist = [cdict["grey"], cdict["orange"], cdict["blue"], cdict["magenta"], "black"]
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 1000)
    for (rs, color) in zip(rslist, colorlist)
        Fs_RPA = 0.0
        Fs_PW = sign_Fsa * get_Fs_PW(rs)
        # if rs > 0.25
        #     @assert Fs_PW < 0 "Incorrect sign for Fs"
        # end
        for (Fs, linestyle) in zip([Fs_RPA, Fs_PW], ["--", "-"])
            if linestyle == "--"
                label = nothing
            else
                rsstr = rs == Inf ? "\\infty" : string(Int(rs))
                label = "\$r_s = $rsstr\$"
            end
            y = [integrand_F1p(xi, rs * alpha_ueg / π, Fs) for xi in x]
            ax.plot(x, y; linestyle=linestyle, color=color, label=label)
        end
    end
    # ax.set_xlabel("\$x = \\left| \\mathbf{k} - \\mathbf{k}^\\prime \\right| / k_F\$")
    ax.set_xlabel("\$x\$")
    ax.set_ylabel("\$I_1(x)\$")
    ax.legend(; fontsize=10, loc="best")
    # tight_layout()
    signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
    fig.savefig("integrand_F1p_$(signstr_Fsa).pdf")
    plt.close("all")
end

function get_analytic_F0p(rslist; plot=false, sign_Fsa=+1.0)
    rs_Fsm1 = 5.24881  # Fs(rs = 5.24881) ≈ -1 (using Perdew-Wang fit)
    F0p_RPA = []
    F0p_KOp = []
    F0p_KOm = []
    F0p_KO = []
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    for rs in rslist
        Fs = sign_Fsa * get_Fs_PW(rs)
        Fa = sign_Fsa * get_Fa_PW(rs)
        # if rs > 0.25
        #     @assert Fs < 0 && Fa < 0 "Incorrect sign for Fsa"
        # end
        # RPA
        y_RPA = [integrand_F0p(x, rs * alpha_ueg / π) for x in xgrid]
        val_RPA = CompositeGrids.Interp.integrate1D(y_RPA, xgrid)
        push!(F0p_RPA, val_RPA)
        # KO+
        y_KOp = [integrand_F0p(x, rs * alpha_ueg / π, Fs) for x in xgrid]
        val_KOp = CompositeGrids.Interp.integrate1D(y_KOp, xgrid)
        push!(F0p_KOp, val_KOp)
        # KO-
        y_KOm = [integrand_F0m(x, Fa) for x in xgrid]
        val_KOm = CompositeGrids.Interp.integrate1D(y_KOm, xgrid)
        push!(F0p_KOm, val_KOm)
        # KO
        y_KO = [integrand_F0(x, rs * alpha_ueg / π, Fs, Fa) for x in xgrid]
        val_KO = CompositeGrids.Interp.integrate1D(y_KO, xgrid)
        push!(F0p_KO, val_KO)
    end
    if plot
        fig, ax = plt.subplots()
        ax.plot(rslist, F0p_RPA; color=cdict["orange"], label="\$W_0\$")
        ax.plot(rslist, F0p_KOp; color=cdict["blue"], label="\$W^\\text{KO}_{0,+}\$")
        ax.plot(rslist, F0p_KOm; color=cdict["cyan"], label="\$W^\\text{KO}_{0,-}\$")
        ax.plot(
            rslist,
            F0p_KO;
            color=cdict["magenta"],
            label="\$W^\\text{KO}_{0} = W^\\text{KO}_{0,+} + 3 W^\\text{KO}_{0,-}\$",
        )
        labelstr =
            sign_Fsa > 0 ? "\$-F^+ = 1 - \\kappa_0 / \\kappa\$" :
            "\$F^+ = \\kappa_0 / \\kappa - 1\$"
        ax.plot(
            rslist,
            sign_Fsa * get_Fs_PW.(rslist);
            color=cdict["grey"],
            label=labelstr,
            zorder=-1,
        )
        legend(; loc="best", fontsize=12)
        xlabel("\$r_s\$")
        ylabel("\$F^+_{0,t}\$")
        # ylabel("\$F^+_{0,t} - F^+\$")
        # ylim(-1.1, 0.6)
        ax.legend(; fontsize=10, loc="best")
        # tight_layout()
        signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
        fig.savefig("analytic_F0p_$(signstr_Fsa).pdf")
        plt.close("all")
    end
    return rslist, F0p_RPA, F0p_KOp, F0p_KOm, F0p_KO
end

function get_analytic_F1p(rslist; plot=false, sign_Fsa=+1.0)
    rs_Fsm1 = 5.24881  # Fs(rs = 5.24881) ≈ -1 (using Perdew-Wang fit)
    F1p_RPA = []
    F1p_KOp = []
    F1p_KOm = []
    F1p_KO = []
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    for rs in rslist
        Fs = sign_Fsa * get_Fs_PW(rs)
        Fa = sign_Fsa * get_Fa_PW(rs)
        # if rs > 0.25
        #     @assert Fs < 0 && Fa < 0 "Incorrect sign for Fsa"
        # end
        # RPA
        y_RPA = [integrand_F1p(x, rs * alpha_ueg / π) for x in xgrid]
        val_RPA = CompositeGrids.Interp.integrate1D(y_RPA, xgrid)
        push!(F1p_RPA, val_RPA)
        # KO+
        y_KOp = [integrand_F1p(x, rs * alpha_ueg / π, Fs) for x in xgrid]
        val_KOp = CompositeGrids.Interp.integrate1D(y_KOp, xgrid)
        push!(F1p_KOp, val_KOp)
        # KO-
        y_KOm = [integrand_F1m(x, Fa) for x in xgrid]
        val_KOm = CompositeGrids.Interp.integrate1D(y_KOm, xgrid)
        push!(F1p_KOm, val_KOm)
        # KO
        y_KO = [integrand_F1(x, rs * alpha_ueg / π, Fs, Fa) for x in xgrid]
        val_KO = CompositeGrids.Interp.integrate1D(y_KO, xgrid)
        push!(F1p_KO, val_KO)
    end
    if plot
        fig, ax = plt.subplots()
        ax.plot(rslist, F1p_RPA; color=cdict["orange"], label="\$W_0\$")
        ax.plot(rslist, F1p_KOp; color=cdict["blue"], label="\$W^\\text{KO}_{0,+}\$")
        ax.plot(rslist, F1p_KOm; color=cdict["cyan"], label="\$W^\\text{KO}_{0,-}\$")
        ax.plot(
            rslist,
            F1p_KO;
            color=cdict["magenta"],
            label="\$W^\\text{KO}_{0} = W^\\text{KO}_{0,+} + 3 W^\\text{KO}_{0,-}\$",
        )
        legend(; loc="best", fontsize=12)
        xlabel("\$r_s\$")
        ylabel("\$F^+_{1,t}\$")
        ylim(-0.21, nothing)
        ax.legend(; fontsize=10, loc="best")
        # tight_layout()
        signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
        fig.savefig("analytic_F1p_$(signstr_Fsa).pdf")
        plt.close("all")
    end
    return rslist, F1p_RPA, F1p_KOp, F1p_KOm, F1p_KO
end

function main()
    # UEG parameters
    beta = 1000.0
    dim = 3

    rs_Fsm1 = 5.24881  # Fs(rs = 5.24881) ≈ -1 (using Perdew-Wang fit)
    rslist = sort(unique([0.01; rs_Fsm1; collect(range(0.125, 10.0; step=0.125))]))

    # Using the ElectronLiquid.jl (v + f) convention ⟹ F± < 0
    sign_Fsa = -1.0
    signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"

    # l=0 plots
    plot_integrand_F0p(; sign_Fsa=sign_Fsa)
    _, F0p_RPA, F0p_KOp, _, F0p_KO = get_analytic_F0p(rslist; plot=true, sign_Fsa=sign_Fsa)
    pushfirst!(F0p_RPA, 0.0)
    pushfirst!(F0p_KOp, 0.0)
    pushfirst!(F0p_KO, 0.0)

    # l=1 plots
    plot_integrand_F1p(; sign_Fsa=sign_Fsa)
    _, F1p_RPA, F1p_KOp, _, F1p_KO = get_analytic_F1p(rslist; plot=true, sign_Fsa=sign_Fsa)
    pushfirst!(F1p_RPA, 0.0)
    pushfirst!(F1p_KOp, 0.0)
    pushfirst!(F1p_KO, 0.0)

    Fp_vs_rs = []
    # F0p_rpa_vs_rs = []
    # F0p_fp_vs_rs = []
    # F0p_fp_fm_vs_rs = []
    F0p_rpa_vs_rs_ueg = []
    F0p_fp_vs_rs_ueg = []
    F0p_fp_fm_vs_rs_ueg = []
    # F1p_rpa_vs_rs = []
    # F1p_fp_vs_rs = []
    # F1p_fp_fm_vs_rs = []
    F1p_rpa_vs_rs_ueg = []
    F1p_fp_vs_rs_ueg = []
    F1p_fp_fm_vs_rs_ueg = []
    for rs in rslist
        param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
        @unpack kF, EF = param
        # DLR parameters
        Euv = 1000 * EF
        rtol = 1e-14
        # ElectronGas.jl defaults for G0W0 self-energy
        maxK = 6 * kF
        minK = 1e-6 * kF
        # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
        Fs = sign_Fsa * get_Fs_PW(rs)
        Fa = sign_Fsa * get_Fa_PW(rs)

        # nu_grid =
        #     CompositeGrid.LogDensedGrid(:gauss, [-1.0, 1.0], [-1.0, 0.0, 1.0], 32, 1e-8, 32)
        # # nu_grid =
        # #     CompositeGrid.LogDensedGrid(:gauss, [-1.0, 1.0], [-1.0, 0.0, 1.0], 20, 1e-6, 12)
        # nus = nu_grid.grid
        # thetas = acos.(nus)

        # # |k - k'| = kF sqrt(2(1 - ν))
        # k_m_kps = @. kF * sqrt(2 * (1 - nus))

        # # Get W0_{+}(p, iωₙ=0) (spin-symmetric static exchange part of RPA interaction)
        # W0_ex_dyn, W0_ex_inst_inv =
        #     Interaction.RPAwrapped(Euv, rtol, k_m_kps, param; regular=false)
        # W0_ex_inst = 1 ./ W0_ex_inst_inv
        # @assert W0_ex_dyn.mesh[2].grid[1] == 0

        # # Get the static, spin-symmetric/antisymmetric parts of the interaction,
        # # converting from exchange interaction (Ws + Wa \sigma\sigma)_ex to a direct interaction Ws'+Wa' \sigma\sigma
        # W0_ex_static_s = W0_ex_dyn[1, 1, :] + W0_ex_inst[1, 1, :]
        # W0_ex_static_a = W0_ex_dyn[2, 1, :] + W0_ex_inst[2, 1, :]
        # W0_plus_static, W0_minus_static = exchange_to_direct(W0_ex_static_s, W0_ex_static_a)
        # # W0_plus_static = W0_ex_static_s
        # # W0_minus_static = W0_ex_static_a
        # println("rs = $rs:")
        # println(
        #     "static W0ex+(q = 0) = $(real(W0_ex_static_s[1])), static W0ex-(q = 0) = $(real(W0_ex_static_a[1]))",
        # )
        # println(
        #     "static W0+(q = 0) = $(real(W0_plus_static[1])), static W0-(q = 0) = $(real(W0_minus_static[1]))",
        # )
        # @assert maximum(imag(W0_plus_static)) ≤ 1e-10

        # # Get R_{+}(p, iωₙ=0) (spin-symmetric static exchange part of KO interaction)
        # R_plus_ex_dyn, R_plus_ex_inst_inv = Interaction.KOwrapped(
        #     Euv,
        #     rtol,
        #     k_m_kps,
        #     param;
        #     regular=false,
        #     int_type=:ko_const,
        #     landaufunc=Interaction.landauParameterConst,
        #     Fs=Fs,
        #     Fa=0.0,
        # )
        # R_plus_ex_inst = 1 ./ R_plus_ex_inst_inv
        # @assert R_plus_ex_dyn.mesh[2].grid[1] == 0

        # # Get the static, spin-symmetric/antisymmetric parts of the interaction,
        # # converting from exchange interaction (Rs + Ra \sigma\sigma)_ex to a direct interaction Rs'+Ra' \sigma\sigma
        # R_plus_ex_static_s = R_plus_ex_dyn[1, 1, :] + R_plus_ex_inst[1, 1, :]
        # R_plus_ex_static_a = R_plus_ex_dyn[2, 1, :] + R_plus_ex_inst[2, 1, :]
        # R_plus_static, R_plus_minus_static =
        #     exchange_to_direct(R_plus_ex_static_s, R_plus_ex_static_a)
        # # R_plus_static = R_plus_ex_static_s
        # # R_minus_static = R_plus_ex_static_a
        # println("rs = $rs:")
        # println(
        #     "(fp != 0, fm = 0) static Rex+(q = 0) = $(real(R_plus_ex_static_s[1])), static Rex-(q = 0) = $(real(R_plus_ex_static_a[1]))",
        # )
        # println(
        #     "(fp != 0, fm = 0) static R+(q = 0) = $(real(R_plus_static[1])), static R-(q = 0) = $(real(R_plus_minus_static[1]))",
        # )
        # @assert maximum(imag(R_plus_static)) ≤ 1e-10

        # # Get R(p, iωₙ=0) (spin-symmetric static exchange part of KO interaction)
        # R_ex_dyn, R_ex_inst_inv = Interaction.KOwrapped(
        #     Euv,
        #     rtol,
        #     k_m_kps,
        #     param;
        #     regular=false,
        #     int_type=:ko_const,
        #     landaufunc=Interaction.landauParameterConst,
        #     Fs=Fs,
        #     Fa=Fa,
        # )
        # R_ex_inst = 1 ./ R_ex_inst_inv
        # @assert R_ex_dyn.mesh[2].grid[1] == 0

        # # Get the static, spin-symmetric/antisymmetric parts of the interaction,
        # # converting from exchange interaction (Rs + Ra \sigma\sigma)_ex to a direct interaction Rs'+Ra' \sigma\sigma
        # R_ex_static_s = R_ex_dyn[1, 1, :] + R_ex_inst[1, 1, :]
        # R_ex_static_a = R_ex_dyn[2, 1, :] + R_ex_inst[2, 1, :]
        # R_static, R_minus_static = exchange_to_direct(R_ex_static_s, R_ex_static_a)
        # # R_static = R_ex_static_s
        # # R_minus_static = R_ex_static_a
        # println("rs = $rs:")
        # println(
        #     "(fp != 0, fm != 0) static Rex+(q = 0) = $(real(R_ex_static_s[1])), static Rex-(q = 0) = $(real(R_ex_static_a[1]))",
        # )
        # println(
        #     "(fp != 0, fm != 0) static R+(q = 0) = $(real(R_static[1])), static R-(q = 0) = $(real(R_minus_static[1]))",
        # )
        # @assert maximum(imag(R_static)) ≤ 1e-10

        # # The F0 angular integrands are W0_{+}(p, iωₙ=0) / 2, R_{+}(p, iωₙ=0) / 2, and (R_{+}(p, iωₙ=0) + 3 R_{-}(p, iωₙ=0)) / 2, respectively
        # F0p_rpa_integrand = @. -0.25 * param.NF * real(W0_plus_static)
        # F0p_fp_integrand = @. -0.25 * param.NF * real(R_plus_static)
        # F0p_fp_fm_integrand = @. -0.25 * param.NF * real(R_static)

        # # The F1 angular integrands are ν W0_{+}(p, iωₙ=0) / 2, ν R_{+}(p, iωₙ=0) / 2, and ν (R_{+}(p, iωₙ=0) + 3 R_{-}(p, iωₙ=0)) / 2, respectively
        # F1p_rpa_integrand = @. -0.5 * param.NF * nus * real(W0_plus_static)
        # F1p_fp_integrand = @. -0.5 * param.NF * nus * real(R_plus_static)
        # F1p_fp_fm_integrand = @. -0.5 * param.NF * nus * real(R_static)

        # # Perform angular integrations ν ∈ [-1, 1]
        # F0p_rpa = CompositeGrids.Interp.integrate1D(F0p_rpa_integrand, nu_grid)
        # F0p_fp = CompositeGrids.Interp.integrate1D(F0p_fp_integrand, nu_grid)
        # F0p_fp_fm = CompositeGrids.Interp.integrate1D(F0p_fp_fm_integrand, nu_grid)
        # F1p_rpa = CompositeGrids.Interp.integrate1D(F1p_rpa_integrand, nu_grid)
        # F1p_fp = CompositeGrids.Interp.integrate1D(F1p_fp_integrand, nu_grid)
        # F1p_fp_fm = CompositeGrids.Interp.integrate1D(F1p_fp_fm_integrand, nu_grid)

        # Use ElectronLiquid.jl to compute the same quantities for constant Fs
        p_rpa = ParaMC(; rs=rs, beta=beta, Fs=0.0, Fa=0.0, order=1, mass2=0.0)
        p_fp = ParaMC(; rs=rs, beta=beta, Fs=Fs, Fa=0.0, order=1, mass2=0.0)
        p_fp_fm = ParaMC(; rs=rs, beta=beta, Fs=Fs, Fa=Fa, order=1, mass2=0.0)
        F0p_rpa_ueg, _ =
            -1 .* Ver4.projected_exchange_interaction(0, p_rpa, Ver4.exchange_interaction)
        F0p_fp_ueg, _ =
            -1 .* Ver4.projected_exchange_interaction(0, p_fp, Ver4.exchange_interaction)
        F0p_fp_fm_ueg, _ =
            -1 .* Ver4.projected_exchange_interaction(0, p_fp_fm, Ver4.exchange_interaction)
        F1p_rpa_ueg, _ =
            -1 .* Ver4.projected_exchange_interaction(1, p_rpa, Ver4.exchange_interaction)
        F1p_fp_ueg, _ =
            -1 .* Ver4.projected_exchange_interaction(1, p_fp, Ver4.exchange_interaction)
        F1p_fp_fm_ueg, _ =
            -1 .* Ver4.projected_exchange_interaction(1, p_fp_fm, Ver4.exchange_interaction)
        # println(
        #     "\nrs = $rs:" *
        #     "\nF^{(RPA)+}_0 = $F0p_rpa" *
        #     "\nF^{(fp)+}_0 = $F0p_fp" *
        #     "\nF^{(fp and fm)+}_0 = $F0p_fp_fm",
        # )
        println(
            "\nElectronLiquid:" *
            "\nF^{(RPA)+}_0 = $F0p_rpa_ueg" *
            "\nF^{(fp)+}_0 = $F0p_fp_ueg",
            "\nF^{(fp and fm)+}_0 = $F0p_fp_fm_ueg",
        )
        # println(
        #     "\nrs = $rs:" *
        #     "\nF^{(RPA)+}_1 = $F1p_rpa" *
        #     "\nF^{(fp)+}_1 = $F1p_fp" *
        #     "\nF^{(fp and fm)+}_1 = $F1p_fp_fm",
        # )
        println(
            "\nElectronLiquid:" *
            "\nF^{(RPA)+}_1 = $F1p_rpa_ueg" *
            "\nF^{(fp)+}_1 = $F1p_fp_ueg",
            "\nF^{(fp and fm)+}_1 = $F1p_fp_fm_ueg",
        )
        push!(Fp_vs_rs, Fs)
        # push!(F0p_rpa_vs_rs, F0p_rpa)
        # push!(F0p_fp_vs_rs, F0p_fp)
        # push!(F0p_fp_fm_vs_rs, F0p_fp_fm)
        push!(F0p_rpa_vs_rs_ueg, F0p_rpa_ueg)
        push!(F0p_fp_vs_rs_ueg, F0p_fp_ueg)
        push!(F0p_fp_fm_vs_rs_ueg, F0p_fp_fm_ueg)
        # push!(F1p_rpa_vs_rs, F1p_rpa)
        # push!(F1p_fp_vs_rs, F1p_fp)
        # push!(F1p_fp_fm_vs_rs, F1p_fp_fm)
        push!(F1p_rpa_vs_rs_ueg, F1p_rpa_ueg)
        push!(F1p_fp_vs_rs_ueg, F1p_fp_ueg)
        push!(F1p_fp_fm_vs_rs_ueg, F1p_fp_fm_ueg)
    end

    # Add points at rs = 0
    pushfirst!(rslist, 0.0)
    pushfirst!(Fp_vs_rs, 0.0)
    # pushfirst!(F0p_rpa_vs_rs, 0.0)
    # pushfirst!(F0p_fp_vs_rs, 0.0)
    # pushfirst!(F0p_fp_fm_vs_rs, 0.0)
    pushfirst!(F0p_rpa_vs_rs_ueg, 0.0)
    pushfirst!(F0p_fp_vs_rs_ueg, 0.0)
    pushfirst!(F0p_fp_fm_vs_rs_ueg, 0.0)
    # pushfirst!(F1p_rpa_vs_rs, 0.0)
    # pushfirst!(F1p_fp_vs_rs, 0.0)
    # pushfirst!(F1p_fp_fm_vs_rs, 0.0)
    pushfirst!(F1p_rpa_vs_rs_ueg, 0.0)
    pushfirst!(F1p_fp_vs_rs_ueg, 0.0)
    pushfirst!(F1p_fp_fm_vs_rs_ueg, 0.0)

    color = [
        [cdict["orange"], cdict["magenta"], cdict["red"]],
        [cdict["blue"], cdict["cyan"], cdict["teal"]],
        [cdict["blue"], cdict["cyan"], cdict["teal"]],
    ]

    function plot_mvsrs(rs, meff_data, color, label, ls="-"; ax1=plt.gca(), zorder=nothing)
        # mfitfunc = interp.PchipInterpolator(rs, meff_data)
        mfitfunc = interp.Akima1DInterpolator(rs, meff_data)
        xgrid = np.arange(0, maximum(rslist) + 0.2, 0.01)
        # xgrid = np.arange(0, 6.2, 0.01)
        # ax1.scatter(rs, meff_data; color=color, marker="o")
        if isnothing(zorder)
            handle, = ax1.plot(xgrid, mfitfunc(xgrid); ls=ls, color=color, label=label)
        else
            handle, = ax1.plot(
                xgrid,
                mfitfunc(xgrid);
                ls=ls,
                color=color,
                label=label,
                zorder=zorder,
            )
        end
        yfit = np.ma.masked_invalid(mfitfunc(xgrid))
        # print(yfit)
        # print("Turning point: rs = ", xgrid[np.argmin(yfit)])
        # print("Effective mass ratio at turning point: ", np.min(yfit))
        return handle
    end

    # Plot F0 vs rs
    fig1 = figure(; figsize=(6, 6))
    ax1 = fig1.add_subplot(111)

    # Fsull F^+(rs)
    labelstr =
        sign_Fsa > 0 ? "\$-F^+ = \\kappa_0 / \\kappa - 1\$" :
        "\$F^+ = \\kappa_0 / \\kappa - 1\$"
    plot_mvsrs(rslist, -sign_Fsa * Fp_vs_rs, cdict["grey"], labelstr, "-")

    # Tree-level RPA
    plot_mvsrs(rslist, F0p_rpa_vs_rs_ueg, cdict["orange"], "\$W_0\$", "-")
    plot_mvsrs(rslist, F0p_RPA, cdict["orange"], nothing, "--")
    # plot_mvsrs(rslist, F0p_rpa_vs_rs, cdict["orange"], "\$W_0\$", "-")
    # plot_mvsrs(rslist, F0p_rpa_vs_rs_ueg, cdict["blue"], "\$W_0\$ (NEFT)", "--")

    # Tree-level KO with fp only
    plot_mvsrs(rslist, F0p_fp_vs_rs_ueg, cdict["blue"], "\$W^\\text{KO}_{0,+}\$", "-")
    plot_mvsrs(rslist, F0p_KOp, cdict["blue"], nothing, "--")
    # plot_mvsrs(rslist, F0p_fp_vs_rs, cdict["red"], "\$W^\\text{KO}_{0,+}\$", "-")
    # plot_mvsrs(rslist, F0p_fp_vs_rs_ueg, cdict["teal"], "\$W^\\text{KO}_{0,+}\$ (NEFT)", "--")

    # Tree-level KO with fp and fm
    plot_mvsrs(rslist, F0p_fp_fm_vs_rs_ueg, cdict["magenta"], "\$W^\\text{KO}_{0}\$", "-")
    plot_mvsrs(rslist, F0p_KO, cdict["magenta"], nothing, "--")
    # plot_mvsrs(rslist, F0p_fp_fm_vs_rs, cdict["magenta"], "\$W^\\text{KO}_{0}\$", "-")
    # plot_mvsrs(rslist, F0p_fp_fm_vs_rs_ueg, cdict["magenta"], "\$W^\\text{KO}_{0}\$ (NEFT)", "--")

    legend(; loc="best", fontsize=10)
    ylabel("\$F^+_{0,t}\$")
    # ylabel("\$F^+_0 \\approx \\langle W(k + k^\\prime - q) \\rangle_\\text{F.S.}\$")
    # ylabel("\$m^* / m\$")
    xlabel("\$r_s\$")
    # ylim(-0.056, 0.034)
    # ylim(-1.1, 0.6)
    tight_layout()
    savefig("F0p_comparisons_ko_const_Fs_Fa_$(signstr_Fsa).pdf")
    # savefig("F1p_comparisons_ko_takada.pdf")
    # savefig("F1p_comparisons_ko_simion_giuliani.pdf")

    # Plot F1 vs rs
    fig1 = figure(; figsize=(6, 6))
    ax1 = fig1.add_subplot(111)

    # Tree-level RPA
    plot_mvsrs(rslist, F1p_rpa_vs_rs_ueg, cdict["orange"], "\$W_0\$", "-")
    plot_mvsrs(rslist, F1p_RPA, cdict["orange"], nothing, "--")
    # plot_mvsrs(rslist, F1p_rpa_vs_rs_ueg, cdict["blue"], "\$W_0\$ (NEFT)", "--")

    # Tree-level KO with fp only
    plot_mvsrs(rslist, F1p_fp_vs_rs_ueg, cdict["blue"], "\$W^\\text{KO}_{0,+}\$", "-")
    plot_mvsrs(rslist, F1p_KOp, cdict["blue"], nothing, "--")
    # plot_mvsrs(rslist, F1p_fp_vs_rs_ueg, cdict["teal"], "\$W^\\text{KO}_{0,+}\$ (NEFT)", "--")

    # Tree-level KO with fp and fm
    plot_mvsrs(rslist, F1p_fp_fm_vs_rs_ueg, cdict["magenta"], "\$W^\\text{KO}_{0}\$", "-")
    plot_mvsrs(rslist, F1p_KO, cdict["magenta"], nothing, "--")
    # plot_mvsrs(rslist, F1p_fp_fm_vs_rs_ueg, cdict["magenta"], "\$W^\\text{KO}_{0}\$ (NEFT)", "--")

    legend(; loc="best", fontsize=10)
    ylabel("\$F^+_{1,t}\$")
    # ylabel(
    #     "\$F^+_1 \\approx \\langle W(k + k^\\prime - q) \\cos\\theta_{k,k^\\prime}\\rangle_\\text{F.S.}\$",
    # )
    xlabel("\$r_s\$")
    # ylim(-0.056, 0.034)
    # ylim(-0.072, 0.034)
    tight_layout()
    savefig("F1p_comparisons_ko_const_Fs_Fa_$(signstr_Fsa).pdf")
    # savefig("F1p_comparisons_ko_takada.pdf")
    # savefig("F1p_comparisons_ko_simion_giuliani.pdf")
    return
end

main()
