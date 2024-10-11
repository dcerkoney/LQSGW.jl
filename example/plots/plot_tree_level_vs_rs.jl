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

# Mapping of interaction types to Landau parameters
const int_type_to_landaufunc = Dict(
    :rpa => Interaction.landauParameter0,
    :ko_const => Interaction.landauParameterConst,
    :ko_takada => Interaction.landauParameterTakada,
    :ko_takada_plus => Interaction.landauParameterTakadaPlus,
    :ko_moroni => Interaction.landauParameterMoroni,
    :ko_simion_giuliani => Interaction.landauParameterSimionGiuliani,
    :ko_simion_giuliani_plus => Interaction.landauParameterSimionGiulianiPlus,
)

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "black" => "black",
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

# Taylor series for m* / m in the high-density limit to leading order in rs
# (c.f. Giuliani & Vignale, Quantum Theory of the Electron Liquid, 2008, p. 500)
function high_density_limit(x)
    return 1 + alpha_ueg * x * np.log(x) / 2π
end

function lindhard(x)
    if abs(x) < 1e-4
        return 1.0 - x^2 / 3 - x^4 / 15
    elseif abs(x - 1) < 1e-5
        return 0.5
    elseif x > 20
        return 1 / (3 * x^2) + 1 / (15 * x^4)
    end
    return 0.5 + ((1 - x^2) / (4 * x)) * log(abs((1 + x) / (1 - x)))
end

function integrand_F0p(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return -2 * x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    NF_times_Rp_ex = coeff / (x^2 + coeff * lindhard(x))
    return -2 * x * NF_times_Rp_ex
end

function integrand_F0m(x, Fa=0.0)
    NF_times_Rm_ex = Fa / (1 + Fa * lindhard(x))
    return -2 * x * NF_times_Rm_ex
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

function plot_Fermi_liquid_parameters(; rs=1.0, sign_Fsa=-1.0)
    # Calculate Fa and Fa under various approximations
    beta = 1000.0
    param = Parameter.rydbergUnit(1.0 / beta, rs, 3)
    @unpack kF, NF = param
    Fs_C = sign_Fsa * get_Fs_PW(rs)
    Fa_C = sign_Fsa * get_Fa_PW(rs)
    if rs > 0.25
        @assert Fs_C < 0 && Fa_C < 0 "Incorrect signs for Fs/Fa!"
    end
    Fs_T_list = []
    Fa_T_list = []
    Fs_SG_list = []
    Fa_SG_list = []
    qlist = np.linspace(1e-7 * kF, 4 * kF, 1000)
    for q in qlist
        fs_T, fa_T = Interaction.landauParameterTakada(q, 0, param)
        fs_SG, fa_SG = Interaction.landauParameterSimionGiuliani(q, 0, param)
        # ElectronLiquid.jl uses the opposite sign 
        # convention to ElectronGas.jl for Fs/Fa!
        push!(Fs_T_list, sign_Fsa * NF * fs_T)
        push!(Fa_T_list, sign_Fsa * NF * fa_T)
        push!(Fs_SG_list, sign_Fsa * NF * fs_SG)
        push!(Fa_SG_list, sign_Fsa * NF * fa_SG)
    end
    fs_SG_inf, fa_SG_inf = Interaction.landauParameterSimionGiuliani(1e13, 0, param)
    Fsa_SG_inf = sign_Fsa * NF * fs_SG_inf

    # Plot Fs and Fa
    fig = figure(; figsize=(6, 6))
    ax = fig.add_subplot(111)
    # Fs
    Fs_labelstr =
        sign_Fsa < 0 ? "\$\\left(\\frac{\\kappa_0}{\\kappa} - 1\\right)_\\text{DMC}\$" :
        "\$\\left(1 - \\frac{\\kappa_0}{\\kappa}\\right)_\\text{DMC}\$"
    # ax.plot(qlist / kF, Fs_T_list; color=cdict["orange"], label="\$F_+^\\text{T}(q)\$")
    ax.plot(qlist / kF, Fs_SG_list; color=cdict["red"], label="\$F_+(q)\$")
    ax.axhline(Fs_C; color=cdict["magenta"], label=Fs_labelstr, linestyle="--")
    # Fa
    Fa_labelstr =
        sign_Fsa < 0 ? "\$\\left(\\frac{\\chi_0}{\\chi} - 1\\right)_\\text{DMC}\$" :
        "\$\\left(1 - \\frac{\\chi_0}{\\chi}\\right)_\\text{DMC}\$"
    # ax.plot(qlist / kF, Fa_T_list; color=cdict["teal"], label="\$F_-^\\text{T}(q)\$")
    ax.plot(qlist / kF, Fa_SG_list; color=cdict["blue"], label="\$F_-(q)\$")
    ax.axhline(Fa_C; color=cdict["cyan"], label=Fa_labelstr, linestyle="--")
    ax.axhline(
        Fsa_SG_inf;
        color=cdict["black"],
        label="\$F_\\pm(\\infty)\$",
        linestyle="--",
    )
    ax.set_xlabel("\$q / k_F\$")
    ax.set_ylabel("\$F_\\pm(q)\$")
    ax.legend(;
        # loc="lower center",
        # loc="upper center",
        loc=sign_Fsa < 0 ? "upper center" : "lower center",
        fontsize=12,
        title="\$r_s = $(Int(round(rs)))\$",
        ncol=3,
        # columnspacing=0.2,
    )
    ax.set_xlim(0, 4)
    if sign_Fsa < 0
        ax.set_ylim(-0.22, 0.06)
    else
        ax.set_ylim(-0.06, 0.22)
    end
    plt.tight_layout()
    signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
    fig.savefig("Fs_and_Fa_comparison_rs=$(rs)_$(signstr_Fsa).pdf")
    return
end

function plot_integrand_F0p(; sign_Fsa=-1.0)
    rslist = [0, 1, 5, 10, 20, Inf]
    colorlist = [
        cdict["grey"],
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        "black",
    ]
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
    ax.set_ylabel("\$I_0[W](x)\$")
    ax.legend(; fontsize=10, loc="best", ncol=2)
    xlim(0, 1)
    ylim(-4.25, 2.25)
    # tight_layout()
    signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
    fig.savefig("integrand_F0p_$(signstr_Fsa).pdf")
    plt.close("all")
end

function plot_integrand_F1p(; sign_Fsa=-1.0)
    rslist = [0, 1, 5, 10, 20, Inf]
    colorlist = [
        cdict["grey"],
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        "black",
    ]
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
    ax.set_ylabel("\$I_1[W](x)\$")
    ax.legend(; fontsize=10, loc="best", ncol=2)
    xlim(0, 1)
    ylim(-2.25, 4.25)
    # tight_layout()
    signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
    fig.savefig("integrand_F1p_$(signstr_Fsa).pdf")
    plt.close("all")
end

function get_analytic_F0p(rslist; plot=false, sign_Fsa=-1.0)
    rs_Fsm1 = 5.24881  # Fs(rs = 5.24881) ≈ -1 (using Perdew-Wang fit)
    F0p_RPA = []
    F0p_KOp = []
    F0p_KOm = []
    F0p_KO = []
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    y0_RPA_inf = [integrand_F0p(x, Inf) for x in xgrid]
    y1_RPA_inf = [integrand_F1p(x, Inf) for x in xgrid]
    F0p_RPA_inf = CompositeGrids.Interp.integrate1D(y0_RPA_inf, xgrid)
    F1p_RPA_inf = CompositeGrids.Interp.integrate1D(y1_RPA_inf, xgrid)
    println("F⁺₀[W₀](∞) = $(F0p_RPA_inf)")
    println("F⁺₁[W₀](∞) = $(F1p_RPA_inf)")
    for rs in rslist
        Fs = sign_Fsa * get_Fs_PW(rs)
        Fa = sign_Fsa * get_Fa_PW(rs)
        # if rs > 0.25
        #     @assert Fs < 0 && Fa < 0 "Incorrect signs for Fs/Fa!"
        # end
        # RPA
        y_RPA = [integrand_F0p(x, rs * alpha_ueg / π) for x in xgrid]
        val_RPA = CompositeGrids.Interp.integrate1D(y_RPA, xgrid)
        push!(F0p_RPA, val_RPA)
        # KO+
        y_KOp = [integrand_F0p(x, rs * alpha_ueg / π, Fs) for x in xgrid]
        val_KOp = Fs + CompositeGrids.Interp.integrate1D(y_KOp, xgrid)
        push!(F0p_KOp, val_KOp)
        # KO-
        y_KOm = [integrand_F0m(x, Fa) for x in xgrid]
        val_KOm = Fa + CompositeGrids.Interp.integrate1D(y_KOm, xgrid)
        push!(F0p_KOm, val_KOm)
        # KO
        Fse = (Fs + 3 * Fa)
        y_KO = [integrand_F0(x, rs * alpha_ueg / π, Fs, Fa) for x in xgrid]
        val_KO = Fse + CompositeGrids.Interp.integrate1D(y_KO, xgrid)
        push!(F0p_KO, val_KO)
    end
    if plot
        fig, ax = plt.subplots()
        ax.plot(rslist, F0p_RPA; color="black", label="\$W_0\$")
        ax.plot(rslist, F0p_KOp; color=cdict["blue"], label="\$W^\\text{KO}_{0,+}\$")
        ax.plot(rslist, F0p_KOm; color=cdict["teal"], label="\$W^\\text{KO}_{0,-}\$")
        ax.plot(
            rslist,
            F0p_KO;
            color=cdict["red"],
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
        xlabel("\$r_s\$")
        ylabel("\$\\widetilde{F}^+_0[W]\$")
        # ylabel("\$\\widetilde{F}^+_0 - \\widetilde{F}^+\$")
        # ylim(-1.1, 0.6)
        xlim(0, 10)
        if sign_Fsa < 0
            ylim(-3.2, 0.6)
        else
            ylim(-1.2, 3.8)
        end
        ax.legend(; fontsize=10, loc="best")
        # tight_layout()
        signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
        fig.savefig("analytic_F0p_$(signstr_Fsa).pdf")
        plt.close("all")
    end
    return F0p_RPA, F0p_KOp, F0p_KO
end

function get_analytic_F1p(rslist; plot=false, sign_Fsa=-1.0)
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
        #     @assert Fs < 0 && Fa < 0 "Incorrect signs for Fs/Fa!"
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
        ax.plot(rslist, F1p_RPA; color="black", label="\$W_0\$")
        ax.plot(rslist, F1p_KOp; color=cdict["blue"], label="\$W^\\text{KO}_{0,+}\$")
        ax.plot(rslist, F1p_KOm; color=cdict["teal"], label="\$W^\\text{KO}_{0,-}\$")
        ax.plot(
            rslist,
            F1p_KO;
            color=cdict["red"],
            label="\$W^\\text{KO}_{0} = W^\\text{KO}_{0,+} + 3 W^\\text{KO}_{0,-}\$",
        )
        xlabel("\$r_s\$")
        ylabel("\$\\widetilde{F}^+_1[W]\$")
        xlim(0, 10)
        if sign_Fsa < 0
            ylim(-0.425, 0.075)
        else
            ylim(-0.12, 0.08)
        end
        ax.legend(; fontsize=10, loc="best")
        # tight_layout()
        signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"
        fig.savefig("analytic_F1p_$(signstr_Fsa).pdf")
        plt.close("all")
    end
    return F1p_RPA, F1p_KOp, F1p_KO
end

function get_Flp_ElectronGas(l::Integer, param::Parameter.Para, Fs, Fa, int_type=:ko_const)
    @unpack kF, EF, NF = param

    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14

    # Dimensionless angular integration grid
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    k_m_kps = @. 2 * kF * xgrid

    # Select the appropriate Landau function for the given interaction type
    landaufunc = int_type_to_landaufunc[int_type]

    # Get R(p, iωₙ=0) (spin-symmetric static exchange part of the KO interaction)
    R_ex_dyn, R_ex_inst_inv = Interaction.KOwrapped(
        Euv,
        rtol,
        k_m_kps,
        param;
        regular=false,
        int_type=int_type,
        landaufunc=landaufunc,
        Fs=Fs,
        Fa=Fa,
    )
    R_ex_inst = 1 ./ R_ex_inst_inv
    @assert R_ex_dyn.mesh[2].grid[1] == 0

    # Get the static, spin-symmetric/antisymmetric parts of the interaction
    R_ex_static_s = R_ex_dyn[1, 1, :] + R_ex_inst[1, 1, :]
    R_ex_static_a = R_ex_dyn[2, 1, :] + R_ex_inst[2, 1, :]

    # Swap spin indices to match external legs
    # NOTE: the extra factor of 2 comes from the spin summation in Flp
    R_plus_static = 2 * exchange_to_direct(R_ex_static_s, R_ex_static_a)[1]
    @assert maximum(imag(R_plus_static)) ≤ 1e-10

    # Perform dimensionless angular integration
    if l == 0
        integrand = @. -real(2 * NF * R_plus_static) * xgrid
    elseif l == 1
        integrand = @. -real(2 * NF * R_plus_static) * xgrid * (1 - 2 * xgrid^2)
    else
        error("l > 1 not yet implemented!")
    end
    Flp = CompositeGrids.Interp.integrate1D(integrand, xgrid)
    return Flp
end

function get_tree_level_Flp_ElectronGas(
    l::Integer,
    param::Parameter.Para;
    int_type=:ko_const,
    sign_Fsa=-1.0,
)
    Fs = Fa = 0.0
    if int_type == :ko_const
        int_type_fp = int_type_fp_fm = int_type
        # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
        Fs = -sign_Fsa * get_Fs_PW(param.rs)
        Fa = -sign_Fsa * get_Fa_PW(param.rs)
        if param.rs > 0.25
            @assert Fs > 0 && Fa > 0 "Incorrect signs for Fs/Fa!"
        end
    elseif int_type == :ko_simion_giuliani
        int_type_fp = :ko_simion_giuliani_plus
        int_type_fp_fm = :ko_simion_giuliani
    elseif int_type == :ko_takada
        int_type_fp = :ko_takada_plus
        int_type_fp_fm = :ko_takada
    end
    Flp_rpa = get_Flp_ElectronGas(l, param, 0.0, 0.0, :rpa)
    Flp_fp = get_Flp_ElectronGas(l, param, Fs, 0.0, int_type_fp)
    Flp_fp_fm = get_Flp_ElectronGas(l, param, Fs, Fa, int_type_fp_fm)
    return Flp_rpa, Flp_fp, Flp_fp_fm
end

function get_tree_level_Flp_ElectronLiquid(l::Integer, param::Parameter.Para; verbose=0)
    @unpack rs, beta, dim = param
    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    # NOTE: ElectronLiquid.jl uses the opposite sign convention from ElectronGas.jl for Fs/Fa!
    Fs = -get_Fs_PW(param.rs)
    Fa = -get_Fa_PW(param.rs)
    if param.rs > 0.25
        @assert Fs < 0 && Fa < 0 "Incorrect signs for Fs/Fa!"
    end
    W_ex = Ver4.exchange_interaction
    p_rpa = ParaMC(; rs=rs, beta=beta, dim=dim, Fs=0.0, Fa=0.0, order=1, mass2=0.0)
    p_fp = ParaMC(; rs=rs, beta=beta, dim=dim, Fs=Fs, Fa=0.0, order=1, mass2=0.0)
    p_fp_fm = ParaMC(; rs=rs, beta=beta, dim=dim, Fs=Fs, Fa=Fa, order=1, mass2=0.0)
    Flp_rpa = -2 * Ver4.projected_exchange_interaction(l, p_rpa, W_ex; verbose=verbose)[1]
    Flp_fp = -2 * Ver4.projected_exchange_interaction(l, p_fp, W_ex; verbose=verbose)[1]
    Flp_fp_fm =
        -2 * Ver4.projected_exchange_interaction(l, p_fp_fm, W_ex; verbose=verbose)[1]
    return Flp_rpa, Flp_fp, Flp_fp_fm
end

function get_meff_ElectronGas(param::Parameter.Para; int_type=:ko_const, sign_Fsa=-1.0)
    @unpack kF, EF, NF = param

    Fs = Fa = 0.0
    if int_type == :ko_const
        int_type_fp = int_type_fp_fm = int_type
        # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
        Fs = get_Fs_PW(param.rs)
        Fa = get_Fa_PW(param.rs)
        # Fs = sign_Fsa * get_Fs_PW(param.rs)
        # Fa = sign_Fsa * get_Fa_PW(param.rs)
        if param.rs > 0.25
            @assert Fs > 0 && Fa > 0 "Incorrect signs for Fs/Fa!"
        end
    elseif int_type == :ko_simion_giuliani
        int_type_fp = :ko_simion_giuliani_plus
        int_type_fp_fm = :ko_simion_giuliani
    elseif int_type == :ko_takada
        int_type_fp = :ko_takada_plus
        int_type_fp_fm = :ko_takada
    end

    # Get the RPA effective mass ratio from the G0W0 self-energy
    Σ_RPA, Σ_ins_RPA = SelfEnergy.G0W0(param; int_type=:rpa)
    meff_RPA = SelfEnergy.massratio(param, Σ_RPA, Σ_ins_RPA)[1]

    # Get the KOp effective mass ratio from the G0W0 self-energy
    Σ_KOp, Σ_ins_KOp = SelfEnergy.G0W0(param; Fs=Fs, Fa=0.0, int_type=int_type_fp)
    meff_KOp = SelfEnergy.massratio(param, Σ_KOp, Σ_ins_KOp)[1]

    # Get the KO effective mass ratio from the G0W0 self-energy
    Σ_KO, Σ_ins_KO = SelfEnergy.G0W0(param; Fs=Fs, Fa=Fa, int_type=int_type_fp_fm)
    meff_KO = SelfEnergy.massratio(param, Σ_KO, Σ_ins_KO)[1]

    return meff_RPA, meff_KOp, meff_KO
end

function plot_vs_rs(
    rslist,
    data,
    color,
    label,
    ls="-";
    ax=plt.gca(),
    zorder=nothing,
    data_rs0=0.0,
    rs_HDL=nothing,
    meff_HDL=nothing,
)
    # Add point at rs = 0
    rslist = unique([0.0; rslist])
    data = unique([data_rs0; data])

    # Add data in the high-density limit to the fit, if provided
    if !isnothing(rs_HDL) && !isnothing(meff_HDL)
        rslist = unique([rslist; rs_HDL])
        data = unique([data; meff_HDL])
    end

    # Re-sort the data after adding the high-density limit data
    P = sortperm(rslist)
    rslist = rslist[P]
    data = data[P]

    # Plot splined data vs rs
    fitfunc = interp.Akima1DInterpolator(rslist, data)
    # fitfunc = interp.PchipInterpolator(rslist, data)
    xgrid = np.arange(0, maximum(rslist) + 0.2, 0.01)
    if isnothing(zorder)
        handle, = ax.plot(xgrid, fitfunc(xgrid); ls=ls, color=color, label=label)
    else
        handle, =
            ax.plot(xgrid, fitfunc(xgrid); ls=ls, color=color, label=label, zorder=zorder)
    end
    return handle
end

function main()
    # UEG parameters
    beta = 1000.0
    dim = 3

    rs_Fsm1 = 5.24881  # Fs(rs = 5.24881) ≈ -1 (using Perdew-Wang fit)
    rslist = sort(unique([0.01; rs_Fsm1; collect(range(0.125, 10.0; step=0.125))]))

    # Using the ElectronLiquid.jl (v + f) convention ⟹ F± < 0
    sign_Fsa = -1.0   # correct sign
    # sign_Fsa = 1.0  # mismatched sign for testing
    signstr_Fsa = sign_Fsa > 0 ? "Fs_Fa_positive" : "Fs_Fa_negative"

    plot_Fermi_liquid_parameters(; rs=1.0, sign_Fsa=sign_Fsa)
    # plot_Fermi_liquid_parameters(; rs=5.0, sign_Fsa=sign_Fsa)
    # return

    # l=0 analytic plots
    plot_integrand_F0p(; sign_Fsa=sign_Fsa)
    get_analytic_F0p(rslist; plot=true, sign_Fsa=sign_Fsa)

    # l=1 analytic plots
    plot_integrand_F1p(; sign_Fsa=sign_Fsa)
    get_analytic_F1p(rslist; plot=true, sign_Fsa=sign_Fsa)

    # load the npz data if it already exists
    if isfile("meff_and_Flp_tree_level_int_type_comparisons_$(signstr_Fsa).npz")
        data = np.load("meff_and_Flp_tree_level_int_type_comparisons_$(signstr_Fsa).npz")
        rslist = data.get("rslist_Flp")
        rslist_small = data.get("rslist_meff")
        meffs_RPA_C = data.get("meffs_RPA_C")
        meffs_KOp_C = data.get("meffs_KOp_C")
        meffs_KO_C = data.get("meffs_KO_C")
        # meffs_RPA_T = data.get("meffs_RPA_T")
        # meffs_KOp_T = data.get("meffs_KOp_T")
        # meffs_KO_T = data.get("meffs_KO_T")
        meffs_RPA_SG = data.get("meffs_RPA_SG")
        meffs_KOp_SG = data.get("meffs_KOp_SG")
        meffs_KO_SG = data.get("meffs_KO_SG")
        F0ps_RPA_C = data.get("F0ps_RPA_C")
        F0ps_KOp_C = data.get("F0ps_KOp_C")
        F0ps_KO_C = data.get("F0ps_KO_C")
        # F0ps_RPA_T = data.get("F0ps_RPA_T")
        # F0ps_KOp_T = data.get("F0ps_KOp_T")
        # F0ps_KO_T = data.get("F0ps_KO_T")
        F0ps_RPA_SG = data.get("F0ps_RPA_SG")
        F0ps_KOp_SG = data.get("F0ps_KOp_SG")
        F0ps_KO_SG = data.get("F0ps_KO_SG")
        F1ps_RPA_C = data.get("F1ps_RPA_C")
        F1ps_KOp_C = data.get("F1ps_KOp_C")
        F1ps_KO_C = data.get("F1ps_KO_C")
        # F1ps_RPA_T = data.get("F1ps_RPA_T")
        # F1ps_KOp_T = data.get("F1ps_KOp_T")
        # F1ps_KO_T = data.get("F1ps_KO_T")
        F1ps_RPA_SG = data.get("F1ps_RPA_SG")
        F1ps_KOp_SG = data.get("F1ps_KOp_SG")
        F1ps_KO_SG = data.get("F1ps_KO_SG")
    else
        meffs_RPA_C = []
        meffs_KOp_C = []
        meffs_KO_C = []
        # meffs_RPA_T = []
        # meffs_KOp_T = []
        # meffs_KO_T = []
        meffs_RPA_SG = []
        meffs_KOp_SG = []
        meffs_KO_SG = []
        rslist_small = sort(unique([0.125; rs_Fsm1; collect(range(0.25, 10.0; step=0.25))]))
        for rs in rslist_small
            print("Computing mass for rs = $(rs)...")
            param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
            # Constant Fs and Fa
            meff_RPA_C, meff_KOp_C, meff_KO_C =
                get_meff_ElectronGas(param; int_type=:ko_const)
            push!(meffs_RPA_C, meff_RPA_C)
            push!(meffs_KOp_C, meff_KOp_C)
            push!(meffs_KO_C, meff_KO_C)
            # # Takada ansatz for Fs and Fa
            # meff_RPA_T, meff_KOp_T, meff_KO_T =
            #     get_meff_ElectronGas(param; int_type=:ko_takada)
            # push!(meffs_RPA_T, meff_RPA_T)
            # push!(meffs_KOp_T, meff_KOp_T)
            # push!(meffs_KO_T, meff_KO_T)
            # Moroni fit for Fs and Simion and Giuliani ansatz for Fa
            meff_RPA_SG, meff_KOp_SG, meff_KO_SG =
                get_meff_ElectronGas(param; int_type=:ko_simion_giuliani)
            push!(meffs_RPA_SG, meff_RPA_SG)
            push!(meffs_KOp_SG, meff_KOp_SG)
            push!(meffs_KO_SG, meff_KO_SG)
            println("done!")
            println("RPA: m*/m = $(meff_RPA_C)")
        end
        F0ps_RPA_C = []
        F0ps_KOp_C = []
        F0ps_KO_C = []
        # F0ps_RPA_T = []
        # F0ps_KOp_T = []
        # F0ps_KO_T = []
        F0ps_RPA_SG = []
        F0ps_KOp_SG = []
        F0ps_KO_SG = []
        F1ps_RPA_C = []
        F1ps_KOp_C = []
        F1ps_KO_C = []
        # F1ps_RPA_T = []
        # F1ps_KOp_T = []
        # F1ps_KO_T = []
        F1ps_RPA_SG = []
        F1ps_KOp_SG = []
        F1ps_KO_SG = []
        for rs in rslist
            print("Computing Fermi liquid parameters for rs = $(rs)...")
            param = Parameter.rydbergUnit(1.0 / beta, rs, dim)

            # Constant Fs and Fa
            # F0p
            F0p_RPA_C, F0p_KOp_C, F0p_KO_C = get_tree_level_Flp_ElectronGas(
                0,
                param;
                int_type=:ko_const,
                sign_Fsa=sign_Fsa,
            )
            push!(F0ps_RPA_C, F0p_RPA_C)
            push!(F0ps_KOp_C, F0p_KOp_C)
            push!(F0ps_KO_C, F0p_KO_C)
            # F1p
            F1p_RPA_C, F1p_KOp_C, F1p_KO_C = get_tree_level_Flp_ElectronGas(
                1,
                param;
                int_type=:ko_const,
                sign_Fsa=sign_Fsa,
            )
            push!(F1ps_RPA_C, F1p_RPA_C)
            push!(F1ps_KOp_C, F1p_KOp_C)
            push!(F1ps_KO_C, F1p_KO_C)

            # # Takada ansatz for Fs and Fa
            # # F0p
            # F0p_RPA_T, F0p_KOp_T, F0p_KO_T =
            #     get_tree_level_Flp_ElectronGas(0, param; int_type=:ko_takada)
            # push!(F0ps_RPA_T, F0p_RPA_T)
            # push!(F0ps_KOp_T, F0p_KOp_T)
            # push!(F0ps_KO_T, F0p_KO_T)
            # # F1p
            # F1p_RPA_T, F1p_KOp_T, F1p_KO_T =
            #     get_tree_level_Flp_ElectronGas(1, param; int_type=:ko_takada)
            # push!(F1ps_RPA_T, F1p_RPA_T)
            # push!(F1ps_KOp_T, F1p_KOp_T)
            # push!(F1ps_KO_T, F1p_KO_T)

            # Moroni fit for Fs and Simion and Giuliani ansatz for Fa
            # F0p
            F0p_RPA_SG, F0p_KOp_SG, F0p_KO_SG =
                get_tree_level_Flp_ElectronGas(0, param; int_type=:ko_simion_giuliani)
            push!(F0ps_RPA_SG, F0p_RPA_SG)
            push!(F0ps_KOp_SG, F0p_KOp_SG)
            push!(F0ps_KO_SG, F0p_KO_SG)
            # F1p
            F1p_RPA_SG, F1p_KOp_SG, F1p_KO_SG =
                get_tree_level_Flp_ElectronGas(1, param; int_type=:ko_simion_giuliani)
            push!(F1ps_RPA_SG, F1p_RPA_SG)
            push!(F1ps_KOp_SG, F1p_KOp_SG)
            push!(F1ps_KO_SG, F1p_KO_SG)
            println("done!")
        end
    end

    ###############################
    # Plot meff comparisons vs rs #
    ###############################

    # High-density limit of the effective mass
    rs_HDL_plot = collect(range(1e-5, 0.35, 101))
    meff_HDL_plot = [high_density_limit(rs) for rs in rs_HDL_plot]

    # Use exact expression in the high-density limit for all effective mass fits
    cutoff_HDL = 0.1
    rs_HDL = rs_HDL_plot[rs_HDL_plot .≤ cutoff_HDL]
    meff_HDL = meff_HDL_plot[rs_HDL_plot .≤ cutoff_HDL]

    fig1 = figure(; figsize=(6, 6))
    ax1 = fig1.add_subplot(111)

    # RPA effective mass from Dyson self-energy
    plot_vs_rs(
        rslist_small,
        meffs_RPA_C,
        "black",
        "\$W_0\$",
        "-";
        data_rs0=1.0,
        rs_HDL=rs_HDL,
        meff_HDL=meff_HDL,
    )
    # plot_vs_rs(
    #     rslist_small,
    #     meffs_RPA_T,
    #     "black",
    #     nothing,
    #     "-.";
    #     data_rs0=1.0,
    #     rs_HDL=rs_HDL,
    #     meff_HDL=meff_HDL,
    # )
    plot_vs_rs(
        rslist_small,
        meffs_RPA_SG,
        "black",
        nothing,
        "--";
        data_rs0=1.0,
        rs_HDL=rs_HDL,
        meff_HDL=meff_HDL,
    )

    # KOp effective mass from Dyson self-energy
    plot_vs_rs(
        rslist_small,
        meffs_KOp_C,
        cdict["blue"],
        "\$W^\\text{KO}_{0,+}[f_\\pm]\$",
        "-";
        data_rs0=1.0,
    )
    # plot_vs_rs(
    #     rslist_small,
    #     meffs_KOp_T,
    #     cdict["blue"],
    #     nothing,
    #     "-.";
    #     data_rs0=1.0,
    #     rs_HDL=rs_HDL,
    #     meff_HDL=meff_HDL,
    # )
    plot_vs_rs(
        rslist_small,
        meffs_KOp_SG,
        cdict["cyan"],
        "\$W^\\text{KO}_{0,+}[f_\\pm(q)]\$",
        "-";
        data_rs0=1.0,
        rs_HDL=rs_HDL,
        meff_HDL=meff_HDL,
    )

    # KO effective mass from Dyson self-energy
    plot_vs_rs(
        rslist_small,
        meffs_KO_C,
        cdict["red"],
        "\$W^\\text{KO}_{0}[f_\\pm]\$",
        "-";
        data_rs0=1.0,
    )
    # plot_vs_rs(
    #     rslist_small,
    #     meffs_KO_T,
    #     cdict["red"],
    #     nothing,
    #     "-.";
    #     data_rs0=1.0,
    #     rs_HDL=rs_HDL,
    #     meff_HDL=meff_HDL,
    # )
    plot_vs_rs(
        rslist_small,
        meffs_KO_SG,
        cdict["orange"],
        "\$W^\\text{KO}_{0}[f_\\pm(q)]\$",
        "-";
        data_rs0=1.0,
        rs_HDL=rs_HDL,
        meff_HDL=meff_HDL,
    )

    # Data taken from Simion & Giuliani (2008), Table 1
    rs_SG_data = collect(1:6)
    m_RPA_SG_data = [0.970, 0.992, 1.016, 1.039, 1.059, 1.078]
    m_Gp_SG_data = [0.952, 0.951, 0.956, 0.962, 0.968, 0.973]
    m_Gpm_SG_data = [0.957, 0.966, 0.983, 1.005, 1.028, 1.055]
    ax1.scatter(rs_SG_data, m_RPA_SG_data, 20; color="black", marker="^")
    ax1.scatter(rs_SG_data, m_Gp_SG_data, 20; color=cdict["cyan"], marker="s")
    ax1.scatter(rs_SG_data, m_Gpm_SG_data, 20; color=cdict["orange"], marker="D")

    legend(; loc="best", fontsize=12, ncol=2)
    ylabel("\$\\left(m^*/m\\right)_D[W]\$")
    xlabel("\$r_s\$")
    # ylim(0.905, 1.2)
    xlim(0, 10)
    tight_layout()
    savefig("meff_int_type_comparisons_$(signstr_Fsa).pdf")

    ##############################
    # Plot Flp comparisons vs rs #
    ##############################

    fig2 = figure(; figsize=(6, 6))
    ax2 = fig2.add_subplot(111)

    F0p_RPA_inf = -1.2980116518203961
    F1p_RPA_inf = 0.1348791193970809

    # Full F^+(rs)
    labelstr =
        sign_Fsa > 0 ? "\$-F^+ = \\kappa_0 / \\kappa - 1\$" :
        "\$F^+ = \\left(\\frac{\\kappa_0}{\\kappa} - 1\\right)_\\text{DMC}\$"
    labelstr2 =
        sign_Fsa > 0 ? "\$-F^+ = \\kappa_0 / \\kappa - 1\$" :
        "\$F^+ + \\widetilde{F}^+_0[W_0](\\infty)\$"
    ax2.axhline(
        F0p_RPA_inf;
        color="black",
        linestyle="--",
        label="\$\\widetilde{F}^+_0[W_0](\\infty)\$",
    )
    plot_vs_rs(
        rslist,
        sign_Fsa * get_Fs_PW.(rslist) .+ F0p_RPA_inf,
        cdict["blue"],
        labelstr2,
        "--";
        data_rs0=F0p_RPA_inf,
    )
    plot_vs_rs(rslist, sign_Fsa * get_Fs_PW.(rslist), cdict["grey"], labelstr, "-")

    # Tree-level RPA
    plot_vs_rs(rslist, F0ps_RPA_C, "black", "\$W_0\$", "-")
    # plot_vs_rs(rslist, F0ps_RPA_T, "black", nothing, "-.")
    # plot_vs_rs(rslist, F0ps_RPA_SG, "black", nothing, "--")

    # Tree-level KO with fp only
    plot_vs_rs(rslist, F0ps_KOp_C, cdict["blue"], "\$W^\\text{KO}_{0,+}[f_\\pm]\$", "-")
    # plot_vs_rs(rslist, F0ps_KOp_T, cdict["blue"], nothing, "-.")
    plot_vs_rs(rslist, F0ps_KOp_SG, cdict["cyan"], "\$W^\\text{KO}_{0,+}[f_\\pm(q)]\$", "-")

    # Tree-level KO with fp and fm
    plot_vs_rs(rslist, F0ps_KO_C, cdict["red"], "\$W^\\text{KO}_{0}[f_\\pm]\$", "-")
    # plot_vs_rs(rslist, F0ps_KO_T, cdict["red"], nothing, "-.")
    plot_vs_rs(rslist, F0ps_KO_SG, cdict["orange"], "\$W^\\text{KO}_{0}[f_\\pm(q)]\$", "-")

    legend(; loc="best", fontsize=10, ncol=3)
    ylabel("\$\\widetilde{F}^+_0[W]\$")
    xlabel("\$r_s\$")
    # ylim(-0.056, 0.034)
    # ylim(-1.1, 0.6)
    xlim(0, 10)
    ylim(-2.6, 0.4)
    tight_layout()
    savefig("F0p_tree_level_int_type_comparisons_$(signstr_Fsa).pdf")

    fig3 = figure(; figsize=(6, 6))
    ax3 = fig3.add_subplot(111)

    ax3.axhline(
        F1p_RPA_inf;
        color="black",
        linestyle="--",
        label="\$\\widetilde{F}^+_1[W_0](\\infty)\$",
    )

    # Tree-level RPA
    plot_vs_rs(rslist, F1ps_RPA_C, "black", "\$W_0\$", "-")
    # plot_vs_rs(rslist, F1ps_RPA_T, "black", nothing, "-.")
    # plot_vs_rs(rslist, F1ps_RPA_SG, "black", nothing, "-")

    # Tree-level KO with fp only
    plot_vs_rs(rslist, F1ps_KOp_C, cdict["blue"], "\$W^\\text{KO}_{0,+}[f_\\pm]\$", "-")
    # plot_vs_rs(rslist, F1ps_KOp_T, cdict["blue"], nothing, "-.")
    plot_vs_rs(rslist, F1ps_KOp_SG, cdict["cyan"], "\$W^\\text{KO}_{0,+}[f_\\pm(q)]\$", "-")

    # Tree-level KO with fp and fm
    plot_vs_rs(rslist, F1ps_KO_C, cdict["red"], "\$W^\\text{KO}_{0}[f_\\pm]\$", "-")
    # plot_vs_rs(rslist, F1ps_KO_T, cdict["red"], nothing, "-.")
    plot_vs_rs(rslist, F1ps_KO_SG, cdict["orange"], "\$W^\\text{KO}_{0}[f_\\pm(q)]\$", "-")

    legend(; loc=(0.02, 0.75), fontsize=12, ncol=2)
    ylabel("\$\\widetilde{F}^+_1[W]\$")
    xlabel("\$r_s\$")
    # ylim(-0.056, 0.034)
    # ylim(-0.072, 0.034)
    xlim(0, 10)
    tight_layout()
    savefig("F1p_tree_level_int_type_comparisons_$(signstr_Fsa).pdf")

    if isfile("meff_and_Flp_tree_level_int_type_comparisons_$(signstr_Fsa).npz") == false
        np.savez(
            "meff_and_Flp_tree_level_int_type_comparisons_$(signstr_Fsa).npz";
            rslist_Flp=rslist,
            rslist_meff=rslist_small,
            # rslist=rslist_small,
            meffs_RPA_C=meffs_RPA_C,
            meffs_KOp_C=meffs_KOp_C,
            meffs_KO_C=meffs_KO_C,
            # meffs_RPA_T=meffs_RPA_T,
            # meffs_KOp_T=meffs_KOp_T,
            # meffs_KO_T=meffs_KO_T,
            meffs_RPA_SG=meffs_RPA_SG,
            meffs_KOp_SG=meffs_KOp_SG,
            meffs_KO_SG=meffs_KO_SG,
            F0ps_RPA_C=F0ps_RPA_C,
            F0ps_KOp_C=F0ps_KOp_C,
            F0ps_KO_C=F0ps_KO_C,
            # F0ps_RPA_T=F0ps_RPA_T,
            # F0ps_KOp_T=F0ps_KOp_T,
            # F0ps_KO_T=F0ps_KO_T,
            F0ps_RPA_SG=F0ps_RPA_SG,
            F0ps_KOp_SG=F0ps_KOp_SG,
            F0ps_KO_SG=F0ps_KO_SG,
            F1ps_RPA_C=F1ps_RPA_C,
            F1ps_KOp_C=F1ps_KOp_C,
            F1ps_KO_C=F1ps_KO_C,
            # F1ps_RPA_T=F1ps_RPA_T,
            # F1ps_KOp_T=F1ps_KOp_T,
            # F1ps_KO_T=F1ps_KO_T,
            F1ps_RPA_SG=F1ps_RPA_SG,
            F1ps_KOp_SG=F1ps_KOp_SG,
            F1ps_KO_SG=F1ps_KO_SG,
        )
    end

    ########################################
    # Benchmark Flp against ElectronLiquid #
    ########################################

    # F0ps_RPA_EG = []
    # F0ps_KOp_EG = []
    # F0ps_KO_EG = []
    # F1ps_RPA_EG = []
    # F1ps_KOp_EG = []
    # F1ps_KO_EG = []
    # if int_type == :ko_const
    #     F0ps_RPA_EL = []
    #     F0ps_KOp_EL = []
    #     F0ps_KO_EL = []
    #     F1ps_RPA_EL = []
    #     F1ps_KOp_EL = []
    #     F1ps_KO_EL = []
    # end
    # for rs in rslist
    #     print("Computing Fermi liquid parameters for rs = $(rs)...")
    #     param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
    #     # Use ElectronGas.jl to compute the Fermi liquid parameters
    #     F0p_RPA_EG, F0p_KOp_EG, F0p_KO_EG =
    #         get_tree_level_Flp_ElectronGas(0, param; int_type=int_type)
    #     F1p_RPA_EG, F1p_KOp_EG, F1p_KO_EG =
    #         get_tree_level_Flp_ElectronGas(1, param; int_type=int_type)
    #     push!(F0ps_RPA_EG, F0p_RPA_EG)
    #     push!(F0ps_KOp_EG, F0p_KOp_EG)
    #     push!(F0ps_KO_EG, F0p_KO_EG)
    #     push!(F1ps_RPA_EG, F1p_RPA_EG)
    #     push!(F1ps_KOp_EG, F1p_KOp_EG)
    #     push!(F1ps_KO_EG, F1p_KO_EG)
    #     # Use ElectronGas.jl to compute the Fermi liquid parameters (requires int_type == :ko_const)
    #     if int_type == :ko_const
    #         F0p_RPA_EL, F0p_KOp_EL, F0p_KO_EL = get_tree_level_Flp_ElectronLiquid(0, param)
    #         F1p_RPA_EL, F1p_KOp_EL, F1p_KO_EL = get_tree_level_Flp_ElectronLiquid(1, param)
    #         push!(F0ps_RPA_EL, F0p_RPA_EL)
    #         push!(F0ps_KOp_EL, F0p_KOp_EL)
    #         push!(F0ps_KO_EL, F0p_KO_EL)
    #         push!(F1ps_RPA_EL, F1p_RPA_EL)
    #         push!(F1ps_KOp_EL, F1p_KOp_EL)
    #         push!(F1ps_KO_EL, F1p_KO_EL)
    #     end
    #     println("done!")
    # end

    return
end

main()
