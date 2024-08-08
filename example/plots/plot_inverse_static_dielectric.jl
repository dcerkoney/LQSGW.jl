using Colors
using CompositeGrids
using ElectronGas
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
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 16
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "Times New Roman"

const colors = [
    "grey",
    cdict["teal"],
    cdict["cyan"],
    cdict["orange"],
    cdict["magenta"],
    cdict["red"],
    cdict["blue"],
    "black",
]
const pts = ["s", "^", "v", "p", "s", "<", "h", "o"]
const reflabels = ["\$^*\$", "\$^\\dagger\$", "\$^\\ddagger\$"]

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via interpolation of the 
compressibility ratio data of Perdew & Wang (1992) [Phys. Rev. B 45, 13244].
"""
@inline function get_Fs_PW(rs)
    # if rs < 1.0 || rs > 5.0
    #     @warn "The Perdew-Wang interpolation for Fs may " *
    #           "be inaccurate outside the metallic regime!"
    # end
    kappa0_over_kappa = 1.0025 - 0.1721rs - 0.0036rs^2
    # NOTE: NEFT uses opposite sign convention for F!
    # -F⁰ₛ = 1 - κ₀/κ
    return 1.0 - kappa0_over_kappa
end

"""
Get the antisymmetric l=0 Fermi-liquid parameter F⁰ₐ via interpolation of the 
susceptibility ratio data (c.f. Kukkonen & Chen, 2021)
"""
@inline function get_Fa_PW(rs)
    chi0_over_chi = 0.9821 - 0.1232rs + 0.0091rs^2
    # NOTE: NEFT uses opposite sign convention for F!
    # -F⁰ₐ = 1 - χ₀/χ
    return 1.0 - chi0_over_chi
end

function spline(x, y, e; xmin=0.0, xmax=x[end])
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(xmin, xmax, 1000))
    yfit = spl(__x)
    return __x, yfit
end

function plot_spline(x, y, idx, label, ax; ls="-", zorder=nothing)
    # mfitfunc = interp.PchipInterpolator(x, y)
    mfitfunc = interp.Akima1DInterpolator(x, y)
    xgrid = np.arange(0, 6.2, 0.02)
    if isnothing(zorder) == false
        handle, = ax.plot(
            xgrid,
            mfitfunc(xgrid);
            ls=ls,
            color=colors[idx],
            label=label,
            zorder=zorder,
        )
    else
        handle, = ax.plot(xgrid, mfitfunc(xgrid); ls=ls, color=colors[idx], label=label)
    end
    return handle
end

function load_w(
    param::Parameter.Para,
    int_type=:rpa,
    max_steps=300;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    key = "W"
    n_max = -1
    data = nothing
    found_data = false
    filename = joinpath(savedir, savename)
    jldopen(filename, "r") do f
        for i in 0:max_steps
            try
                f["$(key)_$(i)"]
            catch
                n_max = i - 1
                found_data = true
                data = f["$(key)_$(n_max)"]
                # println("Found converged data at i = $n_max")
                break
            end
        end
    end
    if found_data == false
        error("LQSGW data not found for key = $key")
    end
    return data, n_max
end

function load_w(
    i_step,
    param::Parameter.Para,
    int_type=:rpa;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    key = "W"
    filename = joinpath(savedir, savename)
    jldopen(filename, "r") do f
        try
            return f["$(key)_$(i_step)"]
        catch
            error("One-shot ata not found for key = $key, i_step = $i_step")
        end
    end
end

function load_w_oneshot(
    param::Parameter.Para,
    int_type=:rpa,
    δK=5e-6;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    return W0 = load_w(0, param, int_type)
end

function load_w_lqsgw(
    param::Parameter.Para,
    int_type=:rpa,
    δK=5e-6,
    max_steps=300;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    return load_w(param, int_type, max_steps)[1]
end

function main()
    # UEG parameters
    beta = 40.0
    rs = 5.0
    dim = 3
    param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
    @unpack kF, EF, β = param

    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    # Momentum grid maxima for G, Π, and Σ
    maxKG = 4.3 * maxK
    maxKP = 2.1 * maxK
    maxKS = maxK

    # Bosonic DLR grid for the problem
    bdlr = DLRGrid(Euv, β, rtol, false, :ph)

    constant_fs = true
    # constant_fs = false

    if constant_fs
        fsstr = "fs_const"
        int_type_fp = :ko_const_p
        int_type_fp_fm = :ko_const_pm
    else
        fsstr = "fs_dmc"
        int_type_fp = :ko_moroni
        int_type_fp_fm = :ko_simion_giuliani
    end
    @assert int_type_fp ∈ [:ko_const_p, :ko_takada_plus, :ko_moroni]
    @assert int_type_fp_fm ∈ [:ko_const_pm, :ko_takada, :ko_simion_giuliani]

    ###########################################################
    # Compute static dielectric function for one-shot methods #
    ###########################################################

    # Get the interactions W_0, W_0^KO_+, and W_0^KO
    # wtilde_static_rpa       =
    # wtilde_static_rpa_fp    =
    # wtilde_static_rpa_fp_fm =

    vq = [Interaction.coulomb(q, param)[1] for q in qPgrid]

    # 1 / ϵ =  W / V, even when f⁺ and/or f⁻ are nonzero
    eps_invs = []
    for w in [wtilde_static_rpa, wtilde_static_rpa_fp, wtilde_static_rpa_fp_fm]
        @assert all(abs.(imag.(w)) .< 1e-10)
        push!(eps_invs, 1 .+ real.(w) ./ vq)
    end

    ########################################################
    # Plot static dielectric function for one-shot methods #
    ########################################################

    fig, ax = plt.subplots(; figsize=(5, 5))

    labels = ["\$G_0 W_0\$", "\$G_0 W^\\text{KO}_{0,+}\$", "\$G_0 W^\\text{KO}_0\$"]
    for (idx, label, eps_inv) in enumerate(zip(labels, eps_invs))
        plot_spline(qPgrid / 2kF, eps_inv, idx, label, ax; ls="--")
    end

    if constant_fs
        ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ax.set_xlabel("\$q / 2 k_F\$")
    ax.set_ylabel("\$1 / \\epsilon(q, i\\nu_m = 0)\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    ax.set_xlim(0, 1.5)
    plt.tight_layout()
    fig.savefig(
        "inverse_static_dielectric_rs=$(round(rs; sigdigits=4))_$(fsstr)_oneshot_zoom.pdf",
    )
    ax.set_xlim(0, maxKP / 2kF)
    plt.tight_layout()
    fig.savefig(
        "inverse_static_dielectric_rs=$(round(rs; sigdigits=4))_$(fsstr)_oneshot.pdf",
    )

    # #####################################################
    # # Load one-shot and LQSGW data computed via LQSGW.jl #
    # #####################################################

    # wtilde_rpa = load_oneshot_data(param)
    # wtilde_rpa_fp = load_oneshot_data(param, int_type_fp)
    # wtilde_rpa_fp_fm = load_oneshot_data(param, int_type_fp_fm)

    # wtilde_lqsgw = load_lqsgw_data(param)
    # wtilde_lqsgw_fp = load_lqsgw_data(param, int_type_fp)
    # wtilde_lqsgw_fp_fm = load_lqsgw_data(param, int_type_fp_fm)

    # # RPA momentum grids for W
    # qPgrid_rpa = wtilde_rpa.mesh[2]
    # qPgrid_rpa_fp = wtilde_rpa_fp.mesh[2]
    # qPgrid_rpa_fp_fm = wtilde_rpa_fp_fm.mesh[2]

    # # LQSGW momentum grids for W
    # qPgrid_lqsgw = wtilde_lqsgw.mesh[2]
    # qPgrid_lqsgw_fp = wtilde_lqsgw_fp.mesh[2]
    # qPgrid_lqsgw_fp_fm = wtilde_lqsgw_fp_fm.mesh[2]

    # ##################################################################
    # # Plot static dielectric function for one-shot and LQSGW methods #
    # ##################################################################

    # fig, ax = plt.subplots(; figsize=(5, 5))

    # # 1 / ϵ =  W / V, even when f⁺ and/or f⁻ are nonzero

    # Wtilde_q_static = data_rpa.W[1, :]
    # @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    # v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_rpa]
    # eps_inv_rpa = 1 .+ real.(Wtilde_q_static) ./ v_q

    # Wtilde_q_static = data_rpa_fp.W[1, :]
    # @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    # v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_rpa_fp]
    # eps_inv_rpa_fp = 1 .+ real.(Wtilde_q_static) ./ v_q

    # Wtilde_q_static = data_rpa_fp_fm.W[1, :]
    # @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    # v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_rpa_fp_fm]
    # eps_inv_rpa_fp_fm = 1 .+ real.(Wtilde_q_static) ./ v_q

    # Wtilde_q_static = data_lqsgw.W[1, :]
    # @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    # v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_lqsgw]
    # eps_inv_lqsgw = 1 .+ real.(Wtilde_q_static) ./ v_q

    # Wtilde_q_static = data_lqsgw_fp.W[1, :]
    # @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    # v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_lqsgw_fp]
    # eps_inv_lqsgw_fp = 1 .+ real.(Wtilde_q_static) ./ v_q

    # Wtilde_q_static = data_lqsgw_fp_fm.W[1, :]
    # @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    # v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_lqsgw_fp_fm]
    # eps_inv_lqsgw_fp_fm = 1 .+ real.(Wtilde_q_static) ./ v_q

    # plot_spline(qPgrid_rpa / 2kF, eps_inv_rpa, 1, "\$G_0 W_0\$", ax; ls="--", zorder=2)
    # plot_spline(
    #     qPgrid_rpa_fp / 2kF,
    #     eps_inv_rpa_fp,
    #     2,
    #     "\$G_0 W^\\text{KO}_{0,+}\$",
    #     ax;
    #     ls="--",
    #     zorder=4,
    # )
    # plot_spline(
    #     qPgrid_rpa_fp_fm / 2kF,
    #     eps_inv_rpa_fp_fm,
    #     3,
    #     "\$G_0 W^\\text{KO}_0\$",
    #     ax;
    #     ls="--",
    #     zorder=6,
    # )

    # plot_spline(qPgrid_lqsgw / 2kF, eps_inv_lqsgw, 4, "LQSGW", ax; zorder=1)
    # plot_spline(
    #     qPgrid_lqsgw_fp / 2kF,
    #     eps_inv_lqsgw_fp,
    #     5,
    #     "LQSGW\$^\\text{KO}_+\$",
    #     ax;
    #     zorder=3,
    # )
    # plot_spline(
    #     qPgrid_lqsgw_fp_fm / 2kF,
    #     eps_inv_lqsgw_fp_fm,
    #     6,
    #     "LQSGW\$^\\text{KO}\$",
    #     ax;
    #     zorder=5,
    # )

    # if constant_fs
    #     ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # else
    #     ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # end
    # ax.set_xlabel("\$q / 2 k_F\$")
    # ax.set_ylabel("\$1 / \\epsilon(q, i\\nu_m = 0)\$")
    # ax.legend(; fontsize=12)
    # # ax.legend(; fontsize=12, ncol=3)
    # ax.set_xlim(0, 1.5)
    # plt.tight_layout()
    # fig.savefig("inverse_static_dielectric_rs=$(round(rs; sigdigits=4))_$(fsstr)_zoom.pdf")
    # ax.set_xlim(0, maxKP / 2kF)
    # plt.tight_layout()
    # fig.savefig("inverse_static_dielectric_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    return
end

main()
