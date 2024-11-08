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

# Data for a single LQSGW iteration
struct LQSGWIteration
    step::Int
    dmu::Float64
    meff::Float64
    zfactor::Float64
    E_k::Vector{Float64}
    Z_k::Vector{Float64}
    G::GreenFunc.MeshArray
    Π::GreenFunc.MeshArray
    W::GreenFunc.MeshArray
    Σ::GreenFunc.MeshArray
    Σ_ins::GreenFunc.MeshArray
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

function plot_errbar(x, y, e, idx, label, ax; zorder=nothing, capsize=4)
    if isnothing(zorder) == false
        handle = ax.errorbar(
            x,
            y,
            e;
            fmt=pts[idx],
            capthick=1,
            capsize=capsize,
            ms=5,
            color=colors[idx],
            label=label,
            zorder=zorder,
        )
    else
        handle = ax.errorbar(
            x,
            y,
            e;
            fmt=pts[idx],
            capthick=1,
            capsize=capsize,
            ms=5,
            color=colors[idx],
            label=label,
        )
    end
    return handle
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

function load_data(
    key,
    param::Parameter.Para,
    int_type=:rpa,
    max_steps=300;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
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

function load_data(
    i_step,
    key,
    param::Parameter.Para,
    int_type=:rpa;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    filename = joinpath(savedir, savename)
    jldopen(filename, "r") do f
        try
            return f["$(key)_$(i_step)"]
        catch
            error("One-shot data not found for key = $key, i_step = $i_step")
        end
    end
end

function load_oneshot_data(
    param::Parameter.Para,
    int_type=:rpa,
    δK=5e-6;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    # Oneshot approach: G0 -> Π0 -> W0 -> Σ1 = Σ_G0W0
    G0 = load_data(0, "G", param, int_type)
    Π0 = load_data(0, "Π", param, int_type)
    W0 = load_data(0, "W", param, int_type)
    Σ_G0W0 = load_data(1, "Σ", param, int_type)
    Σ_G0W0_ins = load_data(1, "Σ_ins", param, int_type)
    # Quasiparticle properties on the Fermi surface derived from Σ_G0W0
    E_k_G0W0 = load_data(1, "E_k", param, int_type)
    Z_k_G0W0 = load_data(1, "Z_k", param, int_type)
    δμ_G0W0 = chemicalpotential(param, Σ_G0W0, Σ_G0W0_ins)
    meff_G0W0 = massratio(param, Σ_G0W0, Σ_G0W0_ins, δK)[1]
    zfactor_G0W0 = zfactor_fermi(param, Σ_G0W0)
    # NOTE: In the one-shot approach, we cite spectral properties *after* the first iteration, i.e., (m*/m)_{G0W0} instead of (m*/m)₀ = 1, etc., hence this is really a mixture of two LQSGW steps.
    return LQSGWIteration(
        1,
        δμ_G0W0,
        meff_G0W0,
        zfactor_G0W0,
        E_k_G0W0,
        Z_k_G0W0,
        G0,
        Π0,
        W0,
        Σ_G0W0,
        Σ_G0W0_ins,
    )
end

function load_lqsgw_data(
    param::Parameter.Para,
    int_type=:rpa,
    δK=5e-6,
    max_steps=300;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    # Converged LQSGW: G_i -> Π_i -> W_i -> Σ_(i+1) = Σ_LQSGW
    G_LQSGW, num_steps = load_data("G", param, int_type, max_steps)
    Π_LQSGW = load_data("Π", param, int_type, max_steps)[1]
    W_LQSGW = load_data("W", param, int_type, max_steps)[1]
    E_k_LQSGW = load_data("E_k", param, int_type, max_steps)[1]
    Z_k_LQSGW = load_data("Z_k", param, int_type, max_steps)[1]
    Σ_LQSGW = load_data("Σ", param, int_type, max_steps)[1]
    Σ_LQSGW_ins = load_data("Σ_ins", param, int_type, max_steps)[1]
    # Quasiparticle properties on the Fermi surface derived from Σ_LQSGW
    δμ_LQSGW = chemicalpotential(param, Σ_LQSGW, Σ_LQSGW_ins)
    meff_LQSGW = massratio(param, Σ_LQSGW, Σ_LQSGW_ins, δK)[1]
    zfactor_LQSGW = zfactor_fermi(param, Σ_LQSGW)
    return LQSGWIteration(
        num_steps,
        δμ_LQSGW,
        meff_LQSGW,
        zfactor_LQSGW,
        E_k_LQSGW,
        Z_k_LQSGW,
        G_LQSGW,
        Π_LQSGW,
        W_LQSGW,
        Σ_LQSGW,
        Σ_LQSGW_ins,
    )
end

# New data
function load_lqsgw_data_new_format(
    param::Parameter.Para,
    int_type,
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2";
)
    local data
    max_step = -1
    filename = joinpath(savedir, savename)
    jldopen(filename, "r") do f
        @assert f["converged"] == true "Specificed save data did not converge!"
        # Find the converged data in JLD2 file
        for i in 0:(LQSGW.MAXIMUM_STEPS)
            if haskey(f, string(i))
                max_step = i
                data = f[string(i)]
            else
                break
            end
        end
        if max_step < 0
            error("No data found in $(savedir)!")
        end
        println(
            "Found converged data with max_step=$(max_step) at rs=$(round(param.rs; sigdigits=4)) for savename $(savename)!",
        )
        return data
    end
end
function load_oneshot_data_new_format(
    param::Parameter.Para,
    int_type,
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="g0w0_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2";
)
    filename = joinpath(savedir, savename)
    jldopen(filename, "r") do f
        if haskey(f, string(0))
            return f[string(0)]
        else
            error("No data found in $(savedir)!")
        end
    end
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

    # #########################################################################################
    # # Load one-shot methods calculated using ElectronGas.jl defaults for Σ_G0W0 = Σ[G0, Π0] #
    # #########################################################################################

    # f_sigma_G0W0 = np.load("data/3d/rpa/meff_3d_sigma_G0W0.npz")
    # rs_G0W0 = f_sigma_G0W0.get("rslist")
    # m_G0W0 = f_sigma_G0W0.get("mefflist")
    # z_G0W0 = f_sigma_G0W0.get("zlist")

    # f_sigma_G0Wp = np.load("data/3d/$(int_type_fp)/meff_3d_sigma_G0Wp.npz")
    # # f_sigma_G0Wp = np.load(
    # #     "data/3d/ko_moroni/meff_3d_sigma_G0Wp.npz")
    # rs_G0Wp = f_sigma_G0Wp.get("rslist")
    # m_G0Wp = f_sigma_G0Wp.get("mefflist")
    # z_G0Wp = f_sigma_G0Wp.get("zlist")

    # f_sigma_G0Wpm = np.load("data/3d/$(int_type_fp_fm)/meff_3d_sigma_G0Wpm.npz")
    # # f_sigma_G0Wpm = np.load(
    # #     "data/3d/ko_simion_giuliani/meff_3d_sigma_G0Wpm.npz")
    # rs_G0Wpm = f_sigma_G0Wpm.get("rslist")
    # m_G0Wpm = f_sigma_G0Wpm.get("mefflist")
    # z_G0Wpm = f_sigma_G0Wpm.get("zlist")

    #####################################################
    # Load one-shot and LQSGW data computed via LQSGW.jl #
    #####################################################

    data_rpa = load_oneshot_data_new_format(param, :rpa)
    data_rpa_fp = load_oneshot_data_new_format(param, int_type_fp)
    data_rpa_fp_fm = load_oneshot_data_new_format(param, int_type_fp_fm)

    data_lqsgw = load_lqsgw_data_new_format(param, :rpa)
    data_lqsgw_fp = load_lqsgw_data_new_format(param, int_type_fp)
    data_lqsgw_fp_fm = load_lqsgw_data_new_format(param, int_type_fp_fm)

    # RPA momentum grids for G, Π, and Σ
    qPgrid_rpa = data_rpa.Π.mesh[2]
    qPgrid_rpa_fp = data_rpa_fp.Π.mesh[2]
    qPgrid_rpa_fp_fm = data_rpa_fp_fm.Π.mesh[2]
    kSgrid_rpa = data_rpa.Σ_ins.mesh[2]
    kSgrid_rpa_fp = data_rpa_fp.Σ_ins.mesh[2]
    kSgrid_rpa_fp_fm = data_rpa_fp_fm.Σ_ins.mesh[2]

    # LQSGW momentum grids for G, Π, and Σ
    qPgrid_lqsgw = data_lqsgw.Π.mesh[2]
    qPgrid_lqsgw_fp = data_lqsgw_fp.Π.mesh[2]
    qPgrid_lqsgw_fp_fm = data_lqsgw_fp_fm.Π.mesh[2]
    kSgrid_lqsgw = data_lqsgw.Σ_ins.mesh[2]
    kSgrid_lqsgw_fp = data_lqsgw_fp.Σ_ins.mesh[2]
    kSgrid_lqsgw_fp_fm = data_lqsgw_fp_fm.Σ_ins.mesh[2]

    ########################################################
    # Plot observables for one-shot and LQSGW methods vs rs #
    ########################################################

    #############################
    # Plot exchange self-energy #
    #############################

    fig, ax = plt.subplots(; figsize=(5, 5))

    sigma_x_rpa = data_rpa.Σ_ins[1, :]
    sigma_x_rpa_fp = data_rpa_fp.Σ_ins[1, :]
    sigma_x_rpa_fp_fm = data_rpa_fp_fm.Σ_ins[1, :]

    # @assert all(abs.(imag.(sigma_x_rpa)) .< 1e-10)
    # @assert all(abs.(imag.(sigma_x_rpa_fp)) .< 1e-10)
    # @assert all(abs.(imag.(sigma_x_rpa_fp_fm)) .< 1e-10)

    sigma_x_lqsgw = data_lqsgw.Σ_ins[1, :]
    sigma_x_lqsgw_fp = data_lqsgw_fp.Σ_ins[1, :]
    sigma_x_lqsgw_fp_fm = data_lqsgw_fp_fm.Σ_ins[1, :]

    # @assert all(abs.(imag.(sigma_x_lqsgw)) .< 1e-10)
    # @assert all(abs.(imag.(sigma_x_lqsgw_fp)) .< 1e-10)
    # @assert all(abs.(imag.(sigma_x_lqsgw_fp_fm)) .< 1e-10)

    plot_spline(
        kSgrid_rpa / kF,
        real.(sigma_x_rpa),
        1,
        "\$G_0 W_0\$",
        ax;
        ls="--",
        zorder=1,
    )
    plot_spline(
        kSgrid_rpa_fp / kF,
        real.(sigma_x_rpa_fp),
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
    )
    plot_spline(
        kSgrid_rpa_fp_fm / kF,
        real.(sigma_x_rpa_fp_fm),
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
    )

    plot_spline(kSgrid_lqsgw / kF, real.(sigma_x_lqsgw), 4, "LQSGW", ax; zorder=2)
    plot_spline(
        kSgrid_lqsgw_fp / kF,
        real.(sigma_x_lqsgw_fp),
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
    )
    plot_spline(
        kSgrid_lqsgw_fp_fm / kF,
        real.(sigma_x_lqsgw_fp_fm),
        6,
        "LQSGW\$^\\text{KO}\$",
        ax;
        zorder=6,
    )

    if constant_fs
        ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ax.set_xlim(0, 2)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\Sigma_x(k)\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("sigma_x_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    #################################
    # Plot total static self-energy #
    #################################

    fig, ax = plt.subplots(; figsize=(5, 5))

    # w0_label = locate(sw1.mesh[1], 0)

    sigma_x_rpa = data_rpa.Σ_ins[1, :]
    sigma_x_rpa_fp = data_rpa_fp.Σ_ins[1, :]
    sigma_x_rpa_fp_fm = data_rpa_fp_fm.Σ_ins[1, :]

    # @assert all(abs.(imag.(sigma_x_rpa)) .< 1e-10)
    # @assert all(abs.(imag.(sigma_x_rpa_fp)) .< 1e-10)
    # @assert all(abs.(imag.(sigma_x_rpa_fp_fm)) .< 1e-10)

    sigma_x_lqsgw = data_lqsgw.Σ_ins[1, :]
    sigma_x_lqsgw_fp = data_lqsgw_fp.Σ_ins[1, :]
    sigma_x_lqsgw_fp_fm = data_lqsgw_fp_fm.Σ_ins[1, :]

    # @assert all(abs.(imag.(sigma_x_lqsgw)) .< 1e-10)
    # @assert all(abs.(imag.(sigma_x_lqsgw_fp)) .< 1e-10)
    # @assert all(abs.(imag.(sigma_x_lqsgw_fp_fm)) .< 1e-10)

    plot_spline(
        kSgrid_rpa / kF,
        real.(sigma_x_rpa),
        1,
        "\$G_0 W_0\$",
        ax;
        ls="--",
        zorder=1,
    )
    plot_spline(
        kSgrid_rpa_fp / kF,
        real.(sigma_x_rpa_fp),
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
    )
    plot_spline(
        kSgrid_rpa_fp_fm / kF,
        real.(sigma_x_rpa_fp_fm),
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
    )

    plot_spline(kSgrid_lqsgw / kF, real.(sigma_x_lqsgw), 4, "LQSGW", ax; zorder=2)
    plot_spline(
        kSgrid_lqsgw_fp / kF,
        real.(sigma_x_lqsgw_fp),
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
    )
    plot_spline(
        kSgrid_lqsgw_fp_fm / kF,
        real.(sigma_x_lqsgw_fp_fm),
        6,
        "LQSGW\$^\\text{KO}\$",
        ax;
        zorder=6,
    )

    if constant_fs
        ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ax.set_xlim(0, 2)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\Sigma_x(k)\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("sigma_static_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    ###################################
    # Plot static dielectric function #
    ###################################

    fig, ax = plt.subplots(; figsize=(5, 5))

    # 1 / ϵ =  W / V, even when f⁺ and/or f⁻ are nonzero

    Wtilde_q_static = data_rpa.W[1, :]
    @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_rpa]
    eps_inv_rpa = 1 .+ real.(Wtilde_q_static) ./ v_q

    Wtilde_q_static = data_rpa_fp.W[1, :]
    @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_rpa_fp]
    eps_inv_rpa_fp = 1 .+ real.(Wtilde_q_static) ./ v_q

    Wtilde_q_static = data_rpa_fp_fm.W[1, :]
    @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_rpa_fp_fm]
    eps_inv_rpa_fp_fm = 1 .+ real.(Wtilde_q_static) ./ v_q

    Wtilde_q_static = data_lqsgw.W[1, :]
    @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_lqsgw]
    eps_inv_lqsgw = 1 .+ real.(Wtilde_q_static) ./ v_q

    Wtilde_q_static = data_lqsgw_fp.W[1, :]
    @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_lqsgw_fp]
    eps_inv_lqsgw_fp = 1 .+ real.(Wtilde_q_static) ./ v_q

    Wtilde_q_static = data_lqsgw_fp_fm.W[1, :]
    @assert all(abs.(imag.(Wtilde_q_static)) .< 1e-10)
    v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid_lqsgw_fp_fm]
    eps_inv_lqsgw_fp_fm = 1 .+ real.(Wtilde_q_static) ./ v_q

    plot_spline(qPgrid_rpa / 2kF, eps_inv_rpa, 1, "\$G_0 W_0\$", ax; ls="--", zorder=2)
    plot_spline(
        qPgrid_rpa_fp / 2kF,
        eps_inv_rpa_fp,
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=4,
    )
    plot_spline(
        qPgrid_rpa_fp_fm / 2kF,
        eps_inv_rpa_fp_fm,
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=6,
    )
    # ax.scatter(
    #     qPgrid_rpa / 2kF,
    #     eps_inv_rpa;
    #     color=colors[1],
    #     s=10,
    #     zorder=1,
    #     facecolor="none",
    # )
    # ax.scatter(
    #     qPgrid_rpa_fp / 2kF,
    #     eps_inv_rpa_fp;
    #     color=colors[2],
    #     s=10,
    #     zorder=3,
    #     facecolor="none",
    # )
    # ax.scatter(
    #     qPgrid_rpa_fp_fm / 2kF,
    #     eps_inv_rpa_fp_fm;
    #     color=colors[3],
    #     s=10,
    #     zorder=5,
    #     facecolor="none",
    # )

    plot_spline(qPgrid_lqsgw / 2kF, eps_inv_lqsgw, 4, "LQSGW", ax; zorder=1)
    plot_spline(
        qPgrid_lqsgw_fp / 2kF,
        eps_inv_lqsgw_fp,
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=3,
    )
    plot_spline(
        qPgrid_lqsgw_fp_fm / 2kF,
        eps_inv_lqsgw_fp_fm,
        6,
        "LQSGW\$^\\text{KO}\$",
        ax;
        zorder=5,
    )
    # ax.scatter(
    #     qPgrid_lqsgw / 2kF,
    #     eps_inv_lqsgw;
    #     color=colors[4],
    #     s=10,
    #     zorder=1,
    #     facecolor="none",
    # )
    # ax.scatter(
    #     qPgrid_lqsgw_fp / 2kF,
    #     eps_inv_lqsgw_fp;
    #     color=colors[5],
    #     s=10,
    #     zorder=3,
    #     facecolor="none",
    # )
    # ax.scatter(
    #     qPgrid_lqsgw_fp_fm / 2kF,
    #     eps_inv_lqsgw_fp_fm;
    #     color=colors[6],
    #     s=10,
    #     zorder=5,
    #     facecolor="none",
    # )

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
    fig.savefig("inverse_static_dielectric_rs=$(round(rs; sigdigits=4))_$(fsstr)_zoom.pdf")
    ax.set_xlim(0, maxKP / 2kF)
    plt.tight_layout()
    fig.savefig("inverse_static_dielectric_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    ##############################
    # Plot quasiparticle residue #
    ##############################

    fig, ax = plt.subplots(; figsize=(5, 5))

    Z_k_rpa = data_rpa.Z_k
    Z_k_rpa_fp = data_rpa_fp.Z_k
    Z_k_rpa_fp_fm = data_rpa_fp_fm.Z_k

    Z_k_lqsgw = data_lqsgw.Z_k
    Z_k_lqsgw_fp = data_lqsgw_fp.Z_k
    Z_k_lqsgw_fp_fm = data_lqsgw_fp_fm.Z_k

    plot_spline(kSgrid_rpa / kF, Z_k_rpa, 1, "\$G_0 W_0\$", ax; ls="--", zorder=1)
    plot_spline(
        kSgrid_rpa_fp / kF,
        Z_k_rpa_fp,
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
    )
    plot_spline(
        kSgrid_rpa_fp_fm / kF,
        Z_k_rpa_fp_fm,
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
    )
    # ax.scatter(kSgrid_rpa / kF, Z_k_rpa, color=colors[1], s=10, zorder=1)
    # ax.scatter(kSgrid_rpa_fp / kF, Z_k_rpa_fp, color=colors[2], s=10, zorder=3)
    # ax.scatter(kSgrid_rpa_fp_fm / kF, Z_k_rpa_fp_fm, color=colors[3], s=10, zorder=5)

    plot_spline(kSgrid_lqsgw / kF, Z_k_lqsgw, 4, "LQSGW", ax; zorder=2)
    plot_spline(
        kSgrid_lqsgw_fp / kF,
        Z_k_lqsgw_fp,
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
    )
    plot_spline(
        kSgrid_lqsgw_fp_fm / kF,
        Z_k_lqsgw_fp_fm,
        6,
        "LQSGW\$^\\text{KO}\$",
        ax;
        zorder=6,
    )
    # ax.scatter(kSgrid_lqsgw / kF, Z_k_lqsgw, color=colors[4], s=10, zorder=2)
    # ax.scatter(kSgrid_lqsgw_fp / kF, Z_k_lqsgw_fp, color=colors[5], s=10, zorder=4)
    # ax.scatter(kSgrid_lqsgw_fp_fm / kF, Z_k_lqsgw_fp_fm, color=colors[6], s=10, zorder=6)

    if constant_fs
        ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ax.set_xlim(0, 4)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$Z(k)\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("zfactor_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    #############################
    # Plot quasiparticle energy #
    #############################

    fig, ax = plt.subplots(; figsize=(5, 5))
    ylimits = Dict(
        5.0 => (-0.17, 0.12),
        # ...
    )

    E_k_rpa = data_rpa.E_k
    E_k_rpa_fp = data_rpa_fp.E_k
    E_k_rpa_fp_fm = data_rpa_fp_fm.E_k

    E_k_lqsgw = data_lqsgw.E_k
    E_k_lqsgw_fp = data_lqsgw_fp.E_k
    E_k_lqsgw_fp_fm = data_lqsgw_fp_fm.E_k

    println(E_k_rpa[end], " ", E_k_lqsgw[end])
    println(E_k_rpa_fp[end], " ", E_k_lqsgw_fp[end])
    println(E_k_rpa_fp_fm[end], " ", E_k_lqsgw_fp_fm[end])

    E_k_0 = bare_energy(param, kSgrid_rpa)
    plot_spline(
        kSgrid_rpa / kF,
        E_k_0,
        8,
        "\$k^2 / 2m - \\epsilon_F\$",
        ax;
        ls="--",
        zorder=100,
    )

    plot_spline(kSgrid_rpa / kF, E_k_rpa, 1, "\$G_0 W_0\$", ax; ls="--", zorder=1)
    plot_spline(
        kSgrid_rpa_fp / kF,
        E_k_rpa_fp,
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
    )
    plot_spline(
        kSgrid_rpa_fp_fm / kF,
        E_k_rpa_fp_fm,
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
    )

    plot_spline(kSgrid_lqsgw / kF, E_k_lqsgw, 4, "LQSGW", ax; zorder=2)
    plot_spline(
        kSgrid_lqsgw_fp / kF,
        E_k_lqsgw_fp,
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
    )
    plot_spline(
        kSgrid_lqsgw_fp_fm / kF,
        E_k_lqsgw_fp_fm,
        6,
        "LQSGW\$^\\text{KO}\$",
        ax;
        zorder=6,
    )

    if constant_fs
        ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ylims = get(ylimits, rs, (nothing, nothing))
    ax.set_xlim(0, 1.5)
    # ax.set_xlim(0, 2)
    ax.set_ylim(ylims)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\mathcal{E}_\\text{qp}(k)\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("quasiparticle_energy_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    # ############################
    # # Plot static polarization #
    # ############################

    # fig, ax = plt.subplots(; figsize=(5, 5))

    # pi_static_rpa = data_rpa.Π[1, :]
    # pi_static_rpa_fp = data_rpa_fp.Π[1, :]
    # pi_static_rpa_fp_fm = data_rpa_fp_fm.Π[1, :]

    # @assert all(abs.(imag.(pi_static_rpa)) .< 1e-10)
    # @assert all(abs.(imag.(pi_static_rpa_fp)) .< 1e-10)
    # @assert all(abs.(imag.(pi_static_rpa_fp_fm)) .< 1e-10)

    # pi_static_lqsgw = data_lqsgw.Π[1, :]
    # pi_static_lqsgw_fp = data_lqsgw_fp.Π[1, :]
    # pi_static_lqsgw_fp_fm = data_lqsgw_fp_fm.Π[1, :]

    # @assert all(abs.(imag.(pi_static_lqsgw)) .< 1e-10)
    # @assert all(abs.(imag.(pi_static_lqsgw_fp)) .< 1e-10)
    # @assert all(abs.(imag.(pi_static_lqsgw_fp_fm)) .< 1e-10)

    # plot_spline(
    #     qPgrid_rpa / kF,
    #     real.(pi_static_rpa),
    #     1,
    #     "\$G_0 W_0\$",
    #     ax;
    #     ls="--",
    #     zorder=1,
    # )
    # plot_spline(
    #     qPgrid_rpa_fp / kF,
    #     real.(pi_static_rpa_fp),
    #     2,
    #     "\$G_0 W^\\text{KO}_{0,+}\$",
    #     ax;
    #     ls="--",
    #     zorder=3,
    # )
    # plot_spline(
    #     qPgrid_rpa_fp_fm / kF,
    #     real.(pi_static_rpa_fp_fm),
    #     3,
    #     "\$G_0 W^\\text{KO}_0\$",
    #     ax;
    #     ls="--",
    #     zorder=5,
    # )

    # plot_spline(qPgrid_lqsgw / kF, real.(pi_static_lqsgw), 4, "LQSGW", ax; zorder=2)
    # plot_spline(
    #     qPgrid_lqsgw_fp / kF,
    #     real.(pi_static_lqsgw_fp),
    #     5,
    #     "LQSGW\$^\\text{KO}_+\$",
    #     ax;
    #     zorder=4,
    # )
    # plot_spline(
    #     qPgrid_lqsgw_fp_fm / kF,
    #     real.(pi_static_lqsgw_fp_fm),
    #     6,
    #     "LQSGW\$^\\text{KO}\$",
    #     ax;
    #     zorder=6,
    # )

    # if constant_fs
    #     ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # else
    #     ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # end
    # ax.set_xlim(0, 4)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.set_ylabel("\$\\Pi(q, i\\nu_m = 0)\$")
    # ax.legend(; fontsize=12)
    # # ax.legend(; fontsize=12, ncol=3)
    # plt.tight_layout()
    # fig.savefig("pi_static_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    return
end

main()
