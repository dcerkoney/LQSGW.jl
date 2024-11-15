using Colors
using CompositeGrids
using ElectronGas
using GreenFunc
using JLD2
using Lehmann
using LsqFit
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

function plot_spline(
    x,
    y,
    idx,
    label,
    ax;
    ls="-",
    zorder=nothing,
    extrapolate=false,
    holes_at=[],
)
    if extrapolate
        mfitfunc = interp.PchipInterpolator(x, y; extrapolate=true)
    else
        mfitfunc = interp.Akima1DInterpolator(x, y)
    end
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
    if isempty(holes_at) == false
        for x_hole in holes_at
            ax.scatter(
                x_hole,
                mfitfunc(x_hole);
                color=colors[idx],
                zorder=zorder + 1,
                s=20,
                facecolor="white",
            )
        end
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

    sigma_x_lqsgw = data_lqsgw.Σ_ins[1, :]
    sigma_x_lqsgw_fp = data_lqsgw_fp.Σ_ins[1, :]
    sigma_x_lqsgw_fp_fm = data_lqsgw_fp_fm.Σ_ins[1, :]

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

    # one-shot methods
    sigma_ins_rpa = data_rpa.Σ_ins[1, :]
    sigma_ins_rpa_fp = data_rpa_fp.Σ_ins[1, :]
    sigma_ins_rpa_fp_fm = data_rpa_fp_fm.Σ_ins[1, :]
    sigma_dyn_rpa = data_rpa.Σ
    sigma_dyn_rpa_fp = data_rpa_fp.Σ
    sigma_dyn_rpa_fp_fm = data_rpa_fp_fm.Σ
    idx_w0_rpa = locate(sigma_dyn_rpa.mesh[1], 0)
    idx_w0_rpa_fp = locate(sigma_dyn_rpa_fp.mesh[1], 0)
    idx_w0_rpa_fp_fm = locate(sigma_dyn_rpa_fp_fm.mesh[1], 0)

    # quasiparticle self-consistent methods
    sigma_ins_lqsgw = data_lqsgw.Σ_ins[1, :]
    sigma_ins_lqsgw_fp = data_lqsgw_fp.Σ_ins[1, :]
    sigma_ins_lqsgw_fp_fm = data_lqsgw_fp_fm.Σ_ins[1, :]
    sigma_dyn_lqsgw = data_lqsgw.Σ
    sigma_dyn_lqsgw_fp = data_lqsgw_fp.Σ
    sigma_dyn_lqsgw_fp_fm = data_lqsgw_fp_fm.Σ
    idx_w0_lqsgw = locate(sigma_dyn_lqsgw.mesh[1], 0)
    idx_w0_lqsgw_fp = locate(sigma_dyn_lqsgw_fp.mesh[1], 0)
    idx_w0_lqsgw_fp_fm = locate(sigma_dyn_lqsgw_fp_fm.mesh[1], 0)

    # static self-energies
    sigma_static_rpa = sigma_ins_rpa .+ sigma_dyn_rpa[idx_w0_rpa, :]
    sigma_static_rpa_fp = sigma_ins_rpa_fp .+ sigma_dyn_rpa_fp[idx_w0_rpa_fp, :]
    sigma_static_rpa_fp_fm = sigma_ins_rpa_fp_fm .+ sigma_dyn_rpa_fp_fm[idx_w0_rpa_fp_fm, :]
    sigma_static_lqsgw = sigma_ins_lqsgw .+ sigma_dyn_lqsgw[idx_w0_lqsgw, :]
    sigma_static_lqsgw_fp = sigma_ins_lqsgw_fp .+ sigma_dyn_lqsgw_fp[idx_w0_lqsgw_fp, :]
    sigma_static_lqsgw_fp_fm =
        sigma_ins_lqsgw_fp_fm .+ sigma_dyn_lqsgw_fp_fm[idx_w0_lqsgw_fp_fm, :]

    partnames = ["Re", "Im"]
    partfuncs = [real, imag]
    for (partname, partfunc) in zip(partnames, partfuncs)
        fig, ax = plt.subplots(; figsize=(5, 5))
        plot_spline(
            kSgrid_rpa / kF,
            partfunc.(sigma_static_rpa),
            1,
            "\$G_0 W_0\$",
            ax;
            ls="--",
            zorder=1,
        )
        plot_spline(
            kSgrid_rpa_fp / kF,
            partfunc.(sigma_static_rpa_fp),
            2,
            "\$G_0 W^\\text{KO}_{0,+}\$",
            ax;
            ls="--",
            zorder=3,
        )
        plot_spline(
            kSgrid_rpa_fp_fm / kF,
            partfunc.(sigma_static_rpa_fp_fm),
            3,
            "\$G_0 W^\\text{KO}_0\$",
            ax;
            ls="--",
            zorder=5,
        )
        plot_spline(
            kSgrid_lqsgw / kF,
            partfunc.(sigma_static_lqsgw),
            4,
            "LQSGW",
            ax;
            zorder=2,
        )
        plot_spline(
            kSgrid_lqsgw_fp / kF,
            partfunc.(sigma_static_lqsgw_fp),
            5,
            "LQSGW\$^\\text{KO}_+\$",
            ax;
            zorder=4,
        )
        plot_spline(
            kSgrid_lqsgw_fp_fm / kF,
            partfunc.(sigma_static_lqsgw_fp_fm),
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
        ax.set_xlim(0, 4)
        ax.set_xlabel("\$k / k_F\$")
        ax.set_ylabel("\$\\text{$(partname)}\\Sigma(k, i\\omega_0)\$")
        # ax.legend(; fontsize=12)
        ax.legend(; fontsize=12, ncol=1)
        plt.tight_layout()
        fig.savefig(
            "$(lowercase(partname))_sigma_static_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf",
        )
    end

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

    Z_k_rpa = LQSGW.zfactor_full(param, data_rpa.Σ)
    Z_k_rpa_fp = LQSGW.zfactor_full(param, data_rpa_fp.Σ)
    Z_k_rpa_fp_fm = LQSGW.zfactor_full(param, data_rpa_fp_fm.Σ)
    Z_k_lqsgw = LQSGW.zfactor_full(param, data_lqsgw.Σ)
    Z_k_lqsgw_fp = LQSGW.zfactor_full(param, data_lqsgw_fp.Σ)
    Z_k_lqsgw_fp_fm = LQSGW.zfactor_full(param, data_lqsgw_fp_fm.Σ)

    # Z_k_rpa = data_rpa.Z_k
    # Z_k_rpa_fp = data_rpa_fp.Z_k
    # Z_k_rpa_fp_fm = data_rpa_fp_fm.Z_k
    # Z_k_lqsgw = data_lqsgw.Z_k
    # Z_k_lqsgw_fp = data_lqsgw_fp.Z_k
    # Z_k_lqsgw_fp_fm = data_lqsgw_fp_fm.Z_k

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

    ###################################
    # Plot dispersion renormalization #
    ###################################

    fig, ax = plt.subplots(; figsize=(5, 5))

    kgrid_plot = collect(range(0; stop=4, length=100))
    dk_plot = kgrid_plot[2] - kgrid_plot[1]

    # NOTE: weird bug when interpolating RPA kSgrids: bound is [minK, maxK] not [0, 2kF]!
    # WORKAROUND: use lqsgw grids for interpolation, which are the same up to the correct bound
    re_sigma_static_rpa_interp =
        Interp.interp1DGrid(real.(sigma_static_rpa), kSgrid_lqsgw, kgrid_plot)
    re_sigma_static_rpa_fp_interp =
        Interp.interp1DGrid(real.(sigma_static_rpa_fp), kSgrid_lqsgw_fp, kgrid_plot)
    re_sigma_static_rpa_fp_fm_interp =
        Interp.interp1DGrid(real.(sigma_static_rpa_fp_fm), kSgrid_lqsgw_fp_fm, kgrid_plot)
    re_sigma_static_lqsgw_interp =
        Interp.interp1DGrid(real.(sigma_static_lqsgw), kSgrid_lqsgw, kgrid_plot)
    re_sigma_static_lqsgw_fp_interp =
        Interp.interp1DGrid(real.(sigma_static_lqsgw_fp), kSgrid_lqsgw_fp, kgrid_plot)
    re_sigma_static_lqsgw_fp_fm_interp =
        Interp.interp1DGrid(real.(sigma_static_lqsgw_fp_fm), kSgrid_lqsgw_fp_fm, kgrid_plot)

    # ∂ₖReΣ(k, iω0) from central difference
    kplot_cd = kgrid_plot[2:(end - 1)]
    dsigma_dk_rpa_cd =
        (re_sigma_static_rpa_interp[3:end] - re_sigma_static_rpa_interp[1:(end - 2)]) /
        (2 * dk_plot)
    dsigma_dk_rpa_fp_cd =
        (
            re_sigma_static_rpa_fp_interp[3:end] -
            re_sigma_static_rpa_fp_interp[1:(end - 2)]
        ) / (2 * dk_plot)
    dsigma_dk_rpa_fp_fm_cd =
        (
            re_sigma_static_rpa_fp_fm_interp[3:end] -
            re_sigma_static_rpa_fp_fm_interp[1:(end - 2)]
        ) / (2 * dk_plot)
    dsigma_dk_lqsgw_cd =
        (re_sigma_static_lqsgw_interp[3:end] - re_sigma_static_lqsgw_interp[1:(end - 2)]) /
        (2 * dk_plot)
    dsigma_dk_lqsgw_fp_cd =
        (
            re_sigma_static_lqsgw_fp_interp[3:end] -
            re_sigma_static_lqsgw_fp_interp[1:(end - 2)]
        ) / (2 * dk_plot)
    dsigma_dk_lqsgw_fp_fm_cd =
        (
            re_sigma_static_lqsgw_fp_fm_interp[3:end] -
            re_sigma_static_lqsgw_fp_fm_interp[1:(end - 2)]
        ) / (2 * dk_plot)

    # D(k) = 1 + (m / k) ∂ₖReΣ(k, iω0)
    D_k_rpa = 1 .+ (param.me ./ kplot_cd) .* dsigma_dk_rpa_cd
    D_k_rpa_fp = 1 .+ (param.me ./ kplot_cd) .* dsigma_dk_rpa_fp_cd
    D_k_rpa_fp_fm = 1 .+ (param.me ./ kplot_cd) .* dsigma_dk_rpa_fp_fm_cd
    D_k_lqsgw = 1 .+ (param.me ./ kplot_cd) .* dsigma_dk_lqsgw_cd
    D_k_lqsgw_fp = 1 .+ (param.me ./ kplot_cd) .* dsigma_dk_lqsgw_fp_cd
    D_k_lqsgw_fp_fm = 1 .+ (param.me ./ kplot_cd) .* dsigma_dk_lqsgw_fp_fm_cd

    plot_spline(
        kplot_cd / kF,
        D_k_rpa,
        1,
        "\$G_0 W_0\$",
        ax;
        ls="--",
        zorder=1,
        extrapolate=true,
    )
    plot_spline(
        kplot_cd / kF,
        D_k_rpa_fp,
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
        extrapolate=true,
    )
    plot_spline(
        kplot_cd / kF,
        D_k_rpa_fp_fm,
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
        extrapolate=true,
    )
    plot_spline(kplot_cd / kF, D_k_lqsgw, 4, "LQSGW", ax; zorder=2, extrapolate=true)
    plot_spline(
        kplot_cd / kF,
        D_k_lqsgw_fp,
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
        extrapolate=true,
    )
    plot_spline(
        kplot_cd / kF,
        D_k_lqsgw_fp_fm,
        6,
        "LQSGW\$^\\text{KO}\$",
        ax;
        zorder=6,
        extrapolate=true,
    )
    if constant_fs
        ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ax.set_xlim(0, 4)
    ax.set_ylim(0.95, 1.75)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$D(k)\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("dfactor_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    #############################
    # Plot quasiparticle energy #
    #############################

    fig, ax = plt.subplots(; figsize=(5, 5))
    ylimits = Dict(
        5.0 => (-0.17, 0.12),
        # ...
    )

    E_k_0 = bare_energy(param, kSgrid_rpa)

    E_k_rpa = data_rpa.E_k
    E_k_rpa_fp = data_rpa_fp.E_k
    E_k_rpa_fp_fm = data_rpa_fp_fm.E_k

    E_k_lqsgw = data_lqsgw.E_k
    E_k_lqsgw_fp = data_lqsgw_fp.E_k
    E_k_lqsgw_fp_fm = data_lqsgw_fp_fm.E_k

    println("\nk = 0")
    println("EF = $(param.EF), E0(k=0) = $(E_k_0[1])")
    println(E_k_rpa[1], " ", E_k_lqsgw[1])
    println(E_k_rpa_fp[1], " ", E_k_lqsgw_fp[1])
    println(E_k_rpa_fp_fm[1], " ", E_k_lqsgw_fp_fm[1])

    println("\nk = 6kF")
    println(E_k_rpa[end], " ", E_k_lqsgw[end])
    println(E_k_rpa_fp[end], " ", E_k_lqsgw_fp[end])
    println(E_k_rpa_fp_fm[end], " ", E_k_lqsgw_fp_fm[end])

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
    # plot_spline(kSgrid_rpa / kF, E_k_rpa ./ Z_k_rpa, 1, "\$G_0 W_0\$", ax; ls="--", zorder=1)
    plot_spline(
        kSgrid_rpa_fp / kF,
        E_k_rpa_fp,
        # E_k_rpa_fp ./ Z_k_rpa_fp,
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
    )
    plot_spline(
        kSgrid_rpa_fp_fm / kF,
        E_k_rpa_fp_fm,
        # E_k_rpa_fp_fm ./ Z_k_rpa_fp_fm,
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
    )

    plot_spline(kSgrid_lqsgw / kF, E_k_lqsgw, 4, "LQSGW", ax; zorder=2)
    # plot_spline(kSgrid_lqsgw / kF, E_k_lqsgw ./ Z_k_lqsgw, 4, "LQSGW", ax; zorder=2)
    plot_spline(
        kSgrid_lqsgw_fp / kF,
        E_k_lqsgw_fp,
        # E_k_lqsgw_fp ./ Z_k_lqsgw_fp,
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
    )
    plot_spline(
        kSgrid_lqsgw_fp_fm / kF,
        E_k_lqsgw_fp_fm,
        # E_k_lqsgw_fp_fm ./ Z_k_lqsgw_fp_fm,
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
    # ax.set_xlim(0, 6)
    ax.set_xlim(0, 1.5)
    # ax.set_ylim(-0.26, 0.12)
    ax.set_ylim(ylims)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\mathcal{E}_\\text{qp}(k)\$")
    # ax.set_ylabel("\$Z^{-1}_k \\cdot \\mathcal{E}_\\text{qp}(k)\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("quasiparticle_energy_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")
    # fig.savefig("quasiparticle_energy_rs=$(round(rs; sigdigits=4))_$(fsstr)_full.pdf")
    # fig.savefig("quasiparticle_energy_times_Zinv_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")
    # fig.savefig("quasiparticle_energy_times_Zinv_rs=$(round(rs; sigdigits=4))_$(fsstr)_full.pdf")

    ###########################################################
    # Plot difference between full dispersion and bare energy #
    ###########################################################

    fig, ax = plt.subplots(; figsize=(5, 5))

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

    # (E_k - E_0)
    energy_ratio_0_rpa = (E_k_rpa - E_k_0)
    energy_ratio_0_rpa_fp = (E_k_rpa_fp - E_k_0)
    energy_ratio_0_rpa_fp_fm = (E_k_rpa_fp_fm - E_k_0)
    energy_ratio_0_lqsgw = (E_k_lqsgw - E_k_0)
    energy_ratio_0_lqsgw_fp = (E_k_lqsgw_fp - E_k_0)
    energy_ratio_0_lqsgw_fp_fm = (E_k_lqsgw_fp_fm - E_k_0)

    # the above function has a hole at k = kF; we need to remove grid points in this neighborhood
    kplot = kSgrid_rpa
    energy_ratio_0_rpa = energy_ratio_0_rpa
    energy_ratio_0_rpa_fp = energy_ratio_0_rpa_fp
    energy_ratio_0_rpa_fp_fm = energy_ratio_0_rpa_fp_fm
    energy_ratio_0_lqsgw = energy_ratio_0_lqsgw
    energy_ratio_0_lqsgw_fp = energy_ratio_0_lqsgw_fp
    energy_ratio_0_lqsgw_fp_fm = energy_ratio_0_lqsgw_fp_fm

    plot_spline(kplot / kF, energy_ratio_0_rpa, 1, "\$G_0 W_0\$", ax; ls="--", zorder=1)
    plot_spline(
        kplot / kF,
        energy_ratio_0_rpa_fp,
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
    )
    plot_spline(
        kplot / kF,
        energy_ratio_0_rpa_fp_fm,
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
    )
    plot_spline(kplot / kF, energy_ratio_0_lqsgw, 4, "LQSGW", ax; zorder=2)
    plot_spline(
        kplot / kF,
        energy_ratio_0_lqsgw_fp,
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
    )
    plot_spline(
        kplot / kF,
        energy_ratio_0_lqsgw_fp_fm,
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
    # ax.set_xlim(0.99, 1.01)
    ax.set_xlim(0, 6)
    ax.set_xlabel("\$k / k_F\$")
    # ax.set_ylabel("\$\\delta\\mathcal{E}^\\text{qp}_k\$")
    ax.set_ylabel("\$\\mathcal{E}^\\text{qp}_k - \\xi_k\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("delta_quasiparticle_energy_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    ###############################################################
    # Plot relative deviation of full dispersion from bare energy #
    ###############################################################

    fig, ax = plt.subplots(; figsize=(5, 5))

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
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_k_0,
    #     8,
    #     "\$k^2 / 2m - \\epsilon_F\$",
    #     ax;
    #     ls="--",
    #     zorder=100,
    # )

    # (E_k - E_0) / E_0
    energy_ratio_0_rpa = (E_k_rpa - E_k_0) ./ E_k_0
    energy_ratio_0_rpa_fp = (E_k_rpa_fp - E_k_0) ./ E_k_0
    energy_ratio_0_rpa_fp_fm = (E_k_rpa_fp_fm - E_k_0) ./ E_k_0
    energy_ratio_0_lqsgw = (E_k_lqsgw - E_k_0) ./ E_k_0
    energy_ratio_0_lqsgw_fp = (E_k_lqsgw_fp - E_k_0) ./ E_k_0
    energy_ratio_0_lqsgw_fp_fm = (E_k_lqsgw_fp_fm - E_k_0) ./ E_k_0

    # the above function has a numerical instability at k = kF;
    # we need to remove grid points in this neighborhood
    mask = abs.(kSgrid_rpa .- kF) .≥ 5e-2
    kplot = kSgrid_rpa[mask]
    insertion_idx = findfirst(.!mask)
    energy_ratio_0_rpa = energy_ratio_0_rpa[mask]
    energy_ratio_0_rpa_fp = energy_ratio_0_rpa_fp[mask]
    energy_ratio_0_rpa_fp_fm = energy_ratio_0_rpa_fp_fm[mask]
    energy_ratio_0_lqsgw = energy_ratio_0_lqsgw[mask]
    energy_ratio_0_lqsgw_fp = energy_ratio_0_lqsgw_fp[mask]
    energy_ratio_0_lqsgw_fp_fm = energy_ratio_0_lqsgw_fp_fm[mask]

    # Analytic result: limit of (E_k - E_0) / E_0 at k -> kF is (m/m* - 1)
    insert!(kplot, insertion_idx, kF)
    insert!(energy_ratio_0_rpa, insertion_idx, (1 / data_rpa.meff) - 1)
    insert!(energy_ratio_0_rpa_fp, insertion_idx, (1 / data_rpa_fp.meff) - 1)
    insert!(energy_ratio_0_rpa_fp_fm, insertion_idx, (1 / data_rpa_fp_fm.meff) - 1)
    insert!(energy_ratio_0_lqsgw, insertion_idx, (1 / data_lqsgw.meff) - 1)
    insert!(energy_ratio_0_lqsgw_fp, insertion_idx, (1 / data_lqsgw_fp.meff) - 1)
    insert!(energy_ratio_0_lqsgw_fp_fm, insertion_idx, (1 / data_lqsgw_fp_fm.meff) - 1)
    println("(rpa) m/m* - 1 = $((1 / data_rpa.meff) - 1)")
    println("(rpa_fp) m/m* - 1 = $((1 / data_rpa_fp.meff) - 1)")
    println("(rpa_fp_fm) m/m* - 1 = $((1 / data_rpa_fp_fm.meff) - 1)")
    println("(lqsgw) m/m* - 1 = $((1 / data_lqsgw.meff) - 1)")
    println("(lqsgw_fp) m/m* - 1 = $((1 / data_lqsgw_fp.meff) - 1)")
    println("(lqsgw_fp_fm) m/m* - 1 = $((1 / data_lqsgw_fp_fm.meff) - 1)")
    for (i, data) in enumerate([
        energy_ratio_0_rpa,
        energy_ratio_0_rpa_fp,
        energy_ratio_0_rpa_fp_fm,
        energy_ratio_0_lqsgw,
        energy_ratio_0_lqsgw_fp,
        energy_ratio_0_lqsgw_fp_fm,
        # data_rpa,
        # data_rpa_fp,
        # data_rpa_fp_fm,
        # data_lqsgw,
        # data_lqsgw_fp,
        # data_lqsgw_fp_fm,
    ])
        # ax.scatter([1.0], [(1 / data.meff) - 1]; color=colors[i], s=10, zorder=10)
        # ax.scatter([kplot[insertion_idx] / kF], [data[insertion_idx]]; color=colors[i], s=10, zorder=10)
    end

    plot_spline(kplot / kF, energy_ratio_0_rpa, 1, "\$G_0 W_0\$", ax; ls="--", zorder=1)
    plot_spline(
        kplot / kF,
        energy_ratio_0_rpa_fp,
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
    )
    plot_spline(
        kplot / kF,
        energy_ratio_0_rpa_fp_fm,
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
    )
    plot_spline(kplot / kF, energy_ratio_0_lqsgw, 4, "LQSGW", ax; zorder=2)
    plot_spline(
        kplot / kF,
        energy_ratio_0_lqsgw_fp,
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
    )
    plot_spline(
        kplot / kF,
        energy_ratio_0_lqsgw_fp_fm,
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
    # ax.set_xlim(0.99, 1.01)
    ax.set_xlim(0, 6)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$(\\mathcal{E}^\\text{qp}_k - \\xi_k) / \\xi_k\$")
    ax.legend(; fontsize=12)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("ratio_quasiparticle_energy_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    # ####################################################################
    # # Plot difference between full dispersion and quasiparticle energy #
    # ####################################################################

    fig, ax = plt.subplots(; figsize=(5, 5))

    # Quasiparticle energy
    E_qp_exact(mstar, mu, kgrid, Z_k) = Z_k .* (kgrid .^ 2 / (2 * mstar) .- mu)
    # E_qp_exact(mstar, mu, kgrid, Z_k) = Z_k .* (kgrid .^ 2 / (2 * param.me) .- mu)

    println("dmu_rpa = $(data_rpa.dmu)")
    println("dmu_rpa_fp = $(data_rpa_fp.dmu)")
    println("dmu_rpa_fp_fm = $(data_rpa_fp_fm.dmu)")
    println("dmu_lqsgw = $(data_lqsgw.dmu)")
    println("dmu_lqsgw_fp = $(data_lqsgw_fp.dmu)")
    println("dmu_lqsgw_fp_fm = $(data_lqsgw_fp_fm.dmu)")

    Z_k_rpa = data_rpa.Z_k
    mstar_rpa = data_rpa.meff * param.me
    mu_rpa = param.μ
    # mu_rpa = param.μ .+ data_rpa.dmu
    E_qp_rpa = E_qp_exact(mstar_rpa, mu_rpa, kSgrid_rpa, Z_k_rpa)

    Z_k_rpa_fp = data_rpa_fp.Z_k
    mstar_rpa_fp = data_rpa_fp.meff * param.me
    mu_rpa_fp = param.μ
    # mu_rpa_fp = param.μ .+ data_rpa_fp.dmu
    E_qp_rpa_fp = E_qp_exact(mstar_rpa_fp, mu_rpa_fp, kSgrid_rpa, Z_k_rpa_fp)

    Z_k_rpa_fp_fm = data_rpa_fp_fm.Z_k
    mstar_rpa_fp_fm = data_rpa_fp_fm.meff * param.me
    mu_rpa_fp_fm = param.μ
    # mu_rpa_fp_fm = param.μ .+ data_rpa_fp_fm.dmu
    E_qp_rpa_fp_fm = E_qp_exact(mstar_rpa_fp_fm, mu_rpa_fp_fm, kSgrid_rpa, Z_k_rpa_fp_fm)

    Z_k_lqsgw = data_lqsgw.Z_k
    mstar_lqsgw = data_lqsgw.meff * param.me
    mu_lqsgw = param.μ
    # mu_lqsgw = param.μ .+ data_lqsgw.dmu
    E_qp_lqsgw = E_qp_exact(mstar_lqsgw, mu_lqsgw, kSgrid_rpa, Z_k_lqsgw)

    Z_k_lqsgw_fp = data_lqsgw_fp.Z_k
    mstar_lqsgw_fp = data_lqsgw_fp.meff * param.me
    mu_lqsgw_fp = param.μ
    # mu_lqsgw_fp = param.μ .+ data_lqsgw_fp.dmu
    E_qp_lqsgw_fp = E_qp_exact(mstar_lqsgw_fp, mu_lqsgw_fp, kSgrid_rpa, Z_k_lqsgw_fp)

    Z_k_lqsgw_fp_fm = data_lqsgw_fp_fm.Z_k
    mstar_lqsgw_fp_fm = data_lqsgw_fp_fm.meff * param.me
    mu_lqsgw_fp_fm = param.μ
    # mu_lqsgw_fp_fm = param.μ .+ data_lqsgw_fp_fm.dmu
    E_qp_lqsgw_fp_fm =
        E_qp_exact(mstar_lqsgw_fp_fm, mu_lqsgw_fp_fm, kSgrid_rpa, Z_k_lqsgw_fp_fm)

    # Verify that the non-interacting density is preserved for the self-consistent methods
    for data in [data_lqsgw, data_lqsgw_fp, data_lqsgw_fp_fm]
        G = data.G
        Z = data.Z_k
        Z_F = data.zfactor
        kGgrid = G.mesh[2]
        G0 = G_0(param, G.mesh[1].representation, kGgrid)
        G0_ins = dlr_to_imtime(to_dlr(G0), [param.β]) * (-1)
        G_qp_ins = dlr_to_imtime(to_dlr(G), [param.β]) * (-1)
        integrand0 = real(G0_ins[1, :]) .* kGgrid.grid .* kGgrid.grid
        integrand = real(G_qp_ins[1, :]) .* kGgrid.grid .* kGgrid.grid
        integrand_with_Zinv = real(G_qp_ins[1, :] ./ Z) .* kGgrid.grid .* kGgrid.grid
        densityG0 = CompositeGrids.Interp.integrate1D(integrand0, kGgrid, [0, maxKG]) / π^2
        densityG = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKG]) / π^2
        densityGZinv =
            CompositeGrids.Interp.integrate1D(integrand_with_Zinv, kGgrid, [0, maxKG]) / π^2
        println("Density using G_qp / Z     (β = $(param.beta)): $(densityGZinv)")
        println("Density using G_qp / Z_F   (β = $(param.beta)): $(densityG / Z_F)")
        println("Density using G_0          (β = $(param.beta)): $(densityG0)")
        println("Exact non-interacting density (T = 0): $(param.n)\n")
    end

    # (E_k - E_qp) / E_qp
    energy_diff_qp_rpa = E_k_rpa - E_qp_rpa
    energy_diff_qp_rpa_fp = E_k_rpa_fp - E_qp_rpa_fp
    energy_diff_qp_rpa_fp_fm = E_k_rpa_fp_fm - E_qp_rpa_fp_fm
    energy_diff_qp_lqsgw = E_k_lqsgw - E_qp_lqsgw
    energy_diff_qp_lqsgw_fp = E_k_lqsgw_fp - E_qp_lqsgw_fp
    energy_diff_qp_lqsgw_fp_fm = E_k_lqsgw_fp_fm - E_qp_lqsgw_fp_fm

    kplot = kSgrid_rpa
    plot_spline(
        kplot / kF,
        energy_diff_qp_rpa,
        1,
        "\$G_0 W_0\$",
        ax;
        ls="--",
        zorder=1,
        holes_at=[],
    )
    plot_spline(
        kplot / kF,
        energy_diff_qp_rpa_fp,
        2,
        "\$G_0 W^\\text{KO}_{0,+}\$",
        ax;
        ls="--",
        zorder=3,
        holes_at=[],
    )
    plot_spline(
        kplot / kF,
        energy_diff_qp_rpa_fp_fm,
        3,
        "\$G_0 W^\\text{KO}_0\$",
        ax;
        ls="--",
        zorder=5,
        holes_at=[],
    )
    plot_spline(kplot / kF, energy_diff_qp_lqsgw, 4, "LQSGW", ax; zorder=2, holes_at=[])
    plot_spline(
        kplot / kF,
        energy_diff_qp_lqsgw_fp,
        5,
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        zorder=4,
        holes_at=[],
    )
    plot_spline(
        kplot / kF,
        energy_diff_qp_lqsgw_fp_fm,
        6,
        "LQSGW\$^\\text{KO}\$",
        ax;
        zorder=6,
        holes_at=[],
    )

    if constant_fs
        ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ax.set_xlim(0, 6)
    # ax.set_ylim(-0.0125, 0.1125)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\mathcal{E}^\\text{qp}_k - \\xi^\\star_k\$")
    ax.legend(; fontsize=12, ncol=2)
    # ax.legend(; fontsize=12, ncol=3)
    plt.tight_layout()
    fig.savefig("quasiparticle_energy_difference_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

    # ##########################################################
    # # Plot full dispersion and quasiparticle energy together #
    # ##########################################################

    # mu_rpa = (param.μ + data_rpa.dmu)
    # mu_rpa_fp = (param.μ + data_rpa_fp.dmu)
    # mu_rpa_fp_fm = (param.μ + data_rpa_fp_fm.dmu)
    # mu_lqsgw = (param.μ + data_lqsgw.dmu)
    # mu_lqsgw_fp = (param.μ + data_lqsgw_fp.dmu)
    # mu_lqsgw_fp_fm = (param.μ + data_lqsgw_fp_fm.dmu)
    # println("mu_rpa = $mu_rpa")
    # println("mu_rpa_fp = $mu_rpa_fp")
    # println("mu_rpa_fp_fm = $mu_rpa_fp_fm")
    # println("mu_lqsgw = $mu_lqsgw")
    # println("mu_lqsgw_fp = $mu_lqsgw_fp")
    # println("mu_lqsgw_fp_fm = $mu_lqsgw_fp_fm")

    # fig, ax = plt.subplots(; figsize=(5, 5))
    # kplot = kSgrid_rpa
    # plot_spline(kplot / kF, E_k_0, 8, "\$k^2 / 2m - \\epsilon_F\$", ax; ls="--", zorder=100)
    # plot_spline(kplot / kF, E_qp_rpa, 1, "\$Z_k \\xi^*_{G_0 W_0}(k)\$", ax; ls="--", zorder=1)
    # plot_spline(
    #     kplot / kF,
    #     E_k_rpa,
    #     1,
    #     "\$\\mathcal{E}^\\text{qp}_{G_0 W_0}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=1,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_qp_rpa_fp,
    #     2,
    #     "\$Z_k \\xi^*_{G_0 W^\\text{KO}_{0,+}}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=3,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_k_rpa_fp,
    #     2,
    #     "\$\\mathcal{E}^\\text{qp}_{G_0 W^\\text{KO}_{0,+}}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=3,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_qp_rpa_fp_fm,
    #     3,
    #     "\$Z_k \\xi^*_{G_0 W^\\text{KO}_0}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=5,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_k_rpa_fp_fm,
    #     3,
    #     "\$\\mathcal{E}^\\text{qp}_{G_0 W^\\text{KO}_0}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=5,
    # )
    # if constant_fs
    #     ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # else
    #     ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # end
    # ax.set_xlim(0, 6)
    # # ax.set_ylim(-0.0125, 0.1125)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig(
    #     "quasiparticle_energy_comparison_oneshot_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf",
    # )

    # fig, ax = plt.subplots(; figsize=(5, 5))
    # kplot = kSgrid_rpa
    # plot_spline(kplot / kF, E_k_0, 8, "\$k^2 / 2m - \\epsilon_F\$", ax; ls="--", zorder=100)
    # plot_spline(
    #     kplot / kF,
    #     E_qp_lqsgw,
    #     4,
    #     "\$Z_k \\xi^*_\\text{LQSGW}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=2,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_k_lqsgw,
    #     4,
    #     "\$\\mathcal{E}^\\text{qp}_\\text{LQSGW}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=2,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_qp_lqsgw_fp,
    #     5,
    #     "\$Z_k \\xi^*_{\\text{LQSGW}^\\text{KO}_+}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=4,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_k_lqsgw_fp,
    #     5,
    #     "\$\\mathcal{E}^\\text{qp}_{\\text{LQSGW}^\\text{KO}_+}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=4,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_qp_lqsgw_fp_fm,
    #     6,
    #     "\$Z_k \\xi^*_{\\text{LQSGW}^\\text{KO}}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=6,
    # )
    # plot_spline(
    #     kplot / kF,
    #     E_k_lqsgw_fp_fm,
    #     6,
    #     "\$\\mathcal{E}^\\text{qp}_{\\text{LQSGW}^\\text{KO}}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=6,
    # )
    # if constant_fs
    #     ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # else
    #     ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # end
    # ax.set_xlim(0, 6)
    # # ax.set_ylim(-0.0125, 0.1125)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig(
    #     "quasiparticle_energy_comparison_lqsgw_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf",
    # )

    # ############################################
    # # Plot full dispersion with quadratic fits #
    # ############################################

    # # Least-squares fits quadratic dispersions, p = (μ, m*)
    # @. Ekstar_model(k, p) = (k^2 / (2 * p[2]) - p[1])

    # # Fit to low-energy behavior
    # cutoff = kF
    # x = kSgrid_rpa[kSgrid_rpa .≤ cutoff]

    # y = E_k_rpa[kSgrid_rpa .≤ cutoff] ./ Z_k_rpa[kSgrid_rpa .≤ cutoff]
    # p0 = [param.μ + data_rpa.dmu, data_rpa.meff]
    # fit_rpa = curve_fit(Ekstar_model, x, y, p0)
    # fitted_model_rpa(x) = Ekstar_model(x, fit_rpa.param)
    # fit_errs = stderror(fit_rpa)
    # println("\nfit_rpa guesses:\t\t\t$(p0)")
    # println("fit_rpa parameters:\t\t\t$(fit_rpa.param)")
    # println("fit_rpa standard errors:\t\t$(fit_errs)")
    # E_qp_fit_rpa = fitted_model_rpa.(kSgrid_rpa) .* Z_k_rpa

    # y = E_k_rpa_fp[kSgrid_rpa .≤ cutoff] ./ Z_k_rpa_fp[kSgrid_rpa .≤ cutoff]
    # p0 = [param.μ + data_rpa_fp.dmu, data_rpa_fp.meff]
    # fit_rpa_fp = curve_fit(Ekstar_model, x, y, p0)
    # fitted_model_rpa_fp(x) = Ekstar_model(x, fit_rpa_fp.param)
    # fit_errs = stderror(fit_rpa_fp)
    # println("\nfit_rpa_fp guesses:\t\t\t$(p0)")
    # println("fit_rpa_fp parameters:\t\t\t$(fit_rpa_fp.param)")
    # println("fit_rpa_fp standard errors:\t\t$(fit_errs)")
    # E_qp_fit_rpa_fp = fitted_model_rpa_fp.(kSgrid_rpa) .* Z_k_rpa_fp

    # y = E_k_rpa_fp_fm[kSgrid_rpa .≤ cutoff] ./ Z_k_rpa_fp_fm[kSgrid_rpa .≤ cutoff]
    # p0 = [param.μ + data_rpa_fp_fm.dmu, data_rpa_fp_fm.meff]
    # fit_rpa_fp_fm = curve_fit(Ekstar_model, x, y, p0)
    # fitted_model_rpa_fp_fm(x) = Ekstar_model(x, fit_rpa_fp_fm.param)
    # fit_errs = stderror(fit_rpa_fp_fm)
    # println("\nfit_rpa_fp_fm guesses:\t\t\t$(p0)")
    # println("fit_rpa_fp_fm parameters:\t\t$(fit_rpa_fp_fm.param)")
    # println("fit_rpa_fp_fm standard errors:\t\t$(fit_errs)")
    # E_qp_fit_rpa_fp_fm = fitted_model_rpa_fp_fm.(kSgrid_rpa) .* Z_k_rpa_fp_fm

    # fig, ax = plt.subplots(; figsize=(5, 5))
    # plot_spline(kSgrid_rpa / kF, E_k_0, 8, "\$k^2 / 2m - \\epsilon_F\$", ax; ls="--", zorder=100)
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_qp_fit_rpa,
    #     1,
    #     "\$Z_k \\xi^*_{G_0 W_0}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=1,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_k_rpa,
    #     1,
    #     "\$\\mathcal{E}^\\text{qp}_{G_0 W_0}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=1,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_qp_fit_rpa_fp,
    #     2,
    #     "\$Z_k \\xi^*_{G_0 W^\\text{KO}_{0,+}}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=3,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_k_rpa_fp,
    #     2,
    #     "\$\\mathcal{E}^\\text{qp}_{G_0 W^\\text{KO}_{0,+}}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=3,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_qp_fit_rpa_fp_fm,
    #     3,
    #     "\$Z_k \\xi^*_{G_0 W^\\text{KO}_0}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=5,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_k_rpa_fp_fm,
    #     3,
    #     "\$\\mathcal{E}^\\text{qp}_{G_0 W^\\text{KO}_0}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=5,
    #     holes_at=[],
    # )
    # if constant_fs
    #     ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # else
    #     ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # end
    # ax.set_xlim(0, 2)
    # ax.set_ylim(-0.2, 0.6)
    # # ax.set_ylim(-0.0125, 0.1125)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig(
    #     "quasiparticle_energy_fits_oneshot_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf",
    # )

    # y = E_k_lqsgw[kSgrid_rpa .≤ cutoff] ./ Z_k_lqsgw[kSgrid_rpa .≤ cutoff]
    # p0 = [param.μ + data_lqsgw.dmu, data_lqsgw.meff]
    # fit_lqsgw = curve_fit(Ekstar_model, x, y, p0)
    # fitted_model_lqsgw(x) = Ekstar_model(x, fit_lqsgw.param)
    # fit_errs = stderror(fit_lqsgw)
    # println("\nfit_lqsgw guesses:\t\t\t$(p0)")
    # println("fit_lqsgw parameters:\t\t\t$(fit_lqsgw.param)")
    # println("fit_lqsgw standard errors:\t\t$(fit_errs)")
    # E_qp_fit_lqsgw = fitted_model_lqsgw.(kSgrid_rpa) .* Z_k_lqsgw

    # y = E_k_lqsgw_fp[kSgrid_rpa .≤ cutoff] ./ Z_k_lqsgw_fp[kSgrid_rpa .≤ cutoff]
    # p0 = [param.μ + data_lqsgw_fp.dmu, data_lqsgw_fp.meff]
    # fit_lqsgw_fp = curve_fit(Ekstar_model, x, y, p0)
    # fitted_model_lqsgw_fp(x) = Ekstar_model(x, fit_lqsgw_fp.param)
    # fit_errs = stderror(fit_lqsgw_fp)
    # println("\nfit_lqsgw_fp guesses:\t\t\t$(p0)")
    # println("fit_lqsgw_fp parameters:\t\t$(fit_lqsgw_fp.param)")
    # println("fit_lqsgw_fp standard errors:\t\t$(fit_errs)")
    # E_qp_fit_lqsgw_fp = fitted_model_lqsgw_fp.(kSgrid_rpa) .* Z_k_lqsgw_fp

    # y = E_k_lqsgw_fp_fm[kSgrid_rpa .≤ cutoff] ./ Z_k_lqsgw_fp_fm[kSgrid_rpa .≤ cutoff]
    # p0 = [param.μ + data_lqsgw_fp_fm.dmu, data_lqsgw_fp_fm.meff]
    # fit_lqsgw_fp_fm = curve_fit(Ekstar_model, x, y, p0)
    # fitted_model_lqsgw_fp_fm(x) = Ekstar_model(x, fit_lqsgw_fp_fm.param)
    # fit_errs = stderror(fit_lqsgw_fp_fm)
    # println("\nfit_lqsgw_fp_fm guesses:\t\t$(p0)")
    # println("fit_lqsgw_fp_fm parameters:\t\t$(fit_lqsgw_fp_fm.param)")
    # println("fit_lqsgw_fp_fm standard errors:\t$(fit_errs)")
    # E_qp_fit_lqsgw_fp_fm = fitted_model_lqsgw_fp_fm.(kSgrid_rpa) .* Z_k_lqsgw_fp_fm

    # fig, ax = plt.subplots(; figsize=(5, 5))
    # plot_spline(kSgrid_rpa / kF, E_k_0, 8, "\$k^2 / 2m - \\epsilon_F\$", ax; ls="--", zorder=100)
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_qp_fit_lqsgw,
    #     4,
    #     "\$\\xi^*_\\text{LQSGW}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=2,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_k_lqsgw,
    #     4,
    #     "\$\\mathcal{E}^\\text{qp}_\\text{LQSGW}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=2,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_qp_fit_lqsgw_fp,
    #     5,
    #     "\$\\xi^*_{\\text{LQSGW}^\\text{KO}_+}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=4,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_k_lqsgw_fp,
    #     5,
    #     "\$\\mathcal{E}^\\text{qp}_{\\text{LQSGW}^\\text{KO}_+}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=4,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_qp_fit_lqsgw_fp_fm,
    #     6,
    #     "\$\\xi^*_{\\text{LQSGW}^\\text{KO}}(k)\$",
    #     ax;
    #     ls="--",
    #     zorder=6,
    #     holes_at=[],
    # )
    # plot_spline(
    #     kSgrid_rpa / kF,
    #     E_k_lqsgw_fp_fm,
    #     6,
    #     "\$\\mathcal{E}^\\text{qp}_{\\text{LQSGW}^\\text{KO}}(k)\$",
    #     ax;
    #     ls="-",
    #     zorder=6,
    #     holes_at=[],
    # )
    # if constant_fs
    #     ax.set_title("Constant \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # else
    #     ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    # end
    # ax.set_xlim(0, 2)
    # ax.set_ylim(-0.2, 0.6)
    # # ax.set_ylim(-0.0125, 0.1125)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig("quasiparticle_energy_fits_lqsgw_rs=$(round(rs; sigdigits=4))_$(fsstr).pdf")

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
