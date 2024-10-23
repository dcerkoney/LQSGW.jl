using Colors
using CompositeGrids
using ElectronGas
using JLD2
using Lehmann
using LQSGW
using Parameters
using PyCall
using PyPlot

@pyimport pandas as pd  # pd.unique
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

pts = ["s", "^", "v", "p", "s", "<", "h", "o"]
colors = [
    cdict["blue"],
    "grey",
    cdict["teal"],
    cdict["cyan"],
    cdict["orange"],
    cdict["magenta"],
    cdict["red"],
    "black",
]
reflabels = ["\$^*\$", "\$^\\dagger\$", "\$^\\ddagger\$"]

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

# function loadmeffnpz(
#     param::Parameter.Para,
#     int_type,
#     δK=5e-6,
#     max_steps=300,
#     savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
#     savename="lqsgw_$(param.dim)d_$(int_type)_merged.npz",
# )
#     meff = massratio(param, s, si, δK)[1]
#     zfactor = zfactor_fermi(param, s)
#     println("(rs = $(param.rs)) Loaded mass data from converged step $n_max")
#     return datadict
# end

function loaddata_old_format(
    param::Parameter.Para,
    int_type,
    δK=5e-6,
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    local n_max, s, si
    filename = joinpath(savedir, savename)
    jldopen(filename, "r") do f
        println(f)
        for i in 0:(LQSGW.MAXIMUM_STEPS)
            try
                s = f["Σ_$i"]
                si = f["Σ_ins_$i"]
            catch
                n_max = i - 1
                break
            end
        end
    end
    meff = massratio(param, s, si, δK)[1]
    zfactor = zfactor_fermi(param, s)
    println("(rs = $(param.rs)) Loaded mass data from converged step $n_max")
    return meff, zfactor
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
        # Ensure that the saved data was convergent
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
        println("Found converged data with max_step=$(max_step) for savename $(savename)!")
    end
    return data.meff, data.zfactor
end
function load_oneshot_data_new_format(
    param::Parameter.Para,
    int_type,
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="g0w0_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2";
)
    local data
    filename = joinpath(savedir, savename)
    jldopen(filename, "r") do f
        if haskey(f, string(1))
            data = f[string(1)]
        else
            error("No one-shot data found in $(savedir)!")
        end
        println("Found one-shot data for savename $(savename)!")
    end
    return data.meff, data.zfactor
end

function plot_landaufunc(
    beta,
    rslist;
    q_kF_plot=collect(LinRange(1e-5, 8, 501)),
    dir=@__DIR__
)
    # Plot F_s and F_a vs q at rs = 2, 4, 6
    fig1 = figure(; figsize=(5, 5))
    ax1 = fig1.add_subplot(111)
    ic = 1
    colors = [
        [cdict["orange"], cdict["blue"]],
        [cdict["magenta"], cdict["cyan"]],
        [cdict["red"], cdict["teal"]],
    ]
    l1_handles = []
    l2_handles = []
    # Store q = 0 values of F_s and F_a for each rs
    Fs_const_q0_rslist = []
    Fa_const_q0_rslist = []
    Fs_q0_rslist = []
    Fa_q0_rslist = []
    for rs in rslist
        param = Parameter.rydbergUnit(1.0 / beta, rs)
        Fs_PW = get_Fs_PW(rs)
        Fa_PW = get_Fa_PW(rs)
        # Calculate the Coulomb interaction for plot momenta q
        Vq = []
        for q in q_kF_plot * param.kF
            Vs, Va = Interaction.coulomb(q, param)
            @assert Va == 0.0  # The Coulomb interaction couples spin-symmetrically
            push!(Vq, Vs)
        end
        if rs == 1.0
            Fs_qlist_const = []
            Fa_qlist_const = []
            Fs_qlist = []
            Fa_qlist = []
            Fs_qlist_v2 = []
            for q_kF in q_kF_plot
                Fs_const, Fa_const =
                    param.NF .* Interaction.landauParameterConst(
                        q_kF * param.kF,
                        0,
                        param;
                        Fs=Fs_PW,
                        Fa=Fa_PW,
                    )
                Fs, Fa =
                    param.NF .*
                    Interaction.landauParameterSimionGiuliani(q_kF * param.kF, 0, param)
                Fs_v2, _ =
                    param.NF .* Interaction.landauParameterMoroni(q_kF * param.kF, 0, param)
                # @assert Fs ≈ Fs_v2
                push!(Fs_qlist_const, Fs_const)
                push!(Fa_qlist_const, Fa_const)
                push!(Fs_qlist, Fs)
                # push!(Fs_qlist, Fs_v2)
                push!(Fa_qlist, Fa)
                push!(Fs_qlist_v2, Fs_v2)
            end
            # Plot F_i
            h1, = ax1.plot(
                q_kF_plot,
                Fs_qlist_const,
                "--";
                label="\$1 - \\frac{\\kappa_0}{\\kappa}\$",
                # label="\$\\textstyle\\frac{1}{\\mathcal{N}_F}\\Big(1 - \\frac{\\kappa_0}{\\kappa}\\Big)\$",
                # label="\$\\frac{1}{\\mathcal{N}_F}\\left(1 - \\frac{\\kappa_0}{\\kappa}\\right)\$ (\$r_s = $(Int(round(rs)))\$)",
                color=colors[ic + 1][1],
            )
            h2, = ax1.plot(
                q_kF_plot,
                Fs_qlist;
                label="\$F^+(q)\$",
                # label="\$F^+(q)\$ (\$r_s = $(Int(round(rs)))\$)",
                color=colors[ic][1],
            )
            h3, = ax1.plot(
                q_kF_plot,
                Fa_qlist_const,
                "--";
                label="\$1 - \\frac{\\chi_0}{\\chi}\$",
                # label="\$\\textstyle\\frac{1}{\\mathcal{N}_F}\\Big(1 - \\frac{\\chi_0}{\\chi}\\Big)\$",
                # label="\$\\frac{1}{\\mathcal{N}_F}\\left(1 - \\frac{\\chi_0}{\\chi}\\right)\$ (\$r_s = $(Int(round(rs)))\$)",
                color=colors[ic + 1][2],
            )
            h4, = ax1.plot(
                q_kF_plot,
                Fa_qlist;
                label="\$F^-(q)\$",
                # label="\$F^-(q)\$ (\$r_s = $(Int(round(rs)))\$)",
                color=colors[ic][2],
            )
            println(h1)
            if rs == 1.0
                append!(l1_handles, [h2, h1, h4, h3])
            elseif rs == 5.0
                append!(l2_handles, [h1, h2, h3, h4])
            end
            # Increment color index
            ic += 1
        end
        Fs_q0_const, Fa_q0_const =
            param.NF .* Interaction.landauParameterConst(1e-8, 0, param; Fs=Fs_PW, Fa=Fa_PW)
        Fs_q0, Fa_q0 = param.NF .* Interaction.landauParameterSimionGiuliani(1e-8, 0, param)
        Fs_q0_v2, _ = param.NF .* Interaction.landauParameterMoroni(1e-8, 0, param)
        push!(Fs_const_q0_rslist, Fs_q0_const)
        push!(Fa_const_q0_rslist, Fa_q0_const)
        push!(Fs_q0_rslist, Fs_q0)
        # push!(Fs_q0_rslist, Fs_q0_v2)
        push!(Fa_q0_rslist, Fa_q0)
    end
    # Finish Fig. 1
    ax1.set_ylim(0, 0.28)
    top_legend = plt.legend(;
        handles=l1_handles,
        loc="upper center",
        fontsize=14,
        ncol=2,
        title="\$r_s = 1\$",
    )
    ax1.add_artist(top_legend)
    # Add text label \$r_s = 1.0\$ to the plot
    # ax1.text(0.05, 0.25, "\$r_s = 1.0\$")
    ax1.set_ylabel("\$F^\\pm(q)\$")
    ax1.set_xlabel("\$q / k_F\$")
    fig1.tight_layout()
    fig1.savefig(joinpath(dir, "fs_and_fa_vs_q_rs=1.0.pdf"))

    # Plot Fs and Fa vs rs at q = 0
    fig4 = figure(; figsize=(5, 5))
    ax4 = fig4.add_subplot(111)
    println(rslist)
    println(Fs_q0_rslist)
    println(Fa_q0_rslist)
    ax4.plot(rslist, Fs_q0_rslist; label="\$F^+(q = 0)\$", color=cdict["orange"])
    ax4.plot(
        rslist,
        Fs_const_q0_rslist,
        "--";
        label="\$\\textstyle1 - \\frac{\\kappa_0}{\\kappa}\$",
        # label="\$\\textstyle\\frac{1}{\\mathcal{N}_F}\\Big(1 - \\frac{\\kappa_0}{\\kappa}\\Big)\$",
        color=cdict["magenta"],
    )
    ax4.plot(rslist, Fa_q0_rslist; label="\$F^-(q = 0)\$", color=cdict["blue"])
    ax4.plot(
        rslist,
        Fa_const_q0_rslist,
        "--";
        label="\$\\textstyle1 - \\frac{\\chi_0}{\\chi}\$",
        # label="\$\\textstyle\\frac{1}{\\mathcal{N}_F}\\Big(1 - \\frac{\\chi_0}{\\chi}\\Big)\$",
        color=cdict["cyan"],
    )
    # ax4.set_xticks(0:5)
    # ax4.set_xlim(0, 1)
    ax4.set_ylim(-0.05, 1.05)
    legend(; loc="best", ncol=2, fontsize=14)
    ylabel("\$F^\\pm(q = 0)\$")
    xlabel("\$r_s\$")
    tight_layout()
    savefig(joinpath(dir, "fs_and_fa_q0_vs_rs.pdf"))
end

function loaddata(
    key,
    param::Parameter.Para,
    int_type=:rpa;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    data = []
    n_max = -1
    filename = joinpath(savedir, savename)
    jldopen(filename, "r") do f
        for i in 0:max_steps
            try
                push!(data, f["$(key)_$(i)"])
            catch
                n_max = i - 1
                break
            end
        end
    end
    return data, n_max
end
function loaddata(
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
            error("Data not found for key = $key, i_step = $i_step")
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
    G0 = loaddata(0, "G", param, int_type)
    Π0 = loaddata(0, "Π", param, int_type)
    Σ_G0W0 = loaddata(1, "Σ", param, int_type)
    Σ_G0W0_ins = loaddata(1, "Σ_ins", param, int_type)
    # Quasiparticle properties on the Fermi surface derived from Σ_G0W0
    E_k_G0W0 = loaddata(1, "E_k", param, int_type)
    Z_k_G0W0 = loaddata(1, "Z_k", param, int_type)
    δμ_G0W0 = chemicalpotential(param, Σ_prev, Σ_ins_prev)
    meff_G0W0 = massratio(param, Σ_prev, Σ_ins_prev, δK)[1]
    zfactor_G0W0 = zfactor_fermi(param, Σ_prev)
    return (Π0, Σ_G0W0, Σ_G0W0_ins, E_k_G0W0, Z_k_G0W0, δμ_G0W0, meff_G0W0, zfactor_G0W0)
end

function load_lqsgw_data(
    param::Parameter.Para,
    int_type=:rpa,
    δK=5e-6,
    max_steps=300;
    savedir="$(LQSGW.DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    # Converged LQSGW: G_i -> Π_i -> W_i -> Σ_(i+1) = Σ_qp
    G_qp, num_steps = loaddata("G", param, int_type)
    Π_qp = loaddata("Π", param, int_type)[1]
    E_k_qp = loaddata("E_k", param, int_type)[1]
    Z_k_qp = loaddata("Z_k", param, int_type)[1]
    Σ_qp, num_steps_p1 = loaddata("Σ", param, int_type)
    Σ_qp_ins = loaddata("Σ_ins", param, int_type)[1]
    @assert num_steps_p1 == num_steps + 1
    # Quasiparticle properties on the Fermi surface derived from Σ_qp
    δμ_qp = chemicalpotential(param, Σ_prev, Σ_ins_prev)
    meff_qp = massratio(param, Σ_prev, Σ_ins_prev, δK)[1]
    zfactor_qp = zfactor_fermi(param, Σ_prev)
    return (Π_qp, Σ_qp, Σ_qp_ins, E_k_qp, Z_k_qp, δμ_qp, meff_qp, zfactor_qp)
end

function errorbar_mvsrs(rs, meff_data, merr, idx, label, ax; zorder=nothing, capsize=4)
    if isnothing(zorder) == false
        handle = ax.errorbar(
            rs,
            meff_data,
            merr;
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
            rs,
            meff_data,
            merr;
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

function plot_mvsrs(
    rs,
    meff_data,
    idx,
    label,
    ax;
    ls="-",
    rs_HDL=nothing,
    meff_HDL=nothing,
    zorder=nothing,
)
    # Add data in the high-density limit to the fit, if provided
    if isnothing(rs_HDL) == false && isnothing(meff_HDL) == false
        _rslist = np.array(pd.unique(np.concatenate([rs, rs_HDL])))
        mdatalist = np.array(pd.unique(np.concatenate([meff_data, meff_HDL])))
        P = np.argsort(_rslist)
        rs = _rslist[P .+ 1]
        meff_data = mdatalist[P .+ 1]
    end
    # println(rs)
    # println(meff_data)

    # mfitfunc = interp.PchipInterpolator(rs, meff_data)
    mfitfunc = interp.Akima1DInterpolator(rs, meff_data)
    # xgrid = np.arange(0, 10.2, 0.02)
    xgrid = np.arange(0, 10.2, 0.02)
    # ax.plot(rs, meff_data, 'o', ms=10, color=colors[idx])
    # ax.plot(xgrid, mfitfunc(xgrid), ls=ls, lw=2, color=colors[idx], label=label)
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
    yfit = np.ma.masked_invalid(mfitfunc(xgrid))
    # println(yfit)
    # println("Turning point: rs = ", xgrid[np.argmin(yfit) + 1])
    # println("Effective mass ratio at turning point: ", np.min(yfit))
    return handle
end

# Taylor series for m* / m in the high-density limit to leading order in rs in 3D
# (c.f. Giuliani & Vignale, Quantum Theory of the Electron Liquid, 2008, p. 500)
function high_density_limit(rs)
    alpha = (4.0 / (9.0 * np.pi))^(1.0 / 3.0)
    return 1 + alpha * rs * np.log(rs) / (2.0 * np.pi)
end

function main()
    # UEG parameters
    # rslist = [0.01; 0.1; 0.5; collect(range(1, 10; step=0.5))]
    rslist = collect(range(1, 10; step=0.5))
    # rslist = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    beta = 40.0
    dim = 3
    constant_fs = true
    # constant_fs = false

    # plot_landaufunc(beta, collect(sort!(unique!([1; LinRange(0.01, 5, 2001)]))); dir="")
    # plot_landaufunc(beta, [1]; dir="")
    # return

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

    # Load LQSGW and one-shot data from jld2 files
    mefflist_os_g0w0 = []
    mefflist_os_fp = []
    mefflist_os_fp_fm = []
    zlist_os_g0w0 = []
    zlist_os_fp = []
    zlist_os_fp_fm = []
    mefflist_g0w0 = []
    mefflist_fp = []
    mefflist_fp_fm = []
    zlist_g0w0 = []
    zlist_fp = []
    zlist_fp_fm = []
    for rs in rslist
        param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
        @unpack kF, EF, β = param
        # Load one-shot data
        meff_os_g0w0, z_os_g0w0 = get_oneshot_data_new_format(param, :rpa)
        meff_os_fp, z_os_fp = get_oneshot_data_new_format(param, int_type_fp)
        meff_os_fp_fm, z_os_fp_fm = get_oneshot_data_new_format(param, int_type_fp_fm)
        push!(mefflist_os_g0w0, meff_os_g0w0)
        push!(mefflist_os_fp, meff_os_fp)
        push!(mefflist_os_fp_fm, meff_os_fp_fm)
        push!(zlist_os_g0w0, z_os_g0w0)
        push!(zlist_os_fp, z_os_fp)
        push!(zlist_os_fp_fm, z_os_fp_fm)
        # Load LQSGW data
        meff_g0w0, z_g0w0 = load_lqsgw_data_new_format(param, :rpa)
        meff_fp, z_fp = load_lqsgw_data_new_format(param, int_type_fp)
        meff_fp_fm, z_fp_fm = load_lqsgw_data_new_format(param, int_type_fp_fm)
        push!(mefflist_g0w0, meff_g0w0)
        push!(mefflist_fp, meff_fp)
        push!(mefflist_fp_fm, meff_fp_fm)
        push!(zlist_g0w0, z_g0w0)
        push!(zlist_fp, z_fp)
        push!(zlist_fp_fm, z_fp_fm)
    end
    pushfirst!(rslist, 0.0)
    pushfirst!(mefflist_os_g0w0, 1.0)
    pushfirst!(mefflist_os_fp, 1.0)
    pushfirst!(mefflist_os_fp_fm, 1.0)
    pushfirst!(mefflist_g0w0, 1.0)
    pushfirst!(mefflist_fp, 1.0)
    pushfirst!(mefflist_fp_fm, 1.0)
    pushfirst!(zlist_os_g0w0, 1.0)
    pushfirst!(zlist_os_fp, 1.0)
    pushfirst!(zlist_os_fp_fm, 1.0)
    pushfirst!(zlist_g0w0, 1.0)
    pushfirst!(zlist_fp, 1.0)
    pushfirst!(zlist_fp_fm, 1.0)

    rs_FlapwMBPT = [1, 2, 3, 4, 5]
    m_FlapwMBPT =
        [1.0072289156626506, 1.0188253012048194, 1.0335843373493976, 1.0562500000000001]
    z_FlapwMBPT = [
        0.8638571947489384,
        0.7707667812974212,
        0.6965831435079731,
        0.6381168758423326,
        0.5894455967871706,
    ]

    rs_VDMC = [0, 0.5, 1, 2, 3, 4, 5, 6]
    # z_VDMC = [1.0, 0.95358, 0.94705, 0.93015, 0.90854, 0.92296, 0.91668, 0.89664]
    # z_VDMC_err = [0, 0.00365, 0.00858, 0.01441, 0.01951, 0.01810, 0.01984, 0.02427]
    m_VDMC = [1.0, 0.95893, 0.94947, 0.95206, 0.96035, 0.9706, 0.97885, 0.98626]
    m_VDMC_err = [0, 0.00037, 0.00019, 0.00084, 0.00016, 0.0014, 0.00036, 0.00229]

    rs_VMC = [0, 1, 2, 4, 5, 10]
    m_SJVMC = [1.0, 0.96, 0.94, 0.94, 1.02, 1.13]
    m_SJVMC_err = [0, 0.01, 0.02, 0.02, 0.02, 0.03]
    z_SJVMC = [1.0, 0.894, 0.82, 0.69, 0.61, 0.45]
    z_SJVMC_err = [0, 0.009, 0.01, 0.01, 0.01, 0.01]

    rs_DMC = [0, 1, 2, 3, 4, 5]
    m_DMC = [1.0, 0.918, 0.879, 0.856, 0.842, 0.791]
    m_DMC_err = [0, 0.006, 0.014, 0.014, 0.017, 0.01]

    # # Old VDMC results to N=5 with no infinite-order error estimation
    # m_VDMC = [1.0, 0.95893, 0.9514, 0.9516, 0.9597, 0.9692, 0.9771, 0.9842]
    # m_VDMC_err = [0, 0.00067, 0.0016, 0.0018, 0.0016, 0.0026, 0.0028, 0.0029]
    # rs_VDMC = [0, 0.5, 1, 2, 3, 4, 5, 6]

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

    # f_tree_level_G0W0 = np.load("data/3d/rpa/meff_3d_tree_level_G0W0.npz")
    # rs_tree_level_G0W0 = f_tree_level_G0W0.get("rslist")
    # m_tree_level_G0W0 = f_tree_level_G0W0.get("mefflist")

    # f_tree_level_G0Wp = np.load("data/3d/$(int_type_fp)/meff_3d_tree_level_G0Wp.npz")
    # rs_tree_level_G0Wp = f_tree_level_G0Wp.get("rslist")
    # m_tree_level_G0Wp = f_tree_level_G0Wp.get("mefflist")

    # High-density limit
    rs_HDL_plot = np.linspace(1e-5, 0.35; num=101)
    meff_HDL_plot = np.array([high_density_limit(rs) for rs in rs_HDL_plot])

    # Use exact expression in the high-density limit for RPA(+FL) and RPT fits
    cutoff_HDL = 0.1
    rs_HDL = rs_HDL_plot[rs_HDL_plot .≤ cutoff_HDL]
    meff_HDL = meff_HDL_plot[rs_HDL_plot .≤ cutoff_HDL]
    # println(rs_HDL)
    # println(meff_HDL)

    # Plot m*/m convergence
    fig, ax = plt.subplots(; figsize=(5, 5))

    # VDMC results from this work
    errorbar_mvsrs(
        rs_VDMC[1:(end - 1)],
        m_VDMC[1:(end - 1)],
        m_VDMC_err[1:(end - 1)],
        8,
        "VDMC",
        ax;
        zorder=0,
    )
    plot_mvsrs(
        rs_VDMC[1:(end - 1)],
        m_VDMC[1:(end - 1)],
        8,
        "",
        ax;
        ls="-",
        rs_HDL=rs_HDL,
        meff_HDL=meff_HDL,
        # zorder=0,
    )

    # Digitized data from Kutepov 3DUEG FlapwMBPT paper
    ax.scatter(
        rs_FlapwMBPT[2:end],
        m_FlapwMBPT,
        15;
        marker="s",
        label="FlapwMBPT",
        # label="FlapwMBPT$(reflabels[1])",
        color=colors[1],
        zorder=100,
    )

    indices = [2, 3, 4]
    meff_oneshot = [mefflist_os_g0w0, mefflist_os_fp, mefflist_os_fp_fm]
    labels = [
        "\$G_0 W_0\$",
        # "\$G_0 \\mathcal{W}^+_{0}\$",
        # "\$G_0 \\mathcal{W}_0\$",
        "\$G_0 W^\\text{KO}_{0,+}\$",
        "\$G_0 W^\\text{KO}_0\$",
        # "\$G_0 W^+_\\text{KO}\$$(reflabels[2])",
        # "\$G_0 W_\\text{KO}\$$(reflabels[2])",
    ]
    for (mefflist, label, idx) in zip(meff_oneshot, labels, indices)
        print("\nPlotting ", label)
        handle = plot_mvsrs(
            rslist,
            mefflist,
            idx,
            label,
            ax;
            ls="--",
            rs_HDL=rs_HDL,
            meff_HDL=meff_HDL,
        )
        # l1_handles.append(handle)
    end

    plot_mvsrs(
        rslist,
        mefflist_g0w0,
        5,
        "LQSGW",
        ax;
        ls="--",
        rs_HDL=rs_HDL,
        meff_HDL=meff_HDL,
        zorder=10,
    )
    # ax.scatter(rslist, mefflist_g0w0, 20; color=colors[5], zorder=10, facecolors="none")

    plot_mvsrs(
        rslist,
        mefflist_fp,
        6,
        # "LQSG\$\\mathcal{W}^+\$",
        "LQSGW\$^\\text{KO}_+\$",
        ax;
        ls="--",
        rs_HDL=rs_HDL,
        meff_HDL=meff_HDL,
        zorder=20,
    )
    # ax.scatter(rslist, mefflist_fp, 20; color=colors[6], zorder=20, facecolors="none")

    plot_mvsrs(
        rslist,
        mefflist_fp_fm,
        7,
        # "LQSG\$\\mathcal{W}\$",
        "LQSGW\$^\\text{KO}\$",
        ax;
        ls="--",
        rs_HDL=rs_HDL,
        meff_HDL=meff_HDL,
        zorder=30,
    )
    # ax.scatter(rslist, mefflist_fp_fm, 20; color=colors[7], zorder=30, facecolors="none")
    # ax.plot(rslist, mefflist_g0w0, "o-"; label="LQSGW", color=color[2])
    # ax.plot(rslist, mefflist_fp, "o-"; label="\$G_0 W^+_\\text{KO}\$", color=color[4])
    # ax.plot(rslist, mefflist_fp_fm, "o-"; label="\$G_0 W_\\text{KO}\$", color=color[5])

    if constant_fs
        ax.set_title(
            "Constant \$F^\\pm(q)\$";
            # "\$F^+(q) \\approx 1 - {\\kappa_0}/{\\kappa},\\; F^-(q) \\approx 1 - {\\chi_0}/{\\chi}\$";
            pad=10,
            fontsize=16,
        )
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ax.set_ylim(0.94, 1.45)
    ax.set_xticks(0:2:10)
    ax.set_xlabel("\$r_s\$")
    ax.set_ylabel("\$m^* / m\$")
    ax.legend(; fontsize=12, loc="upper left")
    plt.tight_layout()
    fig.savefig("meff_lqsgw_vs_rs_$(fsstr).pdf")

    # Plot Z convergence
    fig, ax = plt.subplots(; figsize=(5, 5))

    # # VDMC results from this work (Z does not converge!)
    # errorbar_mvsrs(rs_VDMC, z_VDMC, z_VDMC_err, 8, "VDMC", ax; zorder=1000)
    # plot_mvsrs(
    #     rs_VDMC,
    #     z_VDMC,
    #     8,
    #     "",
    #     ax;
    #     ls="-",
    #     zorder=1000,
    # )

    # VMC results
    errorbar_mvsrs(rs_VMC, z_SJVMC, z_SJVMC_err, 8, "VMC", ax; zorder=1000)
    # plot_mvsrs(rs_VMC, z_SJVMC, 8, "", ax; ls="-", zorder=1000)

    # Digitized data from Kutepov 3DUEG LQSGW paper
    ax.scatter(
        rs_FlapwMBPT,
        z_FlapwMBPT,
        15;
        marker="s",
        label="FlapwMBPT",
        color=colors[1],
        zorder=100,
    )

    indices = [2, 3, 4]
    z_oneshot = [zlist_os_g0w0, zlist_os_fp, zlist_os_fp_fm]
    labels = [
        "\$G_0 W_0\$",
        # "\$G_0 \\mathcal{W}^+_{0}\$",
        # "\$G_0 \\mathcal{W}_0\$",
        "\$G_0 W^\\text{KO}_{0,+}\$",
        "\$G_0 W^\\text{KO}_0\$",
        # "\$G_0 W^+_\\text{KO}\$$(reflabels[2])",
        # "\$G_0 W_\\text{KO}\$$(reflabels[2])",
    ]
    for (zlist, label, idx) in zip(z_oneshot, labels, indices)
        print("\nPlotting ", label)
        handle = plot_mvsrs(rslist, zlist, idx, label, ax; ls="--")
        # l1_handles.append(handle)
    end

    plot_mvsrs(rslist, zlist_g0w0, 5, "LQSGW", ax; ls="--", zorder=10)
    # ax.scatter(rslist, zlist_g0w0, 20; color=colors[5], zorder=10, facecolors="none")

    plot_mvsrs(rslist, zlist_fp, 6, "LQSGW\$^\\text{KO}_+\$", ax; ls="--", zorder=20)
    # ax.scatter(rslist, zlist_fp, 20; color=colors[6], zorder=20, facecolors="none")

    plot_mvsrs(
        rslist,
        zlist_fp_fm,
        7,
        "LQSGW\$^\\text{KO}\$",
        ax;
        ls="--",
        # rs_HDL=rs_HDL,
        # meff_HDL=meff_HDL,
        zorder=30,
    )
    # ax.scatter(rslist, zlist_fp_fm, 20; color=colors[7], zorder=30, facecolors="none")
    # ax.plot(rslist, zlist_g0w0, "o-"; label="LQSGW", color=color[2])
    # ax.plot(rslist, zlist_fp, "o-"; label="\$G_0 W^+_\\text{KO}\$", color=color[4])

    if constant_fs# ax.plot(rslist, zlist_fp_fm, "o-"; label="\$G_0 W_\\text{KO}\$", color=color[5])
        ax.set_title(
            "Constant \$F^\\pm(q)\$";
            # "\$F^+(q) \\approx 1 - {\\kappa_0}/{\\kappa},\\; F^-(q) \\approx 1 - {\\chi_0}/{\\chi}\$";
            pad=10,
            fontsize=16,
        )
    else
        ax.set_title("Momentum-resolved \$F^\\pm(q)\$"; pad=10, fontsize=16)
    end
    ax.set_ylim(0.44, 1.04)
    ax.set_xticks(0:2:10)
    ax.set_xlabel("\$r_s\$")
    ax.set_ylabel("\$Z_F\$")
    ax.legend(; loc="upper right", fontsize=12)
    plt.tight_layout()
    fig.savefig("zfactor_lqsgw_vs_rs_$(fsstr).pdf")

    return
end

main()
