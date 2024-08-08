using Colors
using CompositeGrids
using ElectronGas
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

function main()
    # UEG parameters
    beta = 40.0
    rs = 1.0
    dim = 3
    param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
    @unpack kF, EF, β = param

    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14

    # CompositeGrid parameters
    # Nk, order = 14, 10
    Nk, order = 12, 8
    # Nk, order = 10, 7

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    # Test LQSGW parameters
    max_steps = 150
    int_type = :rpa

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    # Momentum grid maxima for G, Π, and Σ
    maxKG = 4.3 * maxK
    maxKP = 2.1 * maxK
    maxKS = maxK

    # Make sure we do not exceed the DLR energy cutoff
    @assert maxKG^2 / (2 * param.me) < Euv "Max grid momentum exceeds DLR cutoff"

    # Bosonic DLR grid for the problem
    bdlr = DLRGrid(Euv, β, rtol, false, :ph)

    # Momentum grid maxima for G, Π, and Σ
    maxKG = 4.3 * maxK
    maxKP = 2.1 * maxK
    maxKS = maxK

    # Big grid for G
    multiplier = 1
    # multiplier = 2
    # multiplier = 4
    kGgrid = CompositeGrid.LogDensedGrid(
        :cheb,
        [0.0, maxKG],
        [0.0, kF],
        round(Int, multiplier * Nk),
        0.01 * minK,
        round(Int, multiplier * order),
    )

    # Medium grid for Π
    qPgrid =
        CompositeGrid.LogDensedGrid(:cheb, [0.0, maxKP], [0.0, 2 * kF], Nk, minK, order)

    # Small grid for Σ
    kSgrid = CompositeGrid.LogDensedGrid(:cheb, [0.0, maxKS], [0.0, kF], Nk, minK, order)

    function loaddata(
        key;
        savedir="$(LQSGW.DATA_DIR)/$(dim)d/$(int_type)",
        savename="lqsgw_$(dim)d_$(int_type)_rs=$(round(rs; sigdigits=4))_beta=$(beta).jld2",
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

    colors(n) = reverse(hex.((sequential_palette(0, n; s=100))))

    # Plot Σ_ins convergence
    fig, ax = plt.subplots()
    plotdata, n_max = loaddata("Σ_ins")
    for (idx, sigma_ins) in enumerate(plotdata)
        i = idx - 1
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == n_max
            lw = 1.5
            label = "\$N_\\text{iter} = $n_max\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors(n_max)[i])"
        end
        # kgrid = sigma_ins.mesh[2]
        ax.plot(kSgrid / kF, real(sigma_ins[1, :]), fmt; color=color, lw=lw, label=label)
    end
    ax.set_xlim(0, 2)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\Sigma^{(i)}_{x}(k)\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_sigma_ins_rs=$rs.pdf")

    # Plot Π convergence
    fig, ax = plt.subplots()
    plotdata, n_max = loaddata("Π")
    for (idx, Π) in enumerate(plotdata)
        i = idx - 1
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == n_max
            lw = 1.5
            label = "\$N_\\text{iter} = $n_max\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors(n_max)[i])"
        end
        pi_q_static = Π[1, :]
        ax.plot(qPgrid / kF, real.(pi_q_static), fmt; color=color, lw=lw, label=label)
    end
    ax.set_xlim(0, 4)
    ax.set_xlabel("\$q / k_F\$")
    ax.set_ylabel("\$\\Pi^{(i)}(q, i\\nu_m = 0)\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_pi_static_rs=$rs.pdf")

    # Plot dielectric function convergence
    fig, ax = plt.subplots()
    plotdata, n_max = loaddata("Π")
    for (idx, Π) in enumerate(plotdata)
        i = idx - 1
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == n_max
            lw = 1.5
            label = "\$N_\\text{iter} = $n_max\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors(n_max)[i])"
        end
        pi_q_static = Π[1, :]
        v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid]
        inverse_static_dielectric = 1.0 ./ (1.0 .- 2.0 * v_q .* real.(pi_q_static))
        ax.plot(
            qPgrid / kF,
            inverse_static_dielectric,
            fmt;
            color=color,
            lw=lw,
            label=label,
        )
        # ax.plot(qPgrid * param.rs, inverse_static_dielectric, fmt; color=color, lw=lw, label=label)
    end
    # ax.set_xlim(0, 5 * kF)
    # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    # ax.set_ylim(0, 1)
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(0, 3)
    ax.set_xlabel("\$q / k_F\$")
    # ax.set_xlim(0, 5)
    # ax.set_xlabel("\$q * r_s\$")
    ax.set_ylabel("\$1 / \\epsilon^{(i)}(q, i\\nu_m = 0)\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_inverse_dielectric_static_rs=$rs.pdf")

    # Plot Z factor convergence
    fig, ax = plt.subplots()
    plotdata, n_max = loaddata("Z_k")
    for (idx, zfactor_kSgrid) in enumerate(plotdata)
        i = idx - 1
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == n_max
            lw = 1.5
            label = "\$N_\\text{iter} = $n_max\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors(n_max)[i])"
        end
        ax.plot(kSgrid / kF, zfactor_kSgrid, fmt; color=color, lw=lw, label=label)
    end
    ax.set_xlim(0, 4)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$Z^{(i)}_k\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_zfactor_rs=$rs.pdf")

    # Plot E_qp convergence
    ylimits = Dict(
        0.01 => (-4.5, 11),
        0.5 => (-4.5, 11),
        1.0 => (-5, 14),
        2.0 => (-0.25, 0.5),
        3.0 => (-0.2, 0.55),
        4.0 => (-0.2, 0.55),
        5.0 => (-0.2, 0.55),
    )
    fig, ax = plt.subplots()
    plotdata, n_max = loaddata("E_k")
    for (idx, E_qp_kSgrid) in enumerate(plotdata)
        i = idx - 1
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == n_max
            lw = 1.5
            label = "\$N_\\text{iter} = $n_max\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors(n_max)[i])"
        end
        ax.plot(kSgrid / kF, E_qp_kSgrid, fmt; color=color, lw=lw, label=label)
    end
    ax.set_xlim(0, 2)
    ax.set_ylim(ylimits[rs])
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\mathcal{E}^{(i)}_{\\text{qp}}(k)\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_quasiparticle_energy_rs=$rs.pdf")

    return
end

main()
