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
    rs = 5.0
    dim = 3
    param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
    @unpack kF, EF, β = param

    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14

    # CompositeGrid parameters
    Nk, order = 14, 10
    # Nk, order = 12, 8
    # Nk, order = 10, 7

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    # Test LQSGW parameters
    int_type = :rpa
    num_steps = 30  # number of iterations before convergence / maximum step number reached
    atol = 1e-2
    alpha = 0.5
    δK = 5e-6
    Fs = 0.0
    Fa = 0.0

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
        i,
        key;
        savedir="$(LQSGW.DATA_DIR)/$(dim)d/$(int_type)",
        savename="lqsgw_$(dim)d_$(int_type)_rs=$(round(rs; sigdigits=4))_beta=$(beta)",
    )
        filename = joinpath(savedir, savename * "_i=$(i).jld2")
        d = load(filename)
        return d[key]
    end

    colors = reverse(hex.((sequential_palette(0, num_steps; s=100))))

    # Plot Σ_ins convergence
    fig, ax = plt.subplots()
    for i in 0:num_steps
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == num_steps
            lw = 1
            label = "\$N_\\text{iter} = $num_steps\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors[i])"
        end
        # kgrid = sigma_ins.mesh[2]
        sigma_ins = loaddata(i, "Σ_ins")
        ax.plot(kSgrid / kF, real(sigma_ins[1, :]), fmt; color=color, lw=lw, label=label)
    end
    ax.set_xlim(0, 2)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\Sigma^{(i)}_{x}(k)\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_sigma_ins.pdf")

    # Plot Π convergence
    fig, ax = plt.subplots()
    for i in 0:num_steps
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == num_steps
            lw = 1
            label = "\$N_\\text{iter} = $num_steps\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors[i])"
        end
        pi_q_static = loaddata(i, "Π")[1, :]
        ax.plot(qPgrid / kF, real.(pi_q_static), fmt; color=color, lw=lw, label=label)
    end
    ax.set_xlim(0, 4)
    ax.set_xlabel("\$q / k_F\$")
    ax.set_ylabel("\$\\Pi^{(i)}(q, i\\nu_m = 0)\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_pi_static.pdf")

    # Plot Z factor convergence
    fig, ax = plt.subplots()
    for i in 0:num_steps
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == num_steps
            lw = 1
            label = "\$N_\\text{iter} = $num_steps\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors[i])"
        end
        zfactor_kSgrid = loaddata(i, "Z")
        ax.plot(kSgrid / kF, zfactor_kSgrid, fmt; color=color, lw=lw, label=label)
    end
    ax.set_xlim(0, 4)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$Z^{(i)}_k\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_zfactor.pdf")

    # Plot E_qp convergence
    fig, ax = plt.subplots()
    for i in 0:num_steps
        fmt = i == 0 ? "--" : "-"
        if i == 0
            lw = 1
            label = nothing
            color = "black"
        elseif i == num_steps
            lw = 1
            label = "\$N_\\text{iter} = $num_steps\$"
            color = "limegreen"
        else
            lw = 1
            label = nothing
            color = "#$(colors[i])"
        end
        E_qp_kSgrid = loaddata(i, "E_qp")
        ax.plot(kSgrid / kF, E_qp_kSgrid, fmt; color=color, lw=lw, label=label)
    end
    ax.set_xlim(0, 2)
    ax.set_ylim(-4.5, 11)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\mathcal{E}^{(i)}_{\\text{qp}}(k)\$")
    ax.legend(; fontsize=12)
    plt.tight_layout()
    fig.savefig("convergence_quasiparticle_energy.pdf")

    return
end

main()
