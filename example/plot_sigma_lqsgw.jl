using CompositeGrids
using ElectronGas
using JLD2
using Lehmann
using LQSGW
using Parameters
using PyPlot

# using PyCall
# @pyimport numpy as np

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
    # Nk, order = 12, 8
    Nk, order = 10, 7

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    # Test LQSGW parameters
    int_type = :rpa
    max_steps = 3
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
    # multiplier = 1
    # multiplier = 2
    multiplier = 4
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

    # Plot Σ_ins convergence
    fig, ax = plt.subplots()
    for i in 0:max_steps
        fmt = i == 0 ? "--" : "-"
        color = i == 0 ? "black" : "C$(i - 1)"
        # kgrid = sigma_ins.mesh[2]
        sigma_ins = loaddata(i, "Σ_ins")
        ax.plot(kSgrid / kF, real(sigma_ins[1, :]), fmt; label="\$i = $i\$", color=color)
    end
    ax.set_xlim(0, 2)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\Sigma^{(i)}_{x}(k)\$")
    ax.legend()
    plt.tight_layout()
    fig.savefig("convergence_sigma_ins.pdf")

    # Plot Π convergence
    fig, ax = plt.subplots()
    for i in 0:max_steps
        fmt = i == 0 ? "--" : "-"
        color = i == 0 ? "black" : "C$(i - 1)"
        pi_q_static = loaddata(i, "Π")[1, :]
        ax.plot(qPgrid / kF, pi_q_static, fmt; label="\$i = $i\$", color=color)
    end
    ax.set_xlim(0, 4)
    ax.set_xlabel("\$q / k_F\$")
    ax.set_ylabel("\$\\Pi^{(i)}(q, i\\nu_m = 0)\$")
    ax.legend()
    plt.tight_layout()
    fig.savefig("convergence_pi_static.pdf")

    # Plot Z factor convergence
    fig, ax = plt.subplots()
    for i in 0:max_steps
        fmt = i == 0 ? "--" : "-"
        color = i == 0 ? "black" : "C$(i - 1)"
        zfactor_kSgrid = loaddata(i, "Z")
        ax.plot(kSgrid / kF, zfactor_kSgrid, fmt; label="\$i = $i\$", color=color)
    end
    ax.set_xlim(0, 4)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$Z^{(i)}_k\$")
    ax.legend()
    plt.tight_layout()
    fig.savefig("convergence_zfactor.pdf")

    # Plot E_qp convergence
    fig, ax = plt.subplots()
    for i in 0:max_steps
        println("Loading data for i = $i")
        fmt = i == 0 ? "--" : "-"
        color = i == 0 ? "black" : "C$(i - 1)"
        kgrid = i == 0 ? kSgrid : kGgrid
        E_qp_kGgrid = loaddata(i, "E_qp")
        ax.plot(kgrid / kF, E_qp_kGgrid, fmt; label="\$i = $i\$", color=color)
    end
    ax.set_xlim(0, 2)
    ax.set_ylim(-4.5, 11)
    ax.set_xlabel("\$k / k_F\$")
    ax.set_ylabel("\$\\mathcal{E}^{(i)}_{\\text{qp}}(k)\$")
    ax.legend()
    plt.tight_layout()
    fig.savefig("convergence_quasiparticle_energy.pdf")

    return
end

main()
