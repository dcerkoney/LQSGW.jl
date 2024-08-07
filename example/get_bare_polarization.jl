using BenchmarkTools
using CompositeGrids
using ElectronGas
using GreenFunc
using Lehmann
using LQSGW
using MPI
using Parameters
using PyPlot
using PyCall

import LQSGW: print_root, println_root, timed_result_to_string

@pyimport numpy as np

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    root = 0
    rank = MPI.Comm_rank(comm)

    # Plot options
    make_plots = true

    # UEG parameters
    beta = 40.0
    rs = 1.0
    dim = 3
    param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
    @unpack β, kF, EF = param

    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14

    # CompositeGrid parameters
    # Nk, order = 20, 18
    # Nk, order = 14, 10
    Nk, order = 12, 8
    # Nk, order = 10, 7
    # Nk, order = 8, 6

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

    Π0_qw_data = zeros(ComplexF64, length(bdlr.n), length(qPgrid.grid))
    for (qi, q) in enumerate(qPgrid)
        Π0_qw_data[:, qi] = Polarization.Polarization0_FiniteTemp(
            q,
            bdlr.n,
            param;
            maxk=maxKP / kF,
            scaleN=40,
            gaussN=20,
        )
    end
    return

    # Get UEG G0; a large kgrid is required for the self-consistency loop
    @time G0 = SelfEnergy.G0wrapped(Euv, rtol, kGgrid, param)
    G0_dlr = to_dlr(G0)
    G0_ins = dlr_to_imtime(G0_dlr, [β]) * (-1)

    # Verify that the non-interacting density is correct
    if rank == root
        println("Minimum k in G grid: $(minimum(kGgrid.grid))")
        G0_dlr = to_dlr(G0)
        G0_ins = dlr_to_imtime(G0_dlr, [β]) * (-1)
        integrand = real(G0_ins[1, :]) .* kGgrid.grid .* kGgrid.grid
        densityS = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKS]) / π^2
        densityP = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKP]) / π^2
        densityG = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKG]) / π^2
        println("Density from Σ mesh (β = $beta): $(densityS)")
        println("Density from Π mesh (β = $beta): $(densityP)")
        println("Density from G mesh (β = $beta): $(densityG)")
        println("Exact non-interacting density (T = 0): $(param.n)")
        if beta == 1000.0
            # The large discrepancy in density is due to finite-T effects
            # @assert isapprox(param.n, densityG, rtol=3e-5)  # :gauss
            @assert isapprox(param.n, densityG, rtol=3e-3)  # :cheb
        end
        println("G0(k0, 0⁻) = $(G0_ins[1, 1])\n")
    end

    # Initial instant/dynamic self-energies are zero
    Σ0 = zero(GreenFunc.MeshArray(G0.mesh[1], kSgrid; dtype=ComplexF64))
    Σ0_ins = zero(GreenFunc.MeshArray(G0_ins.mesh[1], kSgrid; dtype=ComplexF64))

    # Get gridded quasiparticle energy for Π_qp solver
    print_root("Computing E_qp[Σ_G0W0]...")
    timed_res = @timed LQSGW.E_qp_grid(param, Σ0, Σ0_ins, kGgrid)
    E_qp_kGgrid = timed_res.value
    println_root(timed_result_to_string(timed_res))

    # Calculate the dynamic polarization
    println_root("Computing quasiparticle polarizaion bubble Π(q, iω)...")
    timed_res =
        @timed LQSGW.Π_qp(param, E_qp_kGgrid, kGgrid, Nk, maxKP, minK, order, qPgrid, bdlr)
    Π_qw = timed_res.value
    Π_qt = to_imtime(to_dlr(Π_qw))
    println_root(timed_result_to_string(timed_res))

    # Print the dynamic polarization at q ≈ 0
    Π_qp_0 = real.(Π_qw)[1]
    println_root("Π(q ≈ 0, iω = 0) = $Π_qp_0")

    if rank == root
        # Compare with exactly computed Π0
        ngrid = bdlr.n
        τgrid = bdlr.τ
        Π0_qw_data = zeros(ComplexF64, length(ngrid), length(qPgrid.grid))
        for (qi, q) in enumerate(qPgrid)
            Π0_qw_data[:, qi] =
                Polarization.Polarization0_FiniteTemp(q, ngrid, param; maxk=maxKP / kF)
            # Π0_qw_data[:, qi] = Polarization.Polarization0_FiniteTemp(
            #     q,
            #     ngrid,
            #     param;
            #     maxk=maxKP,
            #     scaleN=50,
            #     gaussN=25,
            # )
        end
        # Build Π0(iν, q) mesh array
        Π0_qw =
            GreenFunc.MeshArray(ImFreq(bdlr), qPgrid; dtype=Float64, data=real(Π0_qw_data))
        Π0_qt = to_imtime(to_dlr(Π0_qw))

        np.savez(
            "pi_qp_cgrids_minK=$(round(minK / kF; sigdigits=4))kF_maxK=$(Int(round(maxK / kF; sigdigits=4)))kF_Nk=$(Nk)_order=$(order)_Euv=$(Int(round(Euv / EF)))EF_rtol=$(rtol).npz";
            Π_qt=Π_qt,
            Π_qw=Π_qw,
            Π0_qt=Π0_qt,
            Π0_qw=Π0_qw,
            kGgrid=kGgrid,
            qPgrid=qPgrid,
            kSgrid=kSgrid,
            τgrid=τgrid,
            ngrid=ngrid,
            param=string(param),
        )

        if make_plots
            fig, ax = plt.subplots()
            for (ni, n) in enumerate(bdlr.n)
                ax.plot(qPgrid / kF, real.(Π_qw)[ni, :]; color="C$ni")
                ax.scatter(qPgrid / kF, real.(Π0_qw)[ni, :]; s=6, color="C$ni")
            end
            ax.set_xlabel("\$q / k_F\$")
            ax.set_ylabel("\$\\Pi_0(q, i\\omega_m \\ne 0)\$")
            plt.tight_layout()
            fig.savefig(
                "pi0_q_minK=$(round(minK / kF; sigdigits=4))kF_maxK=$(round(maxK / kF; sigdigits=4))kF_Nk=$(Nk)_order=$(order)_Euv=$(Int(round(bdlr.Euv / EF)))EF.pdf",
            )

            fig, ax = plt.subplots()
            for (ni, n) in enumerate(bdlr.n)
                ax.plot(
                    qPgrid / kF,
                    abs.(real.(Π_qw) - real.(Π0_qw))[ni, :],
                    "o-";
                    color="C$ni",
                    markersize=4,
                )
            end
            # ax.set_xlim(0, maxKP / kF)
            ax.set_xlim(0, 4)
            ax.set_xlabel("\$q / k_F\$")
            ax.set_ylabel("\$\\text{Abs. err. }\\Pi_0(q, i\\omega_m = 0)\$")
            plt.tight_layout()
            fig.savefig(
                "error_pi0_q_minK=$(round(minK / kF; sigdigits=4))kF_maxK=$(round(maxKP / kF; sigdigits=4))kF_Nk=$(Nk)_order=$(order)_Euv=$(Int(round(Euv / EF)))EF.pdf",
            )
            ax.set_xlim(0, 0.0001)
            mask = qPgrid / kF .≤ 0.0001
            max_yval = 0
            for ni in eachindex(ngrid)
                max_yval_n = maximum(abs.(real.(Π_qw) - real.(Π0_qw))[ni, :][mask])
                max_yval = max(max_yval, max_yval_n)
            end
            ax.set_ylim(0.0, 1.1 * max_yval)
            plt.tight_layout()
            fig.savefig(
                "error_pi0_small_q_minK=$(round(minK / kF; sigdigits=4))kF_maxK=$(round(maxKP / kF; sigdigits=4))kF_Nk=$(Nk)_order=$(order)_Euv=$(Int(round(Euv / EF)))EF.pdf",
            )
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
