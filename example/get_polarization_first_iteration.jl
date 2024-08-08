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
    @unpack kF, EF = param

    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14

    # CompositeGrid parameters
    # Nk, order = 20, 18
    # Nk, order = 14, 10
    # Nk, order = 12, 8
    Nk, order = 10, 7
    # Nk, order = 8, 6

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    int_type = :rpa
    Fs = 0.0
    Fa = 0.0

    # Make sure we are using parameters for the bare UEG theory
    @assert param.Λs == param.Λa == 0.0
    @unpack beta, β, kF = param

    # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
    rs = round(param.rs; sigdigits=13)
    if int_type in [:ko_const_p, :ko_const_pm]
        _int_type = :ko_const
        if verbose && int_type == :ko_const_pm
            println("Fermi liquid parameters at rs = $(rs): Fs = $Fs, Fa = $Fa")
        end
    else
        _int_type = int_type
    end

    # Momentum grid maxima for G, Π, and Σ
    maxKG = 4.3 * maxK
    maxKP = 2.1 * maxK
    maxKS = maxK
    minKS = minK

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

    # Get UEG G0; a large kgrid is required for the self-consistency loop
    @time G0 = SelfEnergy.G0wrapped(Euv, rtol, kGgrid, param)

    # Verify that the non-interacting density is correct
    if rank == root
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

    # Helper function to get the GW self-energy Σ[G, W](iωₙ, k) for a given int_type (W)
    # NOTE: A large momentum grid is required for G and Σ at intermediate steps
    function Σ_GW(G, Π)
        Σ_imtime, Σ_ins, _ = GW(
            param,
            G,
            Π,
            kSgrid;
            Euv=Euv,
            rtol=rtol,
            Nk=Nk,
            maxK=maxKS,
            minK=minKS,
            order=order,
            int_type=_int_type,
            Fs=Fs,
            Fa=Fa,
        )
        Σ = to_imfreq(to_dlr(Σ_imtime))  # Σ(τ, k) → Σ(iωₙ, k)
        return Σ, Σ_ins
    end

    # Use exactly computed Π0 as starting point
    println_root("Computing initial Π0...")
    bdlr = DLRGrid(Euv, β, rtol, false, :ph)
    Π0_qw = GreenFunc.MeshArray(ImFreq(bdlr), qPgrid; dtype=ComplexF64)
    for (qi, q) in enumerate(qPgrid)
        Π0_qw[:, qi] = Polarization.Polarization0_FiniteTemp(
            q,
            bdlr.n,
            param;
            maxk=maxKP / kF,
            scaleN=50,
            gaussN=25,
        )
    end
    Π0_qt = to_imtime(to_dlr(Π0_qw))

    # Use exact Π0 for initial G0W0 self-energy: Σ[G0, Π0](k, τ)
    println_root("Computing initial Σ_G0W0...")
    timed_res = @timed Σ_GW(G0, Π0_qw)
    Σ, Σ_ins = timed_res.value
    println_root(timed_result_to_string(timed_res))

    # Get gridded quasiparticle energy for Π_qp solver
    println_root("Computing E_qp[Σ_G0W0]...")
    timed_res = @timed LQSGW.E_qp_grid(param, Σ, Σ_ins, kGgrid)
    E_qp_kGgrid = timed_res.value
    println_root(timed_result_to_string(timed_res))

    # Get Π_mix = Π_qp[G_mix]
    println_root("Computing Π[E_qp]...")
    timed_res =
        @timed Π_qp(param, E_qp_kGgrid, kGgrid, Nk, maxKP, minK, order, qPgrid, bdlr)
    Π_qw = timed_res.value
    Π_qt = to_imtime(to_dlr(Π_qw))
    println_root(timed_result_to_string(timed_res))

    if rank == root
        # Print the q ≈ 0 values of the Matsubara and imaginary-time polarizations
        Π_qp_dynamic_w0 = real.(Π_qw)[1]
        Π_qp_dynamic_t0 = real.(Π_qt)[1]
        println_root("Π(q ≈ 0, iω = 0) = $Π_qp_dynamic_w0")
        println_root("Π(q ≈ 0, τ = 0) = $Π_qp_dynamic_t0")

        if make_plots
            fig, ax = plt.subplots()
            # τgrid_points = collect(1:length(τgrid)÷10:length(τgrid))
            # for (i, τi) in enumerate(τgrid_points)
            for (i, τ) in enumerate(τgrid)
                pifunc = q -> Interp.interp1D(real(Π_qt)[i, :], qPgrid, q)
                # qs = LinRange(0, 6 * kF, 5000)
                qs = LinRange(0, maxKP, 5000)
                polns = pifunc.(qs)
                # ax.plot(qs / kF, polns, color="C$(i-1)", zorder=10 * i)
                # ax.plot(qs / kF, polns, label="τ / β = $(round(τgrid[τi] / β; sigdigits=4))", color="C$(i-1)")
                ax.plot(qPgrid / kF, real(Π_qt)[i, :]; color="C$(i-1)", zorder=10 * i)
                ax.scatter(
                    qPgrid / kF,
                    real(Π0_qt)[i, :];
                    color="C$(i-1)",
                    s=6,
                    zorder=10 * i + 5,
                )
            end
            # ax.set_xlim(0, 6)
            # ax.set_xlim(0, 10)
            ax.set_xlim(0, maxKP / kF)
            # ax.set_ylim(-0.055, 0.005)
            ax.set_xlabel("\$q / k_F\$")
            ax.set_ylabel("\$\\Pi_0(q, \\tau)\$")
            plt.tight_layout()
            # ax.legend()
            fig.savefig(
                "pi0_qt_minK=$(round(minK / kF; sigdigits=4))kF_maxK=$(Int(round(maxK / kF; sigdigits=4)))kF_Nk=$(Nk)_order=$(order)_Euv=$(Int(round(Euv / EF)))EF.pdf",
            )

            fig, ax = plt.subplots()
            # ngrid_points = collect(1:length(ngrid)÷10:length(ngrid))
            # for (i, ni) in enumerate(ngrid_points)
            for (i, τ) in enumerate(τgrid)
                ax.plot(
                    qPgrid / kF,
                    abs.(real(Π_qt) - real(Π0_qt))[i, :],
                    "o-";
                    color="C$(i-1)",
                    zorder=10 * i,
                    markersize=2,
                )
            end
            # ax.set_xlim(0, 6)
            # ax.set_xlim(0, 10)
            ax.set_xlim(0, maxKP / kF)
            # ax.set_ylim(-0.055, 0.005)
            ax.set_xlabel("\$q / k_F\$")
            ax.set_ylabel("\$|\\Delta\\Pi_0(q, \\tau)|\$")
            plt.tight_layout()
            # ax.legend()
            fig.savefig(
                "error_pi0_qt_minK=$(round(minK / kF; sigdigits=4))kF_maxK=$(Int(round(maxK / kF; sigdigits=4)))kF_Nk=$(Nk)_order=$(order)_Euv=$(Int(round(Euv / EF)))EF.pdf",
            )

            fig, ax = plt.subplots()
            # ngrid_points = collect(1:length(ngrid)÷10:length(ngrid))
            # for (i, ni) in enumerate(ngrid_points)
            for (i, n) in enumerate(ngrid)
                pifunc = q -> Interp.interp1D(real(Π_qw)[i, :], qPgrid, q)
                # qs = LinRange(0, 6 * kF, 5000)
                qs = LinRange(0, maxKP, 5000)
                polns = pifunc.(qs)
                # ax.plot(qs / kF, polns, color="C$(i-1)", zorder=10 * i)
                # ax.plot(qs / kF, polns, label="m = $n", color="C$(i-1)")
                # ax.plot(qPgrid / kF, real(Π_qw)[ni, :], color="C$(i-1)")
                # ax.plot(qPgrid / kF, real(Π_qw)[i, :], "o-", label="m = $ni", color="C$(i-1)")
                # ax.scatter(qPgrid / kF, Polarization.Polarization0_FiniteTemp.(qPgrid, [n], [param]), color="C$(i-1)", s=10)
                ax.plot(
                    qPgrid / kF,
                    real(Π_qw)[i, :];
                    color="C$(i-1)",
                    zorder=10 * i,
                    markersize=2,
                )
                ax.scatter(
                    qPgrid / kF,
                    real(Π0_qw)[i, :];
                    color="C$(i-1)",
                    s=6,
                    zorder=10 * i + 5,
                )
            end
            # ax.set_xlim(0, 6)
            # ax.set_xlim(0, 10)
            ax.set_xlim(0, maxKP / kF)
            # ax.set_ylim(-0.055, 0.005)
            ax.set_xlabel("\$q / k_F\$")
            ax.set_ylabel("\$\\Pi_0(q, i\\nu_m)\$")
            plt.tight_layout()
            # ax.legend()
            fig.savefig(
                "pi0_qw_minK=$(round(minK / kF; sigdigits=4))kF_maxK=$(Int(round(maxK / kF; sigdigits=4)))kF_Nk=$(Nk)_order=$(order)_Euv=$(Int(round(Euv / EF)))EF.pdf",
            )

            fig, ax = plt.subplots()
            # ngrid_points = collect(1:length(ngrid)÷10:length(ngrid))
            # for (i, ni) in enumerate(ngrid_points)
            for (i, n) in enumerate(ngrid)
                # @assert ngrid[1] == 0
                # for (i, n) in enumerate([0])
                ax.plot(
                    qPgrid / kF,
                    abs.(real(Π_qw) - real(Π0_qw))[i, :],
                    "o-";
                    color="C$(i-1)",
                    zorder=10 * i,
                    markersize=2,
                )
            end
            ax.set_xlim(0, 2)
            # ax.set_xlim(0, 6)
            # ax.set_xlim(0, 10)
            # ax.set_xlim(0, maxKP / kF)
            # ax.set_ylim(-0.055, 0.005)
            ax.set_xlabel("\$q / k_F\$")
            ax.set_ylabel("\$|\\Delta\\Pi_0(q, i\\nu_m=0)|\$")
            plt.tight_layout()
            # ax.legend()
            fig.savefig(
                "error_pi0_qw_minK=$(round(minK / kF; sigdigits=4))kF_maxK=$(Int(round(maxK / kF; sigdigits=4)))kF_Nk=$(Nk)_order=$(order)_Euv=$(Int(round(Euv / EF)))EF.pdf",
            )
        end

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
    end
    MPI.Finalize()
    return
end

main()
