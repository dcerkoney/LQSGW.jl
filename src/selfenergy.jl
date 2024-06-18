
function GW(
    param::Parameter.Para,
    G_prev::GreenFunc.MeshArray,
    Π_prev::GreenFunc.MeshArray,
    kgrid::Union{AbstractGrid,AbstractVector,Nothing}=nothing;
    Euv=1000 * param.EF,
    rtol=1e-14,
    Nk=12,
    maxK=6 * param.kF,
    minK=1e-8 * param.kF,
    order=8,
    int_type=:rpa,
    kwargs...,
)
    return GW(
        param,
        G_prev,
        Π_prev,
        Euv,
        rtol,
        Nk,
        maxK,
        minK,
        order,
        int_type,
        kgrid;
        kwargs...,
    )
end

function GW(
    param::Parameter.Para,
    G_prev::GreenFunc.MeshArray,
    Π_prev::GreenFunc.MeshArray,
    Euv,
    rtol,
    Nk,
    maxK,
    minK,
    order,
    int_type,
    kgrid::Union{AbstractGrid,AbstractVector,Nothing}=nothing;
    kwargs...,
)
    dim = param.dim
    if dim == 2
        error("2D case not yet implemented!")
    elseif dim == 3
        if isnothing(kgrid)
            kernel = DCKernel_qp(
                param,
                Π_prev;
                Euv=Euv,
                rtol=rtol,
                Nk=Nk,
                maxK=maxK,
                minK=minK,
                order=order,
                int_type=int_type,
                spin_state=:sigma,
                kwargs...,
            )
        else
            if (kgrid isa AbstractVector)
                kgrid = SimpleG.Arbitrary{eltype(kgrid)}(kgrid)
            end
            kernel = DCKernel_qp(
                param,
                Π_prev;
                Euv=Euv,
                rtol=rtol,
                Nk=Nk,
                maxK=maxK,
                minK=minK,
                order=order,
                int_type=int_type,
                spin_state=:sigma,
                kgrid=kgrid,
                kwargs...,
            )
        end
        Σ, Σ_ins = calcΣ_3d(G_prev, kernel)
    else
        error("No support for GW in $dim dimension!")
    end

    return Σ, Σ_ins, kernel
end

function calcΣ_3d(G::GreenFunc.MeshArray, W::LegendreInteraction.DCKernel)
    @unpack β = W.param

    kgrid = W.kgrid
    qgrids = W.qgrids
    fdlr = G.mesh[1].representation
    bdlr = W.dlrGrid

    G_dlr = to_dlr(G)
    G_imt = to_imtime(G_dlr)

    # prepare kernel, interpolate into τ-space with fdlr.τ
    kernel_bare = W.kernel_bare
    kernel_freq = W.kernel
    kernel = Lehmann.matfreq2tau(bdlr, kernel_freq, fdlr.τ, bdlr.n; axis=3)

    # container of Σ
    Σ = GreenFunc.MeshArray(G_imt.mesh[1], kgrid; dtype=ComplexF64)

    # equal-time green (instant)
    G_ins = dlr_to_imtime(G_dlr, [β]) * (-1)
    Σ_ins = GreenFunc.MeshArray(G_ins.mesh[1], kgrid; dtype=ComplexF64)

    for τi in eachindex(G_imt.mesh[1])
        for ki in eachindex(kgrid)
            k = kgrid[ki]
            Gq = CompositeGrids.Interp.interp1DGrid(
                G_imt[τi, :],
                G_imt.mesh[2],
                qgrids[ki].grid,
            )
            integrand = kernel[ki, 1:(qgrids[ki].size), τi] .* Gq ./ k .* qgrids[ki].grid
            Σ[τi, ki] = CompositeGrids.Interp.integrate1D(integrand, qgrids[ki])
            @assert isfinite(Σ[τi, ki]) "fail Δ at $τi, $ki"
            if τi == 1
                Gq = CompositeGrids.Interp.interp1DGrid(
                    G_ins[1, :],
                    G_ins.mesh[2],
                    qgrids[ki].grid,
                )
                integrand =
                    kernel_bare[ki, 1:(qgrids[ki].size)] .* Gq ./ k .* qgrids[ki].grid
                Σ_ins[1, ki] = CompositeGrids.Interp.integrate1D(integrand, qgrids[ki])
                @assert isfinite(Σ_ins[1, ki]) "fail Δ0 at $ki"
            end
        end
    end

    return Σ / (-4 * π^2), Σ_ins / (-4 * π^2)
end

function Σ_LQSGW(
    param::Parameter.Para;
    Euv=1000 * param.EF,
    rtol=1e-14,
    Nk=14,
    maxK=6 * param.kF,
    minK=1e-8 * param.kF,
    order=10,
    int_type=:rpa,
    max_steps=100,
    atol=1e-7,
    alpha=0.3,
    δK=5e-6,
    Fs=0.0,
    Fa=0.0,
    verbose=false,
    save=false,
    mpi=false,
    savedir="$(DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
)
    return Σ_LQSGW(
        param,
        Euv,
        rtol,
        Nk,
        maxK,
        minK,
        order,
        int_type,
        max_steps,
        atol,
        alpha,
        δK,
        Fs,
        Fa,
        verbose,
        save,
        mpi,
        savedir,
        savename,
    )
end

function Σ_LQSGW(
    param::Parameter.Para,
    Euv,
    rtol,
    Nk,
    maxK,
    minK,
    order,
    int_type,
    max_steps,
    atol,
    alpha,
    δK,
    Fs,
    Fa,
    verbose,
    save,
    mpi,
    savedir,
    savename,
)
    MPI.Init()
    comm = MPI.COMM_WORLD
    root = 0
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    # Make sigma output directory if needed
    if save && rank == root
        mkpath(savedir)
    end

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
    kGgrid = CompositeGrid.LogDensedGrid(
        :cheb,
        [0.0, maxKG],
        [0.0, kF],
        4 * Nk,
        0.01 * minK,
        4 * order,
    )

    # Medium grid for Π
    qPgrid =
        CompositeGrid.LogDensedGrid(:cheb, [0.0, maxKP], [0.0, 2 * kF], Nk, minK, order)

    # Small grid for Σ
    kSgrid = CompositeGrid.LogDensedGrid(:cheb, [0.0, maxKS], [0.0, kF], Nk, minK, order)

    # Get UEG G0; a large kgrid is required for the self-consistency loop
    G0 = SelfEnergy.G0wrapped(Euv, rtol, kGgrid, param)

    # Verify that the non-interacting density is correct
    G0_dlr = to_dlr(G0)
    G0_ins = dlr_to_imtime(G0_dlr, [β]) * (-1)
    integrand = real(G0_ins[1, :]) .* kGgrid.grid .* kGgrid.grid
    densityS = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKS]) / π^2
    densityP = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKP]) / π^2
    densityG = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKG]) / π^2
    if rank == root
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

    # Initial quasiparticle properties
    E_qp_0 = bare_energy(param, kSgrid)
    δμ_prev = 0.0
    meff_prev = 1.0
    zfactor_prev = 1.0

    # Use bare Green's function G0 as starting point
    G_prev = G0

    # Use exactly computed Π0 as starting point
    bdlr = DLRGrid(Euv, β, rtol, false, :ph)
    Π0 = GreenFunc.MeshArray(ImFreq(bdlr), qPgrid; dtype=ComplexF64)
    for (qi, q) in enumerate(qPgrid)
        Π0[:, qi] = Polarization.Polarization0_FiniteTemp(
            q,
            bdlr.n,
            param;
            maxk=maxKP,
            scaleN=40,
            gaussN=20,
        )
    end

    # Use exact Π0 for initial G0W0 self-energy: Σ[G0, Π0](k, τ)
    Σ, Σ_ins = Σ_GW(G0, Π0)

    # Write initial (G0W0) self-energy to JLD2 file, overwriting if it already exists
    if save && rank == root
        jldopen(joinpath(savedir, savename), "w") do file
            file["param"] = string(param)
            file["E_qp_0"] = E_qp_0
            file["Z_0"] = zfactor_prev * ones(length(E_qp_0))
            file["G_0"] = G0
            file["Π_0"] = Π0
            file["Σ_0"] = Σ
            return file["Σ_ins_0"] = Σ_ins
        end
    end

    # Self-consistency loop
    pi_getter = mpi ? Π_qp : Π_qp_serial
    if rank == root
        println("\nBegin self-consistency loop...\n")
    end
    i_step = 0
    while i_step < max_steps
        δμ = chemicalpotential(param, Σ, Σ_ins)
        meff = massratio(param, Σ, Σ_ins, δK)[1]
        zfactor = zfactor_fermi(param, Σ)

        if verbose && rank == root
            print("""\n
            Step $(i_step + 1):
            • m*/m        = \t$(meff)
            • Z           = \t$(zfactor)
            • δμ          = \t$(δμ)
            """)
        end

        # Get the current Z-factor
        Z_kgrid = zfactor_full(param, Σ)

        # Get the current quasiparticle energy up to k=maxKS
        E_qp_kgrid = quasiparticle_energy(param, Σ, Σ_ins)

        # Get interpolated quasiparticle energy for Π_qp solver
        E_qp = E_qp_interp(param, Σ, Σ_ins, kGgrid; maxKS=maxKS)

        # Get interpolated quasiparticle Green's function G_qp
        G = G_qp(param, Σ, Σ_ins, kGgrid; maxKS=maxKS)

        # Mix the new and old Green's functions via linear interpolation:
        # G_mix = (1 - α) * G_prev + α * G
        G_mix = _lerp(G_prev, G, alpha)

        # Get Π_mix = Π_qp[G_mix]
        Π_mix = pi_getter(param, E_qp, Nk, maxKP, minK, order, qPgrid, bdlr)

        # # Get Π_mix = Π0 + δΠ_qp[G0, G_mix]
        # δΠ_mix = get_δΠ_qp_qw(param, G0, G_mix, Euv, rtol, Nk, maxK, minK, order, qPgrid)
        # Π_mix = Π0 + δΠ_mix

        # Get Σ[G_mix, Π_mix](K, τ)
        Σ, Σ_ins = Σ_GW(G_mix, Π_mix)

        # Append self-energy at this step to JLD2 file
        if save && rank == root
            jldopen(joinpath(savedir, savename), "a+") do file
                file["E_qp_$(i_step + 1)"] = E_qp_kgrid
                file["Z_$(i_step + 1)"] = Z_kgrid
                file["G_$(i_step + 1)"] = G_mix
                file["Π_$(i_step + 1)"] = Π_mix
                file["Σ_$(i_step + 1)"] = Σ
                return file["Σ_ins_$(i_step + 1)"] = Σ_ins
            end
        end

        # Test for convergence of quasiparticle properties
        if i_step > 0
            dmeff = abs(meff - meff_prev)
            dzfactor = abs(zfactor - zfactor_prev)
            ddeltamu = abs(δμ - δμ_prev)
            if verbose && rank == root
                print("""
                • |Δ(m* / m)| = \t$(dmeff)
                • |Δ(Z)|      = \t$(dzfactor)
                • |Δ(δμ)|     = \t$(ddeltamu)
                """)
            end
            if all([dmeff, dzfactor, ddeltamu] .< atol)
                if rank == root
                    println("\nConverged to atol = $atol after $i_step steps!")
                end
                break
            end
        end

        # Prepare for next iteration
        i_step += 1
        # G_prev = G
        G_prev = G_mix
        δμ_prev = δμ
        meff_prev = meff
        zfactor_prev = zfactor
    end
    if i_step == max_steps && rank == root
        println(
            "\nWARNING: Convergence to atol = $atol not reached after $max_steps steps!",
        )
    end
    return Σ, Σ_ins
end
