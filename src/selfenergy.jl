
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
    show_progress=false,
    save=false,
    mpi=false,
    savedir="$(DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(param.beta).jld2",
    loaddir="$(DATA_DIR)/$(param.dim)d/$(int_type)",
    loadname=nothing,
)
    @assert max_steps ≤ MAXIMUM_STEPS "max_steps must be ≤ $MAXIMUM_STEPS"
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
        show_progress,
        save,
        mpi,
        savedir,
        savename,
        loaddir,
        loadname,
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
    show_progress,
    save,
    mpi,
    savedir,
    savename,
    loaddir,
    loadname,
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

    # Bosonic DLR grid
    bdlr = DLRGrid(Euv, β, rtol, false, :ph)

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
            # Nk=Nk,
            maxK=maxKS,
            minK=minKS,
            # order=order,
            int_type=_int_type,
            Fs=Fs,
            Fa=Fa,
        )
        Σ = to_imfreq(to_dlr(Σ_imtime))  # Σ(τ, k) → Σ(iωₙ, k)
        return Σ, Σ_ins
    end

    local E_qp_0, Z_0, G0, Π0, W0, Σ_prev, Σ_ins_prev, Σ_mix, Σ_ins_mix
    if isnothing(loaddir) == false && isnothing(loadname) == false
        if rank == root
            # Load starting point from JLD2 file
            jldopen(joinpath(loaddir, loadname), "r") do file
                # Ensure that the load data was convergent
                @assert file["converged"] == true "Specificed starting point data did not converge!"
                # Find the converged data in JLD2 file
                max_step = -1
                for j in 0:MAXIMUM_STEPS
                    if haskey(file, "E_k_$(j)")
                        max_step = j
                    else
                        break
                    end
                end
                if max_step < 0
                    error("No data found in $(savedir)!")
                end
                E_qp_0 = file["E_k_$(max_step)"]
                Z_0 = file["Z_k_$(max_step)"]
                G0 = file["G_$(max_step)"]
                Π0 = file["Π_$(max_step)"]
                W0 = file["W_$(max_step)"]
                Σ_prev = Σ_mix = file["Σ_$(max_step)"]
                Σ_ins_prev = Σ_ins_mix = file["Σ_ins_$(max_step)"]
                println("Found converged data with max_step=$(max_step) for loadname $(loadname)!")
            end
        else
            E_qp_0 = Z_0 = G0 = Π0 = W0 = Σ_prev = Σ_ins_prev = Σ_mix = Σ_ins_mix = nothing
        end
        # Broadcast starting point data to all processes
        E_qp_0 = MPI.bcast(E_qp_0, root, comm)
        Z_0 = MPI.bcast(Z_0, root, comm)
        G0 = MPI.bcast(G0, root, comm)
        Π0 = MPI.bcast(Π0, root, comm)
        W0 = MPI.bcast(W0, root, comm)
        Σ_prev = MPI.bcast(Σ_prev, root, comm)
        Σ_ins_prev = MPI.bcast(Σ_ins_prev, root, comm)
        Σ_mix = MPI.bcast(Σ_mix, root, comm)
        Σ_ins_mix = MPI.bcast(Σ_ins_mix, root, comm)
    else
        # Get UEG G0; a large kgrid is required for the self-consistency loop
        G0 = G_0(param, Euv, rtol, kGgrid; symmetry=:sym)
        # G0 = G_0(param, Euv, rtol, kGgrid; symmetry=:none)  # same as SelfEnergy.G0Wrapped
        # G0 = SelfEnergy.G0wrapped(Euv, rtol, kGgrid, param)

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

        # Initial quasiparticle properties
        E_qp_0 = bare_energy(param, kSgrid)
        δμ_prev = 0.0
        meff_prev = 1.0
        zfactor_prev = 1.0
        Z_0 = zfactor_prev * ones(length(E_qp_0))
        # Use exactly computed Π0 as starting point
        Π0 = GreenFunc.MeshArray(ImFreq(bdlr), qPgrid; dtype=ComplexF64)
        for (qi, q) in enumerate(qPgrid)
            Π0[:, qi] = Polarization.Polarization0_FiniteTemp(
                q,
                bdlr.n,
                param;
                maxk=maxKP / kF,
                scaleN=50,
                gaussN=25,
            )
        end
        # Use exact Π0 for initial G0W0 self-energy: Σ[G0, Π0](k, τ)
        Σ, Σ_ins = Σ_GW(G0, Π0)
        # Use G0W0 self-energy as starting point
        Σ_prev = Σ_mix = Σ
        Σ_ins_prev = Σ_ins_mix = Σ_ins
    end

    # Write initial (G0W0) self-energy to JLD2 file, overwriting
    # if it already exists (G0 -> Π0 -> W0 -> Σ1 = Σ_G0W0)
    if save && rank == root
        jldopen(joinpath(savedir, savename), "w") do file
            # Get W0 = W_qp[Π0] for plotting purposes only
            W0 = W_qp(param, Π0; int_type=_int_type, Fs=Fs, Fa=Fa)
            file["param"] = string(param)
            file["E_k_0"] = E_qp_0
            file["Z_k_0"] = Z_0
            file["G_0"] = G0
            file["Π_0"] = Π0
            file["W_0"] = W0
            file["Σ_0"] = Σ_prev
            file["Σ_ins_0"] = Σ_ins_prev
        end
    end

    # Self-consistency loop
    if rank == root
        println("\nBegin self-consistency loop...\n")
    end
    i_step = 0
    converged = false
    # alpha_first_step = 0.01
    while i_step < max_steps
        # Get quasiparticle properties
        δμ = chemicalpotential(param, Σ_prev, Σ_ins_prev)
        meff = massratio(param, Σ_prev, Σ_ins_prev, δK)[1]
        zfactor = zfactor_fermi(param, Σ_prev)
        if verbose && rank == root
            print("""\n
            Step $(i_step + 1):
            • m*/m        = \t$(meff)
            • Z           = \t$(zfactor)
            • δμ          = \t$(δμ)
            """)
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
                    jldopen(joinpath(savedir, savename), "a") do file
                        file["converged"] = true
                    end
                end
                converged = true
                break
            end
        end

        # Get the current Z-factor
        Z_kSgrid = zfactor_full(param, Σ_prev)

        # Get the current quasiparticle energy
        E_qp_kSgrid = quasiparticle_energy(param, Σ_prev, Σ_ins_prev)

        # Get grid-interpolated quasiparticle energy for Π_qp solver
        E_qp_kGgrid = E_qp_grid(param, Σ_prev, Σ_ins_prev, kGgrid)

        # Get interpolated quasiparticle Green's function G_qp
        G = G_qp(param, Σ_prev, Σ_ins_prev, kGgrid)

        # Get Π = Π_qp[G]
        Π = Π_qp(
            param,
            E_qp_kGgrid,
            kGgrid,
            Nk,
            maxKP,
            minK,
            order,
            qPgrid,
            bdlr;
            verbose=verbose,
            show_progress=show_progress,
        )

        # Get Σ_curr[G, Π](k, τ)
        Σ_curr, Σ_ins_curr = Σ_GW(G, Π)

        # Mix the new and old self energies via linear interpolation:
        # Σ_mix = (1 - α) * Σ_prev + α * Σ    
        Σ_mix = lerp(Σ_prev, Σ_curr, alpha)
        Σ_ins_mix = lerp(Σ_ins_prev, Σ_ins_curr, alpha)

        # # TODO: Benchmark variable alpha
        # _alpha = i_step == 0 ? alpha_first_step : alpha
        # Σ_mix = lerp(Σ_prev, Σ_curr, _alpha)
        # Σ_ins_mix = lerp(Σ_ins_prev, Σ_ins_curr, _alpha)

        # Append data at this step to JLD2 file (G -> Π -> W -> Σ)
        if save && rank == root
            jldopen(joinpath(savedir, savename), "a") do file
                # Get W = W_qp[Π] for plotting purposes only
                W = W_qp(param, Π; int_type=_int_type, Fs=Fs, Fa=Fa)
                file["E_k_$(i_step + 1)"] = E_qp_kSgrid
                file["Z_k_$(i_step + 1)"] = Z_kSgrid
                file["G_$(i_step + 1)"] = G
                file["Π_$(i_step + 1)"] = Π
                file["W_$(i_step + 1)"] = W
                file["Σ_$(i_step + 1)"] = Σ_mix
                file["Σ_ins_$(i_step + 1)"] = Σ_ins_mix
            end
        end

        # Prepare for next iteration
        i_step += 1
        Σ_prev = Σ_mix
        Σ_ins_prev = Σ_ins
        δμ_prev = δμ
        meff_prev = meff
        zfactor_prev = zfactor

        # Explicit garbage collection resolves MPI-related memory leak
        # TODO: Implement globally initialized (variable) MPI buffers.
        GC.gc()
    end
    if i_step == max_steps && rank == root
        println(
            "\nWARNING: Convergence to atol = $atol not reached after $max_steps steps!",
        )
        jldopen(joinpath(savedir, savename), "a") do file
            file["converged"] = false
        end
    end
    return Σ_mix, Σ_ins_mix, i_step, converged
end
