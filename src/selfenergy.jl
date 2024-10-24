# Holds data for a single LQSGW iteration,
# (δμ, m*/m, Z, E) -> G -> Π -> W -> (Σ, Σ_ins)
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

# Construct an LQSGW step from existing data
function LQSGWIteration(lqsgw_iteration::LQSGWIteration, step=lqsgw_iteration.step)
    return LQSGWIteration(
        step,
        lqsgw_iteration.dmu,
        lqsgw_iteration.meff,
        lqsgw_iteration.zfactor,
        lqsgw_iteration.E_k,
        lqsgw_iteration.Z_k,
        lqsgw_iteration.G,
        lqsgw_iteration.Π,
        lqsgw_iteration.W,
        lqsgw_iteration.Σ,
        lqsgw_iteration.Σ_ins,
    )
end

function initialize_g0w0_starting_point(
    param::Parameter.Para,
    bdlr,
    fdlr,
    kGgrid,
    qPgrid,
    kSgrid,
    maxKS,
    maxKP,
    maxKG,
    int_type,
    Fs,
    Fa,
    Σ_GW;
    verbose=false,
)
    @unpack me, β, kF, n, beta = param

    # Get UEG G0; a large kgrid is required for the self-consistency loop
    G0 = G_0(param, fdlr, kGgrid)

    # Verify that the non-interacting density is correct
    # G0_ins = matfreq2tau(fdlr, G0.data, [β]) * (-1)
    G0_ins = dlr_to_imtime(to_dlr(G0), [β]) * (-1)
    integrand = real(G0_ins[1, :]) .* kGgrid.grid .* kGgrid.grid
    densityS = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKS]) / π^2
    densityP = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKP]) / π^2
    densityG = CompositeGrids.Interp.integrate1D(integrand, kGgrid, [0, maxKG]) / π^2
    if verbose
        println("Density from Σ mesh (β = $beta): $(densityS)")
        println("Density from Π mesh (β = $beta): $(densityP)")
        println("Density from G mesh (β = $beta): $(densityG)")
        println("Exact non-interacting density (T = 0): $(n)")
        if beta == 1000.0
            # The large discrepancy in density is due to finite-T effects
            # @assert isapprox(n, densityG, rtol=3e-5)  # :gauss
            @assert isapprox(n, densityG, rtol=3e-3)  # :cheb
        end
        println("G0(k0, 0⁻) = $(G0_ins[1, 1])\n")
    end

    # Initial quasiparticle properties
    E_qp_0 = bare_energy(param, kSgrid)
    δμ_0 = 0.0
    meff_0 = 1.0
    zfactor_0 = 1.0
    Z_0 = zfactor_0 * ones(length(E_qp_0))

    # Use exactly computed Π0 as starting point
    Π0 = GreenFunc.MeshArray(ImFreq(bdlr), qPgrid; dtype=ComplexF64)
    for (qi, q) in enumerate(qPgrid)
        Π0[:, qi] = Polarization.Polarization0_FiniteTemp(
            q,
            bdlr.n,
            param;
            maxk=maxKP / kF,  # argument maxk is specified in units of kF
            scaleN=50,
            gaussN=25,
        )
    end
    W0 = W_qp(param, Π0; int_type=int_type, Fs=Fs, Fa=Fa)

    # Use exact Π0 for initial G0W0 self-energy: Σ[G0, Π0](k, τ)
    Σ_0, Σ_ins_0 = Σ_GW(G0, Π0)

    # Return G0W0 LQSGW starting point
    i_step = 0
    return LQSGWIteration(
        i_step,
        δμ_0,
        meff_0,
        zfactor_0,
        E_qp_0,
        Z_0,
        G0,
        Π0,
        W0,
        Σ_0,
        Σ_ins_0,
    )
end

function load_lqsgw_starting_point(loaddir, loadname)
    MPI.Initialized() == false && MPI.Init()
    comm = MPI.COMM_WORLD
    root = 0
    rank = MPI.Comm_rank(comm)
    if rank == root
        # Load starting point from JLD2 file
        loaddata, loadparam = jldopen(joinpath(loaddir, loadname), "r") do file
            # Ensure that the load data was convergent
            @assert file["converged"] == true "Specificed starting point data did not converge!"
            # Find the converged data in JLD2 file
            max_step = -1
            for i in 0:MAXIMUM_STEPS
                if haskey(file, string(i))
                    max_step = i
                else
                    break
                end
            end
            if max_step < 0
                error("No data found in $(loaddir)!")
            end
            _loaddata = file[string(max_step)]
            _loadparam = Parameter.from_string(file["param"])
            @assert _loaddata.step == max_step "Data step mismatch!"
            println(
                "Found converged data with max_step=$(max_step) for loadname $(loadname)!",
            )
            return _loaddata, _loadparam
        end
    else
        loaddata = loadparam = nothing
    end
    # Broadcast starting point data and param to all processes
    loaddata = MPI.bcast(loaddata, root, comm)
    loadparam = MPI.bcast(loadparam, root, comm)
    # Build the starting point from (rescaled) existing converged LQSGW data
    return LQSGWIteration(loaddata, 0), loadparam
end

# Helper function to get the starting point for the LQSGW loop
function get_starting_point(
    param::Parameter.Para,
    bdlr,
    fdlr,
    kGgrid,
    qPgrid,
    kSgrid,
    maxKS,
    maxKP,
    maxKG,
    int_type,
    Fs,
    Fa,
    Σ_GW;
    rescale=true,
    loaddir=nothing,
    loadname=nothing,
    overwrite=false,
)
    MPI.Initialized() == false && MPI.Init()
    comm = MPI.COMM_WORLD
    root = 0
    rank = MPI.Comm_rank(comm)
    using_existing_lqsgw_data = (
        (overwrite == false) &&
        (isnothing(loaddir) == false) &&
        (isnothing(loadname) == false)
    )
    if using_existing_lqsgw_data
        loaddata, loadparam = load_lqsgw_starting_point(loaddir, loadname)
        # Either relabel or rescale the starting point for Σ to match the current parameters
        if rescale
            loaddata = rescale_starting_point(param, loaddata, loadparam, fdlr, kSgrid)
        else
            loaddata = relabel_starting_point(param, loaddata, loadparam, fdlr, kSgrid)
        end
        starting_point_type = "LQSGW"
    else
        loaddata = initialize_g0w0_starting_point(
            param,
            bdlr,
            fdlr,
            kGgrid,
            qPgrid,
            kSgrid,
            maxKS,
            maxKP,
            maxKG,
            int_type,
            Fs,
            Fa,
            Σ_GW;
            verbose=rank == root,
        )
        loadparam = param
        starting_point_type = "G0W0"
    end
    return loaddata, loadparam, starting_point_type
end

# Helper function to relabel the starting point for Σ to match the current parameters 
function relabel_starting_point(param::Parameter.Para, loaddata, loadparam, fdlr, kSgrid)
    # Relabel starting point for Σ if it was calculated at a different value for rs
    if param.rs != loadparam.rs
        xgrid_old = loaddata.Σ.mesh[2] / loadparam.kF
        xgrid_new = kSgrid / param.kF
        @assert isapprox(xgrid_old, xgrid_new, rtol=1e-7) "Mismatch in dimensionless kgrids for new/old Σ data!"
        Σ_relabeled = GreenFunc.MeshArray(
            ImFreq(fdlr),
            kSgrid;
            dtype=ComplexF64,
            data=loaddata.Σ.data,
        )
        Σ_ins_relabeled = GreenFunc.MeshArray(
            ImTime(fdlr; grid=[param.β]),
            kSgrid;
            dtype=ComplexF64,
            data=loaddata.Σ_ins.data,
        )
        # Reconstruct the starting point using the current parameters
        loaddata = reconstruct(loaddata; Σ=Σ_relabeled, Σ_ins=Σ_ins_relabeled)
    end
    return loaddata
end

# Helper function to rescale the starting point for Σ to the current rs value
function rescale_starting_point(param::Parameter.Para, loaddata, loadparam, fdlr, kSgrid)
    # Rescale starting point for Σ if it was calculated at a different value for rs
    if param.rs != loadparam.rs
        xgrid_old = loaddata.Σ.mesh[2] / loadparam.kF
        xgrid_new = kSgrid / param.kF
        @assert isapprox(xgrid_old, xgrid_new, rtol=1e-7) "Mismatch in dimensionless kgrids for new/old Σ data!"
        #   Σ ~ NF ~ 1 / rs, to leading order
        #   Σ_ins ~ NF ~ 1 / rs, to leading order
        Σ_data = loaddata.Σ.data * (loadparam.rs / param.rs)
        Σ_ins_data = loaddata.Σ_ins.data * (loadparam.rs / param.rs)
        Σ_rescaled =
            GreenFunc.MeshArray(ImFreq(fdlr), kSgrid; dtype=ComplexF64, data=Σ_data)
        Σ_ins_rescaled = GreenFunc.MeshArray(
            ImTime(fdlr; grid=[param.β]),
            kSgrid;
            dtype=ComplexF64,
            data=Σ_ins_data,
        )
        # Reconstruct the starting point using the rescaled self-energy
        loaddata = reconstruct(loaddata; Σ=Σ_rescaled, Σ_ins=Σ_ins_rescaled)
    end
    return loaddata
end

# Helper function to test for convergence of quasiparticle properties.
# We define the LQSGW self-consistency as the maximum absolute difference
# between the current and previous step's quasiparticle properties.
function test_convergence(prev_step, i_step, meff, zfactor, dmu, atol, verbose)
    dmeff = abs(meff - prev_step.meff)
    dzfactor = abs(zfactor - prev_step.zfactor)
    ddeltamu = abs(dmu - prev_step.dmu)
    if verbose
        println_root("""
        \nStep $(i_step):
        • m*/m        = \t$(meff)
        • Z           = \t$(zfactor)
        • δμ          = \t$(dmu)
        • |Δ(m* / m)| = \t$(dmeff)
        • |Δ(Z)|      = \t$(dzfactor)
        • |Δ(δμ)|     = \t$(ddeltamu)
        """)
    end
    return all([dmeff, dzfactor, ddeltamu] .≤ atol)
end

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

function Σ_G0W0(
    param::Parameter.Para;
    Euv=1000 * param.EF,
    rtol=1e-14,
    Nk=14,
    maxK=6 * param.kF,
    minK=1e-8 * param.kF,
    order=10,
    int_type=:rpa,
    δK=5e-6,
    Fs=0.0,
    Fa=0.0,
    verbose=false,
    save=false,
    savedir="$(DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="g0w0_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(round(param.beta; sigdigits=4)).jld2",
)
    return Σ_G0W0(
        param,
        Euv,
        rtol,
        Nk,
        maxK,
        minK,
        order,
        int_type,
        δK,
        Fs,
        Fa,
        verbose,
        save,
        savedir,
        savename,
    )
end

function Σ_G0W0(
    param::Parameter.Para,
    Euv,
    rtol,
    Nk,
    maxK,
    minK,
    order,
    int_type,
    δK,
    Fs,
    Fa,
    verbose,
    save,
    savedir,
    savename,
)
    # Make sigma output directory if needed
    if save
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

    # DLR grids
    # NOTE: `symmetry = :sym` uses a DLR Matsubara grid that is symmetric around n = -1 for improved convergence of the SC loop
    bdlr = DLRGrid(Euv, β, rtol, false, :ph)
    fdlr = DLRGrid(Euv, β, rtol, true, :sym)

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

    # Helper function to get the GW self-energy Σ[G, W](iωₙ, k) for a given int_type (W)
    # NOTE: A large momentum grid is required for G and Σ at intermediate steps
    function Σ_GW(G, Π)
        # Get Σ_dyn(τ, k) and Σ_ins(k)
        Σ_imtime, Σ_ins, _ = GW(
            param,
            G,
            Π,
            kSgrid;
            Euv=Euv,
            rtol=rtol,
            # Nk=Nk,
            maxK=maxKS,
            minK=minK,
            # order=order,
            int_type=_int_type,
            Fs=Fs,
            Fa=Fa,
        )
        # Σ_dyn(τ, k) → Σ_dyn(iωₙ, k)
        Σ = to_imfreq(to_dlr(Σ_imtime))
        return Σ, Σ_ins
    end

    # Helper function to write data to JLD2 file
    function write_to_file(key, val; write_mode="a", compress=true)
        if save
            jldopen(joinpath(savedir, savename), write_mode; compress=compress) do file
                file[string(key)] = val
            end
        end
    end

    # Get one-shot GW result
    oneshot_data = initialize_g0w0_starting_point(
        param,
        bdlr,
        fdlr,
        kGgrid,
        qPgrid,
        kSgrid,
        maxKS,
        maxKP,
        maxKG,
        _int_type,
        Fs,
        Fa,
        Σ_GW;
        verbose=true,
    )

    # Get quasiparticle properties from Σ_G0W0
    dmu_g0w0 = chemicalpotential(param, oneshot_data.Σ, oneshot_data.Σ_ins)
    meff_g0w0 = massratio(param, oneshot_data.Σ, oneshot_data.Σ_ins, δK)[1]
    zfactor_g0w0 = zfactor_fermi(param, oneshot_data.Σ)

    # The one-shot method uses the quasiparticle properties of Σ_G0W0
    oneshot_data =
        reconstruct(oneshot_data; dmu=dmu_g0w0, meff=meff_g0w0, zfactor=zfactor_g0w0)

    if verbose
        println("""
        Calculated one-shot GW self-energy with:
        • rs          = \t$(round(param.rs; sigdigits=13))
        • m*/m        = \t$(oneshot_data.meff)
        • Z           = \t$(oneshot_data.zfactor)
        • δμ          = \t$(oneshot_data.dmu)
        """)
    end

    # Write starting point to JLD2 file, overwriting any existing data
    write_to_file("param", string(param); write_mode="w")
    write_to_file(0, oneshot_data)
    return oneshot_data.Σ, oneshot_data.Σ_ins
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
    savedir="$(DATA_DIR)/$(param.dim)d/$(int_type)",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(round(param.rs; sigdigits=4))_beta=$(round(param.beta; sigdigits=4)).jld2",
    loaddir="$(DATA_DIR)/$(param.dim)d/$(int_type)",
    loadname=nothing,
    overwrite=false,
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
        savedir,
        savename,
        loaddir,
        loadname,
        overwrite,
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
    savedir,
    savename,
    loaddir,
    loadname,
    overwrite,
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

    # DLR grids
    # NOTE: `symmetry = :sym` uses a DLR Matsubara grid that is symmetric around n = -1 for improved convergence of the SC loop
    bdlr = DLRGrid(Euv, β, rtol, false, :ph)
    fdlr = DLRGrid(Euv, β, rtol, true, :sym)

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

    # Helper function to get the GW self-energy Σ[G, W](iωₙ, k) for a given int_type (W)
    # NOTE: A large momentum grid is required for G and Σ at intermediate steps
    function Σ_GW(G, Π)
        # Get Σ_dyn(τ, k) and Σ_ins(k)
        Σ_imtime, Σ_ins, _ = GW(
            param,
            G,
            Π,
            kSgrid;
            Euv=Euv,
            rtol=rtol,
            # Nk=Nk,
            maxK=maxKS,
            minK=minK,
            # order=order,
            int_type=_int_type,
            Fs=Fs,
            Fa=Fa,
        )
        # Σ_dyn(τ, k) → Σ_dyn(iωₙ, k)
        Σ = to_imfreq(to_dlr(Σ_imtime))
        return Σ, Σ_ins
    end

    # Helper function to write data to JLD2 file
    function write_to_file(key, val; write_mode="a", compress=true)
        if save && rank == root
            jldopen(joinpath(savedir, savename), write_mode; compress=compress) do file
                file[string(key)] = val
            end
        end
    end

    # Load or initialize the starting point (step 0)
    prev_step, prev_param, starting_point_type = get_starting_point(
        param,
        bdlr,
        fdlr,
        kGgrid,
        qPgrid,
        kSgrid,
        maxKS,
        maxKP,
        maxKG,
        _int_type,
        Fs,
        Fa,
        Σ_GW;
        loaddir=loaddir,
        loadname=loadname,
        overwrite=overwrite,
    )
    if verbose
        println_root("""
        Using $(starting_point_type) starting point with:
        • rs          = \t$(round(prev_param.rs; sigdigits=13))
        • m*/m        = \t$(prev_step.meff)
        • Z           = \t$(prev_step.zfactor)
        • δμ          = \t$(prev_step.dmu)
        """)
    end

    # Write starting point to JLD2 file, overwriting any existing data
    write_to_file("param", string(param); write_mode="w")
    write_to_file(0, prev_step)

    # Self-consistency loop
    println_root("Beginning self-consistency loop...")
    i_step = 1
    converged = false
    while i_step ≤ max_steps
        i_curr = prev_step.step + 1
        Σ_prev = prev_step.Σ
        Σ_ins_prev = prev_step.Σ_ins

        # Get quasiparticle properties
        dmu_curr = chemicalpotential(param, Σ_prev, Σ_ins_prev)
        meff_curr = massratio(param, Σ_prev, Σ_ins_prev, δK)[1]
        zfactor_curr = zfactor_fermi(param, Σ_prev)

        # Test for convergence of quasiparticle properties
        converged = test_convergence(
            prev_step,
            i_step,
            meff_curr,
            zfactor_curr,
            dmu_curr,
            atol,
            verbose,
        )
        if converged
            println_root("\nConverged to atol = $atol after $i_step step(s)!")
            write_to_file("converged", true)
            break
        end

        # Momentum-resolved Z(k) and E_qp(k) calculated on the self-energy grid (kSgrid)
        Z_curr = zfactor_full(param, Σ_prev)
        E_qp_curr = quasiparticle_energy(param, Σ_prev, Σ_ins_prev)

        # Interpolated quasiparticle energy / G on the Green's function grid (kGgrid)
        E_qp_kGgrid = E_qp_grid(param, Σ_prev, Σ_ins_prev, kGgrid)
        G_curr = G_qp(param, Σ_prev, Σ_ins_prev, kGgrid)

        # Compute the current self-energy using the GW approximation
        Π_curr = Π_qp(
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
        W_curr = W_qp(param, Π_curr; int_type=_int_type, Fs=Fs, Fa=Fa)
        Σ_curr, Σ_ins_curr = Σ_GW(G_curr, Π_curr)  # NOTE: W[Π] is computed internally

        # Linearly interpolate the new and old self energies: Σ_mix = (1 - α) * Σ_prev + α * Σ
        Σ_mix = lerp(Σ_prev, Σ_curr, alpha)
        Σ_ins_mix = lerp(Σ_ins_prev, Σ_ins_curr, alpha)

        # Build the current LQSGW iteration
        curr_step = LQSGWIteration(
            i_curr,
            dmu_curr,
            meff_curr,
            zfactor_curr,
            E_qp_curr,
            Z_curr,
            G_curr,
            Π_curr,
            W_curr,
            Σ_mix,
            Σ_ins_mix,
        )

        # Append data at this step to JLD2 file and prepare for the next iteration
        write_to_file(i_curr, curr_step)
        i_step += 1
        prev_step = curr_step

        # Explicit garbage collection resolves MPI-related memory leak
        # TODO: Implement globally initialized (variable) MPI buffers.
        GC.gc()
    end
    if i_step == max_steps + 1
        println_root(
            "\nWARNING: Convergence to atol = $atol not reached after $max_steps step(s)!",
        )
        write_to_file("converged", false)
    end
    return prev_step.Σ, prev_step.Σ_ins, i_step, converged
end
