
"""Single G0 term with Taylor expansion for low and high frequencies."""
@inline function simple_pole(ω, ϵ)
    # if ω == 0
    #     return -1 / ϵ
    # end
    # Taylor expand for |ϵ / ω| ≪ 1 and |ϵ / ω| ≫ 1
    if abs(ϵ / ω) < 1e-6
        return (-1 + (ϵ / ω)^2) * ϵ / ω^2
    elseif abs(ω / ϵ) < 1e-6
        return (-1 + (ω / ϵ)^2) / ϵ
    else
        return real(1.0 / (im * ω - ϵ))
    end
end

function polar_kernel_integrand(param::Parameter.Para, E_qp, k, q, θ, ω)
    # @assert 0 ≤ θ ≤ π
    # @assert q ≥ 0 && k ≥ 0

    @unpack β, kF, EF, me = param
    if ω < 0
        ω = -ω
    end
    x = cos(θ)
    jacob = sin(θ)

    # Use small angle approximation when θ is near 0 or π
    kp2 = (k + q)^2 - 2k * q * (1 - x)
    # kp2 = k^2 + q^2 + 2k * q * x
    if x ≈ -1 || kp2 ≤ 0
        jacob = π - θ
        kp2 = (k - q)^2 + k * q * (θ - π)^2
    elseif x ≈ 1 || kp2 ≤ 0
        jacob = θ
        kp2 = (k + q)^2 - k * q * θ^2
    end

    # Get quasiparticle energies
    E_k = E_qp(k)
    E_kp = E_qp(√kp2)
    dEp = E_kp - E_k
    if dEp == 0
        return 0.0
    end

    # Build the quasiparticle polarization bubble integrand
    fermi_k = Spectral.fermiDirac(E_k, β)
    if ω == 0
        # fermi_kp = Spectral.fermiDirac(E_kp, β)
        # if abs(dEp) ≤ 1e-12 * EF || q ≤ 1e-6 * kF
        #     res = jacob * Spectral.kernelFermiT_dω(β, E_k, β)
        # elseif q < 3 * kF
        if q < 3 * kF
            fermi_kp = Spectral.fermiDirac(E_kp, β)
            return jacob * (fermi_kp - fermi_k) / dEp
            # return jacob * (fermi_k - fermi_kp) * _simple_pole(ω, dEp)
            # res = jacob * (fermi_k - fermi_kp) * _simple_pole(ω, dEp)
        end
        # return res
    end
    # Use small angle approximation when θ is near 0 or π
    km2 = (k - q)^2 + 2k * q * (1 - x)
    if x ≈ 1 || km2 ≤ 0
        jacob = θ
        km2 = (k - q)^2 + k * q * θ^2
    elseif x ≈ -1 || km2 ≤ 0
        jacob = π - θ
        km2 = (k + q)^2 - k * q * (π - θ)^2
    end
    E_km = E_qp(√km2)
    dEm = E_k - E_km
    if dEm == 0
        return 0.0
    end
    res = jacob * fermi_k * (simple_pole(ω, dEp) - simple_pole(ω, dEm))
    return res
end

"""
Angular integrand for the quasiparticle polarization bubble at ω ≠ 0.
"""
function angular_integrand_dynamic(param::Parameter.Para, E_qp, k, q, θ, ω)
    # @assert 0 ≤ θ ≤ π
    # @assert q ≥ 0 && k ≥ 0

    @unpack β, kF, EF, me = param
    if ω < 0
        ω = -ω
    end
    x = cos(θ)
    jacob = sin(θ)

    # Use small angle approximation when θ is near 0 or π
    kp2 = (k + q)^2 - 2k * q * (1 - x)
    # kp2 = k^2 + q^2 + 2k * q * x
    if x ≈ -1 || kp2 ≤ 0
        jacob = π - θ
        kp2 = (k - q)^2 + k * q * (θ - π)^2
    elseif x ≈ 1 || kp2 ≤ 0
        jacob = θ
        kp2 = (k + q)^2 - k * q * θ^2
    end

    # Get quasiparticle energies
    E_k = E_qp(k)
    E_kp = E_qp(√kp2)
    dEp = E_kp - E_k
    if dEp == 0
        return 0.0
    end

    # Build the quasiparticle polarization bubble integrand
    fermi_k = Spectral.fermiDirac(E_k, β)

    # Use small angle approximation when θ is near 0 or π
    km2 = (k - q)^2 + 2k * q * (1 - x)
    if x ≈ 1 || km2 ≤ 0
        jacob = θ
        km2 = (k - q)^2 + k * q * θ^2
    elseif x ≈ -1 || km2 ≤ 0
        jacob = π - θ
        km2 = (k + q)^2 - k * q * (π - θ)^2
    end
    E_km = E_qp(√km2)
    dEm = E_k - E_km
    if dEm == 0
        return 0.0
    end
    return jacob * fermi_k * (simple_pole(ω, dEp) - simple_pole(ω, dEm))
end
function angular_integrand_dynamic(param::Parameter.Para, E_qp_kGgrid, kGgrid, k, q, θ, ω)
    # @assert 0 ≤ θ ≤ π
    # @assert q ≥ 0 && k ≥ 0

    @unpack β, kF, EF, me = param
    if ω < 0
        ω = -ω
    end
    x = cos(θ)
    jacob = sin(θ)

    # Use small angle approximation when θ is near 0 or π
    kp2 = (k + q)^2 - 2k * q * (1 - x)
    # kp2 = k^2 + q^2 + 2k * q * x
    if x ≈ -1 || kp2 ≤ 0
        jacob = π - θ
        kp2 = (k - q)^2 + k * q * (θ - π)^2
    elseif x ≈ 1 || kp2 ≤ 0
        jacob = θ
        kp2 = (k + q)^2 - k * q * θ^2
    end

    # Get quasiparticle energies
    E_k = Interp.interp1D(E_qp_kGgrid, kGgrid, k)
    E_kp = Interp.interp1D(E_qp_kGgrid, kGgrid, √kp2)
    dEp = E_kp - E_k
    if dEp == 0
        return 0.0
    end

    # Build the quasiparticle polarization bubble integrand
    fermi_k = Spectral.fermiDirac(E_k, β)

    # Use small angle approximation when θ is near 0 or π
    km2 = (k - q)^2 + 2k * q * (1 - x)
    if x ≈ 1 || km2 ≤ 0
        jacob = θ
        km2 = (k - q)^2 + k * q * θ^2
    elseif x ≈ -1 || km2 ≤ 0
        jacob = π - θ
        km2 = (k + q)^2 - k * q * (π - θ)^2
    end
    E_km = Interp.interp1D(E_qp_kGgrid, kGgrid, √km2)
    dEm = E_k - E_km
    if dEm == 0
        return 0.0
    end
    return jacob * fermi_k * (simple_pole(ω, dEp) - simple_pole(ω, dEm))
end

"""
Angular integrand for the quasiparticle polarization bubble at ω = 0.
"""
function angular_integrand_static(param::Parameter.Para, E_qp, k, q, θ)
    # @assert 0 ≤ θ ≤ π
    # @assert q ≥ 0 && k ≥ 0

    @unpack β, kF, EF, me = param
    x = cos(θ)
    jacob = sin(θ)

    # Use small angle approximation when θ is near 0 or π
    kp2 = (k + q)^2 - 2k * q * (1 - x)
    # kp2 = k^2 + q^2 + 2k * q * x
    if x ≈ -1 || kp2 ≤ 0
        jacob = π - θ
        kp2 = (k - q)^2 + k * q * (θ - π)^2
    elseif x ≈ 1 || kp2 ≤ 0
        jacob = θ
        kp2 = (k + q)^2 - k * q * θ^2
    end

    # Get quasiparticle energies
    E_k = E_qp(k)
    E_kp = E_qp(√kp2)
    dEp = E_kp - E_k
    if dEp == 0
        return 0.0
    end

    # Build the quasiparticle polarization bubble integrand
    fermi_k = Spectral.fermiDirac(E_k, β)
    if q < 2.25 * kF
        fermi_kp = Spectral.fermiDirac(E_kp, β)
        return jacob * (fermi_kp - fermi_k) / dEp
        # return jacob * fermi_k * (1 - fermi_kp) * ((1 - exp(β * dEp)) / dEp)
        # return jacob * fermi_k * fermi_kp * (exp(β * E_kp) - exp(β * E_k)) / dEp
    end

    # Use small angle approximation when θ is near 0 or π
    km2 = (k - q)^2 + 2k * q * (1 - x)
    if x ≈ 1 || km2 ≤ 0
        jacob = θ
        km2 = (k - q)^2 + k * q * θ^2
    elseif x ≈ -1 || km2 ≤ 0
        jacob = π - θ
        km2 = (k + q)^2 - k * q * (π - θ)^2
    end
    E_km = E_qp(√km2)
    dEm = E_k - E_km
    if dEm == 0
        return 0.0
    end
    # return jacob * fermi_k * (simple_pole(ω, dEp) - simple_pole(ω, dEm))
    # return jacob * fermi_k * (1.0 / dEm - 1.0 / dEp)
    return jacob * fermi_k * (dEp - dEm) / (dEm * dEp)
end
function angular_integrand_static(param::Parameter.Para, E_qp_kGgrid, kGgrid, k, q, θ)
    # @assert 0 ≤ θ ≤ π
    # @assert q ≥ 0 && k ≥ 0

    @unpack β, kF, EF, me = param
    x = cos(θ)
    jacob = sin(θ)

    # Use small angle approximation when θ is near 0 or π
    kp2 = (k + q)^2 - 2k * q * (1 - x)
    # kp2 = k^2 + q^2 + 2k * q * x
    if x ≈ -1 || kp2 ≤ 0
        jacob = π - θ
        kp2 = (k - q)^2 + k * q * (θ - π)^2
    elseif x ≈ 1 || kp2 ≤ 0
        jacob = θ
        kp2 = (k + q)^2 - k * q * θ^2
    end

    # Get quasiparticle energies
    E_k = Interp.interp1D(E_qp_kGgrid, kGgrid, k)
    E_kp = Interp.interp1D(E_qp_kGgrid, kGgrid, √kp2)
    dEp = E_kp - E_k
    if dEp == 0
        return 0.0
    end

    # Build the quasiparticle polarization bubble integrand
    fermi_k = Spectral.fermiDirac(E_k, β)
    if q < 2.25 * kF
        fermi_kp = Spectral.fermiDirac(E_kp, β)
        return jacob * (fermi_kp - fermi_k) / dEp
        # return jacob * fermi_k * (1 - fermi_kp) * ((1 - exp(β * dEp)) / dEp)
        # return jacob * fermi_k * fermi_kp * (exp(β * E_kp) - exp(β * E_k)) / dEp
    end

    # Use small angle approximation when θ is near 0 or π
    km2 = (k - q)^2 + 2k * q * (1 - x)
    if x ≈ 1 || km2 ≤ 0
        jacob = θ
        km2 = (k - q)^2 + k * q * θ^2
    elseif x ≈ -1 || km2 ≤ 0
        jacob = π - θ
        km2 = (k + q)^2 - k * q * (π - θ)^2
    end
    E_km = Interp.interp1D(E_qp_kGgrid, kGgrid, √km2)
    dEm = E_k - E_km
    if dEm == 0
        return 0.0
    end
    # return jacob * fermi_k * (simple_pole(ω, dEp) - simple_pole(ω, dEm))
    # return jacob * fermi_k * (1.0 / dEm - 1.0 / dEp)
    return jacob * fermi_k * (dEp - dEm) / (dEm * dEp)
end

"""
Compute the quasiparticle polarization Π(q, iω) given the quasiparticle energy E_qp(k).
"""
function Π_qp(
    param::Parameter.Para,
    E_qp_kGgrid,
    kGgrid,
    Nk,
    maxK,
    minK,
    order,
    qgrid,
    bdlr;
    verbose=false,
    show_progress=false,
)
    # Static polarization
    @assert bdlr.n[1] == 0 "Static point missing from frequency grid!"
    verbose &&
        println_root("Computing static quasiparticle polarization bubble Π(q, iω = 0)...")
    timed_res = @timed Π_qp_static(
        param,
        E_qp_kGgrid,
        kGgrid,
        Nk,
        maxK,
        minK,
        order,
        qgrid,
        show_progress,
    )
    Π_qw_static = timed_res.value
    verbose && println_root("done")
    verbose && println_root(timed_result_to_string(timed_res))

    # Dynamic polarization
    verbose && println_root(
        "Computing dynamic quasiparticle polarization bubble Π(q, iω ≠ 0) with $(length(bdlr.n)) frequency points...",
    )
    timed_res = @timed Π_qp_dynamic(
        param,
        E_qp_kGgrid,
        kGgrid,
        Nk,
        maxK,
        minK,
        order,
        qgrid,
        bdlr,
        show_progress,
    )
    Π_qw_dynamic = timed_res.value
    verbose && println_root("done")
    verbose && println_root(timed_result_to_string(timed_res))

    # Return Π(iω, q) as a MeshArray
    return GreenFunc.MeshArray(
        ImFreq(bdlr),
        qgrid;
        data=[Π_qw_static; Π_qw_dynamic],
        dtype=ComplexF64,
    )
end

"""
Compute the quasiparticle polarization Π(q, iω) given the quasiparticle energy E_qp(k).
"""
function Π_qp_static(
    param::Parameter.Para,
    E_qp_kGgrid,
    kGgrid,
    Nk,
    maxK,
    minK,
    order,
    qgrid,
    show_progress=false,
)
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    # A larger momentum grid for internal integrations is needed to accurately
    # compute the static polarization and adequately reduce the error near q=0.
    Nk_integrate = 40
    order_integrate = 30
    # multiplier = 4
    # Nk_integrate = round(Int, multiplier * Nk)
    # order_integrate = round(Int, multiplier * order)

    @unpack kF, β, EF = param

    # Upper bound for adaptive kgrid size
    kgridmax = round(
        Int,
        1.1 * length(
            CompositeGrid.LogDensedGrid(
                :gauss,
                [0.0, maxK],
                [0.5 * maxK, kF],
                Nk_integrate,
                minK,
                order_integrate,
            ).grid,
        ),
    )

    θgrid = CompositeGrid.LogDensedGrid(
        :gauss,
        [0.0, π],
        [0.0, π],
        Nk_integrate,
        minK,
        order_integrate,
    )
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 40, minK, 30)

    # Initialize Π(0, q)
    Nq = length(qgrid.grid)
    Π_q_static_data = zeros(ComplexF64, 1, Nq)

    # Setup buffers for scatter/gather
    counts = split_count(Nq, comm_size)  # number of values per rank
    data_vbuffer = VBuffer(Π_q_static_data, counts)
    if rank == root
        length_ubuf = UBuffer(counts, 1)
        # For global indices
        qi_vbuffer = VBuffer(collect(1:Nq), counts)
    else
        length_ubuf = UBuffer(nothing)
        qi_vbuffer = VBuffer(nothing)
    end

    # Scatter the data to all ranks
    local_length = MPI.Scatter(length_ubuf, Int, root, comm)
    local_qi = MPI.Scatterv!(qi_vbuffer, zeros(Int, local_length), root, comm)
    local_data = MPI.Scatterv!(data_vbuffer, zeros(ComplexF64, local_length), root, comm)

    # Perform the momentum and angular integrations numerically
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress && rank == root,
    )
    local_kernel = Vector{ComplexF64}(undef, kgridmax)
    θ_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    for (i, qi) in enumerate(local_qi)
        # println("rank = $rank: Integrating (q, 0) point $i/$local_length")
        # Get external frequency and momentum at this index
        q = qgrid.grid[qi]
        # Build kgrid which is log-densed at q/2 and kF for each q in qgrid
        kgrid = Polarization.finitetemp_kgrid(
            q,
            kF,
            maxK / kF,
            Nk_integrate,
            minK / kF,
            order_integrate,
        )
        # Integrate over θ and k
        for (ki, k) in enumerate(kgrid.grid)
            for (θi, θ) in enumerate(θgrid.grid)
                θ_integrand[θi] =
                    angular_integrand_static(param, E_qp_kGgrid, kGgrid, k, q, θ)
            end
            # res = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
            # @assert isfinite(res) "fail local_kernel at $qi,$ki,$ni ($q,$k,$n) with $(res)\n$(θ_integrand[isnan.(θ_integrand)])"
            # local_kernel[ki] = res
            local_kernel[ki] = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
        end
        # integrand = local_kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid
        # res = CompositeGrids.Interp.integrate1D(integrand, kgrid)
        # @assert isfinite(res) "fail Δ at $qi,$ni"
        # local_data[i] = res
        local_data[i] = CompositeGrids.Interp.integrate1D(
            local_kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid,
            kgrid,
        )
        next!(progress_meter)
    end
    finish!(progress_meter)

    # Collect the results from all ranks
    MPI.Allgatherv!(local_data, data_vbuffer, comm)
    return Π_q_static_data /= (4 * π^2)
end

"""
Compute the quasiparticle polarization Π(q, iω) given the quasiparticle energy E_qp(k).
"""
function Π_qp_dynamic(
    param::Parameter.Para,
    E_qp_kGgrid,
    kGgrid,
    Nk,
    maxK,
    minK,
    order,
    qgrid,
    bdlr,
    show_progress=false,
)
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    # A larger momentum grid for internal integrations is needed to accurately
    # compute the static polarization and adequately reduce the error near q=0.
    Nk_integrate = 40
    order_integrate = 30
    # multiplier = 4
    # Nk_integrate = round(Int, multiplier * Nk)
    # order_integrate = round(Int, multiplier * order)

    @unpack kF, β, EF = param

    # Upper bound for adaptive kgrid size
    kgridmax = round(
        Int,
        1.1 * length(
            CompositeGrid.LogDensedGrid(
                :gauss,
                [0.0, maxK],
                [0.5 * maxK, kF],
                Nk,
                minK,
                order,
            ).grid,
        ),
    )

    θgrid = CompositeGrid.LogDensedGrid(
        :gauss,
        [0.0, π],
        [0.0, π],
        Nk_integrate,
        minK,
        order_integrate,
    )
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 30, minK, 20)

    # Initialize Π(iω, q)
    wngrid = bdlr.ωn[2:end]  # skip static point
    Nw, Nq = length(wngrid), length(qgrid.grid)
    Π_qw_data = zeros(ComplexF64, Nw, Nq)

    # Setup buffers for scatter/gather
    counts = split_count(Nw * Nq, comm_size)  # number of values per rank
    data_vbuffer = VBuffer(Π_qw_data, counts)
    if rank == root
        # Get global n and q indices for each data point
        indices = vec(CartesianIndices(Π_qw_data))
        w_indices = getindex.(indices, 1)
        q_indices = getindex.(indices, 2)
        length_ubuf = UBuffer(counts, 1)
        # For global indices
        ni_vbuffer = VBuffer(w_indices, counts)
        qi_vbuffer = VBuffer(q_indices, counts)
    else
        length_ubuf = UBuffer(nothing)
        ni_vbuffer = VBuffer(nothing)
        qi_vbuffer = VBuffer(nothing)
    end

    # Scatter the data to all ranks
    local_length = MPI.Scatter(length_ubuf, Int, root, comm)
    local_ni = MPI.Scatterv!(ni_vbuffer, zeros(Int, local_length), root, comm)
    local_qi = MPI.Scatterv!(qi_vbuffer, zeros(Int, local_length), root, comm)
    local_data = MPI.Scatterv!(data_vbuffer, zeros(ComplexF64, local_length), root, comm)

    # Perform the momentum and angular integrations numerically
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress && rank == root,
    )
    local_kernel = Vector{ComplexF64}(undef, kgridmax)
    θ_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    for (i, (ni, qi)) in enumerate(zip(local_ni, local_qi))
        # println("rank = $rank: Integrating (q, ω) point $i/$local_length")
        # Get external frequency and momentum at this index
        wn = wngrid[ni]
        q = qgrid.grid[qi]
        # Build kgrid which is log-densed at q/2 and kF for each q in qgrid
        kgrid = Polarization.finitetemp_kgrid(q, kF, maxK / kF, Nk, minK / kF, order)
        # Integrate over θ and k
        for (ki, k) in enumerate(kgrid.grid)
            for (θi, θ) in enumerate(θgrid.grid)
                θ_integrand[θi] =
                    angular_integrand_dynamic(param, E_qp_kGgrid, kGgrid, k, q, θ, wn)
            end
            # res = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
            # @assert isfinite(res) "fail local_kernel at $qi,$ki,$ni ($q,$k,$n) with $(res)\n$(θ_integrand[isnan.(θ_integrand)])"
            # local_kernel[ki] = res
            local_kernel[ki] = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
        end
        # integrand = local_kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid
        # res = CompositeGrids.Interp.integrate1D(integrand, kgrid)
        # @assert isfinite(res) "fail Δ at $qi,$ni"
        # local_data[i] = res
        local_data[i] = CompositeGrids.Interp.integrate1D(
            local_kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid,
            kgrid,
        )
        next!(progress_meter)
    end
    finish!(progress_meter)

    # Collect the results from all ranks
    MPI.Allgatherv!(local_data, data_vbuffer, comm)
    return Π_qw_data /= (4 * π^2)
end

"""
Compute the quasiparticle polarization Π(q, iω) given the quasiparticle energy E_qp(k).
"""
function Π_qp_serial(
    param::Parameter.Para,
    E_qp_kGgrid,
    kGgrid,
    Nk,
    maxK,
    minK,
    order,
    qgrid,
    bdlr,
    show_progress=false,
)
    @unpack kF, β, EF = param
    # # Build kgrids which are log-densed at q/2 and kF for each q in qgrid
    # kgrids = [
    #     Polarization.finitetemp_kgrid(q, kF, maxK / kF, Nk, minK / kF, order) for
    #     q in qgrid.grid
    # ]
    # kgridmax = maximum([kg.size for kg in kgrids])

    # Upper bound for adaptive kgrid size
    kgridmax = round(
        Int,
        1.1 * length(
            CompositeGrid.LogDensedGrid(
                :gauss,
                [0.0, maxK],
                [0.5 * maxK, kF],
                Nk,
                minK,
                order,
            ).grid,
        ),
    )

    wngrid = bdlr.ωn

    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 2 * Nk, minK, 2 * order)
    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], Nk, minK, order)

    # Get Π(q, ω) by integrating over k
    Π_qw_data = Array{ComplexF64}(undef, length(wngrid), length(qgrid.grid))

    # Evaluate the kernel p[qi, ki], performing the angular and momentum integrations numerically
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress && rank == root,
    )
    kernel = zeros(ComplexF64, kgridmax)
    θ_integrand = zeros(ComplexF64, length(θgrid.grid))
    for (ni, wn) in enumerate(wngrid)
        println("Integrating frequency $ni of $(length(wngrid))")
        for (qi, q) in enumerate(qgrid.grid)
            # println("qi=$qi of $(length(qgrid.grid))")
            # kgrid = kgrids[qi]
            # Build kgrid which is log-densed at q/2 and kF for each q in qgrid
            kgrid = Polarization.finitetemp_kgrid(q, kF, maxK / kF, Nk, minK / kF, order)
            E_qp_kgrid = Interp.interp1DGrid(E_qp_kGgrid, kGgrid, kgrid)
            # Integrate over θ and k
            for ki in eachindex(kgrid.grid)
                for (θi, θ) in enumerate(θgrid.grid)
                    θ_integrand[θi] =
                        angular_integrand_dynamic(param, E_qp_kgrid, kgrid, ki, q, θ, wn)
                end
                # res = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
                # @assert isfinite(res) "fail kernel at $qi,$ki,$ni ($q,$k,$n) with $(res)\n$(θ_integrand[isnan.(θ_integrand)])"
                # kernel[ki] = res
                kernel[ki] = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
            end
            # integrand = kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid
            # res = CompositeGrids.Interp.integrate1D(integrand, kgrid)
            # @assert isfinite(res) "fail Δ at $qi,$ni"
            # Π_qw_data[ni, qi] = res
            Π_qw_data[ni, qi] = CompositeGrids.Interp.integrate1D(
                kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid,
                kgrid,
            )
        end
        next!(progress_meter)
    end
    finish!(progress_meter)

    Π_qw_data /= (4 * π^2)
    Π_qw = GreenFunc.MeshArray(ImFreq(bdlr), qgrid; data=Π_qw_data, dtype=ComplexF64)
    return Π_qw
end

"""
Compute the non-interacting polarization Π₀(q, iω) given the non-interacting Green's function G₀.
"""
function Π0_serial(
    param::Parameter.Para,
    G0::GreenFunc.MeshArray,
    Nk,
    maxK,
    minK,
    order,
    qgrid,
    bdlr,
    show_progress=false,
)
    @unpack kF, β, EF = param
    kGgrid = G0.mesh[2]

    # Initial instant/dynamic self-energies are zero
    G0_dlr = to_dlr(G0)
    G0_imt = to_imtime(G0_dlr)
    G0_ins = dlr_to_imtime(G0_dlr, [β]) * (-1)
    Σ0 = zero(GreenFunc.MeshArray(G0.mesh[1], kGgrid; dtype=ComplexF64))
    Σ0_ins = zero(GreenFunc.MeshArray(G0_ins.mesh[1], kGgrid; dtype=ComplexF64))

    # Get quasiparticle energy
    E_qp_kGgrid = quasiparticle_energy(param, Σ0, Σ0_ins)

    # # Build kgrids which are log-densed at q/2 and kF for each q in qgrid
    # kgrids = [
    #     Polarization.finitetemp_kgrid(q, kF, maxK / kF, Nk, minK / kF, order) for
    #     q in qgrid.grid
    # ]
    # kgridmax = maximum([kg.size for kg in kgrids])

    # Upper bound for adaptive kgrid size
    kgridmax = round(
        Int,
        1.1 * length(
            CompositeGrid.LogDensedGrid(
                :gauss,
                [0.0, maxK],
                [0.5 * maxK, kF],
                Nk,
                minK,
                order,
            ).grid,
        ),
    )

    wngrid = bdlr.ωn

    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 2 * Nk, minK, 2 * order)
    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], Nk, minK, order)
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 20, minK, 14)
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 50, minK, 25)

    # Get Π(q, ω) by integrating over k
    Π_qw_data = Array{ComplexF64}(undef, length(wngrid), length(qgrid.grid))

    # Evaluate the kernel p[qi, ki], performing the angular and momentum integrations numerically
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress,
    )
    kernel = zeros(ComplexF64, kgridmax)
    θ_integrand = zeros(ComplexF64, length(θgrid.grid))
    for (ni, wn) in enumerate(wngrid)
        # println("Integrating frequency $ni of $(length(wngrid))")
        for (qi, q) in enumerate(qgrid.grid)
            # println("qi=$qi of $(length(qgrid.grid))")
            # kgrid = kgrids[qi]
            # Build kgrid which is log-densed at q/2 and kF for each q in qgrid
            kgrid = Polarization.finitetemp_kgrid(q, kF, maxK / kF, Nk, minK / kF, order)
            E_qp_kgrid = Interp.interp1DGrid(E_qp_kGgrid, kGgrid, kgrid)
            # Integrate over θ and k
            for ki in eachindex(kgrid.grid)
                for (θi, θ) in enumerate(θgrid.grid)
                    θ_integrand[θi] =
                        angular_integrand_dynamic(param, E_qp_kgrid, kgrid, ki, q, θ, wn)
                end
                # res = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
                # @assert isfinite(res) "fail kernel at $qi,$ki,$ni ($q,$k,$n) with $(res)\n$(θ_integrand[isnan.(θ_integrand)])"
                # kernel[ki] = res
                kernel[ki] = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
            end
            # integrand = kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid
            # res = CompositeGrids.Interp.integrate1D(integrand, kgrid)
            # @assert isfinite(res) "fail Δ at $qi,$ni"
            # Π_qw_data[ni, qi] = res
            Π_qw_data[ni, qi] = CompositeGrids.Interp.integrate1D(
                kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid,
                kgrid,
            )
        end
        next!(progress_meter)
    end
    finish!(progress_meter)
    Π_qw_data /= (4 * π^2)
    Π_qw = GreenFunc.MeshArray(ImFreq(bdlr), qgrid; data=Π_qw_data, dtype=ComplexF64)
    return Π_qw
end
