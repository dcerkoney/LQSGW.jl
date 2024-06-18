
"""Single G0 term with Taylor expansion for low and high frequencies."""
function _simple_pole(ω, ϵ)
    if ω == 0
        return -1 / ϵ
    end
    # Taylor expand for |ϵ / ω| ≪ 1 and |ϵ / ω| ≫ 1
    if abs(ϵ / ω) < 1e-6
        return (-1 + (ϵ / ω)^2) * ϵ / ω^2
    elseif abs(ω / ϵ) < 1e-6
        return (-1 + (ω / ϵ)^2) / ϵ
    else
        return real(1.0 / (im * ω - ϵ))
    end
end

"""
Angular integrand for the quasiparticle polarization bubble.
"""
function _angular_integrand(param::Parameter.Para, E_qp, k, q, θ, ω)
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
    res = jacob * fermi_k * (_simple_pole(ω, dEp) - _simple_pole(ω, dEm))
    return res
end

"""
Compute the quasiparticle polarization Π(q, iω) given the quasiparticle energy E_qp(k).
"""
function Π_qp(param::Parameter.Para, E_qp, Nk, maxK, minK, order, qgrid, bdlr)
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    @unpack kF, β, EF = param
    # Build kgrids which are log-densed at q/2 and kF for each q in qgrid
    kgrids = [
        Polarization.finitetemp_kgrid(q, kF, maxK / kF, Nk, minK / kF, order) for
        q in qgrid.grid
    ]
    kgridmax = maximum([kg.size for kg in kgrids])

    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 2 * Nk, minK, 2 * order)
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], Nk, minK, order)

    # Initialize Π(iω, q)
    Nw, Nq = length(bdlr.n), length(qgrid.grid)
    Π_qw_data = zeros(ComplexF64, Nw, Nq)

    # Setup buffers for scatter/gather
    counts = _split_count(Nw * Nq, comm_size)  # number of values per rank
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

    # Perform the angular integration and frequency sum numerically
    # local_kernel = zeros(ComplexF64, kgridmax)
    # θ_integrand = zeros(ComplexF64, length(θgrid.grid))
    for (i, (ni, qi)) in enumerate(zip(local_ni, local_qi))
        # println("rank = $rank: Integrating (q, ω) point $i/$local_length")
        # println("rank = $rank: Integrating (q, ω) point $i/$local_length")
        # Get external frequency and momentum at this index
        n = bdlr.n[ni]
        q = qgrid.grid[qi]
        # Integrate over θ and k
        kgrid = kgrids[qi]
        local_kernel = zeros(ComplexF64, kgridmax)
        for (ki, k) in enumerate(kgrids[qi].grid)
            θ_integrand = zeros(ComplexF64, length(θgrid.grid))
            for (θi, θ) in enumerate(θgrid.grid)
                θ_integrand[θi] = _angular_integrand(param, E_qp, k, q, θ, 2π * n / β)
            end
            res = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
            @assert isfinite(res) "fail local_kernel at $qi,$ki,$ni ($q,$k,$n) with $(res)\n$(θ_integrand[isnan.(θ_integrand)])"
            local_kernel[ki] = res
        end
        integrand = local_kernel[1:(kgrid.size)] .* kgrid.grid .* kgrid.grid
        res = CompositeGrids.Interp.integrate1D(integrand, kgrid)
        @assert isfinite(res) "fail Δ at $qi,$ni"
        local_data[i] = res
    end

    # Collect the results from all ranks
    MPI.Allgatherv!(local_data, data_vbuffer, comm)

    # Return Π(iω, q) as a MeshArray
    Π_qw_data /= (4 * π^2)
    return GreenFunc.MeshArray(ImFreq(bdlr), qgrid; data=Π_qw_data, dtype=ComplexF64)
end

"""
Compute the quasiparticle polarization Π(q, iω) given the quasiparticle energy E_qp(k).
"""
function Π_qp_serial(param::Parameter.Para, E_qp, Nk, maxK, minK, order, qgrid, bdlr)
    @unpack kF, β, EF = param
    # Build kgrids which are log-densed at q/2 and kF for each q in qgrid
    kgrids = [
        Polarization.finitetemp_kgrid(q, kF, maxK / kF, Nk, minK / kF, order) for
        q in qgrid.grid
    ]
    kgridmax = maximum([kg.size for kg in kgrids])

    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 2 * Nk, minK, 2 * order)
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], Nk, minK, order)

    # Evaluate the kernel p[qi, ki], performing the angular integration and frequency sum numerically
    kernel = Array{ComplexF64}(undef, length(qgrid.grid), (kgridmax), length(bdlr.n))
    θ_integrand = zeros(ComplexF64, length(θgrid.grid))
    for (ni, n) in enumerate(bdlr.n)
        println("Integrating frequency $ni of $(length(bdlr.n))")
        for (qi, q) in enumerate(qgrid.grid)
            # println("qi=$qi of $(length(qgrid.grid))")
            for (ki, k) in enumerate(kgrids[qi].grid)
                for (θi, θ) in enumerate(θgrid.grid)
                    θ_integrand[θi] = _angular_integrand(param, E_qp, k, q, θ, 2π * n / β)
                end
                res = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
                @assert isfinite(res) "fail kernel at $qi,$ki,$ni ($q,$k,$n) with $(res)\n$(θ_integrand[isnan.(θ_integrand)])"
                kernel[qi, ki, ni] += res
            end
        end
    end
    # Get Π(q, ω) by integrating over k
    Π_qw_data = Array{ComplexF64}(undef, length(bdlr.n), length(qgrid.grid))
    for ni in eachindex(bdlr.n)
        for qi in eachindex(qgrid.grid)
            kgrid = kgrids[qi]
            integrand = kernel[qi, 1:(kgrid.size), ni] .* kgrid.grid .* kgrid.grid
            res = CompositeGrids.Interp.integrate1D(integrand, kgrid)
            @assert isfinite(res) "fail Δ at $qi,$ni"
            Π_qw_data[ni, qi] = res
        end
    end
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
    E_qp_kgrid = quasiparticle_energy(param, Σ0, Σ0_ins)
    E_qp = k -> CompositeGrids.Interp.interp1D(E_qp_kgrid, kGgrid, k)

    # Build kgrids which are log-densed at q/2 and kF for each q in qgrid
    kgrids = [
        Polarization.finitetemp_kgrid(q, kF, maxK / kF, Nk, minK / kF, order) for
        q in qgrid.grid
    ]
    kgridmax = maximum([kg.size for kg in kgrids])

    θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 2 * Nk, minK, 2 * order)
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 20, minK, 14)
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 50, minK, 25)

    # Evaluate the kernel p[qi, ki], performing the angular integration and frequency sum numerically
    kernel = Array{ComplexF64}(undef, length(qgrid.grid), (kgridmax), length(bdlr.n))
    θ_integrand = zeros(ComplexF64, length(θgrid.grid))
    for (ni, n) in enumerate(bdlr.n)
        # println("Integrating frequency $ni of $(length(bdlr.n))")
        for (qi, q) in enumerate(qgrid.grid)
            # println("qi=$qi of $(length(qgrid.grid))")
            for (ki, k) in enumerate(kgrids[qi].grid)
                for (θi, θ) in enumerate(θgrid.grid)
                    θ_integrand[θi] = _angular_integrand(param, E_qp, k, q, θ, 2π * n / β)
                end
                res = CompositeGrids.Interp.integrate1D(θ_integrand, θgrid)
                # θi == 1 && println("$qi, $ki, $ni  $(res)")
                @assert isfinite(res) "fail kernel at $qi,$ki,$ni ($q,$k,$n) with $(res)\n$(θ_integrand[isnan.(θ_integrand)])"
                kernel[qi, ki, ni] += res
            end
        end
    end

    # Get Π(q, ω) by integrating over k
    Π_qw_data = Array{ComplexF64}(undef, length(bdlr.n), length(qgrid.grid))
    for (ni, n) in enumerate(bdlr.n)
        for (qi, q) in enumerate(qgrid.grid)
            kgrid = kgrids[qi]
            integrand = kernel[qi, 1:(kgrid.size), ni] .* kgrid.grid .* kgrid.grid
            res = CompositeGrids.Interp.integrate1D(integrand, kgrid)
            @assert isfinite(res) "fail Δ at $qi,$ni"
            Π_qw_data[ni, qi] = res
        end
    end
    Π_qw_data /= (4 * π^2)
    Π_qw = GreenFunc.MeshArray(ImFreq(bdlr), qgrid; data=Π_qw_data, dtype=ComplexF64)
    return Π_qw
end
