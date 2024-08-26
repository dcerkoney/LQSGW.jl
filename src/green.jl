"""
Compute the noninteracting Green's function G⁰ₖ(iωₙ) on a given momentum grid.
By default (`symmetry = :sym`), uses a DLR Matsubara grid that is symmetric around n = -1.
"""
function G_0(param::Parameter.Para, Euv, rtol, xGgrid; symmetry=:sym)
    @unpack me, β, μ, kF = param
    wnmesh = GreenFunc.ImFreq(β, FERMION; Euv=Euv, rtol=rtol, symmetry=symmetry)
    green = GreenFunc.MeshArray(wnmesh.grid, xGgrid; dtype=ComplexF64)  # G_0(x, n)
    for ind in eachindex(green)
        iw, ik = ind[1], ind[2]
        kGgrid = kF * xGgrid[ik]
        green[ind] = 1 / (im * wnmesh[iw] - (kGgrid^2 / (2 * me) - μ))
    end
    return green
end

"""
Compute the quasiparticle Green's function Gₖ(iωₙ) given Σ grid data.
"""
function G_qp(
    param::Parameter.Para,
    Σ::GreenFunc.MeshArray,
    Σ_ins::GreenFunc.MeshArray,
    xGgrid,
)
    @unpack me, β, μ, kF = param
    # Extract quasiparticle energy and Z-factor from the self-energy
    Z_xSgrid = zfactor_full(param, Σ)
    E_qp_xSgrid = quasiparticle_energy(param, Σ, Σ_ins)
    δμ = chemicalpotential(param, Σ, Σ_ins)
    # Interpolate the quasiparticle energy and Z-factor along the Green's
    # function momentum mesh, masking values where k is larger than the
    # largest k in the self-energy mesh
    Z_k = Vector{Float64}(undef, length(xGgrid))
    E_qp_k = Vector{Float64}(undef, length(xGgrid))
    for (i, x) in enumerate(xGgrid)
        # G_qp → G0(μ + δμ) as x → ∞ (use hard cutoff at largest x in Σ)
        if x > maximum(Σ.mesh[2])
            k = x * kF
            Z_k[i] = 1.0
            E_qp_k[i] = k^2 / (2 * me) - (μ + δμ)
        else
            Z_k[i] = Interp.interp1D(Z_xSgrid, Σ.mesh[2], x)
            E_qp_k[i] = Interp.interp1D(E_qp_xSgrid, Σ.mesh[2], x)
        end
    end
    # Build the Green's function MeshArray
    wnmesh = Σ.mesh[1] * π / β
    green = GreenFunc.MeshArray(Σ.mesh[1], xGgrid; dtype=ComplexF64)  # G_qp(x, n)
    for ind in eachindex(green)
        iw, ik = ind[1], ind[2]
        green[ind] = Z_k[ik] / (im * wnmesh[iw] - E_qp_k[ik])
    end
    return green
end

"""
Compute the quasiparticle Green's function Gₖ(iωₙ) given Z, E_qp, and δμ.
"""
function G_qp(
    param::Parameter.Para,
    Z_xSgrid::T,
    E_qp_xSgrid::T,
    xGgrid::T,
    δμ,
) where {T<:AbstractVector}
    @unpack me, μ, β, kF = param
    @assert length(Z_xSgrid) == length(E_qp_xSgrid) "Z_xSgrid and E_qp_xSgrid must have the same lengths!"
    # Interpolate the quasiparticle energy and Z-factor along the Green's
    # function momentum mesh, masking values where k is larger than the
    # largest k in the self-energy mesh
    Z_k = Vector{Float64}(undef, length(xGgrid))
    E_qp_k = Vector{Float64}(undef, length(xGgrid))
    for (i, x) in enumerate(xGgrid)
        # G_qp → G0(μ + δμ) as x → ∞ (use hard cutoff at largest x in Σ)
        if x > maximum(Σ.mesh[2])
            k = x * kF
            Z_k[i] = 1.0
            E_qp_k[i] = k^2 / (2 * me) - (μ + δμ)
        else
            Z_k[i] = Interp.interp1D(Z_xSgrid, Σ.mesh[2], x)
            E_qp_k[i] = Interp.interp1D(E_qp_xSgrid, Σ.mesh[2], x)
        end
    end
    # Build the Green's function MeshArray
    wnmesh = Σ.mesh[1] * π / β
    green = GreenFunc.MeshArray(Σ.mesh[1], xGgrid; dtype=ComplexF64)
    for ind in eachindex(green)
        iw, ik = ind[1], ind[2]
        green[ind] = Z_k[ik] / (im * wnmesh[iw] - E_qp_k[ik])
    end
    return green
end
