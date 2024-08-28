"""
Compute the noninteracting Green's function G⁰ₖ(iωₙ) on a given momentum grid.
"""
function G_0(param::Parameter.Para, fdlr, kGgrid)
    @unpack me, β, μ, kF = param
    wnmesh = ImFreq(fdlr)
    green = GreenFunc.MeshArray(wnmesh, kGgrid; dtype=ComplexF64)
    for ind in eachindex(green)
        iw, ik = ind[1], ind[2]
        green[ind] = 1 / (im * wnmesh[iw] - (kGgrid[ik]^2 / (2 * me) - μ))
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
    kGgrid,
)
    @unpack me, β, μ, kF = param
    # Extract quasiparticle energy and Z-factor from the self-energy
    Z_kSgrid = zfactor_full(param, Σ)
    E_qp_kSgrid = quasiparticle_energy(param, Σ, Σ_ins)
    δμ = chemicalpotential(param, Σ, Σ_ins)
    # Interpolate the quasiparticle energy and Z-factor along the Green's
    # function momentum mesh, masking values where k is larger than the
    # largest k in the self-energy mesh
    Z_k = Vector{Float64}(undef, length(kGgrid))
    E_qp_k = Vector{Float64}(undef, length(kGgrid))
    for (i, k) in enumerate(kGgrid)
        # G_qp → G0(μ + δμ) as k → ∞ (use hard cutoff at largest k in Σ)
        if k > maximum(Σ.mesh[2])
            Z_k[i] = 1.0
            E_qp_k[i] = k^2 / (2 * me) - (μ + δμ)
        else
            Z_k[i] = Interp.interp1D(Z_kSgrid, Σ.mesh[2], k)
            E_qp_k[i] = Interp.interp1D(E_qp_kSgrid, Σ.mesh[2], k)
        end
    end
    # Build the Green's function MeshArray
    wnmesh = Σ.mesh[1]
    green = GreenFunc.MeshArray(wnmesh, kGgrid; dtype=ComplexF64)  # G_qp(x, n)
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
    Z_kSgrid::T,
    E_qp_kSgrid::T,
    kGgrid::T,
    δμ,
) where {T<:AbstractVector}
    @unpack me, μ, β, kF = param
    @assert length(Z_kSgrid) == length(E_qp_kSgrid) "Z_kSgrid and E_qp_kSgrid must have the same lengths!"
    # Interpolate the quasiparticle energy and Z-factor along the Green's
    # function momentum mesh, masking values where k is larger than the
    # largest k in the self-energy mesh
    Z_k = Vector{Float64}(undef, length(kGgrid))
    E_qp_k = Vector{Float64}(undef, length(kGgrid))
    for (i, k) in enumerate(kGgrid)
        # G_qp → G0(μ + δμ) as k → ∞ (use hard cutoff at largest k in Σ)
        if k > maximum(Σ.mesh[2])
            Z_k[i] = 1.0
            E_qp_k[i] = k^2 / (2 * me) - (μ + δμ)
        else
            Z_k[i] = Interp.interp1D(Z_kSgrid, Σ.mesh[2], k)
            E_qp_k[i] = Interp.interp1D(E_qp_kSgrid, Σ.mesh[2], k)
        end
    end
    # Build the Green's function MeshArray
    wnmesh = Σ.mesh[1]
    green = GreenFunc.MeshArray(wnmesh, kGgrid; dtype=ComplexF64)
    for ind in eachindex(green)
        iw, ik = ind[1], ind[2]
        green[ind] = Z_k[ik] / (im * wnmesh[iw] - E_qp_k[ik])
    end
    return green
end
 