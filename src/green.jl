
"""
Compute the quasiparticle Green's function Gₖ(iωₙ) given Σ grid data.
"""
function G_qp(
    param::Parameter.Para,
    Σ::GreenFunc.MeshArray,
    Σ_ins::GreenFunc.MeshArray,
    kGgrid;
    maxKS=nothing,
)
    @unpack me, μ = param
    # Extract quasiparticle energy and Z-factor from the self-energy
    Z_kSgrid = zfactor_full(param, Σ)
    E_qp_kSgrid = quasiparticle_energy(param, Σ, Σ_ins)
    δμ = chemicalpotential(param, Σ, Σ_ins)

    # Interpolate the quasiparticle energy and Z-factor along the Green's
    # function momentum mesh, masking values where k > maxKS if applicable
    Z_k = []
    E_qp_k = []
    for k in kGgrid
        if isnothing(maxKS) == false && k > maxKS
            # G_qp → G0(μ + δμ) as k → ∞ (use hard cutoff at maxKS)
            Etilde_0 = k^2 / (2 * param.me) - (param.μ + δμ)
            push!(Z_k, 1.0)
            push!(E_qp_k, Etilde_0)
        else
            push!(Z_k, Interp.interp1D(Z_kSgrid, Σ.mesh[2], k))
            push!(E_qp_k, Interp.interp1D(E_qp_kSgrid, Σ.mesh[2], k))
        end
    end

    wnmesh = Σ.mesh[1]
    green = GreenFunc.MeshArray(wnmesh, kGgrid; dtype=ComplexF64)
    for ind in eachindex(green)
        iw, ik = ind[1], ind[2]
        green[ind] = Z_k[ik] / (im * wnmesh[iw] - E_qp_k[ik])
    end
    return green
end
