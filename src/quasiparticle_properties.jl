
function bare_energy(param::Parameter.Para, kgrid)
    @unpack me, μ = param
    return @. kgrid^2 / (2 * me) - μ
end

"""
Compute the linearized quasiparticle dispersion 𝓔ₖ given Σ grid data.
"""
function quasiparticle_energy(
    param::Parameter.Para,
    Σ::GreenFunc.MeshArray,
    Σ_ins::GreenFunc.MeshArray,
)
    @assert Σ.mesh[1] isa ImFreq "Σ must be dynamic self-energy data in the Matsubara frequency domain!"
    @assert 0 in Σ.mesh[1].grid "Static component not found in Σ MeshArray!"
    @unpack me, kF, β, μ = param

    @assert length(Σ_ins.mesh[1]) == 1
    kgrid = Σ.mesh[2]

    if Σ.mesh[1].grid == [0]
        w0_label = 1
    else
        w0_label = locate(Σ.mesh[1], 0)
    end
    kf_label = locate(kgrid, kF)
    Σ_static = Σ[w0_label, :] + Σ_ins[1, :]                     # Σ(k, iω=0)
    δμ = real(Σ[w0_label, kf_label] + Σ_ins[1, kf_label])  # δμ = ReΣ(kF, iω=0)

    xi_k = @. kgrid^2 / (2 * me) - μ
    @assert length(xi_k) == length(Σ_static)

    # NOTE: We need to use the momentum-dependent Z-factor to obtain the correct behavior at large momenta
    Zk = zfactor_full(param, Σ)

    E_qp_kgrid = @. Zk * (xi_k + real(Σ_static) - δμ)
    # return k -> Interp.interp1D(E_qp_kgrid, kgrid, k)
    return E_qp_kgrid
end

function E_0_interp(param::Parameter.Para)
    @unpack me, μ = param
    return k -> k^2 / (2 * me) - μ
end

function E_qp_interp(
    param::Parameter.Para,
    Σ::GreenFunc.MeshArray,
    Σ_ins::GreenFunc.MeshArray,
    kGgrid;
    maxKS=nothing,
)
    @unpack me, μ = param
    # Extract quasiparticle energy and Z-factor from the self-energy
    E_qp_kSgrid = quasiparticle_energy(param, Σ, Σ_ins)
    δμ = chemicalpotential(param, Σ, Σ_ins)

    # Interpolate the quasiparticle energy and Z-factor along the Green's
    # function momentum mesh, masking values where k > maxKS if applicable
    E_qp_k = []
    for k in kGgrid
        if isnothing(maxKS) == false && k > maxKS
            # G_qp → G0(μ + δμ) as k → ∞ (use hard cutoff at maxKS)
            Etilde_0 = k^2 / (2 * param.me) - (param.μ + δμ)
            push!(E_qp_k, Etilde_0)
        else
            push!(E_qp_k, Interp.interp1D(E_qp_kSgrid, Σ.mesh[2], k))
        end
    end
    # return E_qp_k
    return k -> Interp.interp1D(E_qp_k, kGgrid, k)
end

"""
    function chemicalpotential(param, Σ::GreenFunc.MeshArray, Σ_ins::GreenFunc.MeshArray)

Calculate the chemical potential from the self-energy at the Fermi momentum,
```math
    δμ = ReΣ(kF, iω=0)
```
"""
function chemicalpotential(param, Σ::GreenFunc.MeshArray, Σ_ins::GreenFunc.MeshArray)
    @assert Σ.mesh[1] isa ImFreq "Σ must be dynamic self-energy data in the Matsubara frequency domain!"
    @assert 0 in Σ.mesh[1].grid "Static component not found in Σ MeshArray!"
    # one can achieve ~1e-5 accuracy with δK = 5e-6
    @unpack kF, me = param

    if Σ.mesh[1].grid == [0]
        w0_label = 1
    else
        w0_label = locate(Σ.mesh[1], 0)
    end

    kf_label = locate(Σ.mesh[2], kF)
    return real(Σ[w0_label, kf_label] + Σ_ins[1, kf_label])  # δμ = ReΣ(kF, iω=0)
end

"""
    function zfactor_full(param, Σ::GreenFunc.MeshArray)
    
Calculate the momentum-dependent z-factor of the self-energy with improved finite-temperature scaling,
```math
    z_k=\\frac{1}{1-\\frac{\\partial Im\\Sigma(k, 0^+)}{\\partial \\omega}}
```
"""
function zfactor_full(param, Σ::GreenFunc.MeshArray)
    @assert Σ.mesh[1] isa ImFreq "Σ must be dynamic self-energy data in the Matsubara frequency domain!"
    @assert 0 in Σ.mesh[1].grid "Static component not found in Σ MeshArray!"
    @unpack kF, β = param

    if Σ.mesh[1].grid == [0]
        w0_label = 1
    else
        w0_label = locate(Σ.mesh[1], 0)
    end

    ΣI = imag(Σ[w0_label, :])
    ds_dw = ΣI / π * β

    Zk = @. 1 / (1 - ds_dw)
    return Zk
end

"""
    function zfactor_fermi(param, Σ::GreenFunc.MeshArray; kamp=param.kF)
    
Calculate the z-factor of the self-energy at the momentum kamp with improved finite-temperature scaling,
```math
    z_k=\\frac{1}{1-\\frac{\\partial Im\\Sigma(k, 0^+)}{\\partial \\omega}}
```
"""
function zfactor_fermi(param, Σ::GreenFunc.MeshArray; kamp=param.kF)
    @assert Σ.mesh[1] isa ImFreq "Σ must be dynamic self-energy data in the Matsubara frequency domain!"
    @assert 0 in Σ.mesh[1].grid "Static component not found in Σ MeshArray!"
    @unpack kF, β = param

    if Σ.mesh[1].grid == [0]
        w0_label = 1
    else
        w0_label = locate(Σ.mesh[1], 0)
    end

    kF_label = locate(Σ.mesh[2], kamp)
    kamp = Σ.mesh[2][kF_label]

    ΣI = imag(Σ[w0_label, kF_label])
    ds_dw = ΣI[1] / π * β

    Z0 = 1 / (1 - ds_dw)
    return Z0
end

"""
    function massratio(param, Σ::GreenFunc.MeshArray, Σ_ins::GreenFunc.MeshArray, δK=5e-6; kamp=param.kF)
    
Calculate the effective mass of the self-energy at the momentum kamp,
```math
    \\frac{m^*_k}{m}=\\frac{1}{z_k} \\cdot \\left(1+\\frac{m}{k}\\frac{\\partial Re\\Sigma(k, 0)}{\\partial k}\\right)^{-1}
```
"""
function massratio(
    param,
    Σ::GreenFunc.MeshArray,
    Σ_ins::GreenFunc.MeshArray,
    δK=5e-6;
    kamp=param.kF,
)
    @assert Σ.mesh[1] isa ImFreq "Σ must be dynamic self-energy data in the Matsubara frequency domain!"
    @assert 0 in Σ.mesh[1].grid "Static component not found in Σ MeshArray!"
    # one can achieve ~1e-5 accuracy with δK = 5e-6
    @unpack kF, me = param

    δK *= kF
    if Σ.mesh[1].grid == [0]
        w0_label = 1
    else
        w0_label = locate(Σ.mesh[1], 0)
    end
    k_label = locate(Σ.mesh[2], kamp)

    kgrid = Σ.mesh[2]
    kamp = kgrid[k_label]
    z = zfactor_fermi(param, Σ; kamp=kamp)

    k1, k2 = k_label, k_label + 1
    while abs(kgrid[k2] - kgrid[k1]) < δK
        k2 += 1
    end

    # @assert kF < kgrid.grid[k1] < kgrid.grid[k2] "k1 and k2 are not on the same side! It breaks $kF > $(kgrid.grid[k1]) > $(kgrid.grid[k2])"
    sigma1 = real(Σ[w0_label, k1] + Σ_ins[1, k1])
    sigma2 = real(Σ[w0_label, k2] + Σ_ins[1, k2])
    ds_dk = (sigma1 - sigma2) / (kgrid[k1] - kgrid[k2])

    return 1.0 / z / (1 + me / kamp * ds_dk), kamp
end
