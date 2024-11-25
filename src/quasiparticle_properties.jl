
@inline function bare_energy(param::Parameter.Para, kgrid)
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
    Σ_static = Σ[w0_label, :] + Σ_ins[1, :]  # Σ(k, iω=0)

    xi_k = @. kgrid^2 / (2 * me) - μ
    @assert length(xi_k) == length(Σ_static)

    # NOTE: We need to use the momentum-dependent Z-factor to obtain the correct behavior at large momenta
    Zk = zfactor_full(param, Σ)
    E_qp_kgrid = @. Zk * (xi_k + real(Σ_static) - real(Σ_static[kf_label]))  # δμ = ReΣ(kF, iω=0)
    return E_qp_kgrid
end

@inline function E_0_grid(param::Parameter.Para, kGgrid)
    @unpack me, μ = param
    return kGgrid .^ 2 / (2 * me) .- μ
end

@inline function E_0_interp(param::Parameter.Para)
    @unpack me, μ = param
    return k -> k^2 / (2 * me) - μ
end

function E_qp_grid(
    param::Parameter.Para,
    Σ::GreenFunc.MeshArray,
    Σ_ins::GreenFunc.MeshArray,
    kGgrid,
)
    @unpack me, μ = param
    # Extract quasiparticle energy and Z-factor from the self-energy
    E_qp_kSgrid = quasiparticle_energy(param, Σ, Σ_ins)
    δμ = chemicalpotential(param, Σ, Σ_ins)
    # Interpolate the quasiparticle energy and Z-factor along the Green's
    # function momentum mesh, masking values where k is larger than the
    # largest k in the self-energy mesh
    E_qp_k = Vector{Float64}(undef, length(kGgrid))
    # E_qp_k = Vector{eltype(E_qp_kSgrid)}(undef, length(kGgrid))
    for (i, k) in enumerate(kGgrid)
        # G_qp → G0(μ + δμ) as k → ∞ (use hard cutoff at largest k in Σ)
        if k > maximum(Σ.mesh[2])
            E_qp_k[i] = k^2 / (2 * me) - (μ + δμ)
        else
            E_qp_k[i] = Interp.interp1D(E_qp_kSgrid, Σ.mesh[2], k)
        end
    end
    return E_qp_k
end

function E_qp_interp(
    param::Parameter.Para,
    Σ::GreenFunc.MeshArray,
    Σ_ins::GreenFunc.MeshArray,
    kGgrid,
)
    @unpack me, μ = param
    # Extract quasiparticle energy and Z-factor from the self-energy
    E_qp_kSgrid = quasiparticle_energy(param, Σ, Σ_ins)
    δμ = chemicalpotential(param, Σ, Σ_ins)
    # Interpolate the quasiparticle energy and Z-factor along the Green's
    # function momentum mesh, masking values where k is larger than the
    # largest k in the self-energy mesh
    E_qp_k = Vector{Float64}(undef, length(kGgrid))
    # E_qp_k = Vector{eltype(E_qp_kSgrid)}(undef, length(kGgrid))
    for (i, k) in enumerate(kGgrid)
        # G_qp → G0(μ + δμ) as k → ∞ (use hard cutoff at largest k in Σ)
        if k > maximum(Σ.mesh[2])
            E_qp_k[i] = k^2 / (2 * me) - (μ + δμ)
        else
            E_qp_k[i] = Interp.interp1D(E_qp_kSgrid, Σ.mesh[2], k)
        end
    end
    return k -> Interp.interp1D(E_qp_k, kGgrid, k)
end

"""
    function chemicalpotential(param, Σ::GreenFunc.MeshArray, Σ_ins::GreenFunc.MeshArray)

Calculate the chemical potential shift from the self-energy at the Fermi momentum,
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
    
Calculate the momentum-dependent Z-factor of the self-energy with improved finite-temperature scaling,
```math
    Z_k=\\frac{1}{1-\\frac{\\partial Im\\Sigma(k, 0^+)}{\\partial \\omega}}
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
    
Calculate the Z-factor of the self-energy at the momentum kamp with improved finite-temperature scaling,
```math
    Z_k=\\frac{1}{1-\\frac{\\partial Im\\Sigma(k, 0^+)}{\\partial \\omega}}
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
    function dfactor(param, Σ::GreenFunc.MeshArray, δK=5e-6; kamp=param.kF)
    
Calculate the D-factor of the self-energy at the momentum kamp,
```math
    D_k=\\left(1+\\frac{m}{k}\\frac{\\partial Re\\Sigma(k, 0)}{\\partial k}\\right)^{-1}
```
"""
function dfactor(
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
    kamp = Σ.mesh[2][k_label]

    k1, k2 = k_label, k_label + 1
    while abs(kgrid[k2] - kgrid[k1]) < δK
        k2 += 1
    end

    Σ_static1 = real(Σ[w0_label, k1] + Σ_ins[1, k1])
    Σ_static2 = real(Σ[w0_label, k2] + Σ_ins[1, k2])
    ds_dk = (Σ_static1 - Σ_static2) / (kgrid[k1] - kgrid[k2])
    D0 = 1 + (me / kamp) * ds_dk
    return D0
end

"""
    function massratio(param, Σ::GreenFunc.MeshArray, Σ_ins::GreenFunc.MeshArray, δK=5e-6; kamp=param.kF)
    
Calculate the effective mass of the self-energy at the momentum kamp,
```math
    \\frac{m^*_k}{m}=\\frac{1}{Z_k} \\cdot \\left(1+\\frac{m}{k}\\frac{\\partial Re\\Sigma(k, 0)}{\\partial k}\\right)^{-1}
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

    return 1.0 / z / (1 + (me / kamp) * ds_dk), kamp
end

function get_lqsgw_properties(
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
    # No-op at rs = 0
    if param.rs == 0.0
        i_step = 0
        converged = true
        meff = 1.0
        zfactor = 1.0
        dmu = 0.0
    else
        Σ, Σ_ins, i_step, converged = Σ_LQSGW(
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
        #@assert converged "LQSGW loop did not converge!"
        if converged == false
            println_root("LQSGW loop did not converge!")
        end
        meff = massratio(param, Σ, Σ_ins, δK)[1]
        zfactor = zfactor_fermi(param, Σ)
        dmu = chemicalpotential(param, Σ, Σ_ins)
    end
    return (; atol, alpha, i_step, converged, meff, zfactor, dmu)
end

function get_g0w0_properties(
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
    # No-op at rs = 0
    if param.rs == 0.0
        i_step = 0
        converged = true
        meff = 1.0
        zfactor = 1.0
        dmu = 0.0
    else
        Σ, Σ_ins = Σ_G0W0(
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
        meff = massratio(param, Σ, Σ_ins, δK)[1]
        zfactor = zfactor_fermi(param, Σ)
        dmu = chemicalpotential(param, Σ, Σ_ins)
    end
    return (; meff, zfactor, dmu)
end
