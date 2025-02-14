using Colors
using CompositeGrids
using ElectronGas
using ElectronLiquid
using GreenFunc
# using JLD2
using LinearAlgebra
using Lehmann
using LQSGW
using MPI
using Parameters
using ProgressMeter
using PyCall
using PyPlot
using Roots

import LQSGW: split_count, println_root, timed_result_to_string

@pyimport numpy as np   # for saving/loading numpy data
@pyimport scienceplots  # for style "science"
@pyimport scipy.interpolate as interp

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
    "black" => "black",
    "orange" => "#EE7733",
    "blue" => "#0077BB",
    "cyan" => "#33BBEE",
    "magenta" => "#EE3377",
    "red" => "#CC3311",
    "teal" => "#009988",
    "grey" => "#BBBBBB",
]);
style = PyPlot.matplotlib."style"
style.use(["science", "std-colors"])
const color = [
    "black",
    cdict["orange"],
    cdict["blue"],
    cdict["cyan"],
    cdict["magenta"],
    cdict["red"],
    cdict["teal"],
    cdict["grey"],
]
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 16
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "Times New Roman"

const alpha_ueg = (4 / 9π)^(1 / 3)

# Specify the type of momentum and frequency (index) grids explicitly to ensure type stability
const MomInterpGridType = CompositeGrids.CompositeG.Composite{
    Float64,
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound},
    CompositeGrids.CompositeG.Composite{
        Float64,
        CompositeGrids.SimpleG.Log{Float64},
        CompositeGrids.SimpleG.Uniform{Float64,CompositeGrids.SimpleG.ClosedBound},
    },
}
const MomGridType = CompositeGrids.CompositeG.Composite{
    Float64,
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound},
    CompositeGrids.CompositeG.Composite{
        Float64,
        CompositeGrids.SimpleG.Log{Float64},
        CompositeGrids.SimpleG.GaussLegendre{Float64},
    },
}
const MGridType = CompositeGrids.SimpleG.Arbitrary{Int64,CompositeGrids.SimpleG.ClosedBound}
const FreqGridType =
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound}
const AngularGridType = CompositeGrids.CompositeG.Composite{
    Float64,
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound},
    CompositeGrids.CompositeG.Composite{
        Float64,
        CompositeGrids.SimpleG.Log{Float64},
        CompositeGrids.SimpleG.GaussLegendre{Float64},
    },
}

@with_kw mutable struct OneLoopParams
    # UEG parameters
    beta::Float64
    rs::Float64
    dim::Int = 3
    spin::Int = 2
    Fs::Float64 = -0.0

    # mass2::Float64 = 1.0      # large Yukawa screening λ for testing
    mass2::Float64 = 1e-5     # fictitious Yukawa screening λ
    massratio::Float64 = 1.0  # mass ratio m*/m

    basic::Parameter.Para = Parameter.rydbergUnit(1.0 / beta, rs, dim; Λs=mass2, spin=spin)
    paramc::ParaMC =
        UEG.ParaMC(; rs=rs, beta=beta, dim=dim, spin=spin, mass2=mass2, Fs=Fs, basic=basic)
    kF::Float64 = basic.kF
    EF::Float64 = basic.EF
    β::Float64 = basic.β
    me::Float64 = basic.me
    ϵ0::Float64 = basic.ϵ0
    e0::Float64 = basic.e0
    μ::Float64 = basic.μ
    NF::Float64 = basic.NF
    NFstar::Float64 = basic.NF * massratio
    qTF::Float64 = basic.qTF
    fs::Float64 = Fs / NF

    # Momentum grid parameters
    maxK::Float64 = 6 * kF
    maxQ::Float64 = 6 * kF
    Q_CUTOFF::Float64 = 1e-10 * kF

    # nk ≈ 75 is sufficiently converged for all relevant euv/rtol
    Nk::Int = 7
    Ok::Int = 6

    # na ≈ 75 is sufficiently converged for all relevant euv/rtol
    Na::Int = 8
    Oa::Int = 7

    euv::Float64 = 1000.0
    rtol::Float64 = 1e-7

    # We precompute R(q, iνₘ) on a mesh of ~100 k-points
    # NOTE: EL.jl default is `Nk, Ok = 16, 16` (~700 k-points)
    qgrid_interp::MomInterpGridType =
        CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 10, 0.01 * kF, 10)  # sufficient for 1-decimal accuracy
    # CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 12, 0.01 * kF, 12)
    # CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 16, 0.01 * kF, 16)
    # CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], Nk, 0.01 * kF, Ok)

    # Later, we integrate R(q, iνₘ) on a Gaussian mesh of ~100 k-points
    qgrid::MomGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxQ], [0.0, 2 * kF], Nk, 0.01 * kF, Ok)
    # CompositeGrid.LogDensedGrid(:gauss, [0.0, maxQ], [0.0, 2 * kF], 16, 0.01 * kF, 16)

    # Sparse angular grids (~100 points each)
    # NOTE: EL.jl default is `Na, Oa = 16, 32` (~1000 θ/φ-points)
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 1e-6, 32)
    # φgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], 16, 1e-6, 32)
    θgrid::AngularGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], Na, 0.01, Oa)
    # CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], Na, 1e-6, Oa)
    # CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 1e-6, 16)
    φgrid::AngularGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], Na, 0.01, Oa)
    # CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], Na, 1e-6, Oa)
    # CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], 16, 1e-6, 16)

    # Use a sparse DLR grid for the bosonic Matsubara summation (~30-50 iνₘ-points)
    dlr::DLRGrid{Float64,:ph} =
        DLRGrid(; Euv=euv * EF, β=β, rtol=rtol, isFermi=false, symmetry=:ph)
    mgrid::MGridType = SimpleG.Arbitrary{Int64}(dlr.n)
    vmgrid::FreqGridType = SimpleG.Arbitrary{Float64}(dlr.ωn)
    Mmax::Int64 = maximum(mgrid)

    # Incoming momenta k1, k2 and incident scattering angle
    kamp1::Float64 = basic.kF
    kamp2::Float64 = basic.kF
    θ12::Float64 = π / 2

    # Lowest non-zero Matsubara frequencies
    iw0 = im * π / β  # fermionic
    iv1 = im * 2π / β  # bosonic

    # R grid data is precomputed in an initialization step
    initialized::Bool   = false
    dR::Matrix{Float64} = Matrix{Float64}(undef, length(qgrid_interp), length(mgrid))
    R::Matrix{Float64}  = Matrix{Float64}(undef, length(qgrid_interp), length(mgrid))
end

function spline(x, y, e; xmin=0.0, xmax=x[end])
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(xmin, xmax, 1000))
    yfit = spl(__x)
    return __x, yfit
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via Corradini's fit
to the DMC compressibility enhancement [doi: 10.1103/PhysRevB.57.14569].
"""
@inline function get_Fs(basic_param::Parameter.Para)
    kappa0_over_kappa = Interaction.compressibility_enhancement(basic_param)
    # NOTE: NEFT uses opposite sign convention for F!
    # -F⁰ₛ = 1 - κ₀/κ
    return 1.0 - kappa0_over_kappa
end

"""
Get the antisymmetric l=0 Fermi-liquid parameter F⁰ₐ via  Corradini's fit
to the DMC susceptibility enhancement [doi: 10.1103/PhysRevB.57.14569].
"""
@inline function get_Fa(param::Parameter.Para)
    chi0_over_chi = Interaction.spin_susceptibility_enhancement(param)
    # NOTE: NEFT uses opposite sign convention for F!
    # -F⁰ₐ = 1 - χ₀/χ
    return 1.0 - chi0_over_chi
end

function lindhard(x)
    if abs(x) < 1e-4
        return 1.0 - x^2 / 3 - x^4 / 15
    elseif abs(x - 1) < 1e-7
        return 0.5
    elseif x > 20
        return 1 / (3 * x^2) + 1 / (15 * x^4)
    end
    return 0.5 + ((1 - x^2) / (4 * x)) * log(abs((1 + x) / (1 - x)))
end

function integrand_F1(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return -x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    # NF (R + f)
    NF_times_Rpf_ex = coeff / (x^2 + coeff * lindhard(x))
    return -x * NF_times_Rpf_ex
end

"""
Solve I0[F+] = F+ / 2 to obtain the tree-level self-consistent value for F⁰ₛ.
"""
function get_tree_level_self_consistent_Fs(rs::Float64)
    function I0_R(x, y)
        ts = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
        integrand = [integrand_F1(t, x * alpha_ueg / π, y) for t in ts]
        integral = Interp.integrate1D(integrand, ts)
        return integral
    end
    F1_sc = find_zero(Fp -> I0_R(rs, Fp) - Fp / 2, (-20.0, 20.0))
    return F1_sc
end
function get_tree_level_self_consistent_Fs(param::OneLoopParams)
    @unpack rs = param
    function I0_R(x, y)
        ts = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
        integrand = [integrand_F1(t, x * alpha_ueg / π, y) for t in ts]
        integral = Interp.integrate1D(integrand, ts)
        return integral
    end
    F1_sc = find_zero(Fp -> I0_R(rs, Fp) - Fp / 2, (-20.0, 20.0))
    return F1_sc
end

"""
The one-loop (GW) self-energy Σ₁.
"""
function Σ1(param::OneLoopParams, kgrid::KGT) where {KGT<:AbstractVector}
    @unpack kF, EF, Fs, basic = param
    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14
    # Based on ElectronGas.jl defaults for G0W0 self-energy (here, minK *= 100)
    maxK = 6 * kF
    minK = 1e-6 * kF
    # Get the one-loop self-energy
    Σ_imtime, _ = SelfEnergy.G0W0(
        basic,
        kgrid;
        Euv=Euv,
        rtol=rtol,
        maxK=maxK,
        minK=minK,
        int_type=:ko_const,
        Fs=Fs,
        Fa=-0.0,
    )
    # Σ_dyn(τ, k) → Σ_dyn(iωₙ, k)
    Σ = to_imfreq(to_dlr(Σ_imtime))
    return Σ
end

"""
Leading-order (one-loop) correction to Z_F.
"""
function get_Z1(param::OneLoopParams, kgrid::KGT) where {KGT<:AbstractVector}
    sigma1 = Σ1(param, kgrid)
    return zfactor_fermi(param.basic, sigma1)  # compute Z_F using improved finite-temperature scaling
end

"""
Tree-level estimate of F⁺₀ ~ ⟨R(k - k', 0)⟩.
"""
function get_F1(param::OneLoopParams)
    @unpack rs, kF, EF, NF, Fs, basic = param
    rstilde = rs * alpha_ueg / π
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 16, 1e-8, 16)
    y = [integrand_F1(x, rstilde, Fs) for x in xgrid]
    F1 = (Fs / 2) + Interp.integrate1D(y, xgrid)
    return F1
end

function x_NF_R0(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    # NF (R + f)
    NF_times_Rpf_ex = coeff / (x^2 + coeff * lindhard(x))
    # NF R = NF (R + f) - Fs
    NF_times_Rp_ex = NF_times_Rpf_ex - Fs
    return x * NF_times_Rp_ex
end

# 2R(z1 - f1 Π0) - f1 Π0 f1
function one_loop_counterterms(param::OneLoopParams; kwargs...)
    @unpack rs, kF, EF, NF, Fs, basic = param
    rstilde = rs * alpha_ueg / π

    # x = |k - k'| / 2kF
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 16, 1e-8, 16)

    # NF * ⟨R⟩ = -x N_F R(2kF x, 0)
    F1 = get_F1(param)

    # x R(2kF x, 0)
    x_R0 = [x_NF_R0(x, rstilde, Fs) / NF for x in xgrid]

    # Z_1(kF)
    z1 = get_Z1(param, 2 * kF * xgrid)

    # Π₀(q, iν=0) = -NF * 𝓁(q / 2kF)
    Π0 = -NF * lindhard.(xgrid)

    # A = z₁ + 2 ∫₀¹ dx x R(x, 0) Π₀(x, 0)
    A = z1 + Interp.integrate1D(2 * x_R0 .* Π0, xgrid)

    # B = ∫₀¹ dx x Π₀(x, 0) / NF
    B = Interp.integrate1D(xgrid .* Π0 / NF, xgrid)

    # 2R(z1 - f1 Π0) - f1 Π0 f1 = 2 F1 A + F1² B
    vertex_cts = -(2 * F1 * A + F1^2 * B)
    # vertex_cts = 2 * F1 * A + F1^2 * B
    return vertex_cts
end

"""
Regularized KO interaction δR(q, iνₘ) = R(q, iνₘ) / r_q, where r_q = v_q + f.
"""
function dR_data(param::OneLoopParams)
    @unpack paramc, qgrid_interp, mgrid, Mmax, dlr, fs = param
    Nq, Nw = length(qgrid_interp), length(mgrid)
    Pi = zeros(Float64, (Nq, Nw))
    dRs = zeros(Float64, (Nq, Nw))
    for (ni, n) in enumerate(mgrid)
        for (qi, q) in enumerate(qgrid_interp)
            rq = UEG.KOinstant(q, paramc)
            invrq = 1.0 / rq
            # Rq = (vq + f) / (1 - (vq + f) Π0) - f
            # δRq = Rq / rq =  1 / (1 - (vq + f) Π0) - f / (vq + f)
            Pi[qi, ni] = UEG.polarKW(q, n, paramc)
            dRs[qi, ni] = 1 / (1 - rq * Pi[qi, ni]) - fs * invrq
        end
    end
    # upsample to full frequency grid with indices ranging from 0 to M
    dRs = matfreq2matfreq(dlr, dRs, collect(0:Mmax); axis=2)
    return real.(dRs)  # δR(q, iνₘ) = δR(q, -iνₘ) ⟹ δR is real
end

"""
Unregularized KO interaction R(q, iνₘ)
"""
function R_data(param::OneLoopParams)
    @unpack paramc, qgrid_interp, mgrid, Mmax, dlr, fs = param
    Nq, Nw = length(qgrid_interp), length(mgrid)
    Pi = zeros(Float64, (Nq, Nw))
    Rs = zeros(Float64, (Nq, Nw))
    for (ni, n) in enumerate(mgrid)
        for (qi, q) in enumerate(qgrid_interp)
            rq = UEG.KOinstant(q, paramc)
            invrq = 1.0 / rq
            # Rq = (vq + f) / (1 - (vq + f) Π0) - f
            Pi[qi, ni] = UEG.polarKW(q, n, paramc)
            Rs[qi, ni] = 1 / (invrq - Pi[qi, ni]) - fs
        end
    end
    # upsample to full frequency grid with indices ranging from 0 to M
    Rs = matfreq2matfreq(dlr, Rs, collect(0:Mmax); axis=2)
    return real.(Rs)  # R(q, iνₘ) = R(q, -iνₘ) ⟹ R is real
end

function initialize_one_loop_params!(param::OneLoopParams)
    param.dR = dR_data(param)
    param.R = R_data(param)
    param.initialized = true
end

"""
G₀(k, iωₙ)
"""
function G0(param::OneLoopParams, k, iwn)
    @unpack me, μ, β = param
    return 1.0 / (iwn - k^2 / (2 * me) + μ)
end

"""
R(q, n) = r(q) δR(q, n) via multilinear interpolation of δR, where n indexes bosonic Matsubara frequencies iνₙ.
"""
function R_from_dR(param::OneLoopParams, q, n)
    if q > param.maxQ
        return 0.0
    end
    if q ≤ param.Q_CUTOFF
        q = param.Q_CUTOFF
    end
    rq = UEG.KOinstant(q, param.paramc)
    return rq * UEG.linear2D(param.dR, param.qgrid_interp, param.mgrid, q, n)
end

"""
R(q, n) via multilinear interpolation, where n indexes bosonic Matsubara frequencies iνₙ.
"""
function R(param::OneLoopParams, q, n)
    if q > param.maxQ
        return 0.0
    end
    if q ≤ param.Q_CUTOFF
        q = param.Q_CUTOFF
    end
    return UEG.linear2D(param.R, param.qgrid_interp, param.mgrid, q, n)
end

"""
δR(q, n) = R(q, n) / r(q) via multilinear interpolation of R, where n indexes bosonic Matsubara frequencies iνₙ.
"""
function dR_from_R(param::OneLoopParams, q, n)
    if q > param.maxQ
        return 0.0
    end
    if q ≤ param.Q_CUTOFF
        q = param.Q_CUTOFF
    end
    rq = UEG.KOinstant(q, param.paramc)
    return UEG.linear2D(param.R, param.qgrid_interp, param.mgrid, q, n) / rq
end

"""
δR(q, n) via multilinear interpolation, where n indexes bosonic Matsubara frequencies iνₙ.
"""
function dR(param::OneLoopParams, q, n)
    if q > param.maxQ
        return 0.0
    end
    if q ≤ param.Q_CUTOFF
        q = param.Q_CUTOFF
    end
    return UEG.linear2D(param.dR, param.qgrid_interp, param.mgrid, q, n)
end

function vertex_matsubara_summand(param::OneLoopParams, q, θ, φ)
    @unpack β, NF, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax, iw0 = param
    # p1 = |k + q'|, p2 = |k' + q'|
    k1vec = [0, 0, kamp1]
    k2vec = kamp2 * [sin(θ12), 0, cos(θ12)]
    qvec = q * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    vec_p1 = k1vec + qvec
    vec_p2 = k2vec + qvec
    p1 = norm(vec_p1)
    p2 = norm(vec_p2)
    # S(iνₘ) = dR(q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p2, iω₀ + iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        s_ivm[i] = (
            -q^2 *
            NF *
            R(param, q, m) *
            G0(param, p1, iw0 + im * vm) *
            G0(param, p2, iw0 + im * vm) / β
        )
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

function box_matsubara_summand(param::OneLoopParams, q, θ, φ, ftype)
    @assert ftype in ["Fs", "Fa"]
    @unpack β, NF, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax, iw0 = param
    # p1 = |k + q'|, p2 = |k' + q'|, p3 = |k' - q'|, qex = |k - k' + q'|
    k1vec = [0, 0, kamp1]
    k2vec = kamp2 * [sin(θ12), 0, cos(θ12)]
    qvec = q * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    vec_p1 = k1vec + qvec
    vec_p2 = k2vec + qvec
    vec_p3 = k2vec - qvec
    vec_qex = k1vec - k2vec + qvec
    p1 = norm(vec_p1)
    p2 = norm(vec_p2)
    p3 = norm(vec_p3)
    qex = norm(vec_qex)
    # S(iνₘ) = dR(q', iν'ₘ) * dR(k - k' + q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p3, iω₀ - iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        # Fermionized frequencies, iω₀ ± iνₘ
        ivm_Fp = iw0 + im * vm
        ivm_Fm = iw0 - im * vm
        # Ex (spin factor = 1/2)
        s_ivm_inner =
            -R(param, qex, m) * (G0(param, p1, ivm_Fp) + G0(param, p3, ivm_Fm)) / 2
        if ftype == "Fs"
            # Di (spin factor = 1)
            s_ivm_inner += R(param, q, m) * (G0(param, p2, ivm_Fp) + G0(param, p3, ivm_Fm))
        end
        s_ivm[i] = q^2 * NF^2 * R(param, q, m) * G0(param, p1, ivm_Fp) * s_ivm_inner / β
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

function vertex_matsubara_sum(param::OneLoopParams, q, θ, φ)
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    summand = vertex_matsubara_summand(param, q, θ, φ)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

function box_matsubara_sum(param::OneLoopParams, q, θ, φ; ftype="fs")
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    summand = box_matsubara_summand(param, q, θ, φ, ftype)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

# 2RΛ₁
function one_loop_vertex_corrections(param::OneLoopParams; show_progress=false)
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    @unpack qgrid, θgrid, φgrid, θ12, rs, Fs, kF, NF, basic, paramc = param

    # Initialize vertex integrand
    Nq = length(qgrid.grid)
    q_integrand = zeros(ComplexF64, Nq)

    # Setup buffers for scatter/gather
    counts = split_count(Nq, comm_size)  # number of values per rank
    data_vbuffer = VBuffer(q_integrand, counts)
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

    # Compute the integrand over loop momentum magnitude q in parallel
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress && rank == root,
    )
    θ_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    φ_integrand = Vector{ComplexF64}(undef, length(φgrid.grid))
    for (i, qi) in enumerate(local_qi)
        # println("rank = $rank: Integrating (q, 0) point $i/$local_length")
        # Get external frequency and momentum at this index
        q = qgrid.grid[qi]
        for (iθ, θ) in enumerate(θgrid)
            for (iφ, φ) in enumerate(φgrid)
                φ_integrand[iφ] = vertex_matsubara_sum(param, q, θ, φ)
            end
            θ_integrand[iθ] = Interp.integrate1D(φ_integrand, φgrid)
        end
        local_data[i] = Interp.integrate1D(θ_integrand .* sin.(θgrid.grid), θgrid)
        next!(progress_meter)
    end
    finish!(progress_meter)

    # Collect q_integrand subresults from all ranks
    MPI.Allgatherv!(local_data, data_vbuffer, comm)

    # total integrand ~ NF
    vertex_integrand = q_integrand / (2π)^3

    # Integrate over q
    k_m_kp = kF * sqrt(2 * (1 - cos(θ12)))
    # Fᵥ(θ₁₂) = Λ₁(θ₁₂) R(|k₁ - k₂|, 0)
    result = Interp.integrate1D(vertex_integrand, qgrid) * R(param, k_m_kp, 0)
    return result
end

# gg'RR' + exchange counterpart
function one_loop_box_diagrams(param::OneLoopParams; show_progress=false, ftype="fs")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @assert ftype in ["Fs", "Fa"]
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    comm_size = MPI.Comm_size(comm)

    @unpack qgrid, θgrid, φgrid, θ12, rs, Fs, kF, NF, basic, paramc = param

    # Initialize vertex integrand
    Nq = length(qgrid.grid)
    q_integrand = zeros(ComplexF64, Nq)

    # Setup buffers for scatter/gather
    counts = split_count(Nq, comm_size)  # number of values per rank
    data_vbuffer = VBuffer(q_integrand, counts)
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

    # Compute the integrand over loop momentum magnitude q in parallel
    progress_meter = Progress(
        local_length;
        desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress && rank == root,
    )
    θ_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    φ_integrand = Vector{ComplexF64}(undef, length(φgrid.grid))
    for (i, qi) in enumerate(local_qi)
        # println("rank = $rank: Integrating (q, 0) point $i/$local_length")
        # Get external frequency and momentum at this index
        q = qgrid.grid[qi]
        for (iθ, θ) in enumerate(θgrid)
            for (iφ, φ) in enumerate(φgrid)
                φ_integrand[iφ] = box_matsubara_sum(param, q, θ, φ; ftype=ftype)
            end
            θ_integrand[iθ] = Interp.integrate1D(φ_integrand, φgrid)
        end
        local_data[i] = Interp.integrate1D(θ_integrand .* sin.(θgrid.grid), θgrid)
        next!(progress_meter)
    end
    finish!(progress_meter)

    # Collect q_integrand subresults from all ranks
    MPI.Allgatherv!(local_data, data_vbuffer, comm)

    # total integrand ~ NF
    box_integrand = q_integrand / (NF * (2π)^3)

    # Integrate over q
    result = Interp.integrate1D(box_integrand, qgrid)
    return result
end

# linear interpolation with mixing parameter α: x * (1 - α) + y * α
function lerp(x, y, alpha)
    return (1 - alpha) * x + alpha * y
end

function get_one_loop_Fs(param::OneLoopParams; verbose=false, ftype="Fs", kwargs...)
    function one_loop_total(param, verbose; kwargs...)
        if verbose
            F1 = get_F1(param)
            println_root("F1 = ($(F1))ξ")

            F2v = real(one_loop_vertex_corrections(param; kwargs...))
            println_root("F2v = ($(F2v))ξ²")

            F2b = real(one_loop_box_diagrams(param; ftype=ftype, kwargs...))
            println_root("F2b = ($(F2b))ξ²")

            F2ct = real(one_loop_counterterms(param; kwargs...))
            println_root("F2ct = ($(F2ct))ξ²")

            F2 = F2v + F2b + F2ct
            println_root("F2 = ($(F1))ξ + ($(F2))ξ²")
            return F1, F2v, F2b, F2ct, F2
        else
            F1 = get_F1(param)
            F2v = real(one_loop_vertex_corrections(param; kwargs...))
            F2b = real(one_loop_box_diagrams(param; kwargs...))
            F2ct = real(one_loop_counterterms(param; kwargs...))
            F2 = F2v + F2b + F2ct
            return F1, F2v, F2b, F2ct, F2
        end
    end
    return one_loop_total(param, verbose; kwargs...)
end

function check_sign_Fs(param::OneLoopParams)
    # ElectronLiquid.jl sign convention: Fs < 0
    @unpack Fs, paramc = param
    if param.rs > 0.25
        @assert Fs ≤ 0 "Fs = $Fs must be negative in the ElectronLiquid convention!"
        @assert paramc.Fs ≤ 0 "Fs = $Fs must be negative in the ElectronLiquid convention!"
    else
        println("WARNING: when rs is nearly zero, we cannot check the sign of Fs!")
    end
end

function check_signs_Fs_Fa(rs, Fs, Fa)
    # ElectronLiquid.jl sign convention: Fs < 0
    if rs > 0.25
        @assert Fs ≤ 0 "Fs = $Fs must be negative in the ElectronLiquid convention!"
        @assert Fa ≤ 0 "Fa = $Fa must be negative in the ElectronLiquid convention!"
    else
        println("WARNING: when rs is nearly zero, we cannot check the signs of Fs/Fa!")
    end
end

function testdlr(rs, euv, rtol; rpa=false, verbose=false)
    verbose && println("(rs = $rs) Testing DLR grid with Euv / EF = $euv, rtol = $rtol")
    param = Parameter.rydbergUnit(1.0 / 40.0, rs, 3)
    @unpack β, kF, EF, NF = param
    if rpa
        Fs = 0.0
        fs = 0.0
    else
        Fs = -get_Fs(param)
        fs = Fs / NF
    end
    paramc = ParaMC(; rs=rs, beta=40.0, dim=3, Fs=Fs)

    qgrid_interp = CompositeGrid.LogDensedGrid(
        :uniform,
        [0.0, 6 * kF],
        [0.0, 2 * kF],
        16,
        0.01 * kF,
        16,
    )

    dlr = DLRGrid(; Euv=euv * EF, β=β, rtol=rtol, isFermi=false, symmetry=:ph)
    mgrid = SimpleG.Arbitrary{Int64}(dlr.n)
    Mmax = maximum(mgrid)

    verbose && println("Nw = $(length(dlr.n)), Mmax = $Mmax")

    Nq, Nw = length(qgrid_interp), length(mgrid)
    Pi = zeros(Float64, (Nq, Nw))
    Rs = zeros(Float64, (Nq, Nw))
    for (ni, n) in enumerate(mgrid)
        for (qi, q) in enumerate(qgrid_interp)
            rq = UEG.KOinstant(q, paramc)
            invrq = 1.0 / rq
            # Rq = (vq + f) / (1 - (vq + f) Π0) - f
            Pi[qi, ni] = UEG.polarKW(q, n, paramc)
            Rs[qi, ni] = 1 / (invrq - Pi[qi, ni]) - fs
        end
    end
    # upsample to full frequency grid with indices ranging from 0 to M
    Rs = matfreq2matfreq(dlr, Rs, collect(0:Mmax); axis=2)
    return Rs, qgrid_interp, paramc
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    root = 0
    rank = MPI.Comm_rank(comm)

    # rslist = [[0.01, 0.1, 0.25, 0.5]; 1:1:10]
    rslist = [[0.01, 0.1, 0.25, 0.5]; 1:0.5:10]
    beta = 40.0

    # ftype = "Fs"  # f^{Di} + f^{Ex} / 2
    ftype = "Fa"  # f^{Ex} / 2
    ftypestr = ftype == "Fs" ? "F^{s}" : "F^{a}"

    plots = true
    debug = true
    verbose = true
    show_progress = true

    # nk ≈ na ≈ 75 is sufficiently converged for all relevant euv/rtol
    Nk, Ok = 7, 6
    Na, Oa = 8, 7

    # DLR parameters for which R(q, 0) is smooth in the q → 0 limit (tested for rs = 1, 10)
    euv = 10.0
    rtol = 1e-7

    Fs_DMCs = []
    Fa_DMCs = []
    F1s = []
    F2vs = []
    F2bs = []
    F2cts = []
    F2s = []
    for (i, rs) in enumerate(rslist)
        if debug && rank == root
            testdlr(rs, euv, rtol; verbose=verbose)
        end
        basic_param = Parameter.rydbergUnit(1.0 / beta, rs, 3)
        Fs_DMC = -get_Fs(basic_param)
        Fa_DMC = -get_Fa(basic_param)
        println_root("\nrs = $rs:")
        println_root("F+ from DMC: $(Fs_DMC)")
        param = OneLoopParams(;
            rs=rs,
            beta=beta,
            Fs=Fs_DMC,
            euv=euv,
            rtol=rtol,
            Nk=Nk,
            Ok=Ok,
            Na=Na,
            Oa=Oa,
        )
        if debug && rank == root && rs > 0.25
            check_sign_Fs(param)
            check_signs_Fs_Fa(rs, Fs_DMC, Fa_DMC)
        end
        if verbose && rank == root && i == 1
            println_root("nk=$(length(param.qgrid)), na=$(length(param.θgrid))")
            println_root("nk=$(length(param.qgrid)), na=$(length(param.θgrid))")
            println_root("euv=$(param.euv), rtol=$(param.rtol)")
            println_root(
                "\nrs=$(param.rs), beta=$(param.beta), Fs=$(Fs_DMC), Fa=$(Fa_DMC)",
            )
        end
        initialize_one_loop_params!(param)  # precompute the interaction interpoland R(q, iνₘ)
        F1, F2v, F2b, F2ct, F2 = get_one_loop_Fs(
            param;
            verbose=verbose,
            show_progress=show_progress,
            ftype=ftype,
        )
        push!(Fs_DMCs, Fs_DMC)
        push!(Fa_DMCs, Fa_DMC)
        push!(F1s, F1)
        push!(F2vs, F2v)
        push!(F2bs, F2b)
        push!(F2cts, F2ct)
        push!(F2s, F2)
        GC.gc()
    end
    if plots && rank == root
        println(Fs_DMCs)
        println(Fa_DMCs)
        println(F1s)
        println(F2vs)
        println(F2bs)
        println(F2cts)
        println(F2s)

        # Plot spline fits to data vs rs
        fig, ax = plt.subplots(; figsize=(5, 5))
        error = 1e-6 * ones(length(rslist))
        flabel = ftype == "Fs" ? "\$F^{s}_{\\text{DMC}}\$" : "\$F^{a}_{\\text{DMC}}\$"
        fdata = ftype == "Fs" ? Fs_DMCs : Fa_DMCs
        ax.plot(spline(rslist, fdata, error)...; label=flabel, color=cdict["grey"])
        ax.plot(spline(rslist, F1s, error)...; label="\$F_1 \\xi\$", color=cdict["orange"])
        ax.plot(
            spline(rslist, F1s .+ F2s, error)...;
            label="\$F_1 \\xi + F_2 \\xi^2\$",
            color=cdict["blue"],
        )
        ax.plot(
            spline(rslist, F2vs, error)...;
            label="\${$ftypestr}_{\\text{v},2}\$",
            color=cdict["cyan"],
        )
        ax.plot(
            spline(rslist, F2bs, error)...;
            label="\${$ftypestr}_{\\text{b},2}\$",
            color=cdict["magenta"],
        )
        ax.plot(
            spline(rslist, F2cts, error)...;
            label="\${$ftypestr}_{\\text{ct},2}\$",
            color=cdict["teal"],
        )
        ax.set_xlabel("\$r_s\$")
        ax.set_xlim(0, maximum(rslist))
        ax.set_ylim(-5.5, 5.5)
        ax.legend(;
            ncol=2,
            loc="upper left",
            fontsize=14,
            title_fontsize=16,
            title="\$\\Lambda_\\text{UV} = $(Int(round(euv)))\\epsilon_F, \\varepsilon = 10^{$(Int(round(log10(rtol))))}\$",
        )
        fig.tight_layout()
        fig.savefig("oneshot_one_loop_$(ftype)_vs_rs_euv=$(euv)_rtol=$(rtol).pdf")
        plt.close(fig)
    end
    MPI.Finalize()
    return
end

main()
