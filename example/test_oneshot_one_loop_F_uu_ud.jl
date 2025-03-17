using AbstractTrees
using Colors
using CompositeGrids
using ElectronGas
using ElectronLiquid
using FeynmanDiagram
using GreenFunc
# using JLD2
using LinearAlgebra
using Lehmann
using LQSGW
using Measurements
using MPI
using Parameters
using ProgressMeter
using PyCall
using PyPlot
using Roots
using Test

import LQSGW: split_count, println_root, timed_result_to_string

import FeynmanDiagram.FrontEnds: TwoBodyChannel, Alli, PHr, PHEr, PPr, AnyChan
import FeynmanDiagram.FrontEnds:
    Filter, NoHartree, NoFock, DirectOnly, Wirreducible, Girreducible, NoBubble, Proper
import FeynmanDiagram.FrontEnds: Response, Composite, ChargeCharge, SpinSpin, UpUp, UpDown
import FeynmanDiagram.FrontEnds: AnalyticProperty, Instant, Dynamic

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

    # true:  KO+ interaction
    # false: Yukawa interaction
    isDynamic = true

    basic::Parameter.Para = Parameter.rydbergUnit(1.0 / beta, rs, dim; Λs=mass2, spin=spin)
    paramc::ParaMC = UEG.ParaMC(;
        rs=rs,
        beta=beta,
        dim=dim,
        spin=spin,
        mass2=mass2,
        Fs=Fs,
        basic=basic,
        isDynamic=isDynamic,
    )
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

    # We precompute r(q, iνₘ) on a mesh of ~100 k-points
    # NOTE: EL.jl default is `Nk, Ok = 16, 16` (~700 k-points)
    qgrid_interp::MomInterpGridType =
        CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 10, 0.01 * kF, 10)  # sufficient for 1-decimal accuracy
    # CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 12, 0.01 * kF, 12)
    # CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 16, 0.01 * kF, 16)
    # CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], Nk, 0.01 * kF, Ok)

    # Later, we integrate r(q, iνₘ) on a Gaussian mesh of ~100 k-points
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

    # r grid data is precomputed in an initialization step
    initialized::Bool = false
    r::Matrix{Float64} = Matrix{Float64}(undef, length(qgrid_interp), length(mgrid))
    r_yukawa::Array{Float64} = Array{Float64}(undef, length(qgrid_interp))
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
    if param.isDynamic == false
        # the one-loop self-energy is frequency independent for the Thomas-Fermi interaction
        return 0.0
    end
    sigma1 = Σ1(param, kgrid)
    return zfactor_fermi(param.basic, sigma1)  # compute Z_F using improved finite-temperature scaling
end
function get_Z1(param::OneLoopParams)
    if param.isDynamic == false
        # the one-loop self-energy is frequency independent for the Thomas-Fermi interaction
        return 0.0
    end
    @unpack kF = param
    kgrid =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, 2.0 * kF], [0.0, kF], 16, 1e-8 * kF, 16)
    sigma1 = Σ1(param, kgrid)
    return zfactor_fermi(param.basic, sigma1)  # compute Z_F using improved finite-temperature scaling
end

"""
Tree-level estimate of F⁺₀ ~ ⟨R(k - k', 0)⟩.
"""
function get_F1(param::OneLoopParams)
    if param.isDynamic == false
        return get_F1_TF(param.rs)
    end
    @unpack rs, kF, EF, NF, Fs, basic = param
    rstilde = rs * alpha_ueg / π
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 16, 1e-8, 16)
    y = [integrand_F1(x, rstilde, Fs) for x in xgrid]
    F1 = (Fs / 2) + Interp.integrate1D(y, xgrid)
    return F1
end

"""
Tree-level estimate of F⁺₀ ~ ⟨V_TF(k - k', 0)⟩ for the Thomas-Fermi interaction.
"""
function get_F1_TF(rs)
    if isinf(rs)
        return -0.5
    elseif rs == 0
        return -0.0
    end
    rstilde = rs * alpha_ueg / π
    F1 = (rstilde / 2) * log(rstilde / (rstilde + 1))
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

function x_NF_VTF(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return x
    elseif rs_tilde == 0
        return 0.0
    end
    NF_times_VTF_ex = rs_tilde / (x^2 + rs_tilde)
    return x * NF_times_VTF_ex
end

# 2R(z1 - f1 Π0) - f1 Π0 f1
function one_loop_counterterms(param::OneLoopParams; kwargs...)
    @unpack rs, kF, EF, NF, Fs, basic, isDynamic = param
    if isDynamic == false
        @assert param.mass2 ≈ param.qTF^2 "Counterterms currently only implemented for the Thomas-Fermi interaction (Yukawa mass = qTF)!"
    end
    rstilde = rs * alpha_ueg / π

    # x = |k - k'| / 2kF
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 16, 1e-8, 16)

    # NF * ⟨R⟩ = -x N_F R(2kF x, 0)
    F1 = get_F1(param)

    # x R(2kF x, 0)
    x_NF_R = isDynamic ? x_NF_R0 : x_NF_VTF
    x_R0 = [x_NF_R(x, rstilde, Fs) / NF for x in xgrid]

    # Z_1(kF)
    z1 = get_Z1(param, 2 * kF * xgrid)

    # Π₀(q, iν=0) = -NF * 𝓁(q / 2kF)
    Π0 = -NF * lindhard.(xgrid)

    # BUGGY!
    # # A = z₁ + 2 ∫₀¹ dx x R(x, 0) Π₀(x, 0)
    # A = z1 + Interp.integrate1D(2 * x_R0 .* Π0, xgrid)

    # A = z₁ + ∫₀¹ dx x R(x, 0) Π₀(x, 0)
    A = z1 + Interp.integrate1D(x_R0 .* Π0, xgrid)

    # B = ∫₀¹ dx x Π₀(x, 0) / NF
    B = Interp.integrate1D(xgrid .* Π0 / NF, xgrid)

    # (NF/2)*⟨2R(z1 - f1 Π0) - f1 Π0 f1⟩ = -2 F1 A - F1² B
    vertex_cts = -(2 * F1 * A + F1^2 * B)
    # vertex_cts = 2 * F1 * A + F1^2 * B
    return vertex_cts
end

"""
Dimensionless KO interaction, r(q, iνₘ) = NF * R(q, iνₘ)
"""
function r_data(param::OneLoopParams)
    @unpack paramc, qgrid_interp, mgrid, Mmax, dlr, Fs, NF = param
    Nq, Nw = length(qgrid_interp), length(mgrid)
    Pi = zeros(Float64, (Nq, Nw))
    rq = zeros(Float64, (Nq, Nw))
    for (ni, n) in enumerate(mgrid)
        for (qi, q) in enumerate(qgrid_interp)
            vq_plus_f = UEG.KOinstant(q, paramc)
            inv_vq_plus_f = 1.0 / vq_plus_f
            # rq = NF (vq + f) / (1 - (vq + f) Π0) - F
            Pi[qi, ni] = UEG.polarKW(q, n, paramc)
            rq[qi, ni] = NF / (inv_vq_plus_f - Pi[qi, ni]) - Fs
        end
    end
    # upsample to full frequency grid with indices ranging from 0 to M
    rq = matfreq2matfreq(dlr, rq, collect(0:Mmax); axis=2)
    return real.(rq)  # r(q, iνₘ) = r(q, -iνₘ) ⟹ r is real
end

"""
Dimensionless Yukawa interaction, r_yukawa(q) = NF * V_yukawa(q) = qTF^2 / (q^2 + mass2)
"""
function r_yukawa_data(param::OneLoopParams)
    @unpack qgrid_interp, qTF, mass2 = param
    @assert mass2 ≈ qTF^2
    r_yukawa(q) = qTF^2 / (q^2 + mass2)
    return r_yukawa.(qgrid_interp)
end

function initialize_one_loop_params!(param::OneLoopParams)
    if param.isDynamic
        # Dimensionless KO+ interaction, NF R = NF (v + f) / (1 - (v + f) Π₀) - f
        param.r = r_data(param)
    else
        # Dimensionless Yukawa interaction with mass √λ, NF V = qTF^2 / (q^2 + λ)
        param.r_yukawa = r_yukawa_data(param)
    end
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
r(q, n) via multilinear interpolation, where n indexes bosonic Matsubara frequencies iνₙ.
"""
function r_interp(param::OneLoopParams, q, n)
    if q > param.maxQ
        return 0.0
    end
    if q ≤ param.Q_CUTOFF
        q = param.Q_CUTOFF
    end
    if param.isDynamic
        v_q_n = UEG.linear2D(param.r, param.qgrid_interp, param.mgrid, q, n)
    else
        v_q_n = Interp.interp1D(param.r_yukawa, param.qgrid_interp, q)
    end
    return v_q_n
end

"""
δr(q, n) = r(q, n) / (v(q) + f) via multilinear interpolation of r, where n indexes bosonic Matsubara frequencies iνₙ.
"""
function dr_from_r(param::OneLoopParams, q, n)
    @assert param.isDynamic "δr(q, n) is only relevant when `isDynamic=true`!"
    if q > param.maxQ
        return 0.0
    end
    if q ≤ param.Q_CUTOFF
        q = param.Q_CUTOFF
    end
    vq_plus_f = UEG.KOinstant(q, param.paramc)
    return UEG.linear2D(param.r, param.qgrid_interp, param.mgrid, q, n) / vq_plus_f
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
    # S(iνₘ) = r(q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p2, iω₀ + iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        s_ivm[i] = (
            -q^2 *
            r_interp(param, q, m) *
            G0(param, p1, iw0 + im * vm) *
            G0(param, p2, iw0 + im * vm) / β
        )
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
    # S(iνₘ) = R(q', iν'ₘ) * R(k - k' + q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p3, iω₀ - iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        # Fermionized frequencies, iω₀ ± iνₘ
        ivm_Fp = iw0 + im * vm
        ivm_Fm = iw0 - im * vm
        # Ex (spin factor = 1/2)
        s_ivm_inner =
            -r_interp(param, qex, m) * (G0(param, p1, ivm_Fp) + G0(param, p3, ivm_Fm)) / 2
        if ftype == "Fs"
            # Di (spin factor = 1)
            s_ivm_inner +=
                r_interp(param, q, m) * (G0(param, p2, ivm_Fp) + G0(param, p3, ivm_Fm))
        end
        s_ivm[i] = q^2 * r_interp(param, q, m) * G0(param, p1, ivm_Fp) * s_ivm_inner / β
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

function box_matsubara_sum(param::OneLoopParams, q, θ, φ; ftype="fs")
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    summand = box_matsubara_summand(param, q, θ, φ, ftype)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

function direct_box_matsubara_summand(param::OneLoopParams, q, θ, φ; which="both")
    @assert which in ["both", "ladder", "crossed"]
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
    # S(iνₘ) = r(q', iν'ₘ) * r(k - k' + q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p3, iω₀ - iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        # Fermionized frequencies, iω₀ ± iνₘ
        ivm_Fp = iw0 + im * vm
        ivm_Fm = iw0 - im * vm
        # Ex (spin factor = 1/2)
        if which == "both"
            s_ivm_inner =
                r_interp(param, q, m) * (G0(param, p2, ivm_Fp) + G0(param, p3, ivm_Fm))
        elseif which == "crossed"
            s_ivm_inner = r_interp(param, q, m) * G0(param, p2, ivm_Fp)
        elseif which == "ladder"
            s_ivm_inner = r_interp(param, q, m) * G0(param, p3, ivm_Fm)
        end
        s_ivm[i] = q^2 * r_interp(param, q, m) * G0(param, p1, ivm_Fp) * s_ivm_inner / β
    end
    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

function direct_box_matsubara_sum(param::OneLoopParams, q, θ, φ; which="both")
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    @assert which in ["both", "ladder", "crossed"]
    summand = direct_box_matsubara_summand(param, q, θ, φ; which=which)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

function plot_vertex_matsubara_summand(param::OneLoopParams)
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, isDynamic = param
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
        coordinates = [
            [q, 0, rand(0:(2π))],  # q || k1 (equivalent to q || k2)
            [q, π, rand(0:(2π))],  # q || -k1 (equivalent to q || -k2)
            [q, 3π / 4, π],      # q maximally spaced from (anti-bisects) k1 & k2
            [q, π / 4, 0],       # q bisects k1 & k2
            [q, π / 2, π / 2],   # q || y-axis
            [q, 2π / 3, π / 3],  # general asymmetrically placed q #1
        ]
        labels = [
            "\$\\theta=0, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\pi, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\frac{3\\pi}{4}, \\varphi=\\pi\$",
            "\$\\theta=\\frac{\\pi}{4}, \\varphi=0\$",
            "\$\\theta=\\frac{\\pi}{2}, \\varphi=\\frac{\\pi}{2}\$",
            "\$\\theta=\\frac{2\\pi}{3}, \\varphi=\\frac{\\pi}{3}\$",
        ]
        # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
        fig, ax = plt.subplots(; figsize=(5, 5))
        vms = (0:Mmax) * (2π / β)
        for (i, (label, coord)) in enumerate(zip(labels, coordinates))
            summand = vertex_matsubara_summand(param, coord...)
            ax.plot(
                vms / EF,
                real(summand);
                color=color[i],
                label=label,
                marker="o",
                markersize=4,
                markerfacecolor="none",
            )
        end
        ax.set_xlabel("\$i\\nu_m / \\epsilon_F\$")
        ax.set_ylabel("\$S^\\text{v}_\\mathbf{q}(i\\nu_m)\$")
        ax.set_xlim(0, 4)
        ax.legend(;
            loc="best",
            fontsize=14,
            title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
        )
        # fig.tight_layout()
        kindstr = param.Fs == 0.0 ? "rpa" : "kop"
        interactionstr = isDynamic ? "" : "yukawa"
        fig.savefig("vertex_matsubara_summand_q=$(qstr)_$(kindstr)_$(interactionstr).pdf")
    end
    return
end

function plot_vertex_matsubara_sum(param::OneLoopParams)
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, θgrid, isDynamic = param
    clabels = ["Re", "Im"]
    cparts = [real, imag]
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (clabel, cpart) in zip(clabels, cparts)
        for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
            phis = [0, π / 4, π / 2, 3π / 4, π]
            labels = [
                "\$\\varphi=0\$",
                "\$\\varphi=\\frac{\\pi}{4}\$",
                "\$\\varphi=\\frac{\\pi}{2}\$",
                "\$\\varphi=\\frac{3\\pi}{4}\$",
                "\$\\varphi=\\pi\$",
            ]
            # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
            fig, ax = plt.subplots(; figsize=(5, 5))
            for (i, (label, φ)) in enumerate(zip(labels, phis))
                matsubara_sum_vs_θ =
                    [vertex_matsubara_sum(param, q, θ, φ) for θ in θgrid.grid]
                ax.plot(
                    θgrid.grid,
                    cpart(matsubara_sum_vs_θ);
                    color=color[i],
                    label=label,
                    # marker="o",
                    # markersize=4,
                    # markerfacecolor="none",
                )
            end
            ax.set_xlabel("\$\\theta\$")
            ax.set_ylabel("\$S_\\text{v}(q, \\theta, \\phi; \\theta_{12} = \\pi / 2)\$")
            ax.set_xlim(0, π)
            ax.set_xticks([0, π / 4, π / 2, 3π / 4, π])
            ax.set_xticklabels([
                "0",
                "\$\\frac{\\pi}{4}\$",
                "\$\\frac{\\pi}{2}\$",
                "\$\\frac{3\\pi}{4}\$",
                "\$\\pi\$",
            ])
            ax.legend(;
                loc="best",
                fontsize=14,
                title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
                ncol=2,
            )
            # fig.tight_layout()
            kindstr = param.Fs == 0.0 ? "rpa" : "kop"
            interactionstr = isDynamic ? "" : "yukawa"
            fig.savefig(
                "$(clabel)_vertex_matsubara_sum_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
            )
        end
    end
end

function plot_box_matsubara_summand(param::OneLoopParams; ftype="Fs")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, isDynamic = param
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
        coordinates = [
            [q, 0, rand(0:(2π))],  # q || k1 (equivalent to q || k2)
            [q, π, rand(0:(2π))],  # q || -k1 (equivalent to q || -k2)
            [q, 3π / 4, π],      # q maximally spaced from (anti-bisects) k1 & k2
            [q, π / 4, 0],       # q bisects k1 & k2
            [q, π / 2, π / 2],   # q || y-axis
            [q, 2π / 3, π / 3],  # general asymmetrically placed q #1
        ]
        labels = [
            "\$\\theta=0, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\pi, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\frac{3\\pi}{4}, \\varphi=\\pi\$",
            "\$\\theta=\\frac{\\pi}{4}, \\varphi=0\$",
            "\$\\theta=\\frac{\\pi}{2}, \\varphi=\\frac{\\pi}{2}\$",
            "\$\\theta=\\frac{2\\pi}{3}, \\varphi=\\frac{\\pi}{3}\$",
        ]
        # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
        fig, ax = plt.subplots(; figsize=(5, 5))
        vms = (0:Mmax) * (2π / β)
        for (i, (label, coord)) in enumerate(zip(labels, coordinates))
            summand = box_matsubara_summand(param, coord..., ftype)
            ax.plot(
                vms / EF,
                real(summand);
                color=color[i],
                label=label,
                marker="o",
                markersize=4,
                markerfacecolor="none",
            )
        end
        ax.set_xlabel("\$i\\nu_m / \\epsilon_F\$")
        ax.set_ylabel("\$S^\\text{b}_\\mathbf{q}(i\\nu_m)\$")
        ax.set_xlim(0, 4)
        ax.legend(;
            loc="best",
            fontsize=14,
            title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
        )
        # fig.tight_layout()
        kindstr = param.Fs == 0.0 ? "rpa" : "kop"
        interactionstr = isDynamic ? "" : "yukawa"
        fig.savefig(
            "box_matsubara_summand_$(ftype)_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
        )
    end
    return
end

function plot_box_matsubara_sum(param::OneLoopParams; ftype="Fs")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, θgrid, isDynamic = param
    clabels = ["Re", "Im"]
    cparts = [real, imag]
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (clabel, cpart) in zip(clabels, cparts)
        for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
            phis = [0, π / 4, π / 2, 3π / 4, π]
            labels = [
                "\$\\varphi=0\$",
                "\$\\varphi=\\frac{\\pi}{4}\$",
                "\$\\varphi=\\frac{\\pi}{2}\$",
                "\$\\varphi=\\frac{3\\pi}{4}\$",
                "\$\\varphi=\\pi\$",
            ]
            # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
            fig, ax = plt.subplots(; figsize=(5, 5))
            for (i, (label, φ)) in enumerate(zip(labels, phis))
                matsubara_sum_vs_θ =
                    [box_matsubara_sum(param, q, θ, φ; ftype=ftype) for θ in θgrid.grid]
                ax.plot(
                    θgrid.grid,
                    cpart(matsubara_sum_vs_θ);
                    color=color[i],
                    label=label,
                    # marker="o",
                    # markersize=4,
                    # markerfacecolor="none",
                )
            end
            ax.set_xlabel("\$\\theta\$")
            ax.set_ylabel("\$S_\\text{b}(q, \\theta, \\phi; \\theta_{12} = \\pi / 2)\$")
            ax.set_xlim(0, π)
            ax.set_xticks([0, π / 4, π / 2, 3π / 4, π])
            ax.set_xticklabels([
                "0",
                "\$\\frac{\\pi}{4}\$",
                "\$\\frac{\\pi}{2}\$",
                "\$\\frac{3\\pi}{4}\$",
                "\$\\pi\$",
            ])
            ax.legend(;
                loc="best",
                fontsize=14,
                title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
                ncol=2,
            )
            # fig.tight_layout()
            kindstr = param.Fs == 0.0 ? "rpa" : "kop"
            interactionstr = isDynamic ? "" : "yukawa"
            fig.savefig(
                "$(clabel)_box_matsubara_sum_$(ftype)_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
            )
        end
    end
end

function plot_direct_box_matsubara_summand(param::OneLoopParams; which="both")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @assert which in ["both", "ladder", "crossed"]
    @unpack β, kF, EF, Mmax, Q_CUTOFF, isDynamic = param
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
        coordinates = [
            [q, 0, rand(0:(2π))],  # q || k1 (equivalent to q || k2)
            [q, π, rand(0:(2π))],  # q || -k1 (equivalent to q || -k2)
            [q, 3π / 4, π],      # q maximally spaced from (anti-bisects) k1 & k2
            [q, π / 4, 0],       # q bisects k1 & k2
            [q, π / 2, π / 2],   # q || y-axis
            [q, 2π / 3, π / 3],  # general asymmetrically placed q #1
        ]
        labels = [
            "\$\\theta=0, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\pi, \\varphi \\in [0, 2\\pi]\$",
            "\$\\theta=\\frac{3\\pi}{4}, \\varphi=\\pi\$",
            "\$\\theta=\\frac{\\pi}{4}, \\varphi=0\$",
            "\$\\theta=\\frac{\\pi}{2}, \\varphi=\\frac{\\pi}{2}\$",
            "\$\\theta=\\frac{2\\pi}{3}, \\varphi=\\frac{\\pi}{3}\$",
        ]
        # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
        fig, ax = plt.subplots(; figsize=(5, 5))
        vms = (0:Mmax) * (2π / β)
        for (i, (label, coord)) in enumerate(zip(labels, coordinates))
            summand = direct_box_matsubara_summand(param, coord...; which=which)
            ax.plot(
                vms / EF,
                real(summand);
                color=color[i],
                label=label,
                marker="o",
                markersize=4,
                markerfacecolor="none",
            )
        end
        ax.set_xlabel("\$i\\nu_m / \\epsilon_F\$")
        ax.set_ylabel("\$S^\\text{b,Di}_\\mathbf{q}(i\\nu_m)\$")
        ax.set_xlim(0, 4)
        ax.legend(;
            loc="best",
            fontsize=14,
            title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
        )
        # fig.tight_layout()
        kindstr = param.Fs == 0.0 ? "rpa" : "kop"
        interactionstr = isDynamic ? "" : "yukawa"
        fig.savefig(
            "$(which)_direct_box_matsubara_summand_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
        )
    end
    return
end

function plot_direct_box_matsubara_sum(param::OneLoopParams; which="both")
    @assert param.initialized "R(q, iνₘ) data not yet initialized!"
    @assert which in ["both", "ladder", "crossed"]
    @unpack β, kF, EF, Mmax, Q_CUTOFF, θgrid, isDynamic = param
    clabels = ["Re", "Im"]
    cparts = [real, imag]
    plot_qs = [Q_CUTOFF, kF, 2 * kF]
    plot_qstrs = ["Q_CUTOFF", "kF", "2kF"]
    plot_qlabels = ["q \\approx 0", "q = k_F", "q = 2k_F"]
    for (clabel, cpart) in zip(clabels, cparts)
        for (q, qstr, qlabel) in zip(plot_qs, plot_qstrs, plot_qlabels)
            phis = [0, π / 4, π / 2, 3π / 4, π]
            labels = [
                "\$\\varphi=0\$",
                "\$\\varphi=\\frac{\\pi}{4}\$",
                "\$\\varphi=\\frac{\\pi}{2}\$",
                "\$\\varphi=\\frac{3\\pi}{4}\$",
                "\$\\varphi=\\pi\$",
            ]
            # Plot the Matsubara summand vs iνₘ for fixed q, θ, φ
            fig, ax = plt.subplots(; figsize=(5, 5))
            for (i, (label, φ)) in enumerate(zip(labels, phis))
                matsubara_sum_vs_θ = [
                    direct_box_matsubara_sum(param, q, θ, φ; which=which) for
                    θ in θgrid.grid
                ]
                ax.plot(
                    θgrid.grid,
                    cpart(matsubara_sum_vs_θ);
                    color=color[i],
                    label=label,
                    # marker="o",
                    # markersize=4,
                    # markerfacecolor="none",
                )
            end
            ax.set_xlabel("\$\\theta\$")
            ax.set_ylabel("\$S_\\text{b,Di}(q, \\theta, \\phi; \\theta_{12} = \\pi / 2)\$")
            ax.set_xlim(0, π)
            ax.set_xticks([0, π / 4, π / 2, 3π / 4, π])
            ax.set_xticklabels([
                "0",
                "\$\\frac{\\pi}{4}\$",
                "\$\\frac{\\pi}{2}\$",
                "\$\\frac{3\\pi}{4}\$",
                "\$\\pi\$",
            ])
            ax.legend(;
                loc="best",
                fontsize=14,
                title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
                ncol=2,
            )
            # fig.tight_layout()
            kindstr = param.Fs == 0.0 ? "rpa" : "kop"
            interactionstr = isDynamic ? "" : "yukawa"
            fig.savefig(
                "$(clabel)_$(which)_direct_box_matsubara_sum_q=$(qstr)_$(kindstr)_$(interactionstr).pdf",
            )
        end
    end
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
    vertex_integrand = q_integrand / (NF * (2π)^3)

    # Integrate over q
    k_m_kp = kF * sqrt(2 * (1 - cos(θ12)))
    # Fᵥ(θ₁₂) = Λ₁(θ₁₂) r(|k₁ - k₂|, 0)
    result = Interp.integrate1D(vertex_integrand, qgrid) * r_interp(param, k_m_kp, 0)
    return result
end

# gg'RR' + exchange counterpart
function one_loop_box_diagrams(param::OneLoopParams; show_progress=false, ftype="fs")
    @assert param.initialized "r(q, iνₘ) data not yet initialized!"
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

# gg'RR' + exchange counterpart
function one_loop_direct_box_diagrams(
    param::OneLoopParams;
    show_progress=false,
    which="both",
)
    @assert param.initialized "r(q, iνₘ) data not yet initialized!"
    @assert which in ["both", "ladder", "crossed"]
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
                φ_integrand[iφ] = direct_box_matsubara_sum(param, q, θ, φ; which=which)
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

function get_one_loop_Fs(
    param::OneLoopParams;
    verbose=false,
    ftype="Fs",
    z_renorm=false,
    kwargs...,
)
    function one_loop_total(param, verbose; kwargs...)
        if verbose
            F1 = param.isDynamic ? get_F1(param) : get_F1_TF(param.rs)
            println_root("F1 = ($(F1))ξ")

            F2v = real(one_loop_vertex_corrections(param; kwargs...))
            println_root("F2v = ($(F2v))ξ²")

            F2b = real(one_loop_box_diagrams(param; ftype=ftype, kwargs...))
            println_root("F2b = ($(F2b))ξ²")

            F2ct = real(one_loop_counterterms(param; kwargs...))
            println_root("F2ct = ($(F2ct))ξ²")

            # z²⟨Γ⟩ = (1 + z₁ξ + ...)²⟨Γ⟩ = 2z₁F₁ξ²
            if z_renorm
                z1 = get_Z1(param)
                F2z = 2 * z1 * F1
                println_root("F2z = ($(F2z))ξ²")
            else
                F2z = 0.0
            end

            F2 = F2v + F2b + F2ct + F2z
            println_root("F2 = ($(F1))ξ + ($(F2))ξ²")
            return F1, F2v, F2b, F2ct, F2z, F2
        else
            F1 = get_F1(param)
            F2v = real(one_loop_vertex_corrections(param; kwargs...))
            F2b = real(one_loop_box_diagrams(param; kwargs...))
            F2ct = real(one_loop_counterterms(param; kwargs...))
            if z_renorm
                z1 = get_Z1(param)
                F2z = 2 * z1 * F1
            else
                F2z = 0.0
            end
            F2 = F2v + F2b + F2ct + F2z
            return F1, F2v, F2b, F2ct, F2z, F2
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

function compare(data, expect, ratio=5)
    println(data, ", ", expect)
    @test isapprox(data.val, expect, atol=ratio * data.err)
end

function test_yukawa_tree_level_neft()
    seed = 1234

    # Tree-level => no internal loops or counterterms
    p = (0, 0, 0)

    dummy_paramc = ElectronLiquid.ParaMC(;
        rs=0.01,
        beta=40.0,
        Fs=0.0,
        order=0,
        mass2=0.0,
        isDynamic=false,
    )

    # returns: (partition, diagpara, FeynGraphs, extT_labels, spin_conventions)
    diagrams = Diagram.diagram_parquet_response(
        :vertex4,
        dummy_paramc,
        [p];
        filter=[Proper, NoHartree],  # Proper => only exchange part contribute to F₁
        transferLoop=zeros(3),       # (Q,Ω) → 0
    )
    graphs = diagrams[3]
    println("Tree-level computational graphs from Diagram.diagram_parquet_response:")
    print_tree(graphs)

    rslist = [0.01, 0.1, 1.0, 10.0]
    for rs in rslist
        println("\nTesting for rs = $rs:")
        qTF = Parameter.rydbergUnit(1.0 / 40.0, rs, 3).qTF
        paramc = ElectronLiquid.ParaMC(;
            rs=rs,
            beta=40.0,
            Fs=0.0,
            order=0,
            mass2=qTF^2,  # for a particularly simple dimensionless interaction
            isDynamic=false,
        )
        @unpack e0, NF, mass2 = paramc

        ########## test l=0 PH averged Yukawa interaction ############
        data, result =
            Ver4.lavg(paramc, diagrams; neval=1e5, l=[0], n=[-1, 0, 0], seed=seed, print=-1)
        obs = data[p]
        println("up-up: $(obs[1]), up-down: $(obs[2])")

        # Why are these swapped?
        Fp_MCMC = -real(obs[1] - obs[2]) / 2 # (upup - updn) / 2, extra minus sign, factor of 1/2 already included in lavg
        Fm_MCMC = -real(obs[1] + obs[2]) / 2 # (upup + updn) / 2, extra minus sign, factor of 1/2 already included in lavg
        println("Fp = $Fp_MCMC, Fm = $Fm_MCMC")

        Fp, Fm = Ver4.projected_exchange_interaction(0, paramc, Ver4.exchange_Coulomb)
        println("MCMC for exchange vs NEFT quadrature")
        compare(Fp_MCMC, -Fp)
        compare(Fm_MCMC, -Fm)

        # expect = 0.0
        # compare(real(obs[1]), expect)
        expect = -4π * e0^2 / (mass2) * NF
        println("MCMC for exchange up-down vs exact")
        compare(real(obs[2]), expect)
    end
end

function test_yukawa_one_loop_neft()
    seed = 1234

    # One-loop => only interaction-type counterterms
    p = UEG.partition(1; offset=0)  # TEST: G-type counterterms should be removed by Diagram.diagram_parquet_response!
    # p = [(1, 0, 0), (0, 0, 1)]

    dummy_paramc = ElectronLiquid.ParaMC(;
        rs=0.01,
        beta=40.0,
        Fs=0.0,
        order=1,
        mass2=0.0,
        isDynamic=false,
    )

    # NOTE: we need to include the bubble diagram, as it will be cancelled numerically by the interaction counterterm
    filter = [Proper, NoHartree]  # Proper => only exchange and box-type direct diagrams contribute to F₂

    # returns: (partition, diagpara, FeynGraphs, extT_labels, spin_conventions)
    diagrams = Diagram.diagram_parquet_response(
        :vertex4,
        dummy_paramc,
        p;
        filter=filter,
        transferLoop=zeros(16),  # (Q,Ω) → 0
    )
    graphs = diagrams[3]
    println("Tree-level computational graphs from Diagram.diagram_parquet_response:")
    print_tree(graphs)

    rslist = [0.01, 0.1, 1.0, 10.0]
    for rs in rslist
        println("\nTesting for rs = $rs:")
        qTF = Parameter.rydbergUnit(1.0 / 40.0, rs, 3).qTF
        paramc = ElectronLiquid.ParaMC(;
            rs=rs,
            beta=40.0,
            Fs=0.0,
            order=1,
            mass2=qTF^2,  # for a particularly simple dimensionless interaction
            isDynamic=false,
        )
        @unpack e0, NF, mass2 = paramc

        ########## test l=0 PH averged Yukawa interaction ############
        data, result =
            Ver4.lavg(paramc, diagrams; neval=1e6, l=[0], n=[-1, 0, 0], seed=seed, print=-1)

        # (0, 0, 0)
        obs_tl = real(data[p[1]])

        # (1, 0, 0)
        obs = real(data[p[2]])

        # # (0, 1, 0)
        # obs_gct = real(data[p[3]])
        # println("(G counterterms)\tup-up: $(obs_gct[1]), up-down: $(obs_gct[2])")

        # (0, 0, 1)
        obs_ict = real(data[p[4]])

        # Fp = -real(obs[1] + obs[2]) # upup + updn, extra minus sign, factor of 1/2 already included in lavg
        # Fm = -real(obs[1] - obs[2]) # upup - updn, extra minus sign, factor of 1/2 already included in lavg
        # println("Fp = $Fp, Fm = $Fm")

        exchange_tl = obs_tl[1] - obs_tl[2] # exchange = upup - updn
        exchange = obs[1] - obs[2] # exchange = upup - updn
        exchange_ct = obs_ict[1] - obs_ict[2] # exchange = upup - updn
        println(
            "(tree-level)\tup-up: $(obs_tl[1]), up-down: $(obs_tl[2]), exchange: $(exchange_tl)",
        )
        println(
            "(one-loop diagrams)\tup-up: $(obs[1]), up-down: $(obs[2]), exchange: $(exchange)",
        )
        println(
            "(one-loop counterterms)\tup-up: $(obs_ict[1]), up-down: $(obs_ict[2]), exchange: $(exchange_ct)",
        )

        F1 = exchange_tl
        F2 = exchange + exchange_ct
        println("F = ($(F1))ξ + ($(F2))ξ² + O(ξ³)")

        # TODO: check this—is the test written incorrectly? how can (upup - updn) correspond to Fp?
        Wp, Wm, θgrid = Ver4.exchange_Coulomb(paramc) # Wp = exchanged Coulomb interaction, Wm = 0
        Fp = Ver4.Legrendre(0, Wp, θgrid)
        println("MCMC for tree-level exchange vs NEFT quadrature")
        compare(real(exchange_tl), Fp)

        # expect = 0.0
        # compare(real(obs[1]), expect)
        expect = -4π * e0^2 / (mass2) * NF
        println("MCMC for tree-level up-down vs exact")
        compare(real(obs_tl[2]), expect)
    end
end

function get_yukawa_one_loop_neft(rslist, beta; neval=1e6, seed=1234)
    # One-loop => only interaction-type counterterms
    p = UEG.partition(1; offset=0)  # TEST: G-type counterterms should be removed by Diagram.diagram_parquet_response!

    dummy_paramc = ElectronLiquid.ParaMC(;
        rs=0.01,
        beta=beta,
        Fs=0.0,
        order=1,
        mass2=0.0,
        isDynamic=false,
    )

    # NOTE: we need to include the bubble diagram, as it will be cancelled numerically by the interaction counterterm
    filter = [Proper, NoHartree]  # Proper => only exchange and box-type direct diagrams contribute to F₂

    # returns: (partition, diagpara, FeynGraphs, extT_labels, spin_conventions)
    # graphs contain: {{↑↑ diags}, {↑↓ diags}}
    diagrams = Diagram.diagram_parquet_response(
        :vertex4,
        dummy_paramc,
        p;
        filter=filter,
        transferLoop=zeros(16),  # (Q,Ω) → 0
    )
    graphs = diagrams[3]
    isUpUp =
        Dict(P => [g.properties.response == UpUp for g in graphs[P]] for P in keys(graphs))
    isUpDown = Dict(
        P => [g.properties.response == UpDown for g in graphs[P]] for P in keys(graphs)
    )
    println("Tree-level computational graphs from Diagram.diagram_parquet_response:")
    print_tree(graphs)
    println("isUpUp: $isUpUp")
    println("isUpDown: $isUpDown")

    F1s = []
    Fs2s = []
    Fa2s = []
    Fuu2s = []
    Fud2s = []
    for rs in rslist
        println("\nrs = $rs:")
        qTF = Parameter.rydbergUnit(1.0 / beta, rs, 3).qTF
        paramc = ElectronLiquid.ParaMC(;
            rs=rs,
            beta=beta,
            Fs=0.0,
            order=1,
            mass2=qTF^2,  # for a particularly simple dimensionless interaction
            isDynamic=false,
        )
        @unpack e0, NF, mass2, basic = paramc

        Fs_DMC = -get_Fs(basic)
        Fa_DMC = -get_Fa(basic)
        println_root("F+ from DMC: $(Fs_DMC), F- from DMC: $(Fa_DMC)")

        ########## test l=0 PH averged Yukawa interaction ############
        data, result = Ver4.lavg(
            paramc,
            diagrams;
            neval=neval,
            l=[0],
            n=[-1, 0, 0],
            seed=seed,
            print=-1,
        )

        # (0, 0, 0)
        obs_tl = real(data[p[1]])

        # (1, 0, 0)
        obs = real(data[p[2]])

        # # (0, 1, 0)
        # obs_gct = real(data[p[3]])
        # println("(G counterterms)\tup-up: $(obs_gct[1]), up-down: $(obs_gct[2])")

        # (0, 0, 1)
        obs_ict = real(data[p[4]])

        @assert length(obs_tl) == 2                  # obs = {↑↑Ins, ↑↓Ins}
        @assert length(obs) == length(obs_ict) == 6  # obs = {PHr ↑↑Dyn, PPr ↑↑Dyn, PHEr ↑↑Dyn, PHr ↑↓Dyn, PPr ↑↓Dyn, PHEr ↑↓Dyn}

        F1uu = -isUpUp' * obs_tl
        F1ud = -isUpDown' * obs_tl

        F2uu = -isUpUp' * obs
        F2ud = -isUpDown' * obs

        F2ctuu = -isUpUp' * obs_ict
        F2ctud = -isUpDown' * obs_ict

        println("(tree-level)\tup-up: $F1uu, up-down: $F1ud")
        println("(one-loop diagrams)\tup-up: $F2uu, up-down: $F2ud")
        println("(one-loop counterterms)\tup-up: $F2ctuu, up-down: $F2ctud")

        Fp(Fuu, Fud) = (Fuu + Fud) / 2
        Fm(Fuu, Fud) = (Fuu - Fud) / 2

        F1p = Fp(F1uu, F1ud)
        F1m = Fm(F1uu, F1ud)
        F1 = F1p
        @assert F1p ≈ F1m "F1p = $F1p, F1m = $F1m (should be equal!)"

        F2p = Fp(F2uu, F2ud)
        F2m = Fm(F2uu, F2ud)
        F2ctp = Fp(F2ctuu, F2ctud)
        F2ctm = Fm(F2ctuu, F2ctud)
        F2s = F2p + F2ctp
        F2a = F2m + F2ctm
        F2totaluu = F2uu + F2ctuu
        F2totalud = F2ud + F2ctud
        println("Fs = ($(F1))ξ + ($(F2s))ξ² + O(ξ³)")
        println("Fa = ($(F1))ξ + ($(F2a))ξ² + O(ξ³)")
        println("F↑↑ = ($(F1uu))ξ + ($(F2totaluu))ξ² + O(ξ³)")
        println("F↑↓ = ($(F1ud))ξ + ($(F2totalud))ξ² + O(ξ³)")

        push!(FsDMCs, Fs_DMC)
        push!(FaDMCs, Fa_DMC)
        push!(F1s, F1)
        push!(Fs2s, F2s)
        push!(Fa2s, F2a)
        push!(Fuu2s, F2totaluu)
        push!(Fud2s, F2totalud)
        GC.gc()
    end
    return FsDMCs, FaDMCs, F1s, Fs2s, Fa2s, Fuu2s, Fud2s
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    root = 0
    rank = MPI.Comm_rank(comm)

    # rslist = [1, 10]

    # For fast Fa
    rslist = [[0.01, 0.1, 0.5]; 1:2:10]

    # # Standard rslist
    # rslist = [[0.01, 0.1, 0.25, 0.5]; 1:1:10]

    # rslist = [[0.01, 0.1, 0.25, 0.5]; 1:0.5:10]
    beta = 40.0

    ### Static (Yukawa) interaction

    # test_yukawa_tree_level_neft()
    # test_yukawa_one_loop_neft()
    # return

    z_renorm = false
    plots = true
    debug = true
    verbose = true
    show_progress = true

    # Yukawa interaction
    isDynamic = false

    # ftype = "Fs"  # f^{Di} + f^{Ex} / 2
    ftype = "Fa"  # f^{Ex} / 2
    ftypestr = ftype == "Fs" ? "F^{s}" : "F^{a}"

    interactionstr = isDynamic ? "" : "_yukawa"
    zstr = z_renorm ? "_z_renorm" : ""

    # nk ≈ na ≈ 100 is sufficiently converged for all relevant euv/rtol
    Nk, Ok = 7, 6
    Na, Oa = 8, 7

    # DLR parameters for which r(q, 0) is smooth in the q → 0 limit (tested for rs = 1, 10)
    euv = 10.0
    rtol = 1e-7

    FsDMCs, FaDMCs, F1NEFTs, Fs2NEFTs, Fa2NEFTs, Fuu2NEFTs, Fud2NEFTs =
        get_yukawa_one_loop_neft(rslist, beta; neval=1e5)

    FstotalNEFTs = F1NEFTs .+ Fs2NEFTs
    FatotalNEFTs = F1NEFTs .+ Fa2NEFTs
    FuutotalNEFTs = F1NEFTs .+ Fuu2NEFTs
    FudtotalNEFTs = F1NEFTs .+ Fud2NEFTs
    F2NEFTs = ftype == "Fs" ? Fs2NEFTs : Fa2NEFTs
    FtotalNEFTs = ftype == "Fs" ? FstotalNEFTs : FatotalNEFTs

    # Get Thomas-Fermi result for F1 using exact expression
    rs_exact = LinRange(0, 10, 1000)
    F1s_exact = get_F1_TF.(rs_exact)

    # Get Thomas-Fermi result for F1 using exact expression
    function F2ct_exact(rs)
        xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
        rstilde = rs * alpha_ueg / π
        F1 = get_F1_TF(rs)
        A = Interp.integrate1D(
            lindhard.(xgrid) .* (xgrid * rstilde) ./ (xgrid .^ 2 .+ rstilde),
            xgrid,
        )
        B = Interp.integrate1D(lindhard.(xgrid) .* xgrid, xgrid)
        return 2 * F1 * A + F1^2 * B
    end
    F2cts_exact = F2ct_exact.(rs_exact)

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    ax.errorbar(
        rslist,
        Measurements.value.(F1NEFTs),
        Measurements.uncertainty.(F1NEFTs);
        label="\$F_1 \\xi\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["black"],
    )
    ax.errorbar(
        rslist,
        Measurements.value.(Fuu2NEFTs),
        Measurements.uncertainty.(Fuu2NEFTs);
        label="\$F^{\\uparrow\\uparrow}_2 \\xi^2\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["blue"],
    )
    ax.errorbar(
        rslist,
        Measurements.value.(Fud2NEFTs),
        Measurements.uncertainty.(Fud2NEFTs);
        label="\$F^{\\uparrow\\downarrow}_2 \\xi^2\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["cyan"],
    )
    ax.errorbar(
        rslist,
        Measurements.value.(FuutotalNEFTs),
        Measurements.uncertainty.(FuutotalNEFTs);
        label="\$F^{\\uparrow\\uparrow}\$ to \$\\mathcal{O}(\\xi^2)\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["orange"],
    )
    ax.errorbar(
        rslist,
        Measurements.value.(FudtotalNEFTs),
        Measurements.uncertainty.(FudtotalNEFTs);
        label="\$F^{\\uparrow\\downarrow}\$ to \$\\mathcal{O}(\\xi^2)\$",
        capthick=1,
        capsize=4,
        fmt="o-",
        ms=5,
        color=cdict["magenta"],
    )
    ax.set_xlabel("\$r_s\$")
    # ax.set_ylabel("\$F^{\\sigma_1 \\sigma_2}\$")
    if isDynamic == false && ftype == "Fs"
        ax.set_ylim(-0.42, 0.42)
    end
    # ax.set_ylabel("\$F^s\$")
    ax.legend(; ncol=2, fontsize=12, loc="best", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_F_uu_and_ud_yukawa_vs_rs$(interactionstr)_neft.pdf")
    plt.close(fig)
    return

    fig, ax = plt.subplots(; figsize=(5, 5))
    # NEFT benchmark
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(F1NEFTs),
        Measurements.value.(F1NEFTs),
        Measurements.uncertainty.(F1NEFTs);
        label="\${$ftypestr}_1 \\xi\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["cyan"],
    )
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(F2NEFTs),
        Measurements.value.(F2NEFTs),
        Measurements.uncertainty.(F2NEFTs);
        label="\${$ftypestr}_2 \\xi^2\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["magenta"],
    )
    ax.errorbar(
        rslist,
        # 1 .+ Measurements.value.(FtotalNEFTs),
        Measurements.value.(FtotalNEFTs),
        Measurements.uncertainty.(FtotalNEFTs);
        label="\${$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$ (NEFT)",
        # label="\$\\kappa_0 / \\kappa \\approx 1 + \${$ftypestr}_1 \\xi + \${$ftypestr}_2 \\xi^2\$ (NEFT)",
        capthick=1,
        capsize=4,
        fmt="o",
        ms=5,
        color=cdict["teal"],
    )
    ax.set_xlabel("\$r_s\$")
    ax.set_ylabel("\$$ftypestr\$")
    # ax.set_ylabel("\${\\kappa_0}/{\\kappa} \\approx 1 + F^s\$")
    if isDynamic == false && ftype == "Fs"
        # ax.set_ylim(0.58, 1.42)
        ax.set_ylim(-0.42, 0.42)
    end
    # ax.set_ylabel("\$F^s\$")
    ax.legend(; ncol=2, fontsize=12, loc="best", columnspacing=0.5)
    plt.tight_layout()
    fig.savefig("oneshot_one_loop_$(ftype)_yukawa_vs_rs$(interactionstr)_benchmark.pdf")
    plt.close(fig)

    Fs_DMCs = []
    Fa_DMCs = []
    F1s = []
    F2vs = []
    F2bs = []
    F2cts = []
    F2zs = []
    F2s = []
    for (i, rs) in enumerate(rslist)
        if debug && rank == root
            testdlr(rs, euv, rtol; verbose=verbose)
        end
        basic_param = Parameter.rydbergUnit(1.0 / beta, rs, 3)
        mass2 = isDynamic ? 1e-5 : basic_param.qTF^2
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
            isDynamic=isDynamic,
            mass2=mass2,
        )
        if debug && rank == root && rs > 0.25
            check_sign_Fs(param)
            check_signs_Fs_Fa(rs, Fs_DMC, Fa_DMC)
        end
        if verbose && rank == root && i == 1
            println_root("nk=$(length(param.qgrid)), na=$(length(param.θgrid))")
            println_root("nk=$(length(param.qgrid)), na=$(length(param.θgrid))")
            println_root("euv=$(param.euv), rtol=$(param.rtol)")
            println_root("\nrs=$(param.rs), beta=$(param.beta), Fs=$(Fs_DMC), Fa=$(Fa_DMC)")
        end
        initialize_one_loop_params!(param)  # precompute the interaction interpoland r(q, iνₘ)
        F1, F2v, F2b, F2ct, F2z, F2 = get_one_loop_Fs(
            param;
            verbose=verbose,
            show_progress=show_progress,
            ftype=ftype,
            z_renorm=z_renorm,
        )
        push!(Fs_DMCs, Fs_DMC)
        push!(Fa_DMCs, Fa_DMC)
        push!(F1s, F1)
        push!(F2vs, F2v)
        push!(F2bs, F2b)
        push!(F2cts, F2ct)
        push!(F2zs, F2z)
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
        println(F2zs)
        println(F2s)

        FsDMCs, FaDMCs, F1NEFTs, Fs2NEFTs, Fa2NEFTs =
            get_yukawa_one_loop_neft(rslist, beta; neval=1e5)
        FstotalNEFTs = F1NEFTs .+ Fs2NEFTs
        FatotalNEFTs = F1NEFTs .+ Fa2NEFTs
        F2NEFTs = ftype == "Fs" ? Fs2NEFTs : Fa2NEFTs
        FtotalNEFTs = ftype == "Fs" ? FstotalNEFTs : FatotalNEFTs

        # Get Thomas-Fermi result for F1 using exact expression
        rs_exact = LinRange(0, 10, 1000)
        F1s_exact = get_F1_TF.(rs_exact)

        # Get Thomas-Fermi result for F1 using exact expression
        function F2ct_exact(rs)
            xgrid =
                CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
            rstilde = rs * alpha_ueg / π
            F1 = get_F1_TF(rs)
            A = Interp.integrate1D(
                lindhard.(xgrid) .* (xgrid * rstilde) ./ (xgrid .^ 2 .+ rstilde),
                xgrid,
            )
            B = Interp.integrate1D(lindhard.(xgrid) .* xgrid, xgrid)
            return 2 * F1 * A + F1^2 * B
        end
        F2cts_exact = F2ct_exact.(rs_exact)

        fig, ax = plt.subplots(; figsize=(5, 5))
        # NEFT benchmark
        ax.errorbar(
            rslist,
            # 1 .+ Measurements.value.(F1NEFTs),
            Measurements.value.(F1NEFTs),
            Measurements.uncertainty.(F1NEFTs);
            label="\${$ftypestr}_1 \\xi\$ (NEFT)",
            capthick=1,
            capsize=4,
            fmt="o",
            ms=5,
            color=cdict["cyan"],
        )
        ax.errorbar(
            rslist,
            # 1 .+ Measurements.value.(F2NEFTs),
            Measurements.value.(F2NEFTs),
            Measurements.uncertainty.(F2NEFTs);
            label="\${$ftypestr}_2 \\xi^2\$ (NEFT)",
            capthick=1,
            capsize=4,
            fmt="o",
            ms=5,
            color=cdict["magenta"],
        )
        ax.errorbar(
            rslist,
            # 1 .+ Measurements.value.(FtotalNEFTs),
            Measurements.value.(FtotalNEFTs),
            Measurements.uncertainty.(FtotalNEFTs);
            label="\${$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$ (NEFT)",
            # label="\$\\kappa_0 / \\kappa \\approx 1 + \${$ftypestr}_1 \\xi + \${$ftypestr}_2 \\xi^2\$ (NEFT)",
            capthick=1,
            capsize=4,
            fmt="o",
            ms=5,
            color=cdict["teal"],
        )
        # Our results
        error = 1e-6 * ones(length(rslist))
        fp1label = ftype == "Fs" ? "\$\\kappa_0 / \\kappa\$" : "\$\\chi_0 / \\chi\$"
        fdata = ftype == "Fs" ? Fs_DMCs : Fa_DMCs
        fp1type = ftype == "Fs" ? "kappa0_over_kappa" : "chi0_over_chi"
        if isDynamic
            ax.plot(
                # spline(rslist, 1.0 .+ fdata, error)...;
                spline(rslist, fdata, error)...;
                label=fp1label,
                color=cdict["grey"],
            )
        end
        ax.plot(
            # spline(rslist, 1.0 .+ F1s, error)...;
            # label="\$1 + {$ftypestr}_1 \\xi\$",
            spline(rslist, F1s, error)...;
            label="\${$ftypestr}_1 \\xi\$",
            color=cdict["orange"],
        )
        ax.plot(
            # spline(rslist, 1.0 .+ F2s, error)...;
            # label="\$1 + {$ftypestr}_2 \\xi^2\$",
            spline(rslist, F2s, error)...;
            label="\${$ftypestr}_2 \\xi^2\$",
            color=cdict["blue"],
        )
        ax.plot(
            # spline(rslist, 1.0 .+ F1s .+ F2s, error)...;
            # label="\$1 + {$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$",
            spline(rslist, F1s .+ F2s, error)...;
            label="\${$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$",
            color=cdict["black"],
        )
        ax.set_xlabel("\$r_s\$")
        ax.set_ylabel("\$$ftypestr\$")
        # ax.set_ylabel("\${\\kappa_0}/{\\kappa} \\approx 1 + F^s\$")
        if isDynamic == false && ftype == "Fs"
            # ax.set_ylim(0.58, 1.42)
            ax.set_ylim(-0.42, 0.42)
        end
        # ax.set_ylabel("\$F^s\$")
        ax.legend(; ncol=2, fontsize=12, loc="best", columnspacing=0.5)
        plt.tight_layout()
        fig.savefig("oneshot_one_loop_$(ftype)_yukawa_vs_rs$(interactionstr).pdf")
        # fig.savefig("kappa0_over_kappa_yukawa_vs_rs.pdf")
        plt.close(fig)

        # Plot spline fits to data vs rs
        fig, ax = plt.subplots(; figsize=(5, 5))
        error = 1e-6 * ones(length(rslist))
        flabel = ftype == "Fs" ? "\$F^{s}_{\\text{DMC}}\$" : "\$F^{a}_{\\text{DMC}}\$"
        fdata = ftype == "Fs" ? Fs_DMCs : Fa_DMCs
        if isDynamic
            ax.plot(spline(rslist, fdata, error)...; label=flabel, color=cdict["grey"])
        end
        ax.plot(
            spline(rslist, F1s, error)...;
            label="\${$ftypestr}_1 \\xi\$",
            color=cdict["orange"],
        )
        ax.plot(
            spline(rslist, F2s, error)...;
            label="\${$ftypestr}_2 \\xi^2\$",
            color=cdict["blue"],
        )
        ax.plot(
            spline(rslist, F1s .+ F2s, error)...;
            label="\${$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$",
            color=cdict["black"],
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
        # Exact CT expression
        ax.plot(
            rs_exact,
            F2cts_exact,
            "--";
            label="\${$ftypestr}_{\\text{ct},2}\$ (exact)",
            color=cdict["red"],
        )
        if z_renorm
            ax.plot(
                spline(rslist, F2zs, error)...;
                label="\${$ftypestr}_{\\text{z},2}\$",
                color=cdict["red"],
            )
        end
        ax.set_xlabel("\$r_s\$")
        ax.set_xlim(0, maximum(rslist))
        if isDynamic && ftype == "Fs"
            ax.set_ylim(-5.5, 5.5)
        elseif ftype == "Fs"
            ax.set_ylim(-0.42, 0.42)
        end
        ax.legend(;
            ncol=2,
            loc="upper left",
            fontsize=12,
            title_fontsize=16,
            title=isDynamic ?
                  "\$\\Lambda_\\text{UV} = $(Int(round(euv)))\\epsilon_F, \\varepsilon = 10^{$(Int(round(log10(rtol))))}\$" :
                  nothing,
        )
        fig.tight_layout()
        fig.savefig(
            "oneshot_one_loop_$(ftype)_vs_rs_euv=$(euv)_rtol=$(rtol)$(zstr)$(interactionstr).pdf",
        )
        plt.close(fig)

        # Plot spline fits to just 1+FDMC, 1+F1, 1+F2 and 1+F1+F2 vs rs
        fig, ax = plt.subplots(; figsize=(5, 5))
        error = 1e-6 * ones(length(rslist))
        fp1label = ftype == "Fs" ? "\$\\kappa_0 / \\kappa\$" : "\$\\chi_0 / \\chi\$"
        fdata = ftype == "Fs" ? Fs_DMCs : Fa_DMCs
        fp1type = ftype == "Fs" ? "kappa0_over_kappa" : "chi0_over_chi"
        if isDynamic
            # ax.plot(spline(rslist, 1.0 .+ fdata, error)...; label=flabel, color=cdict["grey"])
            ax.plot(
                spline(rslist, 1.0 .+ fdata, error)...;
                label=fp1label,
                color=cdict["grey"],
            )
        end
        ax.plot(
            spline(rslist, 1.0 .+ F1s, error)...;
            label="\$1 + {$ftypestr}_1 \\xi\$",
            color=cdict["orange"],
        )
        ax.plot(
            spline(rslist, 1.0 .+ F2s, error)...;
            label="\$1 + {$ftypestr}_2 \\xi^2\$",
            color=cdict["blue"],
        )
        ax.plot(
            spline(rslist, 1.0 .+ F1s .+ F2s, error)...;
            label="\$1 + {$ftypestr}_1 \\xi + {$ftypestr}_2 \\xi^2\$",
            color=cdict["black"],
        )
        ax.set_xlabel("\$r_s\$")
        ax.set_xlim(0, maximum(rslist))
        if isDynamic && ftype == "Fs"
            ax.set_ylim(-0.5, 2.0)
        end
        ax.legend(;
            ncol=2,
            loc="best",
            fontsize=12,
            title_fontsize=16,
            title=isDynamic ?
                  "\$\\Lambda_\\text{UV} = $(Int(round(euv)))\\epsilon_F, \\varepsilon = 10^{$(Int(round(log10(rtol))))}\$" :
                  nothing,
        )
        fig.tight_layout()
        fig.savefig(
            # "oneshot_one_loop_$(ftype)_vs_rs_euv=$(euv)_rtol=$(rtol)$(zstr)$(interactionstr)_zoom.pdf",
            "oneshot_one_loop_$(fp1type)_vs_rs_euv=$(euv)_rtol=$(rtol)$(zstr)$(interactionstr)_zoom.pdf",
        )
        plt.close(fig)
    end
    MPI.Finalize()
    return
end

main()
