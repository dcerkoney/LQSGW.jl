using BenchmarkTools
using Colors
using CompositeGrids
using ElectronGas
using ElectronLiquid
using GreenFunc
using JLD2
using LinearAlgebra
using LsqFit
using Lehmann
using LQSGW
using Parameters
using ProgressMeter
using PyCall
using PyPlot
using Roots

@pyimport numpy as np   # for saving/loading numpy data
@pyimport scienceplots  # for style "science"
@pyimport scipy.interpolate as interp

const alpha_ueg = (4 / 9π)^(1 / 3)

# Mapping of interaction types to Landau parameters
const int_type_to_landaufunc = Dict(
    :rpa => Interaction.landauParameter0,
    :ko_const => Interaction.landauParameterConst,
    :ko_moroni => Interaction.landauParameterMoroni,
    :ko_simion_giuliani_plus => Interaction.landauParameterSimionGiulianiPlus,
)

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
    mass2::Float64 = 1e-6     # fictitious Yukawa screening λ
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

    Nk::Int = 7
    Ok::Int = 6

    Na::Int = 8
    Oa::Int = 7

    # We precompute δR(q, iνₘ) on a mesh of ~100 k-points
    # NOTE: EL.jl default is `Nk, Ok = 16, 16` (~700 k-points)
    qgrid_interp::MomInterpGridType =
        CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 16, 0.01 * kF, 16)
    # CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], Nk, 0.01 * kF, Ok)

    # Later, we integrate δR(q, iνₘ) on a Gaussian mesh of ~100 k-points
    qgrid::MomGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxQ], [0.0, 2 * kF], Nk, 0.01 * kF, Ok)
    # CompositeGrid.LogDensedGrid(:gauss, [0.0, maxQ], [0.0, 2 * kF], 16, 0.01 * kF, 16)

    # Sparse angular grids (~100 points each)
    # NOTE: EL.jl default is `Na, Oa = 16, 32` (~1000 θ/φ-points)
    # θgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 1e-6, 32)
    # φgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], 16, 1e-6, 32)
    θgrid::AngularGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], Na, 1e-6, Oa)
    # CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 16, 1e-6, 16)
    φgrid::AngularGridType =
        CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], Na, 1e-6, Oa)
    # CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], 16, 1e-6, 16)

    # Use a sparse DLR grid for the bosonic Matsubara summation (~30-50 iνₘ-points)
    dlr::DLRGrid{Float64,:ph} =
        DLRGrid(; Euv=100 * EF, β=β, rtol=1e-10, isFermi=false, symmetry=:ph)
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
    NF_times_Rp_ex = coeff / (x^2 + coeff * lindhard(x))
    return -x * NF_times_Rp_ex
end

function NF_times_R_static(param::OneLoopParams, q)
    @unpack rs, kF, NF, Fs = param
    x = q / (2 * kF)
    rs_tilde = rs * alpha_ueg / π
    if isinf(rs_tilde)
        return -x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    NF_times_R_static_exact = coeff / (x^2 + coeff * lindhard(x)) - Fs
    return NF_times_R_static_exact
end

function plot_integrand_F1(param::OneLoopParams)
    @unpack beta, dim = param
    rslist = [0, 1, 5, 10, 20, Inf]
    colorlist = [
        cdict["grey"],
        cdict["orange"],
        cdict["blue"],
        cdict["cyan"],
        cdict["magenta"],
        "black",
    ]
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 1000)
    for (rs, color) in zip(rslist, colorlist)
        param_this_rs = Parameter.rydbergUnit(1.0 / beta, rs, dim)
        Fs_RPA = 0.0
        Fs = -get_Fs(param_this_rs)
        if rs > 0.25 && rs != Inf
            @assert Fs ≤ 0 "Incorrect sign for Fs"
        end
        for (Fs, linestyle) in zip([Fs_RPA, Fs], ["--", "-"])
            if linestyle == "--"
                label = nothing
            else
                rsstr = rs == Inf ? "\\infty" : string(Int(rs))
                label = "\$r_s = $rsstr\$"
            end
            y = [integrand_F1(xi, rs * alpha_ueg / π, Fs) for xi in x]
            ax.plot(x, y; linestyle=linestyle, color=color, label=label)
        end
    end
    ax.set_xlabel("\$x\$")
    ax.set_ylabel("\$I_0[W](x)\$")
    ax.legend(; fontsize=10, loc="best", ncol=2)
    xlim(0, 1)
    ylim(-2.125, 1.125)
    # tight_layout()
    fig.savefig("integrand_F1.pdf")
    plt.close("all")
end

function get_analytic_F1(param::OneLoopParams, rslist; plot=false)
    @unpack beta, dim = param
    F1_RPA = []
    F1_R = []
    Fslist = []
    Fssclist = []
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    y0_RPA_inf = [integrand_F1(x, Inf) for x in xgrid]
    F1_RPA_inf = Interp.integrate1D(y0_RPA_inf, xgrid)
    println("F⁺₀[W₀](∞) = $(F1_RPA_inf)")
    for rs in rslist
        param_this_rs = Parameter.rydbergUnit(1.0 / beta, rs, dim)
        rstilde = rs * alpha_ueg / π
        Fs = -get_Fs(param_this_rs)
        Fs_sc = get_tree_level_self_consistent_Fs(param_this_rs)
        if rs > 0.25
            @assert Fs ≤ 0 "Incorrect sign for Fs!"
        end
        # RPA
        y_RPA = [integrand_F1(x, rstilde) for x in xgrid]
        val_RPA = Interp.integrate1D(y_RPA, xgrid)
        push!(F1_RPA, val_RPA)
        # KO+
        y_R = [integrand_F1(x, rstilde, Fs) for x in xgrid]
        val_R = (Fs / 2) + Interp.integrate1D(y_R, xgrid)
        push!(F1_R, val_R)
        push!(Fslist, Fs)
        push!(Fssclist, Fs_sc)
    end
    if plot
        fig, ax = plt.subplots()
        ax.plot(rslist, F1_RPA; color=cdict["red"], label="\$W_0\$")
        ax.plot(rslist, F1_R; color=cdict["blue"], label="\$R\$")
        ax.plot(rslist, Fssclist; color=cdict["teal"], label="\$F^+_\\text{tl-sc}\$")
        ax.plot(
            rslist,
            Fslist;
            color=cdict["grey"],
            label="\$F^+ = \\kappa_0 / \\kappa - 1\$",
            zorder=-1,
        )
        xlabel("\$r_s\$")
        ylabel("\$\\widetilde{F}^+_0[W]\$")
        xlim(0, 10)
        ylim(-1.6, 0.3)
        ax.legend(; fontsize=10, loc="best")
        # tight_layout()
        fig.savefig("analytic_F1.pdf")
        plt.close("all")
    end
    return F1_RPA, F1_R
end

"""
Solve I0[F+] = F+ / 2 to obtain the tree-level self-consistent value for F⁰ₛ.
"""
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
    sigma = SelfEnergy.G0W0(
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
    return sigma
end

"""
Leading-order (one-loop) correction to Z_F.
"""
function Z1(param::OneLoopParams, kgrid::KGT) where {KGT<:AbstractVector}
    sigma1 = Σ1(param, kgrid)
    return zfactor_fermi(param.basic, sigma1)  # compute Z_F using improved finite-temperature scaling
end

"""
Tree-level estimate of F⁺₀ ~ ⟨R(k - k', 0)⟩.
"""
function F1(param::OneLoopParams)
    @unpack rs, kF, EF, NF, Fs, basic = param
    rstilde = rs * alpha_ueg / π
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    y = [integrand_F1(x, rstilde, Fs) for x in xgrid]
    F1 = (Fs / 2) + Interp.integrate1D(y, xgrid)
    return F1
end

function integrand_F1(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return -x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    NF_times_Rp_ex = coeff / (x^2 + coeff * lindhard(x))
    return -x * NF_times_Rp_ex
end

# 2R(z1 - f1 Π0) - f1 Π0 f1
function one_loop_counterterms(param::OneLoopParams; kwargs...)
    @unpack rs, kF, EF, NF, Fs, basic = param
    rstilde = rs * alpha_ueg / π
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)

    # -x N_F R(2kF x, 0), where x = |k - k'| / 2kF
    integrand = [integrand_F1(x, rstilde, Fs) for x in xgrid]
    F1 = (Fs / 2) + Interp.integrate1D(integrand, xgrid)  # NF * ⟨R⟩

    # x R(2kF x, 0)
    x_R0 = x_NF_R0 / NF

    # Z_1(kF)
    z1 = Z1(param, 2 * kF * xgrid)

    # Π₀(q, iν=0) = -NF * 𝓁(q / 2kF)
    Π0 = -NF * lindhard.(xgrid)

    # α = z₁ + 2 ∫₀¹ dx x R(x, 0) Π₀(x, 0)
    alpha = z1 + Interp.integrate1D(2 * x_R0 .* Π0, xgrid)

    # β = ∫₀¹ dx x Π₀(x, 0) / NF
    beta = Interp.integrate1D(xgrid .* Π0 / NF, xgrid)

    # 2R(z1 - f1 Π0) - f1 Π0 f1 = 2 F1 α + F1² β
    vertex_cts = 2 * F1 * alpha + F1^2 * beta
    return vertex_cts
end

function Π0_static(param::OneLoopParams, qgrid::QGT) where {QGT<:AbstractGrid}
    @unpack kF, NF = param
    xgrid = qgrid.mesh[1].grid / (2 * kF)
    return -NF * lindhard.(xgrid)
end

"""
The non-interacting (renormalized) Green's function G₀.
As a first approximation, we neglect the mass renormalization, setting m* = m.
"""
function G0_data(
    param::OneLoopParams,
    kgrid::KGT,
    wgrid::WGT,
) where {KGT<:AbstractGrid,WGT<:AbstractGrid}
    @unpack me, β, μ, kF = param
    g0 = GreenFunc.MeshArray(wgrid, kgrid; dtype=ComplexF64)
    for ind in eachindex(g0)
        iw, ik = ind[1], ind[2]
        g0[ind] = 1.0 / (im * wgrid[iw] - kgrid[ik]^2 / (2 * me) + μ)
    end
    return g0
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
    @unpack β, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax, iw0 = param

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
        # println(dR(param, q, m))
        s_ivm[i] = (
            R(param, q, m) * G0(param, p1, iw0 + im * vm) * G0(param, p2, iw0 + im * vm) / β
        )
    end

    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

function box_matsubara_summand(param::OneLoopParams, q, θ, φ)
    @unpack β, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax, iw0 = param

    # p1 = |k + q'|, p2 = |k' - q'|, qex = |k - k' + q'|
    k1vec = [0, 0, kamp1]
    k2vec = kamp2 * [sin(θ12), 0, cos(θ12)]
    qvec = q * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    vec_p1 = k1vec + qvec
    vec_p2 = k2vec - qvec
    vec_qex = k1vec - k2vec + qvec
    p1 = norm(vec_p1)
    p2 = norm(vec_p2)
    qex = norm(vec_qex)

    # S(iνₘ) = dR(q', iν'ₘ) * dR(k - k' + q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p2, iω₀ - iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        # println(dR(param, q, m))
        s_ivm[i] = (
            R(param, q, m) *
            R(param, qex, m) *
            G0(param, p1, iw0 + im * vm) *
            G0(param, p2, iw0 + im * vm) / β
        )
    end

    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    summand = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)
    return summand
end

function g1g2_pp_matsubara_summand(param::OneLoopParams, q, θ, φ)
    @unpack β, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax, iw0 = param

    # p1 = |k + q'|, p2 = |k' + q'|
    vec_p1 = [0, 0, kamp1] + q * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    vec_p2 = kamp2 * [sin(θ12), 0, cos(θ12)] + q * [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
    p1 = norm(vec_p1)
    p2 = norm(vec_p2)

    # S(iνₘ) = g(p1, iω₀ + iν'ₘ) * g(p2, iω₀ + iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, vm) in enumerate(vmgrid)
        s_ivm[i] = G0(param, p1, iw0 + im * vm) * G0(param, p2, iw0 + im * vm) / β
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

function box_matsubara_sum(param::OneLoopParams, q, θ, φ)
    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    summand = box_matsubara_summand(param, q, θ, φ)
    matsubara_sum = summand[1] + 2 * sum(summand[2:end])
    return matsubara_sum
end

function plot_g1g2_pp_matsubara_summand(param::OneLoopParams)
    @assert param.initialized "δR(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF = param
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
            summand = g1g2_pp_matsubara_summand(param, coord...)
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
        ax.set_ylabel("\$S_\\mathbf{q}(i\\nu_m) = g(k_1 + q) g(k_2 + q)\$")
        ax.set_xlim(0, 4)
        ax.legend(;
            loc="best",
            fontsize=14,
            title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
        )
        fig.savefig("vertex_matsubara_summand_Gp1_Gp2_q=$qstr.pdf")
    end
    return
end

function plot_vertex_matsubara_summand(param::OneLoopParams)
    @assert param.initialized "δR(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF = param
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
        # ax.set_ylabel("\$S^\\mathbf{q}_\\mathbf{k,k^\\prime}(i\\nu_m)\$")
        ax.set_xlim(0, 4)
        ax.legend(;
            loc="best",
            fontsize=14,
            title="\$\\mathbf{k}_1 = k_F\\mathbf{\\hat{z}}, \\mathbf{k}_2 = k_F\\mathbf{\\hat{x}}, $qlabel\$",
        )
        # fig.tight_layout()
        fig.savefig("vertex_matsubara_summand_q=$qstr.pdf")
    end
    return
end

function plot_vertex_matsubara_sum(param::OneLoopParams)
    @assert param.initialized "δR(q, iνₘ) data not yet initialized!"
    @unpack β, kF, EF, Mmax, Q_CUTOFF, θgrid = param
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
            # ax.set_ylabel("\$S_\\mathbf{q}(i\\nu_m)\$")
            # ax.set_ylabel("\$S^\\mathbf{q}_\\mathbf{k,k^\\prime}(i\\nu_m)\$")
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
            fig.savefig("$(clabel)_vertex_matsubara_sum_q=$(qstr).pdf")
        end
    end
end

# 2RΛ₁
function one_loop_vertex_corrections(param::OneLoopParams; show_progress=true)
    @assert param.initialized "δR(q, iνₘ) data not yet initialized!"
    @unpack qgrid, θgrid, φgrid, θ12, rs, Fs, kF, NF, basic, paramc = param
    q_integrand = Vector{ComplexF64}(undef, length(qgrid.grid))
    θ_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    φ_integrand = Vector{ComplexF64}(undef, length(φgrid.grid))
    # integrate over loop momentum q
    progress_meter = Progress(
        length(qgrid.grid) * length(θgrid.grid) * length(φgrid.grid);
        # desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress,
    )
    for (iq, q) in enumerate(qgrid)
        for (iθ, θ) in enumerate(θgrid)
            for (iφ, φ) in enumerate(φgrid)
                φ_integrand[iφ] = vertex_matsubara_sum(param, q, θ, φ)
                next!(progress_meter)
            end
            θ_integrand[iθ] = Interp.integrate1D(φ_integrand, φgrid)
        end
        q_integrand[iq] = Interp.integrate1D(θ_integrand .* sin.(θgrid.grid), θgrid)
    end
    finish!(progress_meter)
    vertex_integrand = q_integrand .* qgrid.grid .* qgrid.grid / (2π)^3

    # N_F R(2kF x), where x = |k - k'| / 2kF
    v = cos(θ12)
    x = sqrt((1 - v) / 2.0)  # 
    rs_tilde = rs * alpha_ueg / π
    NF_R = NF_times_R_static(param, 2 * kF * x)

    result = Interp.integrate1D(vertex_integrand, qgrid) * NF_R
    println(NF_R)
    println(result)

    v = cos(θ12)
    x = sqrt((1 - v) / 2.0)  # 
    NF_R = R(param, 2 * kF * x, 0) * NF

    result = Interp.integrate1D(vertex_integrand, qgrid) * NF_R
    println(NF_R)
    println(result)
    return result
end

# gg'RR' + exchange counterpart
function one_loop_box_diagrams(param::OneLoopParams; show_progress=true)
    @assert param.initialized "δR(q, iνₘ) data not yet initialized!"
    @unpack qgrid, θgrid, φgrid, basic, paramc = param
    q_integrand = Vector{ComplexF64}(undef, length(qgrid.grid))
    θ_integrand = Vector{ComplexF64}(undef, length(θgrid.grid))
    φ_integrand = Vector{ComplexF64}(undef, length(φgrid.grid))
    # integrate over loop momentum q
    progress_meter = Progress(
        length(qgrid.grid) * length(θgrid.grid) * length(φgrid.grid);
        # desc="Progress (rank = 0): ",
        output=stdout,
        showspeed=true,
        enabled=show_progress,
    )
    for (iq, q) in enumerate(qgrid)
        for (iθ, θ) in enumerate(θgrid)
            for (iφ, φ) in enumerate(φgrid)
                φ_integrand[iφ] = box_matsubara_sum(param, q, θ, φ)
                next!(progress_meter)
            end
            θ_integrand[iθ] = Interp.integrate1D(φ_integrand, φgrid)
        end
        q_integrand[iq] = Interp.integrate1D(θ_integrand .* sin.(θgrid.grid), θgrid)
    end
    finish!(progress_meter)
    rq = [UEG.KOinstant(q, paramc) for q in qgrid.grid]
    box_integrand = q_integrand .* qgrid.grid .* qgrid.grid * param.NF / (2π)^3
    result = Interp.integrate1D(box_integrand, qgrid)
    return result
end

function test_vertex_integral(param::OneLoopParams; show_progress=true)
    result = one_loop_vertex_corrections(param; show_progress=show_progress)
    return result
end

function test_box_integral(param::OneLoopParams; show_progress=true)
    result = one_loop_box_diagrams(param; show_progress=show_progress)
    return result
end

# linear interpolation with mixing parameter α: x * (1 - α) + y * α
function lerp(x, y, alpha)
    return (1 - alpha) * x + alpha * y
end

function get_one_loop_integrand(param::OneLoopParams; kwargs...)
    function one_loop_integrand(param; kwargs...)
        return one_loop_vertex_corrections(param; kwargs...) +
               one_loop_box_diagrams(param; kwargs...) +
               one_loop_counterterms(param; kwargs...)
    end
    return one_loop_integrand
end

function get_one_loop_Fs(param::OneLoopParams; kwargs...)
    integrand = get_one_loop_integrand(param; kwargs...)
    F2 = missing
    return F2
end

function plot_F2v_convergence_vs_nk(; rs=1.0, beta=40.0, verbose=true)
    NOk_pairs = [
        # (3, 4),  # Nq ≈ 25
        (4, 4),  # Nq ≈ 30
        (5, 4),  # Nq ≈ 50
        (5, 5),  # 
        (6, 5),  # Nq ≈ 75
        (6, 6),  # 
        (7, 6),  # Nq ≈ 100
        (9, 9),  
    ]
    NOa_pairs = [
        # (4, 4),  # Nθ = Nϕ ≈ 25
        (5, 4),  # Nθ = Nϕ ≈ 30
        (6, 5),  # Nθ = Nϕ ≈ 50
        (6, 6),  # 
        (7, 6),  # Nθ = Nϕ ≈ 75
        (7, 7),  # 
        (8, 7),  # Nθ = Nϕ ≈ 100
        (10, 10),  
    ]
    nks = []
    F2s_rpa = []
    F2s_kop = []
    for ((Nk, Ok), (Na, Oa)) in zip(NOk_pairs, NOa_pairs)
        # RPA
        param_rpa = OneLoopParams(; rs=rs, beta=beta, Nk=Nk, Ok=Ok, Na=Na, Oa=Oa)

        # KO+
        Fs = -get_Fs(param_rpa.basic)  # ~ -2 when rs=10, -0.95 when rs=5, and -0.17 when rs=1
        param_kop = OneLoopParams(; rs=rs, beta=beta, Fs=Fs, Nk=Nk, Ok=Ok, Na=Na, Oa=Oa)
        check_sign_Fs(param_kop)

        # Total number of momentum  amplitude and angular grid points
        nk = length(param_rpa.qgrid)
        na = length(param_rpa.θgrid)

        # Precompute the interaction δR(q, iνₘ)
        initialize_one_loop_params!(param_rpa)
        initialize_one_loop_params!(param_kop)

        verbose && println("(nk = $nk, na = $na) One-loop vertex part:")
        F1_rpa = F1(param_rpa)
        F1_kop = F1(param_kop)
        F2_rpa = test_vertex_integral(param_rpa)
        verbose && println("(RPA) $(F1_rpa) ξ + $F2_rpa ξ²")
        F2_kop = test_vertex_integral(param_kop)
        verbose && println("(KO+) $(F1_kop) ξ + $(F2_kop) ξ²")

        push!(nks, nk)
        push!(F2s_rpa, F2_rpa)
        push!(F2s_kop, F2_kop)
    end

    # Convergence plot of F2 vs nk
    fig, ax = plt.subplots()
    ax.plot(
        nks,
        real(F2s_rpa);
        label="RPA",
        marker="o",
        markersize=4,
        markerfacecolor="none",
    )
    ax.plot(
        nks,
        real(F2s_kop);
        label="\$\\text{KO}_+\$",
        marker="o",
        markersize=4,
        markerfacecolor="none",
    )
    ax.set_xlabel("\$N_k\$")
    ax.set_ylabel("\$F^\\text{v}_2(\\theta_{12} = \\pi / 2)\$")
    ax.legend(; loc="best", fontsize=14)
    # fig.tight_layout()
    fig.savefig("F2_vs_nk_rs=$(rs)_beta=$(beta).pdf")

    # RPA convergence plot of F2 vs nk
    fig, ax = plt.subplots()
    ax.plot(
        nks,
        real(F2s_rpa);
        label="RPA",
        marker="o",
        markersize=4,
        markerfacecolor="none",
    )
    ax.set_xlabel("\$N_k\$")
    ax.set_ylabel("\$F^\\text{v}_2(\\theta_{12} = \\pi / 2)\$")
    ax.legend(; loc="best", fontsize=14)
    # fig.tight_layout()
    fig.savefig("F2_vs_nk_rs=$(rs)_beta=$(beta)_rpa.pdf")

    # KO+ convergence plot of F2 vs nk
    fig, ax = plt.subplots()
    ax.plot(
        nks,
        real(F2s_kop);
        label="\$\\text{KO}_+\$",
        marker="o",
        markersize=4,
        markerfacecolor="none",
    )
    ax.set_xlabel("\$N_k\$")
    ax.set_ylabel("\$F^\\text{v}_2(\\theta_{12} = \\pi / 2)\$")
    ax.legend(; loc="best", fontsize=14)
    # fig.tight_layout()
    fig.savefig("F2_vs_nk_rs=$(rs)_beta=$(beta)_kop.pdf")
    return
end

function check_sign_Fs(param::OneLoopParams)
    # ElectronLiquid.jl sign convention: Fs < 0
    @unpack Fs, paramc = param
    if param.rs > 0.25
        @assert Fs < 0 "Fs = $Fs must be negative in the ElectronLiquid convention!"
        @assert paramc.Fs < 0 "Fs = $Fs must be negative in the ElectronLiquid convention!"
    else
        println("WARNING: when rs is nearly zero, we cannot check the sign of Fs!")
    end
end

function plot_R_and_W0(param_rpa::OneLoopParams, param_kop::OneLoopParams)
    fig, ax = plt.subplots()
    labels = ["\$W_0\$", "\$R\$"]
    params = [param_rpa, param_kop]
    colorlists = [[cdict["red"], cdict["orange"]], [cdict["blue"], cdict["teal"]]]
    for (label, colors, param) in zip(labels, colorlists, params)
        @unpack kF, NF, θ12, rs, Fs, θgrid, maxQ = param
        qgrid_fine = LinRange(0.0, maxQ, 2000)
        NF_times_R_static_exact = [NF_times_R_static(param, q) for q in qgrid_fine]
        NF_times_R_static_interp_R = [R(param, q, 0) * NF for q in qgrid_fine]
        ax.plot(qgrid_fine / kF, NF_times_R_static_exact; color=colors[1], label=label)
        ax.plot(
            qgrid_fine / kF,
            NF_times_R_static_interp_R;
            color=colors[2],
            linestyle="--",
        )
        xlim(0, maxQ / kF)
    end
    xlabel("\$q / k_F\$")
    ylabel("\$\\mathcal{N}_F W(q, i\\nu_m = 0)\$")
    ax.legend(; loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig("NF_times_R_and_W0_comparison.pdf")
    plt.close("all")
end

function plot_R(param::OneLoopParams)
    @unpack kF, NF, θ12, rs, Fs, θgrid, maxQ = param
    qgrid_fine = LinRange(0.0, maxQ, 2000)
    intnname = Fs == 0.0 ? "W_0" : "R"
    intnstr = Fs == 0.0 ? "W0" : "R"
    NF_times_R_static_exact = [NF_times_R_static(param, q) for q in qgrid_fine]
    NF_times_R_static_interp_R = [R(param, q, 0) * NF for q in qgrid_fine]
    NF_times_R_static_interp_dR = [R_from_dR(param, q, 0) * NF for q in qgrid_fine]
    fig, ax = plt.subplots()
    ax.plot(
        qgrid_fine / kF,
        NF_times_R_static_interp_R;
        color=cdict["blue"],
        label="interp",
    )
    # ax.plot(
    #     qgrid_fine / kF,
    #     NF_times_R_static_interp_dR;
    #     color=cdict["teal"],
    #     label="interp \$\\delta $(intnname)\$",
    # )
    ax.plot(
        qgrid_fine / kF,
        NF_times_R_static_exact;
        color=cdict["red"],
        label="exact",
        linestyle="--",
    )
    xlabel("\$q / k_F\$")
    ylabel("\$\\mathcal{N}_F $(intnname)(q, i\\nu_m = 0)\$")
    xlim(0, maxQ / kF)
    ax.legend(; loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig("NF_times_$(intnstr)_comparison.pdf")
    plt.close("all")
end

function plot_dR(param::OneLoopParams)
    @unpack kF, NF, θ12, rs, Fs, θgrid, maxQ, paramc = param
    qgrid_fine = LinRange(0.0, maxQ, 2000)
    intnname = Fs == 0.0 ? "W_0" : "R"
    intnstr = Fs == 0.0 ? "W0" : "R"
    R_static_exact = [NF_times_R_static(param, q) / NF for q in qgrid_fine]
    r_exact = [UEG.KOinstant(q, paramc) for q in qgrid_fine]
    dR_static_exact = R_static_exact ./ r_exact
    dR_static_interp_dR = [dR(param, q, 0) for q in qgrid_fine]
    dR_static_interp_R = [dR_from_R(param, q, 0) for q in qgrid_fine]
    fig, ax = plt.subplots()
    # ax.plot(
    #     qgrid_fine / kF,
    #     dR_static_interp_dR;
    #     color=cdict["blue"],
    #     label="interp \$\\delta $(intnname)\$",
    # )
    ax.plot(qgrid_fine / kF, dR_static_interp_R; color=cdict["teal"], label="interp")
    ax.plot(
        qgrid_fine / kF,
        dR_static_exact;
        color=cdict["red"],
        label="exact",
        linestyle="--",
    )
    xlabel("\$q / 2k_F\$")
    ylabel("\$\\delta $(intnname)(q, i\\nu_m = 0)\$")
    xlim(0, maxQ / kF)
    ax.legend(; loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig("d$(intnstr)_comparison.pdf")
    plt.close("all")
end

function main()
    rs = 10.0
    beta = 40.0
    verbose = true

    plot_F2v_convergence_vs_nk(; rs=rs, beta=beta, verbose=true)
    return

    # # nk ≈ 50
    # Nk, Ok = 5, 4
    # Na, Oa = 6, 5

    # # nk ≈ 100
    # Nk, Ok = 7, 6
    # Na, Oa = 8, 7

    # RPA
    param_rpa = OneLoopParams(; rs=rs, beta=beta, Nk=Nk, Ok=Ok, Na=Na, Oa=Oa)

    # ~ -2 when rs=10, -0.95 when rs=5, and -0.17 when rs=1
    Fs = -get_Fs(param_rpa.basic)

    # Fs = -1.0

    # KO+
    param_kop = OneLoopParams(; rs=rs, beta=beta, Fs=Fs, Nk=Nk, Ok=Ok, Na=Na, Oa=Oa)
    check_sign_Fs(param_kop)

    # Total number of momentum  amplitude and angular grid points
    nq = length(param_rpa.qgrid)
    nt = length(param_rpa.θgrid)
    np = length(param_rpa.θgrid)
    println("nq = $nq, nt = $nt, np = $np")

    # Precompute the interaction δR(q, iνₘ)
    initialize_one_loop_params!(param_rpa)
    initialize_one_loop_params!(param_kop)

    plot_vertex_matsubara_summand(param_rpa)
    plot_vertex_matsubara_sum(param_rpa)

    # Plots of the total effective interaction R = r δR = (v + f) δR
    plot_R(param_rpa)
    plot_R(param_kop)

    # Plot of R and W0 together
    plot_R_and_W0(param_rpa, param_kop)

    println("(nq = $nq) One-loop vertex part:")
    F1_rpa = F1(param_rpa)
    F1_kop = F1(param_kop)

    F2_rpa = test_vertex_integral(param_rpa)
    println("(RPA) $(F1_rpa) ξ + $(F2_rpa) ξ²")

    F2_kop = test_vertex_integral(param_kop)
    println("(KO+) $(F1_kop) ξ + $(F2_kop) ξ²")

    # result_rpa = test_box_integral(param_rpa)
    # result_kop = test_box_integral(param_kop)
    # println("One-loop box part:\n(RPA) $result_rpa\n(KO+) $result_kop")
    return

    # # l=0 analytic plots
    # rs_Fsm1 = 5.24881  # Fs(rs = 5.24881) ≈ -1 (using Perdew-Wang fit)
    # rslist = sort(unique([0.01; 0.025; 0.05; rs_Fsm1; collect(range(0.1, 10.0; step=0.1))]))
    # plot_integrand_F1(param)
    # get_analytic_F1(param, rslist; plot=true)
    # return

    # # Dimensionless angular momentum grid for k - k'
    # xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    # k_m_kps = @. 2 * kF * xgrid

    # # Use the tree-level self-consistent result as a starting point
    # # TODO: add a relative tolerance variable and a check for convergence
    # alpha_mix = 0.5
    # num_self_consistency_iterations = 1
    # Fs_prev = get_tree_level_self_consistent_Fs(param)
    # println("Using tree-level self-consistent starting point...")
    # println("Fs(0) = $Fs_prev")
    # for i in 1:num_self_consistency_iterations
    #     param = Parameter.rydbergUnit(1.0 / beta, rs, 3; Fs=Fs_prev)
    #     Fs_curr = get_one_loop_Fs(
    #         param,
    #         # kwargs...
    #     )
    #     Fs_mix = lerp(Fs_prev, Fs_curr, alpha_mix)
    #     println("Fs($i) = $Fs_mix")
    #     Fs_prev = Fs_mix
    # end
    # println("Fs($(num_self_consistency_iterations)) = $Fs_prev")
    # return
end

main()
