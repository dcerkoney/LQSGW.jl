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
using PyCall
using PyPlot
using Roots

@pyimport numpy as np   # for saving/loading numpy data
@pyimport scienceplots  # for style "science"
@pyimport scipy.interpolate as interp

const iw0 = im * π / β
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
]
rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["font.size"] = 16
rcParams["mathtext.fontset"] = "cm"
rcParams["font.family"] = "Times New Roman"

# Specify the type of momentum and frequency (index) grids explicitly to ensure type stability
const MomGridType = CompositeGrids.CompositeG.Composite{
    Float64,
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound},
    CompositeGrids.CompositeG.Composite{
        Float64,
        CompositeGrids.SimpleG.Log{Float64},
        CompositeGrids.SimpleG.Uniform{Float64,CompositeGrids.SimpleG.ClosedBound},
    },
}
const MGridType = CompositeGrids.SimpleG.Arbitrary{Int64,CompositeGrids.SimpleG.ClosedBound}
const FreqGridType =
    CompositeGrids.SimpleG.Arbitrary{Float64,CompositeGrids.SimpleG.ClosedBound}

@with_kw mutable struct OneLoopParams
    # UEG parameters
    beta::Float64
    rs::Float64
    Fs::Float64 = -0.0
    Fa::Float64 = -0.0

    mass2::Float64 = 1e-6     # fictitious Yukawa screening λ
    massratio::Float64 = 1.0  # mass ratio m*/m

    dim::Int = 3
    spin::Int = 2

    basic::Parameter.Para = Parameter.rydbergUnit(1.0 / beta, rs, dim; Λs=mass2, spin=spin)

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
    fa::Float64 = Fa / NF

    # Momentum grid parameters
    maxK::Float64 = 6 * kF
    maxQ::Float64 = 6 * kF
    Q_CUTOFF::Float64 = 1e-10 * kF

    # Use ~100 k-points for the momentum grids
    # NOTE: EL.jl default is `Nk, order = 16, 16` (~700 k-points)
    Nk, order = 7, 7
    qgrid::MomGridType =
        CompositeGrid.LogDensedGrid(:uniform, [0.0, maxQ], [0.0, 2 * kF], 7, 0.01 * kF, 7)

    # Use a sparse DLR grid for the bosonic Matsubara summation (~30-50 iνₘ-points)
    dlr::DLRGrid{Float64,:ph} =
        DLRGrid(; Euv=10 * EF, β=β, rtol=1e-10, isFermi=false, symmetry=:ph)
    mgrid::MGridType = SimpleG.Arbitrary{Int64}(dlr.n)
    vmgrid::Vector{Float64} = SimpleG.Arbitrary{Float64}(dlr.ωn)
    Mmax::Int64 = maximum(mgrid)

    # Incoming momenta k1, k2 and incident scattering angle
    kamp1::Float64 = basic.kF
    kamp2::Float64 = basic.kF
    θ12::Float64 = π

    # R grid data is precomputed in an initialization step
    initialized::Bool = false
    R::Matrix{Float64} = Matrix{Float64}(undef, length(qgrid), length(mgrid))
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
@inline function get_Fs(param::OneLoopParams)
    kappa0_over_kappa = Interaction.compressibility_enhancement(param.basic)
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

function integrand_F0p(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return -x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    NF_times_Rp_ex = coeff / (x^2 + coeff * lindhard(x))
    return -x * NF_times_Rp_ex
end

function plot_integrand_F0p(param::OneLoopParams)
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
            y = [integrand_F0p(xi, rs * alpha_ueg / π, Fs) for xi in x]
            ax.plot(x, y; linestyle=linestyle, color=color, label=label)
        end
    end
    ax.set_xlabel("\$x\$")
    ax.set_ylabel("\$I_0[W](x)\$")
    ax.legend(; fontsize=10, loc="best", ncol=2)
    xlim(0, 1)
    ylim(-2.125, 1.125)
    # tight_layout()
    fig.savefig("integrand_F0p.pdf")
    plt.close("all")
end

function get_analytic_F0p(param::OneLoopParams, rslist; plot=false)
    @unpack beta, dim = param
    F0p_RPA = []
    F0p_R = []
    Fslist = []
    Fssclist = []
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    y0_RPA_inf = [integrand_F0p(x, Inf) for x in xgrid]
    F0p_RPA_inf = CompositeGrids.Interp.integrate1D(y0_RPA_inf, xgrid)
    println("F⁺₀[W₀](∞) = $(F0p_RPA_inf)")
    for rs in rslist
        param_this_rs = Parameter.rydbergUnit(1.0 / beta, rs, dim)
        rstilde = rs * alpha_ueg / π
        Fs = -get_Fs(param_this_rs)
        Fs_sc = get_tree_level_self_consistent_Fs(param_this_rs)
        if rs > 0.25
            @assert Fs ≤ 0 "Incorrect sign for Fs!"
        end
        # RPA
        y_RPA = [integrand_F0p(x, rstilde) for x in xgrid]
        val_RPA = CompositeGrids.Interp.integrate1D(y_RPA, xgrid)
        push!(F0p_RPA, val_RPA)
        # KO+
        y_R = [integrand_F0p(x, rstilde, Fs) for x in xgrid]
        val_R = (Fs / 2) + CompositeGrids.Interp.integrate1D(y_R, xgrid)
        push!(F0p_R, val_R)
        push!(Fslist, Fs)
        push!(Fssclist, Fs_sc)
    end
    if plot
        fig, ax = plt.subplots()
        ax.plot(rslist, F0p_RPA; color=cdict["red"], label="\$W_0\$")
        ax.plot(rslist, F0p_R; color=cdict["blue"], label="\$R\$")
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
        fig.savefig("analytic_F0p.pdf")
        plt.close("all")
    end
    return F0p_RPA, F0p_R
end

"""
Solve I0[F+] = F+ / 2 to obtain the tree-level self-consistent value for F⁰ₛ.
"""
function get_tree_level_self_consistent_Fs(param::OneLoopParams)
    @unpack rs = param
    function I0_R(x, y)
        ts = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
        integrand = [integrand_F0p(t, x * alpha_ueg / π, y) for t in ts]
        integral = CompositeGrids.Interp.integrate1D(integrand, ts)
        return integral
    end
    F0p_sc = find_zero(Fp -> I0_R(rs, Fp) - Fp / 2, (-20.0, 20.0))
    return F0p_sc
end

"""
The one-loop (GW) self-energy Σ₁.
"""
function Σ1(
    param::OneLoopParams,
    kgrid::KGT;
    Fs=-0.0,
    int_type=:rpa,
) where {KGT<:AbstractGrid}
    @unpack kF, EF = param
    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14
    # Based on ElectronGas.jl defaults for G0W0 self-energy (here, minK *= 100)
    maxK = 6 * kF
    minK = 1e-6 * kF
    # Get the One-loop self-energy
    sigma = Sigma.G0W0(
        param.basic,
        kgrid;
        Euv=Euv,
        rtol=rtol,
        maxK=maxK,
        minK=minK,
        int_type=int_type,
        Fs=Fs,
        Fa=-0.0,
    )
    return sigma
end

"""
Leading-order (one-loop) correction to Z_F.
"""
function Z1(
    param::OneLoopParams,
    kgrid::KGT;
    Fs=-0.0,
    int_type=:rpa,
) where {KGT<:AbstractGrid}
    sigma1 = Σ1(param.basic, kgrid; Fs=Fs, int_type=int_type)
    # Z_F using improved finite-temperature scaling
    return zfactor_fermi(param.basic, sigma1)
end

"""
Tree-level estimate of F⁺₀.
"""
function F1(param::OneLoopParams; int_type=:rpa)
    @unpack rs, kF, EF, NF = param
    rstilde = rs * alpha_ueg / π
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    if int_type == :rpa
        Fs = 0.0
    else
        Fs = -get_Fs(param)
    end
    y = [integrand_F0p(x, rstilde, Fs) for x in xgrid]
    F0p_tree_level = (Fs / 2) + CompositeGrids.Interp.integrate1D(y, xgrid)
    return F0p_tree_level
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

function R_data(param::OneLoopParams)
    @unpack basic, qgrid, mgrid, Mmax, fs = param
    Nq, Nw = length(qgrid), length(mgrid)
    R = zeros(Float64, (Nq, Nw))
    for (ni, n) in enumerate(mgrid)
        for (qi, q) in enumerate(qgrid)
            invKOinstant = 1.0 / UEG.KOinstant(q, basic)
            # R = (vq + f) / (1 - (vq + f) Π0) - f
            Pi[qi, ni] = UEG.polarKW(q, n, basic)
            R[qi, ni] = 1 / (invKOinstant - Pi[qi, ni]) - fs
        end
    end
    # upsample to full frequency grid with indices ranging from -N to N
    R = matfreq2matfreq(dlr, R, collect((-Mmax):Mmax); axis=2)
    return real.(R)  # R(q, iνₘ) = R(q, -iνₘ) ⟹ R is real
end

function initialize_R!(param::OneLoopParams)
    param.R = R_data(param)
    param.initialized = true
end

"""
G₀(k, iωₙ)
"""
function G0(param::OneLoopParams, k, wn)
    @unpack me, μ, β = param
    return 1.0 / (im * wn - k^2 / (2 * me) + μ)
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
    return UEG.linear2D(param.R, param.qgrid, param.mgrid, q, n)
end

function vertex_matsubara_sum(param::OneLoopParams, q, θq, φq)
    @unpack kF, β, EF, kamp1, kamp2, θ12, mgrid, vmgrid, Mmax = param

    # p1 = |k + q'|, p2 = |k' + q'|
    vec_p1 = [0.0, 0.0, kamp1] + q * [sin(θq) * cos(φq), sin(θq) * sin(φq), cos(θq)]
    vec_p2 = [kamp2, 0.0, 0.0] + q * [sin(θq) * cos(φq), sin(θq) * sin(φq), cos(θq)]
    p1 = norm(vec_p1)
    p2 = norm(vec_p2)

    # S(iνₘ) = R(q', iν'ₘ) * g(p1, iω₀ + iν'ₘ) * g(p2, iω₀ + iν'ₘ)
    s_ivm = Vector{ComplexF64}(undef, length(mgrid))
    for (i, (m, vm)) in enumerate(zip(mgrid, vmgrid))
        s_ivm[i] =
            R(param, q, m) * G0(param, p1, iw0 + im * vm) * G0(param, p2, iw0 + im * vm)
    end

    # interpolate data for S(iνₘ) over entire frequency mesh from 0 to Mmax
    s_ivm_full = Interp.interp1DGrid(s_ivm, mgrid, 0:Mmax)

    # sum over iνₘ including negative frequency terms (S(iνₘ) = S(-iνₘ))
    matsubara_sum = (s_ivm_full[0] + 2 * sum(s_ivm_full[2:end])) / β
    return matsubara_sum
end

# 2RΛ₁
function one_loop_vertex_corrections(param::OneLoopParams)

    # integrate over loop momentum q
    qs = [kF]
    θqs = CompositeGrid.LogDensedGrid(:gauss, [0.0, π], [0.0, π], 6, 1e-6, 6)
    φqs = CompositeGrid.LogDensedGrid(:gauss, [0.0, 2π], [0.0, 2π], 6, 1e-6, 6)
    φq_integrand = Vector{Complex64}(undef, length(θqs.grid))
    for (iq, q) in enumerate(qs)
        for (iθq, θq) in enumerate(θqs)
            for (iφq, φq) in enumerate(φqs)
                res[iq, iθq, iφq] = vertex_matsubara_sum(param, q, θq, φq)
            end
        end
    end

    one_loop_vertex_corrections = missing
    return one_loop_vertex_corrections
end

function test_vertex_integral(param::Parameter.Para; args...)
    vertex_integrand = one_loop_vertex_corrections(param; args...)
    vertex_integrator = missing
    result = missing
    eval_time = missing
    return result, eval_time
end

# linear interpolation with mixing parameter α: x * (1 - α) + y * α
function lerp(x, y, alpha)
    return (1 - alpha) * x + alpha * y
end

function get_one_loop_integrand(param::Parameter.Para; args...)
    function one_loop_integrand(param; args...)
        return one_loop_vertex_corrections(param; args...) +
               one_loop_box_diagrams(param; args...) +
               one_loop_counterterms(param; args...)
    end
    return one_loop_integrand
end

function get_one_loop_Fs(param::Parameter.Para; args...)
    integrand = get_one_loop_integrand(param; args...)
    F0p_ol = missing
    return F0p_ol
end

function main()
    rs = 1.0
    beta = 40.0
    param = Parameter.rydbergUnit(1.0 / beta, rs, 3)

    @unpack kF, EF, β = param

    # Get coarse grid data for R(q, iνₘ)
    Rs = R_data(param, qgrid, mgrid)

    res, time = test_vertex_integral(
        param,
        #  args...
    )

    println("Result: $res")
    println("Evaluation time: $time")
    return

    # # l=0 analytic plots
    # rs_Fsm1 = 5.24881  # Fs(rs = 5.24881) ≈ -1 (using Perdew-Wang fit)
    # rslist = sort(unique([0.01; 0.025; 0.05; rs_Fsm1; collect(range(0.1, 10.0; step=0.1))]))
    # plot_integrand_F0p(param)
    # get_analytic_F0p(param, rslist; plot=true)
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
    #         # args...
    #     )
    #     Fs_mix = lerp(Fs_prev, Fs_curr, alpha_mix)
    #     println("Fs($i) = $Fs_mix")
    #     Fs_prev = Fs_mix
    # end
    # println("Fs($(num_self_consistency_iterations)) = $Fs_prev")
    # return
end

main()
