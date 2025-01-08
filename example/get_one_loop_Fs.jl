using Colors
using CompositeGrids
using ElectronGas
using ElectronLiquid
using GreenFunc
using JLD2
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

function spline(x, y, e; xmin=0.0, xmax=x[end])
    # generate knots with spline without constraints
    w = 1.0 ./ e
    spl = interp.UnivariateSpline(x, y; w=w, k=3)
    __x = collect(LinRange(xmin, xmax, 1000))
    yfit = spl(__x)
    return __x, yfit
end

# exchange interaction (Ws + Wa \sigma\sigma)_ex to a direct interaction Ws'+Wa' \sigma\sigma 
# # exchange S/A interaction projected to the spin-symmetric and antisymmetric parts
# # NOTE: since Wae = 0, this just divides Ws by 2...
function exchange_to_direct(Wse, Wae)
    Ws = (Wse + 3 * Wae) / 2
    Wa = (Wse - Wae) / 2
    return Ws, Wa
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via Corradini's fit
to the DMC compressibility enhancement [doi: 10.1103/PhysRevB.57.14569].
"""
@inline function get_Fs(param::Parameter.Para)
    kappa0_over_kappa = Interaction.compressibility_enhancement(param)
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

function integrand_F0p(x, rs_tilde, Fs=0.0)
    if isinf(rs_tilde)
        return -x / lindhard(x)
    end
    coeff = rs_tilde + Fs * x^2
    NF_times_Rp_ex = coeff / (x^2 + coeff * lindhard(x))
    return -x * NF_times_Rp_ex
end

function plot_integrand_F0p(param::Parameter.Para)
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
        param_this_rs = Parameter.rydbergUnit(1.0 / param.beta, rs, param.dim)
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

function get_analytic_F0p(param, rslist; plot=false)
    F0p_RPA = []
    F0p_R = []
    Fslist = []
    Fssclist = []
    xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    y0_RPA_inf = [integrand_F0p(x, Inf) for x in xgrid]
    F0p_RPA_inf = CompositeGrids.Interp.integrate1D(y0_RPA_inf, xgrid)
    println("F⁺₀[W₀](∞) = $(F0p_RPA_inf)")
    for rs in rslist
        param_this_rs = Parameter.rydbergUnit(1.0 / param.beta, rs, param.dim)
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
function get_tree_level_self_consistent_Fs(param::Parameter.Para)
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

function Π0_static(param::Parameter.Para, qgrid::QGT) where {QGT<:AbstractGrid}
    @unpack kF, NF = param
    xgrid = qgrid.mesh[1].grid / (2 * kF)
    return -param.NF * lindhard.(xgrid)
end

"""
The non-interacting (renormalized) Green's function G₀.
As a first approximation, we neglect the mass renormalization, setting m* = m.
"""
function G0(
    param::Parameter.Para,
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
G₀(k, iω)
"""
function G0(param::Parameter.Para, k, w)
    @unpack me, μ = param
    return 1.0 / (im * w - k^2 / (2 * me) + μ)
end

"""
The one-loop (GW) self-energy Σ₁.
"""
function Σ1(
    param::Parameter.Para,
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
        param,
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
    param::Parameter.Para,
    kgrid::KGT;
    Fs=-0.0,
    int_type=:rpa,
) where {KGT<:AbstractGrid}
    sigma1 = Σ1(param, kgrid; Fs=Fs, int_type=int_type)
    # Z_F using improved finite-temperature scaling
    return zfactor_fermi(param, sigma1)
end

"""
Tree-level estimate of F⁺₀.
"""
function F1(param::Parameter.Para; int_type=:rpa)
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

function R_data(param::Parameter.Para)
    @unpack kF, EF, β = param
    dlr = DLRGrid(Euv=10 * EF, β=β, rtol=1e-10, isFermi=false, symmetry=:ph)
    ngrid = dlr.n
    qgrid = CompositeGrid.LogDensedGrid(:uniform, [0.0, 6 * kF], [0.0, 2 * kF], 16, 0.01 * kF, 16)
    # Evaluate R(q, iνₘ) on a sparse DLR grid
    Nmax = maximum(ngrid)
    Nq, Nw = length(qgrid), length(ngrid)
    Rs = zeros(Float64, (Nq, Nw))
    for (ni, n) in enumerate(ngrid)
        for (qi, q) in enumerate(qgrid)
            invKOinstant = 1.0 / ElectronLiquid.KOinstant(q, para)
            # Rs = (vq+f)Π0/(1-(vq+f)Π0)
            Pi[qi, ni] = ElectronLiquid.polarKW(q, n, para)
            Rs[qi, ni] = Pi[qi, ni] / (invKOinstant - Pi[qi, ni])
        end
    end
    # upsample to full bosonic Matsubara grid with indices ranging from -N to N
    Rs = matfreq2matfreq(dlr, Rs, collect((-Nmax):Nmax); axis=2)
    return real.(Rs)  # R(q, iνₘ) = R(q, -iνₘ) ⟹ R is real
end

# 2RΛ₁
function one_loop_vertex_corrections(
    param::Parameter.Para;
    kamp1=param.kF,
    kamp2=param.kF,
    θ12=π,
)
    @unpack kF, β, EF = param

    function vertex_matsubara_sum(param::Parameter.Para, q, θq, φq)
        s_ivm = []
        # mlist = ... # sparse bosonic mesh {iνₘ} from DLR
        for m in mlist
            # A(iνₘ) = R(q', iν'ₘ) * g(k + q', iω₀ + iν'ₘ) * g(k' + q', iω₀ + iν'ₘ)
            # gg = ...
        end
        # interpolate data for A(iνₘ) over entire frequency mesh from -M to M
        mlist_full = collect((-M):M) * (im * 2π / β)
        s_ivm_full = Interp.interp1DGrid(s_ivm, mlist, mlist_full)
        # sum over iνₘ
        matsubara_sum = s_ivm_full[0] + 2 * sum(s_ivm_full[2:end])
        return matsubara_sum / β
    end

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

    # # l=0 analytic plots
    # rs_Fsm1 = 5.24881  # Fs(rs = 5.24881) ≈ -1 (using Perdew-Wang fit)
    # rslist = sort(unique([0.01; 0.025; 0.05; rs_Fsm1; collect(range(0.1, 10.0; step=0.1))]))
    # plot_integrand_F0p(param)
    # get_analytic_F0p(param, rslist; plot=true)
    # return

    # # Dimensionless angular momentum grid for k - k'
    # xgrid = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
    # k_m_kps = @. 2 * kF * xgrid

    res, time = test_vertex_integral(
        param,
        #  args...
    )
    println("Result: $res")
    println("Evaluation time: $time")
    return

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
