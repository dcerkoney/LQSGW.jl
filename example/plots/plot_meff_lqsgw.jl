using Colors
using CompositeGrids
using ElectronGas
using GreenFunc
using JLD2
using Lehmann
using LQSGW
using Parameters
using PyCall
using PyPlot

@pyimport numpy as np   # for saving/loading numpy data
@pyimport scienceplots  # for style "science"
@pyimport scipy.interpolate as interp

# Vibrant qualitative colour scheme from https://personal.sron.nl/~pault/
const cdict = Dict([
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

function plot_spline(
    x,
    y;
    ax=plt.gca(),
    color="k",
    label=nothing,
    ls="-",
    lw=nothing,
    extrapolate=true,
)
    if extrapolate
        fitfunc = interp.PchipInterpolator(x, y; extrapolate=true)
    else
        fitfunc = interp.Akima1DInterpolator(x, y)
    end
    xgrid = np.arange(0, 6.2, 0.02)
    if isnothing(lw) == false
        handle, = ax.plot(xgrid, fitfunc(xgrid); ls=ls, color=color, label=label, lw=lw)
    else
        handle, = ax.plot(xgrid, fitfunc(xgrid); ls=ls, color=color, label=label)
    end
    return handle
end

function main()
    # UEG parameters
    beta = 40.0
    rs = 5.0
    dim = 3
    param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
    paramold = Parameter.rydbergUnit(1.0 / beta, 4.5, dim)  # param for previous rs
    @unpack kF, EF, β = param

    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14

    # CompositeGrid parameters
    # Nk, order = 14, 10
    Nk, order = 12, 8
    # Nk, order = 10, 7

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    # Test LQSGW parameters
    int_type = :rpa

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    # Momentum grid maxima for G, Π, and Σ
    maxKG = 4.3 * maxK
    maxKP = 2.1 * maxK
    maxKS = maxK

    # Make sure we do not exceed the DLR energy cutoff
    @assert maxKG^2 / (2 * param.me) < Euv "Max grid momentum exceeds DLR cutoff"

    # Bosonic DLR grid for the problem
    bdlr = DLRGrid(Euv, β, rtol, false, :ph)

    # Momentum grid maxima for G, Π, and Σ
    maxKG = 4.3 * maxK
    maxKP = 2.1 * maxK
    maxKS = maxK

    # Big grid for G
    multiplier = 1
    # multiplier = 2
    # multiplier = 4
    kGgrid = CompositeGrid.LogDensedGrid(
        :cheb,
        [0.0, maxKG],
        [0.0, kF],
        round(Int, multiplier * Nk),
        0.01 * minK,
        round(Int, multiplier * order),
    )

    # Medium grid for Π
    qPgrid =
        CompositeGrid.LogDensedGrid(:cheb, [0.0, maxKP], [0.0, 2 * kF], Nk, minK, order)

    # Small grid for Σ
    kSgrid = CompositeGrid.LogDensedGrid(:cheb, [0.0, maxKS], [0.0, kF], Nk, minK, order)

    # Legacy data
    function loaddata_old(
        key;
        savedir="$(LQSGW.DATA_DIR)/$(dim)d/$(int_type)",
        savename="lqsgw_$(dim)d_$(int_type)_rs=$(round(rs; sigdigits=4))_beta=$(round(beta; sigdigits=4)).jld2",
    )
        data = []
        n_max = -1
        filename = joinpath(savedir, savename)
        jldopen(filename, "r") do f
            for i in 0:max_steps
                try
                    push!(data, f["$(key)_$(i)"])
                catch
                    n_max = i - 1
                    break
                end
            end
        end
        return data, n_max
    end

    # New data
    function loaddata(;
        savedir="$(LQSGW.DATA_DIR)/$(dim)d/$(int_type)",
        savename="lqsgw_$(dim)d_$(int_type)_rs=$(round(rs; sigdigits=4))_beta=$(round(beta; sigdigits=4)).jld2",
    )
        data = []
        max_step = -1
        jldopen(joinpath(savedir, savename), "r") do file
            # Ensure that the saved data was convergent
            @assert file["converged"] == true "Specificed save data did not converge!"
            # Find the converged data in JLD2 file
            for i in 0:(LQSGW.MAXIMUM_STEPS)
                if haskey(file, string(i))
                    max_step = i
                    push!(data, file[string(i)])
                else
                    break
                end
            end
            if max_step < 0
                error("No data found in $(savedir)!")
            end
            println(
                "Found converged data with max_step=$(max_step) for savename $(savename)!",
            )
        end
        return data, max_step
    end

    colors(n) = reverse(hex.((colormap("Reds", n; mid=0.8, logscale=true))))
    # colors(n) = reverse(hex.((sequential_palette(0, n; s=1))))

    # # Load the data using legacy format
    # Σinsqp, nstep = loaddata_old("Σ_ins")
    # Σqp, nstep = loaddata_old("Σ")
    # Eqp, nstep = loaddata_old("E_k")
    # Zqp, nstep = loaddata_old("Z_k")
    # Gqp, nstep = loaddata_old("G")
    # Πqp, nstep = loaddata_old("Π")
    # Wqp, nstep = loaddata_old("W")
    # lqsgw_data = []
    # for (i, (s, si, e, z, g, p, w)) in enumerate(zip(Σinsqp, Σqp, Eqp, Zqp, Gqp, Πqp, Wqp))
    #     push!(lqsgw_data, LQSGW.LQSGWIteration(i, -1, -1, -1, e, z, g, p, w, s, si))
    # end

    # Load the data using new format
    savename1 = "lqsgw_convergence_$(int_type)_rs=$(rs)_no_prev_rs.pdf"
    savename2 = "lqsgw_convergence_$(int_type)_rs=$(rs)_prev_rs_without_rescale.pdf"
    savename3 = "lqsgw_convergence_$(int_type)_rs=$(rs)_prev_rs_with_rescale.pdf"
    lqsgw_data1, nstep1 = loaddata(; savedir="$(LQSGW.DATA_DIR)/test_osp")
    lqsgw_data2, nstep2 = loaddata(; savedir="$(LQSGW.DATA_DIR)/test_nsp_no_rescale")
    lqsgw_data3, nstep3 = loaddata(; savedir="$(LQSGW.DATA_DIR)/test_nsp_rescale")
    for (savename, lqsgw_data, nstep) in zip(
        [savename1, savename2, savename3],
        [lqsgw_data1, lqsgw_data2, lqsgw_data3],
        [nstep1, nstep2, nstep3],
    )
        # Plot Z, Sigma, and E_qp from first iteration side-by-side
        nrows, ncols, wspace = 2, 3, 0.4
        plt.figure(; figsize=(5 * ncols + wspace * (ncols), 5 * nrows))
        ax1 = plt.subplot(nrows, ncols, 1)
        ax2 = plt.subplot(nrows, ncols, 2)
        ax3 = plt.subplot(nrows, ncols, 3)
        ax4 = plt.subplot(nrows, ncols, 4)
        ax5 = plt.subplot(nrows, ncols, 5)
        ax6 = plt.subplot(nrows, ncols, 6)

        if lqsgw_data[1].Σ.mesh[1].grid == [0]
            w0_label = 1
        else
            w0_label = locate(lqsgw_data[1].Σ.mesh[1], 0)
        end
        println("Σ: w0_label = $w0_label")
        Σins0 = real(lqsgw_data[1].Σ_ins[1, :])
        Σins1 = real(lqsgw_data[2].Σ_ins[1, :])
        Σins2 = real(lqsgw_data[3].Σ_ins[1, :])
        Σinssc = real(lqsgw_data[end].Σ_ins[1, :])
        Σ0 = real((lqsgw_data[1].Σ)[w0_label, :])
        Σ1 = real((lqsgw_data[2].Σ)[w0_label, :])
        Σ2 = real((lqsgw_data[3].Σ)[w0_label, :])
        Σsc = real((lqsgw_data[end].Σ)[w0_label, :])
        Σ0_static = Σins0 + Σ0
        Σ1_static = Σins1 + Σ1
        Σ2_static = Σins2 + Σ2
        Σsc_static = Σinssc + Σsc
        E0 = lqsgw_data[1].E_k
        E1 = lqsgw_data[2].E_k
        E2 = lqsgw_data[3].E_k
        Esc = lqsgw_data[end].E_k
        Z0 = lqsgw_data[1].Z_k
        Z1 = lqsgw_data[2].Z_k
        Z2 = lqsgw_data[3].Z_k
        Zsc = lqsgw_data[end].Z_k

        # Compute D(k) from ReΣ(k, iω0)
        kgrid_plot = collect(range(0; stop=4, length=200))
        dk_plot = kgrid_plot[2] - kgrid_plot[1]
        kplot_cd = kgrid_plot[2:(end - 1)]
        Σ1_static_interp = Interp.interp1DGrid(Σ1_static, kSgrid, kgrid_plot)
        Σ2_static_interp = Interp.interp1DGrid(Σ2_static, kSgrid, kgrid_plot)
        Σsc_static_interp = Interp.interp1DGrid(Σsc_static, kSgrid, kgrid_plot)
        dΣ1_dk_static =
            (Σ1_static_interp[3:end] - Σ1_static_interp[1:(end - 2)]) / (2 * dk_plot)
        dΣ2_dk_static =
            (Σ2_static_interp[3:end] - Σ2_static_interp[1:(end - 2)]) / (2 * dk_plot)
        dΣsc_dk_static =
            (Σsc_static_interp[3:end] - Σsc_static_interp[1:(end - 2)]) / (2 * dk_plot)
        Dk1 = 1 .+ (param.me ./ kplot_cd) .* dΣ1_dk_static
        Dk2 = 1 .+ (param.me ./ kplot_cd) .* dΣ2_dk_static
        Dksc = 1 .+ (param.me ./ kplot_cd) .* dΣsc_dk_static

        # Compute (m*/m)(k) = 1 / (Z(k) * D(k))
        Zk1_interp = Interp.interp1DGrid(Z1, kSgrid, kplot_cd)
        Zk2_interp = Interp.interp1DGrid(Z2, kSgrid, kplot_cd)
        Zksc_interp = Interp.interp1DGrid(Zsc, kSgrid, kplot_cd)
        meff1 = 1 ./ (Zk1_interp .* Dk1)
        meff2 = 1 ./ (Zk2_interp .* Dk2)
        meffsc = 1 ./ (Zksc_interp .* Dksc)
        meff1[isinf.(meff1)] .= 1.0
        meff2[isinf.(meff2)] .= 1.0
        meffsc[isinf.(meffsc)] .= 1.0

        if lqsgw_data[1].Π.mesh[1].grid == [0]
            w0_label = 1
        else
            w0_label = locate(lqsgw_data[1].Π.mesh[1], 0)
        end
        println("Π: w0_label = $w0_label")
        # Π0 = real((lqsgw_data[1].Π)[w0_label, :])
        Π1 = real((lqsgw_data[2].Π)[w0_label, :])
        Π2 = real((lqsgw_data[3].Π)[w0_label, :])
        Πsc = real((lqsgw_data[end].Π)[w0_label, :])
        if lqsgw_data[1].W.mesh[1].grid == [0]
            w0_label = 1
        else
            w0_label = locate(lqsgw_data[1].W.mesh[1], 0)
        end
        println("W: w0_label = $w0_label")
        # W0 = real((lqsgw_data[1].W)[w0_label, :])
        W1 = real((lqsgw_data[2].W)[w0_label, :])
        W2 = real((lqsgw_data[3].W)[w0_label, :])
        Wsc = real((lqsgw_data[end].W)[w0_label, :])
        V = [Interaction.coulomb(q, param)[1] for q in qPgrid]

        for (i, d) in enumerate(lqsgw_data)
            i == 1 && continue
            if lqsgw_data[1].Σ.mesh[1].grid == [0]
                w0_label = 1
            else
                w0_label = locate(lqsgw_data[1].Σ.mesh[1], 0)
            end
            Σ_static = real(d.Σ_ins[1, :]) + real(d.Σ[w0_label, :])
            plot_spline(kSgrid / kF, Σ_static; ax=ax1, color="#$(colors(nstep + 1)[i])")
        end
        plot_spline(
            kSgrid / kF,
            Σ0_static;
            ax=ax1,
            ls="--",
            color="k",
            lw=1.5,
            label="\$i=0\$",
        )
        plot_spline(
            kSgrid / kF,
            Σ1_static;
            ax=ax1,
            color=cdict["grey"],
            lw=1.5,
            label="\$i=1\$",
        )
        plot_spline(
            kSgrid / kF,
            Σ2_static;
            ax=ax1,
            color=cdict["red"],
            lw=1.5,
            label="\$i=2\$",
        )
        plot_spline(
            kSgrid / kF,
            Σsc_static;
            ax=ax1,
            color=cdict["teal"],
            lw=1.5,
            label="self-consistent",
        )
        # for (i, d) in enumerate(lqsgw_data)
        #     i == 1 && continue
        #     ax3.plot(kSgrid / kF, real(d.Σ_ins[1, :]); color="#$(colors(nstep + 1)[i])")
        # end
        # ax3.plot(kSgrid / kF, Σins0, "--"; color="k", lw=1.5, label="\$i=0\$")
        # ax3.plot(kSgrid / kF, Σins1; color=cdict["grey"], lw=1.5, label="\$i=1\$")
        # ax3.plot(kSgrid / kF, Σins2; color=cdict["red"], lw=1.5, label="\$i=2\$")
        # ax3.plot(kSgrid / kF, Σinssc; color=cdict["teal"], lw=1.5, label="self-consistent")
        # for (i, d) in enumerate(lqsgw_data)
        #     i == 1 && continue
        #     if lqsgw_data[1].Σ.mesh[1].grid == [0]
        #         w0_label = 1
        #     else
        #         w0_label = locate(lqsgw_data[1].Σ.mesh[1], 0)
        #     end
        #     ax1.plot(kSgrid / kF, real(d.Σ[w0_label, :]); color="#$(colors(nstep + 1)[i])")
        # end
        # ax1.plot(kSgrid / kF, Σ0, "--"; color="k", lw=1.5, label="\$i=0\$")
        # ax1.plot(kSgrid / kF, Σ1; color=cdict["grey"], lw=1.5, label="\$i=1\$")
        # ax1.plot(kSgrid / kF, Σ2; color=cdict["red"], lw=1.5, label="\$i=2\$")
        # ax1.plot(kSgrid / kF, Σsc; color=cdict["teal"], lw=1.5, label="self-consistent")

        for (i, d) in enumerate(lqsgw_data)
            i == 1 && continue
            plot_spline(kSgrid / kF, d.Z_k; ax=ax2, color="#$(colors(nstep + 1)[i])")
        end
        # plot_spline(kSgrid / kF, Z0, "--"; ax=ax2, color="k", lw=1.5, label="\$i=0\$")
        plot_spline(kSgrid / kF, Z1; ax=ax2, color=cdict["grey"], lw=1.5, label="\$i=1\$")
        plot_spline(kSgrid / kF, Z2; ax=ax2, color=cdict["red"], lw=1.5, label="\$i=2\$")
        plot_spline(
            kSgrid / kF,
            Zsc;
            ax=ax2,
            color=cdict["teal"],
            lw=1.5,
            label="self-consistent",
        )

        for (i, d) in enumerate(lqsgw_data)
            i == 1 && continue
            if lqsgw_data[1].Σ.mesh[1].grid == [0]
                w0_label = 1
            else
                w0_label = locate(lqsgw_data[1].Σ.mesh[1], 0)
            end
            Σ_static = real(d.Σ_ins[1, :]) + real(d.Σ[w0_label, :])
            Σ_static_interp = Interp.interp1DGrid(Σ_static, kSgrid, kgrid_plot)
            dΣ_dk_static =
                (Σ_static_interp[3:end] - Σ_static_interp[1:(end - 2)]) / (2 * dk_plot)
            D_k = 1 .+ (param.me ./ kplot_cd) .* dΣ_dk_static
            plot_spline(kplot_cd / kF, D_k; ax=ax3, color="#$(colors(nstep + 1)[i])")
        end
        plot_spline(
            kplot_cd / kF,
            Dk1;
            ax=ax3,
            color=cdict["grey"],
            lw=1.5,
            label="\$i=1\$",
        )
        plot_spline(kplot_cd / kF, Dk2; ax=ax3, color=cdict["red"], lw=1.5, label="\$i=2\$")
        plot_spline(
            kplot_cd / kF,
            Dksc;
            ax=ax3,
            color=cdict["teal"],
            lw=1.5,
            label="self-consistent",
        )

        for (i, d) in enumerate(lqsgw_data)
            i == 1 && continue
            plot_spline(kSgrid / kF, d.E_k; ax=ax4, color="#$(colors(nstep + 1)[i])")
        end
        # plot_spline(
        #     kSgrid / kF,
        #     E0 * (paramold.rs / rs)^2,
        #     "--";
        #     color="k",
        #     lw=1.5,
        #     label="\$i=0\$",
        # )
        plot_spline(kSgrid / kF, E1; ax=ax4, color=cdict["grey"], lw=1.5, label="\$i=1\$")
        plot_spline(kSgrid / kF, E2; ax=ax4, color=cdict["red"], lw=1.5, label="\$i=2\$")
        plot_spline(
            kSgrid / kF,
            Esc;
            ax=ax4,
            color=cdict["teal"],
            lw=1.5,
            label="self-consistent",
        )

        for (i, d) in enumerate(lqsgw_data)
            i == 1 && continue
            plot_spline(
                qPgrid / kF,
                real(d.Π[1, :]);
                ax=ax5,
                color="#$(colors(nstep + 1)[i])",
            )
        end
        # plot_spline(qPgrid / kF, Π0 * (paramold.rs / rs), "--"; ax=ax5, color="k", lw=1.5, label="\$i=0\$")
        plot_spline(qPgrid / kF, Π1; ax=ax5, color=cdict["grey"], lw=1.5, label="\$i=1\$")
        plot_spline(qPgrid / kF, Π2; ax=ax5, color=cdict["red"], lw=1.5, label="\$i=2\$")
        plot_spline(
            qPgrid / kF,
            Πsc;
            ax=ax5,
            color=cdict["teal"],
            lw=1.5,
            label="self-consistent",
        )

        for (i, d) in enumerate(lqsgw_data)
            i == 1 && continue
            if lqsgw_data[1].Σ.mesh[1].grid == [0]
                w0_label = 1
            else
                w0_label = locate(lqsgw_data[1].Σ.mesh[1], 0)
            end
            Σ_static = real(d.Σ_ins[1, :]) + real(d.Σ[w0_label, :])
            Σ_static_interp = Interp.interp1DGrid(Σ_static, kSgrid, kgrid_plot)
            dΣ_dk_static =
                (Σ_static_interp[3:end] - Σ_static_interp[1:(end - 2)]) / (2 * dk_plot)
            D_k = 1 .+ (param.me ./ kplot_cd) .* dΣ_dk_static
            Z_k = Interp.interp1DGrid(d.Z_k, kSgrid, kplot_cd)
            meff_k = 1 ./ (Z_k .* D_k)
            maskinf = isinf.(meff_k)
            meff_k[maskinf] .= 1.0
            # println(meff_k)
            plot_spline(kplot_cd / kF, meff_k; ax=ax6, color="#$(colors(nstep + 1)[i])")
        end
        plot_spline(
            kplot_cd / kF,
            meff1;
            ax=ax6,
            color=cdict["grey"],
            lw=1.5,
            label="\$i=1\$",
        )
        plot_spline(kplot_cd / kF, meff2; ax=ax6, color=cdict["red"], lw=1.5, label="\$i=2\$")
        plot_spline(
            kplot_cd / kF,
            meffsc;
            ax=ax6,
            color=cdict["teal"],
            lw=1.5,
            label="self-consistent",
        )
        # for (i, d) in enumerate(lqsgw_data)
        #     i == 1 && continue
        #     plot_spline(
        #         qPgrid / kF,
        #         1 .+ real(d.W[1, :]) ./ V;
        #         ax=ax6,
        #         color="#$(colors(nstep + 1)[i])",
        #     )
        # end
        # # plot_spline(qPgrid / kF, 1 .+ W0 ./ V, "--"; ax=ax6, color="k", lw=1.5, label="\$i=0\$")
        # plot_spline(
        #     qPgrid / kF,
        #     1 .+ W1 ./ V;
        #     ax=ax6,
        #     color=cdict["grey"],
        #     lw=1.5,
        #     label="\$i=1\$",
        # )
        # plot_spline(
        #     qPgrid / kF,
        #     1 .+ W2 ./ V;
        #     ax=ax6,
        #     color=cdict["red"],
        #     lw=1.5,
        #     label="\$i=2\$",
        # )
        # plot_spline(
        #     qPgrid / kF,
        #     1 .+ Wsc ./ V;
        #     ax=ax6,
        #     color=cdict["teal"],
        #     lw=1.5,
        #     label="self-consistent",
        # )

        ax1.set_ylabel("\$\\text{Re}\\Sigma(k, i\\omega_0)\$")
        ax2.set_ylabel("\$Z(k)\$")
        ax3.set_ylabel("\$D(k)\$")
        ax4.set_ylabel("\$\\mathcal{E}_\\text{qp}(k)\$")
        ax5.set_ylabel("\$\\Pi_\\text{qp}(q, i\\nu_m = 0)\$")
        ax6.set_ylabel("\$\\frac{m*}{m}(k)\$")
        # ax6.set_ylabel("\$\\epsilon^{-1}_\\text{qp}(q, i\\nu_m = 0)\$")
        # ax6.set_ylabel("\$\\epsilon^{-1}_\\text{qp}(q, 0) = W_\\text{qp}(q, 0) / V(q)\$")

        ax1.set_xlim(0, 4)
        ax1.set_ylim(nothing, -0.04)
        # ax3.set_ylim(-0.46, 0.02)
        # ax1.set_xlim(0, 2)
        # ax1.set_ylim(-5.2e-8, 0.2e-8)
        ax2.set_xlim(0, 4)
        ax3.set_xlim(0, 4)
        ax4.set_xlim(0, 2)
        ax4.set_ylim(-0.16, 0.22)
        ax5.set_xlim(0, 4)
        ax6.set_xlim(0, 4)
        ax6.set_ylim(0.99, 1.16)
        # ax5.set_xlim(0, 0.1)

        # x-axis on last subplot
        ax1.set_xlabel("\$k / k_F\$")
        ax2.set_xlabel("\$k / k_F\$")
        ax3.set_xlabel("\$k / k_F\$")
        ax4.set_xlabel("\$k / k_F\$")
        ax5.set_xlabel("\$q / k_F\$")
        ax6.set_xlabel("\$q / k_F\$")

        ax1.legend(; loc="lower right", fontsize=14)
        ax2.legend(; loc="lower right", fontsize=14)
        ax3.legend(; loc="upper right", fontsize=14)
        ax4.legend(; loc="lower right", fontsize=14)
        ax5.legend(; loc="lower right", fontsize=14)
        ax6.legend(; loc="upper right", fontsize=14)

        plt.subplots_adjust(; wspace=wspace)
        plt.savefig(savename)
        # plt.savefig("lqsgw_convergence_$(int_type)_rs=$(rs)_prev_rs_without_rescale.pdf")
        # plt.savefig("lqsgw_convergence_$(int_type)_rs=$(rs)_prev_rs_with_rescale.pdf")
        # plt.savefig("lqsgw_convergence_$(int_type)_rs=$(rs)_no_prev_rs.pdf")
    end
    return

    # # Plot Σ_ins convergence
    # fig, ax = plt.subplots()
    # plotdata, n_max = loaddata("Σ_ins")
    # for (idx, sigma_ins) in enumerate(plotdata)
    #     i = idx - 1
    #     fmt = i == 0 ? "--" : "-"
    #     if i == 0
    #         lw = 1
    #         label = nothing
    #         color = "black"
    #     elseif i == n_max
    #         lw = 1.5
    #         label = "\$N_\\text{iter} = $n_max\$"
    #         color = "limegreen"
    #     else
    #         lw = 1
    #         label = nothing
    #         color = "#$(colors(n_max)[i])"
    #     end
    #     # kgrid = sigma_ins.mesh[2]
    #     ax.plot(kSgrid / kF, real(sigma_ins[1, :]), fmt; color=color, lw=lw, label=label)
    # end
    # ax.set_xlim(0, 2)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.set_ylabel("\$\\Sigma^{(i)}_{x}(k)\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig("convergence_sigma_ins_rs=$rs.pdf")

    # # Plot Π convergence
    # fig, ax = plt.subplots()
    # plotdata, n_max = loaddata("Π")
    # for (idx, Π) in enumerate(plotdata)
    #     i = idx - 1
    #     fmt = i == 0 ? "--" : "-"
    #     if i == 0
    #         lw = 1
    #         label = nothing
    #         color = "black"
    #     elseif i == n_max
    #         lw = 1.5
    #         label = "\$N_\\text{iter} = $n_max\$"
    #         color = "limegreen"
    #     else
    #         lw = 1
    #         label = nothing
    #         color = "#$(colors(n_max)[i])"
    #     end
    #     pi_q_static = Π[1, :]
    #     ax.plot(qPgrid / kF, real.(pi_q_static), fmt; color=color, lw=lw, label=label)
    # end
    # ax.set_xlim(0, 4)
    # ax.set_xlabel("\$q / k_F\$")
    # ax.set_ylabel("\$\\Pi^{(i)}(q, i\\nu_m = 0)\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig("convergence_pi_static_rs=$rs.pdf")

    # # Plot dielectric function convergence
    # fig, ax = plt.subplots()
    # plotdata, n_max = loaddata("Π")
    # for (idx, Π) in enumerate(plotdata)
    #     i = idx - 1
    #     fmt = i == 0 ? "--" : "-"
    #     if i == 0
    #         lw = 1
    #         label = nothing
    #         color = "black"
    #     elseif i == n_max
    #         lw = 1.5
    #         label = "\$N_\\text{iter} = $n_max\$"
    #         color = "limegreen"
    #     else
    #         lw = 1
    #         label = nothing
    #         color = "#$(colors(n_max)[i])"
    #     end
    #     pi_q_static = Π[1, :]
    #     v_q = [Interaction.coulomb(q, param)[1] for q in qPgrid]
    #     inverse_static_dielectric = 1.0 ./ (1.0 .- 2.0 * v_q .* real.(pi_q_static))
    #     ax.plot(
    #         qPgrid / kF,
    #         inverse_static_dielectric,
    #         fmt;
    #         color=color,
    #         lw=lw,
    #         label=label,
    #     )
    #     # ax.plot(qPgrid * param.rs, inverse_static_dielectric, fmt; color=color, lw=lw, label=label)
    # end
    # # ax.set_xlim(0, 5 * kF)
    # # ax.set_xticks([0, 0.5, 1, 1.5, 2])
    # # ax.set_ylim(0, 1)
    # # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # ax.set_xlim(0, 3)
    # ax.set_xlabel("\$q / k_F\$")
    # # ax.set_xlim(0, 5)
    # # ax.set_xlabel("\$q * r_s\$")
    # ax.set_ylabel("\$1 / \\epsilon^{(i)}(q, i\\nu_m = 0)\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig("convergence_inverse_dielectric_static_rs=$rs.pdf")

    # # Plot Z factor convergence
    # fig, ax = plt.subplots()
    # plotdata, n_max = loaddata("Z_k")
    # for (idx, zfactor_kSgrid) in enumerate(plotdata)
    #     i = idx - 1
    #     fmt = i == 0 ? "--" : "-"
    #     if i == 0
    #         lw = 1
    #         label = nothing
    #         color = "black"
    #     elseif i == n_max
    #         lw = 1.5
    #         label = "\$N_\\text{iter} = $n_max\$"
    #         color = "limegreen"
    #     else
    #         lw = 1
    #         label = nothing
    #         color = "#$(colors(n_max)[i])"
    #     end
    #     ax.plot(kSgrid / kF, zfactor_kSgrid, fmt; color=color, lw=lw, label=label)
    # end
    # ax.set_xlim(0, 4)
    # ax.set_xlabel("\$k / k_F\$")
    # ax.set_ylabel("\$Z^{(i)}_k\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig("convergence_zfactor_rs=$rs.pdf")

    # # Plot E_qp convergence
    # ylimits = Dict(
    #     0.01 => (-4.5, 11),
    #     0.5 => (-4.5, 11),
    #     1.0 => (-5, 14),
    #     2.0 => (-0.25, 0.5),
    #     3.0 => (-0.2, 0.55),
    #     4.0 => (-0.2, 0.55),
    #     5.0 => (-0.2, 0.55),
    # )
    # fig, ax = plt.subplots()
    # plotdata, n_max = loaddata("E_k")
    # for (idx, E_qp_kSgrid) in enumerate(plotdata)
    #     i = idx - 1
    #     fmt = i == 0 ? "--" : "-"
    #     if i == 0
    #         lw = 1
    #         label = nothing
    #         color = "black"
    #     elseif i == n_max
    #         lw = 1.5
    #         label = "\$N_\\text{iter} = $n_max\$"
    #         color = "limegreen"
    #     else
    #         lw = 1
    #         label = nothing
    #         color = "#$(colors(n_max)[i])"
    #     end
    #     ax.plot(kSgrid / kF, E_qp_kSgrid, fmt; color=color, lw=lw, label=label)
    # end
    # ax.set_xlim(0, 2)
    # ax.set_ylim(ylimits[rs])
    # ax.set_xlabel("\$k / k_F\$")
    # ax.set_ylabel("\$\\mathcal{E}^{(i)}_{\\text{qp}}(k)\$")
    # ax.legend(; fontsize=12)
    # plt.tight_layout()
    # fig.savefig("convergence_quasiparticle_energy_rs=$rs.pdf")

    return
end

main()
