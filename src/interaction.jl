
function DCKernel_qp(
    param::Parameter.Para,
    Π_qp::GreenFunc.MeshArray;
    Euv=param.EF * 100,
    rtol=1e-10,
    Nk=8,
    maxK=param.kF * 10,
    minK=param.kF * 1e-7,
    order=4,
    int_type=:rpa,
    spin_state=:auto,
    kwargs...,
)
    return DCKernel_qp(
        param::Parameter.Para,
        Π_qp::GreenFunc.MeshArray,
        Euv,
        rtol,
        Nk,
        maxK,
        minK,
        order,
        int_type,
        spin_state;
        kwargs...,
    )
end

function DCKernel_qp(
    param::Parameter.Para,
    Π_qp::GreenFunc.MeshArray,
    Euv,
    rtol,
    Nk,
    maxK,
    minK,
    order,
    int_type,
    spin_state=:auto;
    kgrid=CompositeGrid.LogDensedGrid(:cheb, [0.0, maxK], [0.0, param.kF], Nk, minK, order),
    kwargs...,
)
    # use helper function
    @unpack kF, β = param
    channel = 0

    if spin_state == :sigma
        # for self-energy, always use ℓ=0
        channel = 0
    elseif spin_state == :auto
        # automatically assign spin_state, triplet for even, singlet for odd channel
        spin_state = (channel % 2 == 0) ? (:triplet) : (:singlet)
    end

    bdlr = DLRGrid(Euv, β, rtol, false, :ph)
    qgrids = [
        CompositeGrid.LogDensedGrid(:gauss, [0.0, maxK], [k, kF], Nk, minK, order) for
        k in kgrid.grid
    ]
    qgridmax = maximum([qg.size for qg in qgrids])
    #println(qgridmax)

    kernel_bare = zeros(Float64, (length(kgrid.grid), (qgridmax)))
    kernel = zeros(Float64, (length(kgrid.grid), (qgridmax), length(bdlr.n)))

    helper_grid = CompositeGrid.LogDensedGrid(
        :cheb,
        [0.0, 2.1 * maxK],
        [0.0, 2kF],
        2Nk,
        0.01minK,
        2order,
    )
    intgrid = CompositeGrid.LogDensedGrid(
        :cheb,
        [0.0, helper_grid[end]],
        [0.0, 2kF],
        2Nk,
        0.01minK,
        2order,
    )

    # dynamic
    for (ni, n) in enumerate(bdlr.n)
        # Quasiparticle pifunc defined via 1D interpolation of Π_qp over qgrid
        pifunc_qp(q, n, param; kwargs...) = Interp.interp1D(Π_qp[ni, :], Π_qp.mesh[2], q)
        # pifunc_qp(q, n, param; kwargs...) = Interp.interp1D(Π_qp[ni, :], Π_qp.mesh[2], q; method=Interp.LinearInterp())

        # Get W_qp[Π_qp]
        helper = LegendreInteraction.helper_function_grid(
            helper_grid,
            intgrid,
            1,
            u -> LegendreInteraction.interaction_dynamic(
                u,
                n,
                param,
                int_type,
                spin_state;
                pifunc=pifunc_qp,
                kwargs...,
            ),
            param,
        )
        for (ki, k) in enumerate(kgrid.grid)
            for (pi, p) in enumerate(qgrids[ki].grid)
                Hp, Hm = Interp.interp1D(helper, helper_grid, k + p),
                Interp.interp1D(helper, helper_grid, abs(k - p))
                kernel[ki, pi, ni] = (Hp - Hm)
            end
        end
    end

    # instant
    helper = LegendreInteraction.helper_function_grid(
        helper_grid,
        intgrid,
        1,
        u -> LegendreInteraction.interaction_instant(u, param, spin_state; kwargs...),
        param,
    )
    for (ki, k) in enumerate(kgrid.grid)
        for (pi, p) in enumerate(qgrids[ki].grid)
            Hp, Hm = Interp.interp1D(helper, helper_grid, k + p),
            Interp.interp1D(helper, helper_grid, abs(k - p))
            kernel_bare[ki, pi] = (Hp - Hm)
        end
    end

    return LegendreInteraction.DCKernel(
        int_type,
        spin_state,
        channel,
        param,
        kgrid,
        qgrids,
        bdlr,
        kernel_bare,
        kernel,
    )
end

function W_qp(
    param::Parameter.Para,
    Π_qp::GreenFunc.MeshArray;
    int_type=:rpa,
    spin_state=:sigma,
    kwargs...,
)
    @assert Π_qp.mesh[1] isa ImFreq "Π_qp must be quasiparticle polarization data in the Matsubara frequency domain!"
    W_qp = similar(Π_qp)
    for (ni, n) in enumerate(Π_qp.mesh[1].grid)
        for (qi, q) in enumerate(Π_qp.mesh[2].grid)
            pifunc_qp(q, n, param; kwargs...) = Π_qp[ni, qi]  # Dummy pifunc
            W_qp[ni, qi] = LegendreInteraction.interaction_dynamic(
                q,
                n,
                param,
                int_type,
                spin_state;
                pifunc=pifunc_qp,
                kwargs...,
            )
        end
    end
    return W_qp
end
