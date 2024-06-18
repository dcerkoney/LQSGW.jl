using ElectronGas
using LQSGW
using MPI
using Parameters

"""
Get the LQSGW effective mass ratio from the self-energy.
"""
function meff_from_Σ_LQSGW(
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
    max_steps=10,
    atol=1e-7,
    alpha=0.3,
    verbose=true,
    save=false,
    mpi=true,
    savedir="$(DATA_DIR)/$(dim)d",
    savename="lqsgw_$(param.dim)d_$(int_type)_rs=$(param.rs)_beta=$(beta).jld2",
)
    Σ, Σ_ins = Σ_LQSGW(
        param;
        Euv=Euv,
        rtol=rtol,
        Nk=Nk,
        maxK=maxK,
        minK=minK,
        order=order,
        int_type=int_type,
        δK=δK,
        Fs=Fs,
        Fa=Fa,
        max_steps=max_steps,
        atol=atol,
        alpha=alpha,
        verbose=verbose,
        save=save,
        mpi=mpi,
        savedir=savedir,
        savename=savename,
    )
    return massratio(param, Σ, Σ_ins, δK)[1]
end

function main()
    MPI.Init()

    # UEG parameters
    beta = 40.0
    rs = 1.0
    dim = 3
    param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
    @unpack kF, EF = param

    # DLR parameters
    Euv = 1000 * EF
    rtol = 1e-14

    # CompositeGrid parameters
    Nk, order = 14, 10
    # Nk, order = 12, 8
    # Nk, order = 10, 6

    # ElectronGas.jl defaults for G0W0 self-energy
    maxK = 6 * kF
    minK = 1e-6 * kF

    # Test LQSGW parameters
    max_steps = 50
    atol = 1e-7
    alpha = 0.5
    δK = 5e-6
    Fs = 0.0
    Fa = 0.0
    verbose = true
    save = true
    mpi = true

    # Calculate the LQSGW effective mass ratio
    meff = get_meff_from_Σ_LQSGW(
        param;
        Euv=Euv,
        rtol=rtol,
        Nk=Nk,
        maxK=maxK,
        minK=minK,
        order=order,
        int_type=:rpa,
        max_steps=max_steps,
        atol=atol,
        alpha=alpha,
        δK=δK,
        Fs=Fs,
        Fa=Fa,
        verbose=verbose,
        save=save,
        mpi=mpi,
    )
    println("UEG parameters:\tβ = $beta, rs = $rs, dim = $dim")
    println("LQSGW effective mass ratio:\tm*/m = $meff")
    MPI.Finalize()
    return
end

main()
