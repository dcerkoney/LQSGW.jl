using ElectronGas
using LQSGW
using MPI
using Parameters

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

    Σ_LQSGW(
        param::Parameter.Para;
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
    MPI.Finalize()
    return
end

main()
