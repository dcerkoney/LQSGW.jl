using ElectronGas
using JLD2
using LQSGW
using MPI
using Parameters
using PyCall

import LQSGW: println_root, DATA_DIR

@pyimport numpy as np   # for saving/loading numpy data

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via interpolation of the 
compressibility ratio data of Perdew & Wang (1992) [Phys. Rev. B 45, 13244].
"""
@inline function get_Fs_PW(rs)
    # if rs < 1.0 || rs > 5.0
    #     @warn "The Perdew-Wang interpolation for Fs may " *
    #           "be inaccurate outside the metallic regime!"
    # end
    kappa0_over_kappa = 1.0025 - 0.1721rs - 0.0036rs^2
    # NOTE: NEFT uses opposite sign convention for F!
    # -F⁰ₛ = 1 - κ₀/κ
    return 1.0 - kappa0_over_kappa
end

"""
Get the antisymmetric l=0 Fermi-liquid parameter F⁰ₐ via interpolation of the 
susceptibility ratio data (c.f. Kukkonen & Chen, 2021)
"""
@inline function get_Fa_PW(rs)
    chi0_over_chi = 0.9821 - 0.1232rs + 0.0091rs^2
    # NOTE: NEFT uses opposite sign convention for F!
    # -F⁰ₐ = 1 - χ₀/χ
    return 1.0 - chi0_over_chi
end

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    root = 0
    rank = MPI.Comm_rank(comm)

    # UEG parameters
    beta = 40.0
    dim = 3

    # CompositeGrid parameters
    # Nk, order = 14, 10
    Nk, order = 12, 8
    # Nk, order = 10, 7

    # LQSGW parameters
    max_steps = 200
    atol = 1e-5
    #alpha = 0.2
    δK = 5e-6
    verbose = true
    save = true
    constant_fs = true
    #constant_fs = false

    rslist = round.([[0.01, 0.25, 0.5, 0.75]; LinRange(1.0, 10.0, 19)]; sigdigits=13)
    alphalist = 0.3 * ones(length(rslist))

    # alphalist = [0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]
    #rslist = [0.25, 0.75, 1.5, 2.5, 3.5, 4.5, 5.5]
    # rslist = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    #alphalist = [0.1, 0.1, 0.1, 0.1, 0.1]
    #rslist = [6.0, 7.0, 8.0, 9.0, 10.0]

    # # rslist = [0.001; collect(LinRange(0.0, 1.1, 111))[2:end]]  # for accurate 2D HDL
    # # rslist = [0.005; collect(LinRange(0.0, 5.0, 101))[2:end]]  # for 2D
    # # rslist = [0.01; collect(LinRange(0.0, 10.0, 101))[2:end]]  # for 3D

    # rslist = [0.001; collect(LinRange(0.0, 0.25, 5))[2:end]]
    # # rslist = [0.001; collect(range(0.0, 1.1, step=0.05))[2:end]]  # for accurate 2D HDL
    # # rslist = [0.01; collect(range(0.0, 10.0, step=0.5))[2:end]]  # for 3D
    # # rslist = [1.0, 3.0]

    # NOTE: int_type ∈ [:ko_const, :ko_takada_plus, :ko_takada, :ko_moroni, :ko_simion_giuliani] 
    # NOTE: KO interaction using G+ and/or G- is currently only available in 3D
    if constant_fs
        int_type_fp = :ko_const_p
        int_type_fp_fm = :ko_const_pm
    else
        int_type_fp = :ko_moroni
        int_type_fp_fm = :ko_simion_giuliani
    end
    @assert int_type_fp ∈ [:ko_const_p, :ko_takada_plus, :ko_moroni]
    @assert int_type_fp_fm ∈ [:ko_const_pm, :ko_takada, :ko_simion_giuliani]

    # Helper function to calculate LQSGW quasiparticle properties
    function run_lqsgw(param, Euv, rtol, maxK, minK, alpha, int_type=:rpa, Fs=-0.0, Fa=-0.0)
        return get_lqsgw_properties(
            param;
            Euv=Euv,
            rtol=rtol,
            Nk=Nk,
            maxK=maxK,
            minK=minK,
            order=order,
            int_type=int_type,
            max_steps=max_steps,
            atol=atol,
            alpha=alpha,
            δK=δK,
            Fs=int_type in [:ko_const_p, :ko_const_pm] ? Fs : -0.0,
            Fa=int_type == :ko_const_pm ? Fa : -0.0,
            verbose=verbose,
            save=save,
        )
    end

    # Calculate LQSGW effective mass ratios
    # rs => (i, alpha, meff, z, dmu)
    datadict_rpa = Dict(0.0 => (0, 1.0, 1.0, 1.0, 0.0))
    datadict_fp = Dict(0.0 => (0, 1.0, 1.0, 1.0, 0.0))
    datadict_fp_fm = Dict(0.0 => (0, 1.0, 1.0, 1.0, 0.0))
    for (rs, alpha) in zip(rslist, alphalist)
        param = Parameter.rydbergUnit(1.0 / beta, rs, dim)
        @unpack kF, EF = param
        # DLR parameters
        Euv = 1000 * EF
        rtol = 1e-14
        # ElectronGas.jl defaults for G0W0 self-energy
        maxK = 6 * kF
        minK = 1e-6 * kF
        # Get Fermi liquid parameter F⁰ₛ(rs) from Perdew-Wang fit
        Fs = get_Fs_PW(rs)
        Fa = get_Fa_PW(rs)
        # Compute LQSGW quasiparticle properties
        println_root("Calculating LQSGW quasiparticle properties for rs = $rs...")
        data_rpa = run_lqsgw(param, Euv, rtol, maxK, minK, alpha)
        data_fp = run_lqsgw(param, Euv, rtol, maxK, minK, alpha, int_type_fp, Fs)
        data_fp_fm = run_lqsgw(param, Euv, rtol, maxK, minK, alpha, int_type_fp_fm, Fs, Fa)
        # Save data for this rs to dictionaries
        _rs = round(rs; sigdigits=13)
        for (dd, data) in zip(
            [datadict_rpa, datadict_fp, datadict_fp_fm],
            [data_rpa, data_fp, data_fp_fm],
        )
            haskey(dd, _rs) && error("Duplicate rs = $rs!")
            dd[_rs] = (alpha, data...)
        end
        println_root("Done.\n")
    end

    # Save the data
    if rank == root
        for (int_type, datadict) in zip(
            [:rpa, int_type_fp, int_type_fp_fm],
            [datadict_rpa, datadict_fp, datadict_fp_fm],
        )
            # Make output directory if needed
            dir = "$(DATA_DIR)/$(dim)d/$(int_type)"
            mkpath(dir)
            # Avoid overwriting existing data
            i = 0
            f = "lqsgw_$(dim)d_$(int_type).jld2"
            while isfile(joinpath(dir, f))
                i += 1
                f = "lqsgw_$(dim)d_$(int_type)_$(i).jld2"
            end
            # Save to JLD2
            jldopen(joinpath(dir, f), "w") do file
                for (k, v) in datadict
                    file[string(k)] = v
                end
            end
        end
    end
    MPI.Finalize()
    return
end

main()
