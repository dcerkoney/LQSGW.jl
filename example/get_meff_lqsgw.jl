using ElectronGas
using LQSGW
using MPI
using Parameters
using PyCall

import LQSGW: println_root

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
    # F⁰ₛ = κ₀/κ - 1
    return kappa0_over_kappa - 1.0
end

"""
Get the antisymmetric l=0 Fermi-liquid parameter F⁰ₐ via interpolation of the 
susceptibility ratio data (c.f. Kukkonen & Chen, 2021)
"""
@inline function get_Fa_PW(rs)
    chi0_over_chi = 0.9821 - 0.1232rs + 0.0091rs^2
    # F⁰ₐ = χ₀/χ - 1
    return chi0_over_chi - 1.0
end

function main()
    MPI.Init()

    # UEG parameters
    beta = 40.0
    dim = 3

    # CompositeGrid parameters
    # Nk, order = 14, 10
    # Nk, order = 12, 8
    Nk, order = 10, 7

    # LQSGW parameters
    max_steps = 200
    atol = 1e-5
    alpha = 0.3
    δK = 5e-6
    verbose = true
    save = true
    constant_fs = true

    rslist = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    # rslist = [1.0, 3.0]

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

    rpa_dirstr = "rpa"
    if int_type_fp_fm == :ko_const_pm
        ko_dirstr = "const"
    elseif int_type_fp_fm == :ko_takada
        ko_dirstr = "takada"
    elseif int_type_fp_fm == :ko_simion_giuliani
        ko_dirstr = "simion_giuliani"
    end

    # Output directory
    mkpath("results/$(dim)d/$(rpa_dirstr)")
    mkpath("results/$(dim)d/$(ko_dirstr)")
    dir = joinpath(@__DIR__, "results/$(dim)d")

    # Helper function to calculate LQSGW quasiparticle properties
    function run_lqsgw(param, Euv, rtol, maxK, minK, int_type=:rpa, Fs=0.0, Fa=0.0)
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
            Fs=Fs,
            Fa=Fa,
            verbose=verbose,
            save=save,
        )
    end

    # Calculate LQSGW effective mass ratios
    mefflist_rpa = []
    mefflist_fp = []
    mefflist_fp_fm = []
    zlist_rpa = []
    zlist_fp = []
    zlist_fp_fm = []
    dmulist_rpa = []
    dmulist_fp = []
    dmulist_fp_fm = []
    for rs in rslist
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
        meff_rpa, z_rpa, dmu_rpa       = run_lqsgw(param, Euv, rtol, maxK, minK)
        meff_fp, z_fp, dmu_fp          = run_lqsgw(param, Euv, rtol, maxK, minK, int_type_fp, Fs, Fa)
        meff_fp_fm, z_fp_fm, dmu_fp_fm = run_lqsgw(param, Euv, rtol, maxK, minK, int_type_fp_fm, Fs, Fa)
        push!(mefflist_rpa, meff_rpa)
        push!(mefflist_fp, meff_fp)
        push!(mefflist_fp_fm, meff_fp_fm)
        push!(zlist_rpa, z_rpa)
        push!(zlist_fp, z_fp)
        push!(zlist_fp_fm, z_fp_fm)
        push!(dmulist_rpa, dmu_rpa)
        push!(dmulist_fp, dmu_fp)
        push!(dmulist_fp_fm, dmu_fp_fm)
        println_root("Done.\n")
    end

    # Add points at rs = 0
    pushfirst!(rslist, 0.0)
    pushfirst!(mefflist_rpa, 1.0)
    pushfirst!(mefflist_fp, 1.0)
    pushfirst!(mefflist_fp_fm, 1.0)
    pushfirst!(zlist_rpa, 1.0)
    pushfirst!(zlist_fp, 1.0)
    pushfirst!(zlist_fp_fm, 1.0)
    pushfirst!(dmulist_rpa, 0.0)
    pushfirst!(dmulist_fp, 0.0)
    pushfirst!(dmulist_fp_fm, 0.0)

    # Save the data
    f1 = "$(rpa_dirstr)/lqsgw_$(dim)d_rpa.npz"
    f2 = "$(ko_dirstr)/lqsgw_$(dim)d_fp.npz"
    f3 = "$(ko_dirstr)/lqsgw_$(dim)d_fp_fm.npz"
    i1 = i2 = i3 = 0
    while isfile(f1)
        i1 += 1
        f1 = "$(rpa_dirstr)/lqsgw_$(dim)d_rpa_$(i1).npz"
    end
    while isfile(f2)
        i2 += 1
        f2 = "$(rpa_dirstr)/lqsgw_$(dim)d_rpa_$(i2).npz"
    end
    while isfile(f3)
        i3 += 1
        f3 = "$(rpa_dirstr)/lqsgw_$(dim)d_rpa_$(i3).npz"
    end
    np.savez(
        joinpath(dir, f1);
        rslist=rslist,
        mefflist=mefflist_rpa,
        zlist=zlist_rpa,
        dmulist=dmulist_rpa,
    )
    np.savez(
        joinpath(dir, f2);
        rslist=rslist,
        mefflist=mefflist_fp,
        zlist=zlist_fp,
        dmulist=dmulist_fp,
    )
    np.savez(
        joinpath(dir, f3);
        rslist=rslist,
        mefflist=mefflist_fp_fm,
        zlist=zlist_fp_fm,
        dmulist=dmulist_fp_fm,
    )
    MPI.Finalize()
    return
end

main()
