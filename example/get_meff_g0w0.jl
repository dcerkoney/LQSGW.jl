using ElectronGas
using JLD2
using LQSGW
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
    # UEG parameters
    beta = 40.0
    dim = 3

    # CompositeGrid parameters
    # Nk, order = 14, 10
    Nk, order = 12, 8
    # Nk, order = 10, 7

    verbose = true
    save = true
    constant_fs = true

    calculate = Dict("rpa" => true, "fp" => true, "fp_fm" => true)
    rslist = [0.01; 0.25; 0.5; 0.75; collect(range(1.0, 10.0; step=0.5))]

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

    # Helper function to calculate one-shot GW quasiparticle properties
    function run_g0w0(param, Euv, rtol, maxK, minK, int_type=:rpa, Fs=-0.0, Fa=-0.0;)
        return get_g0w0_properties(
            param;
            Euv=Euv,
            rtol=rtol,
            Nk=Nk,
            maxK=maxK,
            minK=minK,
            order=order,
            int_type=int_type,
            Fs=int_type in [:ko_const_p, :ko_const_pm] ? Fs : -0.0,
            Fa=int_type == :ko_const_pm ? Fa : -0.0,
            verbose=verbose,
            save=save,
        )
    end

    # Calculate one-shot GW effective mass ratios
    datadict_rpa = Dict()
    datadict_fp = Dict()
    datadict_fp_fm = Dict()
    for (i, rs) in enumerate(rslist)
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
        # Compute one-shot GW quasiparticle properties
        println("Calculating one-shot GW quasiparticle properties for rs = $rs...")
        data_rpa = []
        data_fp = []
        data_fp_fm = []
        if calculate["rpa"]
            data_rpa = run_g0w0(param, Euv, rtol, maxK, minK)
        end
        if calculate["fp"]
            data_fp = run_g0w0(param, Euv, rtol, maxK, minK, int_type_fp, Fs)
        end
        if calculate["fp_fm"]
            data_fp_fm = run_g0w0(param, Euv, rtol, maxK, minK, int_type_fp_fm, Fs, Fa)
        end
        # Save data for this rs to dictionaries
        _rs = round(rs; sigdigits=13)
        for (dd, data) in zip(
            [datadict_rpa, datadict_fp, datadict_fp_fm],
            [data_rpa, data_fp, data_fp_fm],
        )
            haskey(dd, _rs) && error("Duplicate rs = $(rs) found in one-shot GW run!")
            dd[_rs] = data
        end
        println("Done.\n")
    end

    # Save the data
    for (int_type, datadict, calc) in zip(
        [:rpa, int_type_fp, int_type_fp_fm],
        [datadict_rpa, datadict_fp, datadict_fp_fm],
        [calculate["rpa"], calculate["fp"], calculate["fp_fm"]],
     )
        !calc && continue
        # Make output directory if needed
        dir = "$(DATA_DIR)/$(dim)d/$(int_type)"
        mkpath(dir)
        # Avoid overwriting existing data
        i = 0
        f = "oneshot_gw_$(dim)d_$(int_type).jld2"
        while isfile(joinpath(dir, f))
            i += 1
            f = "oneshot_gw_$(dim)d_$(int_type)_$(i).jld2"
        end
        # Save to JLD2
        jldopen(joinpath(dir, f), "w") do file
            for (k, v) in datadict
                file[string(k)] = v
            end
        end
    end
    return
end

main()
