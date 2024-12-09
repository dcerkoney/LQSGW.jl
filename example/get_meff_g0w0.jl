using CompositeGrids
using ElectronGas
using JLD2
using LQSGW
using Parameters
using PyCall
using Roots

import LQSGW: println_root, DATA_DIR

@pyimport numpy as np   # for saving/loading numpy data

const alpha_ueg = (4 / 9π)^(1 / 3)

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

"""
Solve I0[F+] = F+ / 2 to obtain a tree-level self-consistent value for F⁰ₛ.
"""
function get_self_consistent_Fs(param::Parameter.Para)
    @unpack rs = param
    function I0_KOp(x, y)
        ts = CompositeGrid.LogDensedGrid(:gauss, [0.0, 1.0], [0.0, 1.0], 32, 1e-8, 32)
        integrand = [integrand_F0p(t, x * alpha_ueg / π, y) for t in ts]
        integral = CompositeGrids.Interp.integrate1D(integrand, ts)
        return integral
    end
    F0p_sc = find_zero(Fp -> I0_KOp(rs, Fp) - Fp / 2, (-10.0, 10.0))
    # NOTE: NEFT uses opposite sign convention for F!
    return -F0p_sc
end

"""
Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via Corradini's fit
to the DMC compressibility enhancement [doi: 10.1103/PhysRevB.57.14569].
"""
@inline function get_Fs_new(param::Parameter.Para)
    kappa0_over_kappa = Interaction.compressibility_enhancement(param)
    # NOTE: NEFT uses opposite sign convention for F!
    # -F⁰ₛ = 1 - κ₀/κ
    return 1.0 - kappa0_over_kappa
end

"""
Get the antisymmetric l=0 Fermi-liquid parameter F⁰ₐ via  Corradini's fit
to the DMC susceptibility enhancement [doi: 10.1103/PhysRevB.57.14569].
"""
@inline function get_Fa_new(param::Parameter.Para)
    chi0_over_chi = Interaction.spin_susceptibility_enhancement(param)
    # NOTE: NEFT uses opposite sign convention for F!
    # -F⁰ₐ = 1 - χ₀/χ
    return 1.0 - chi0_over_chi
end

# """
# Get the symmetric l=0 Fermi-liquid parameter F⁰ₛ via a fit to the DMC compressibility
# enhancement following Kukkonen & Chen (2021) [doi: 10.48550/arXiv.2101.10508].
# """
# @inline function get_Fs_PW(rs)
#     if rs < 2.0 || rs > 5.0
#         @warn "The simple quadratic interpolation for Fs may " *
#               "be inaccurate outside the metallic regime rs = 2–5!"
#     end
#     kappa0_over_kappa = 1.0025 - 0.1721rs - 0.0036rs^2
#     # NOTE: NEFT uses opposite sign convention for F!
#     # -F⁰ₛ = 1 - κ₀/κ
#     return 1.0 - kappa0_over_kappa
# end

# """
# Get the antisymmetric l=0 Fermi-liquid parameter F⁰ₐ via a fit to the DMC susceptibility
# enhancement following Kukkonen & Chen (2021) [doi: 10.48550/arXiv.2101.10508].
# """
# @inline function get_Fa_PW(rs)
#     if rs < 2.0 || rs > 5.0
#         @warn "The simple quadratic interpolation for Fa may " *
#               "be inaccurate outside the metallic regime rs = 2–5!"
#     end
#     chi0_over_chi = 0.9821 - 0.1232rs + 0.0091rs^2
#     # NOTE: NEFT uses opposite sign convention for F!
#     # -F⁰ₐ = 1 - χ₀/χ
#     return 1.0 - chi0_over_chi
# end

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
    self_consistent_fs = true

    calculate = Dict("rpa" => false, "fp" => true, "fp_fm" => false)
    rslist = [0.01; 0.25; 0.5; 0.75; collect(range(1.0, 10.0; step=0.5))]

    if self_consistent_fs
        @assert constant_fs == true "We require `int_type = :ko_const`` for the self-consistent Fs calculation!"
        @assert calculate["fp"] == true "Self-consistent calculation requires f+ run!"
        @assert calculate["fp_fm"] == false "Self-consistent calculation with Fa is not currently supported!"
    end

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
    sc_string = self_consistent_fs && int_type_fp == :ko_const_p ? "_sc" : ""

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
        if self_consistent_fs
            # Get Fermi liquid parameter F⁰ₛ(rs) from tree-level self-consistent calculation
            Fs = get_self_consistent_Fs(param)
            Fa = 0.0
        else
            # Get Fermi liquid parameters F⁰ₛ(rs) and F⁰ₐ(rs) from Corradini fits
            Fs = get_Fs_new(param)
            Fa = get_Fa_new(param)
            # # Get Fermi liquid parameters F⁰ₛ(rs) and F⁰ₐ(rs) from Kun & Kukkonen fits
            # Fs = get_Fs_PW(rs)
            # Fa = get_Fa_PW(rs)
        end
        if param.rs > 0.25
            @assert Fs > 0 && Fa > 0 "Signs for Fs/Fa should be positive to match the ElectronGas.jl convention!"
        end
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
        dir = "$(DATA_DIR)/$(dim)d/$(int_type)$(sc_string)"
        mkpath(dir)
        # Avoid overwriting existing data
        i = 0
        f = "oneshot_gw_$(dim)d_$(int_type)$(sc_string).jld2"
        while isfile(joinpath(dir, f))
            i += 1
            f = "oneshot_gw_$(dim)d_$(int_type)$(sc_string)_$(i).jld2"
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
