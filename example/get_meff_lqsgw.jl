using CompositeGrids
using ElectronGas
using JLD2
using LQSGW
using MPI
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
    return F0p_sc
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
    save_qp = true
    constant_fs = true
    self_consistent_fs = true

    # Use data at previous rs as initial guess for next rs or not?
    use_prev_rs = true

    # Use G0W0 for the first rs datapoint, or existing data?
    overwrite = true

    # calculate = Dict("rpa" => true, "fp" => false, "fp_fm" => true)
    # rslist = [5.0]
    # alphalist = [0.3]

    # # f+ and f-
    # calculate = Dict("rpa" => false, "fp" => false, "fp_fm" => true)
    # rslist = [7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0]
    # alphalist = [0.1, 0.1, 0.1, 0.1, 0.1, 0.075, 0.075, 0.075, 0.05, 0.05]

    # Full calculation self-consistent f+
    calculate = Dict("rpa" => false, "fp" => true, "fp_fm" => false)
    rslist = [0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    alphalist = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]

    # # Full calculation f-
    # calculate = Dict("rpa" => false, "fp" => false, "fp_fm" => true)
    # rslist = [0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
    # alphalist = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2]

    # # Full calculation rpa and f+
    # calculate = Dict("rpa" => false, "fp" => false, "fp_fm" => true)
    # rslist = [0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    # alphalist = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
    #
    # rslist = [0.01, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    # alphalist = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]

    #calculate = Dict("rpa" => true, "fp" => true, "fp_fm" => false)
    #rslist = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    #alphalist = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

    # calculate = Dict("rpa" => false, "fp" => false, "fp_fm" => true)
    #rslist = [0.01, 0.25, 0.5]
    #alphalist = [0.3, 0.3, 0.3]

    #calculate = Dict("rpa" => false, "fp" => false, "fp_fm" => true)
    #rslist = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    #alphalist = [0.3, 0.3, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]

    #calculate = Dict("rpa" => false, "fp" => true, "fp_fm" => false)
    #rslist = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    #alphalist = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

    # # NOTE: this data point takes forever to converge. why?
    # calculate = Dict("rpa" => true, "fp" => false, "fp_fm" => false)
    # rslist = [10.0]
    # alphalist = [0.1]

    # # NOTE: α=0.2 converges up to rs=9.5 for RPA
    # calculate = Dict("rpa" => true, "fp" => false, "fp_fm" => false)
    # rslist = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
    # alphalist = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    # # NOTE: α=0.3 converges up to rs=6 for all int_types
    # calculate = Dict("rpa" => true, "fp" => true, "fp_fm" => true)
    # rslist = round.([[0.0, 0.01, 0.25, 0.5, 0.75]; LinRange(1.0, 6.0, 11)]; sigdigits=13)
    # alphalist = 0.3 * ones(length(rslist))

    # alphalist = [0.3, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1]
    # rslist = [0.25, 0.75, 1.5, 2.5, 3.5, 4.5, 5.5]
    # rslist = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    # alphalist = [0.1, 0.1, 0.1, 0.1, 0.1]
    # rslist = [6.0, 7.0, 8.0, 9.0, 10.0]

    # # rslist = [0.001; collect(LinRange(0.0, 1.1, 111))[2:end]]  # for accurate 2D HDL
    # # rslist = [0.005; collect(LinRange(0.0, 5.0, 101))[2:end]]  # for 2D
    # # rslist = [0.01; collect(LinRange(0.0, 10.0, 101))[2:end]]  # for 3D

    # rslist = [0.001; collect(LinRange(0.0, 0.25, 5))[2:end]]
    # # rslist = [0.001; collect(range(0.0, 1.1, step=0.05))[2:end]]  # for accurate 2D HDL
    # # rslist = [0.01; collect(range(0.0, 10.0, step=0.5))[2:end]]  # for 3D
    # # rslist = [1.0, 3.0]

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

    # Helper function to calculate LQSGW quasiparticle properties
    function run_lqsgw(
        param,
        Euv,
        rtol,
        maxK,
        minK,
        alpha,
        int_type=:rpa,
        Fs=-0.0,
        Fa=-0.0;
        loadname=nothing,
        overwrite=overwrite,
    )
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
            loadname=loadname,
            overwrite=overwrite,
            # savedir="$(LQSGW.DATA_DIR)/test_osp",
        )
    end

    function get_loadname(i, int_type)
        if i == 1 || !use_prev_rs
            return nothing
        else
            prev_rs = rslist[i - 1]
            return "lqsgw_$(dim)d_$(int_type)$(sc_string)_rs=$(round(prev_rs; sigdigits=4))_beta=$(round(beta; sigdigits=4)).jld2"
        end
    end

    # Calculate LQSGW effective mass ratios
    datadict_rpa = Dict()
    datadict_fp = Dict()
    datadict_fp_fm = Dict()
    for (i, (rs, alpha)) in enumerate(zip(rslist, alphalist))
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
            @assert Fs > 0 && Fa > 0 "Incorrect signs for Fs/Fa!"
        end
        # Compute LQSGW quasiparticle properties
        println_root("Calculating LQSGW quasiparticle properties for rs = $rs...")
        data_rpa = []
        data_fp = []
        data_fp_fm = []
        if calculate["rpa"]
            data_rpa = run_lqsgw(
                param,
                Euv,
                rtol,
                maxK,
                minK,
                alpha;
                loadname=get_loadname(i, :rpa),
                overwrite=overwrite && i == 1,
            )
        end
        if calculate["fp"]
            data_fp = run_lqsgw(
                param,
                Euv,
                rtol,
                maxK,
                minK,
                alpha,
                int_type_fp,
                Fs;
                loadname=get_loadname(i, int_type_fp),
                overwrite=overwrite && i == 1,
            )
        end
        if calculate["fp_fm"]
            data_fp_fm = run_lqsgw(
                param,
                Euv,
                rtol,
                maxK,
                minK,
                alpha,
                int_type_fp_fm,
                Fs,
                Fa;
                loadname=get_loadname(i, int_type_fp_fm),
                overwrite=overwrite && i == 1,
            )
        end
        # Save data for this rs to dictionaries
        _rs = round(rs; sigdigits=13)
        for (dd, data) in zip(
            [datadict_rpa, datadict_fp, datadict_fp_fm],
            [data_rpa, data_fp, data_fp_fm],
        )
            haskey(dd, _rs) && error("Duplicate rs = $(rs) found in LQSGW run!")
            dd[_rs] = data
        end
        println_root("Done.\n")
    end

    # Save the data
    if rank == root && save_qp
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
            f = "lqsgw_$(dim)d_$(int_type)$(sc_string).jld2"
            while isfile(joinpath(dir, f))
                i += 1
                f = "lqsgw_$(dim)d_$(int_type)$(sc_string)_$(i).jld2"
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
