using CodecZlib
using ElectronLiquid
using JLD2
using LQSGW
using Measurements
using PyCall

@pyimport numpy as np   # for saving/loading numpy data

const available_int_types =
    ["rpa", "ko_const_p", "ko_const_pm", "ko_moroni", "ko_simion_giuliani"]

const npz_keys = ["rslist", "mefflist", "zlist", "dmulist"]

function merge_data(int_type, dim, overwrite, saveto)
    @assert int_type in available_int_types "Unrecognized int_type subdirectory name '$(int_type)'! Choose from one of the following: $(available_int_types)."

    @assert saveto in ["jld2", "npz"] "Unrecognized saveto format '$(saveto)'! Choose from one of the following: ['jld2', 'npz']."

    data_path = joinpath(LQSGW.DATA_DIR, "$(dim)d/$(int_type)")
    println("Merging LQSGW npz data in subdirectory $(data_path)...")

    i = 0
    suffix = overwrite ? "" : "_merged"
    datadict = Dict()  # rs => (i, meff, z, dmu)
    while true
        try
            istr = i == 0 ? "" : "_$(i)"
            d = np.load(joinpath(data_path, "lqsgw_$(dim)d_$(int_type)$(istr).npz"))
            println("Found $(int_type) data #$(i+1) in subdirectory $(data_path)")

            for (rs, meff, z, dmu) in zip((d.get(k) for k in npz_keys)...)
                println("rs = $(rs), meff = $(meff), z = $(z), dmu = $(dmu)")
                _rs = round(rs; sigdigits=13)
                if haskey(datadict, _rs)
                    i_old = datadict[_rs][1]
                    println(
                        "WARNING: Overwriting existing data for rs=$(_rs) (i = $i_old -> $i)!",
                    )
                end
                datadict[_rs] = (i, meff, z, dmu)
            end
        catch e
            # println(e)
            if i == 0
                error("No $(int_type) data found in subdirectory $(data_path)")
            elseif i == 1
                println(
                    "Found 1 $(int_type) data file in subdirectory $(data_path), no merge required!",
                )
                return
            else
                print(
                    "Found $(i+1) $(int_type) data files in subdirectory $(data_path), saving merged data...",
                )
                if saveto == "jld2"
                    jldopen(
                        joinpath(data_path, "lqsgw_$(dim)d_$(int_type)$(suffix).jld2"),
                        "w",
                    ) do file
                        for (rs, data) in datadict
                            i, meff, z, dmu = data
                            file[string(rs)] = [i, meff, z, dmu]
                        end
                        return
                    end
                else
                    # Build merged data arrays
                    rslist, mefflist, zlist, dmulist = [], [], [], []
                    for rs in sort(collect(keys(datadict)))
                        i, meff, z, dmu = datadict[rs]
                        push!(rslist, rs)
                        push!(mefflist, meff)
                        push!(zlist, z)
                        push!(dmulist, dmu)
                    end
                    println()
                    println("Merged data:\n")
                    println("rslist = ", rslist)
                    println("mefflist = ", mefflist)
                    println("zlist = ", zlist)
                    println("dmulist = ", dmulist)
                    println()
                    np.savez(
                        joinpath(data_path, "lqsgw_$(dim)d_$(int_type)$(suffix).npz");
                        rslist=rslist,
                        mefflist=mefflist,
                        zlist=zlist,
                        dmulist=dmulist,
                    )
                end
                println("done!")
            end
            break
        end
        i += 1
    end
end

function merge_all_data(; dim=3, overwrite=false, saveto="jld2")
    for int_type in available_int_types
        merge_data(int_type, dim, overwrite, saveto)
    end
    return
end

merge_all_data()
