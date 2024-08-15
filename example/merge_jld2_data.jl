using CodecZlib
using ElectronLiquid
using JLD2
using LQSGW
using Measurements
using PyCall

const available_int_types =
    ["rpa", "ko_const_p", "ko_const_pm", "ko_moroni", "ko_simion_giuliani"]

function merge_data(int_type, dim, overwrite)
    @assert int_type in available_int_types "Unrecognized int_type subdirectory name '$(int_type)'! Choose from one of the following: $(available_int_types)."

    data_path = joinpath(LQSGW.DATA_DIR, "$(dim)d/$(int_type)")
    println("Merging LQSGW JLD2 data in subdirectory $(data_path)...")
    i = 0
    suffix = overwrite ? "" : "_merged"
    datadict = Dict()  # rs => [(; atol, alpha, i_step, converged, meff, zfactor, dmu), ...]
    while true
        try
            istr = i == 0 ? "" : "_$(i)"
            p = joinpath(data_path, "lqsgw_$(dim)d_$(int_type)$(istr).jld2")
            isfile(p)
            println("Found $(int_type) data #$(i+1) in subdirectory $(data_path)")
            jldopen(p, "r") do file
                for rs in keys(file)
                    # Append to existing data if already existing for this rs
                    if haskey(datadict, rs)
                        println("Appending data to entry at rs=$rs")
                        push!(datadict[rs], file[rs])
                    else
                        println("Creating new entry at rs=$rs")
                        datadict[rs] = [file[rs]]
                    end
                end
            end
        catch e
            println(e)
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
                jldopen(
                    joinpath(data_path, "lqsgw_$(dim)d_$(int_type)$(suffix)_new.jld2"),
                    "w",
                ) do file
                    for (k, v) in datadict
                        file[string(k)] = unique(v)  # remove duplicate entries, if any
                    end
                end
                println("done!")
            end
            break
        end
        i += 1
    end
end

function merge_all_data(; dim=3, overwrite=false)
    for int_type in available_int_types
        merge_data(int_type, dim, overwrite)
    end
    return
end

merge_all_data()
