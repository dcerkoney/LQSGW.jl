"""
Calculate the linearized quasiparticle self-consistent GW (LQSGW) self-energy in 3D
"""
module LQSGW

using CodecZlib
using CompositeGrids
using ElectronGas
using GreenFunc
using JLD2
using Lehmann
using MCIntegration
using MPI
using ProgressMeter
using Parameters

# using ..Parameter, ..Convention, ..Polarization, ..Interaction, ..LegendreInteraction, ..SelfEnergy

const MAXIMUM_STEPS = 1000
const PROJECT_ROOT = pkgdir(LQSGW)
const DATA_DIR = joinpath(PROJECT_ROOT, "data")

include("quasiparticle_properties.jl")
export bare_energy,
    quasiparticle_energy,
    # E_0_interp,
    # E_qp_interp,
    chemicalpotential,
    zfactor_fermi,
    zfactor_full,
    dfactor,
    massratio,
    get_lqsgw_properties,
    get_g0w0_properties

include("green.jl")
export G_0, G_qp

include("polarization.jl")
export Π_qp

include("interaction.jl")
export W_qp

include("selfenergy.jl")
export GW, Σ_LQSGW

# export ...

"""
    function round8(float::Float64)

Round a floating point number to 8 significant digits (precision ~√eps).
"""
function round8(float::Float64)
    return round(float; sigdigits=8)
end

"""
    function uniqueperm(a)

Return the indices of unique elements of `a` in the order 
they first appear, such that `a[uniqueperm(a)] == unique(a)`.
"""
function uniqueperm(a)
    return unique(i -> a[i], eachindex(a))
end

"""
    lerp(M_start, M_end, alpha)

Helper function for linear interpolation with mixing parameter α.
"""
function lerp(M_start, M_end, alpha)
    return (1 - alpha) * M_start + alpha * M_end
end

# """
#     get_instantaneous(param::Parameter.Para, fdlr, imfreq_data)

# Given imaginary-frequency data, use the DLR to obtain the corresponding instantaneous (τ = 0⁻) data.
# """
# function get_instantaneous(param::Parameter.Para, fdlr, imfreq_data)
#     return matfreq2tau(fdlr, imfreq_data, [param.β]) * (-1)
# end

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
Used to chunk polarization for MPI parallelization.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i in 1:n]
end

function println_root(io::IO, msg)
    MPI.Initialized() == false && MPI.Init()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(io, msg)
    end
end

function println_root(msg)
    MPI.Initialized() == false && MPI.Init()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        println(msg)
    end
end

function print_root(io::IO, msg)
    MPI.Initialized() == false && MPI.Init()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        print(io, msg)
    end
end

function print_root(msg)
    MPI.Initialized() == false && MPI.Init()
    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        print(msg)
    end
end

const TimedResultType{T} = @NamedTuple{
    value::T,
    time::Float64,
    bytes::Int64,
    gctime::Float64,
    gcstats::Base.GC_Diff,
} where {T}

function timed_result_to_string(timed_res::TimedResultType)
    time = round(timed_res.time; sigdigits=3)
    num_bytes = Base.format_bytes(timed_res.bytes)
    num_allocs = timed_res.gcstats.malloc + timed_res.gcstats.poolalloc
    return "  $time seconds ($num_allocs allocations: $num_bytes)"
end

end  # module LQSGW
