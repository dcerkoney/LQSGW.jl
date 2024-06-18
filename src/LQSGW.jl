"""
Calculate the linearized quasiparticle self-consistent GW (LQSGW) self-energy in 3D
"""
module LQSGW

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
    massratio

include("green.jl")
export G_qp

include("polarization.jl")
export Π_qp

include("interaction.jl")
export W_qp

include("selfenergy.jl")
export GW, Σ_LQSGW

# export ...

"""
    _lerp(M_start, M_end, alpha)

Helper function for linear interpolation with mixing parameter α.
"""
function _lerp(M_start, M_end, alpha)
    return (1 - alpha) * M_start + alpha * M_end
end

"""
    _split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`.
Used to chunk polarization for MPI parallelization.
"""
function _split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i in 1:n]
end

end
