module Numerics

using LinearAlgebra

export mkfdstencil,
differentiate_pointwise,
differentiate

include("finite_differences.jl")

end # module