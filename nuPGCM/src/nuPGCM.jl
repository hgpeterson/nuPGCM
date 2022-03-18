module nuPGCM

if VERSION != v"1.6.5"
    println("\nNote that this version of νPGCM.jl was only tested for Julia v1.6.5\n")
end

export 
    # constants
    secs_in_day,
    secs_in_year,
    out_folder

# global constants
const secs_in_day = 86400
const secs_in_year = 360*86400
const out_folder = "output/"

# submodules
include("Numerics/Numerics.jl")
include("OneDimensionalModel/OneDimensionalModel.jl")
include("TwoDimensionalModel/TwoDimensionalModel.jl")
include("ThreeDimensionalModel/ThreeDimensionalModel.jl")

end # module
