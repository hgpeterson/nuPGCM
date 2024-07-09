module NonhydroPG
    using Gridap
    using GridapGmsh
    using Gmsh: gmsh
    using PyPlot

    include("utils.jl")

    export 
    chebyshev_nodes,
    MyGrid,
    get_p_t,
    get_p_to_t,
    nan_eval,
    unpack_fefunction,
    quick_plot
end