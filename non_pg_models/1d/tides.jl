using nuPGCM, PyPlot, PyCall, SpecialFunctions, HDF5, Printf

# libraries
include("structs.jl")
include("plotting.jl")
include("utils.jl")
include("evolution.jl")
include("steady.jl")

function plot_oscillatory_component(folder)
    # S and H
    # S = "1e-3"
    S = "5e-1"
    H = "2km"

    # setup
    m = loadSetup1D(string(folder, "/S", S, "/H", H, "/tides/setup.h5"))

    for i=1:100
       # no tide state 
       s_nt = loadState1D(string(folder, "/S", S, "/H", H, "/no_tides/state$i.h5"))

       # tide state 
       s_t = loadState1D(string(folder, "/S", S, "/H", H, "/tides/state$i.h5"))

       # oscillatory component
       u = s_t.u - s_nt.u
       v = s_t.v - s_nt.v
       b = s_t.b - s_nt.b
       ∂ₓP = s_t.∂ₓP - s_nt.∂ₓP

       # plot
       imgFile = @sprintf("tides_%03d.png", i)
       profilePlot(m, u, v, b, ∂ₓP, s_t.i, imgFile)
    end
end

path = "../../sims/"
plot_oscillatory_component(string(path, "sim045"))