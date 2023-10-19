### Model state

"""
    s = ModelSetup3D(b, ωx, ωy, χx, χy, Ψ, i)

State of 3D model.
"""
struct ModelState3D{I, F1, F2, FS1}
    # buoyancy
	b::F2 # FEField on second order grid

    # vorticity
	ωx::F1 # FEField on first order grid
	ωy::F1

    # streamfunction
    χx::F1
    χy::F1

    # barotropic streamfunction
    Ψ::FS1 # FEField on first order surface grid

    # iteration
    i::I # integer
end

### Model parameters

struct Params{FT}
    # Ekman number
    ε²::FT # float

    # Prandtl number
    μ::FT

    # Burger number
    ϱ::FT

    # timestep 
    Δt::FT

    # Coriolis parameter
    f::FT

    # meridional gradient of Coriolis
    β::FT
end

"""
    params = Params(; ε², μ, ϱ, Δt, f, β)

Set of numerical parameters for 3D model.
"""
function Params(; ε², μ, ϱ, Δt, f, β)
    return Params(ε², μ, ϱ, Δt, f, β)
end

### Model geometry

struct Geometry{F1, F2, GS, VI, G, GC, VF, I}
    # depth and its derivatives
    H::F1 # `FEField` on g_sfc1
    Hx::F2 # `FEField` on g_sfc2
    Hy::F2

    # surface grids in x and y (first and second order)
    g_sfc1::GS # `Grid` of `Triangle`s
    g_sfc2::GS

    # indices of interior nodes for g_sfc1 and g_sfc2, resp.
    in_nodes1::VI # vector of integers
    in_nodes2::VI

    # grids in x, y, σ (first and second order)
    g1::G # `Grid` of `Wedge`s
    g2::G

    # 1D grid in σ (first order)
    g_col::GC # `Grid` of `Line`s

    # nodes in σ (equivalent to g_col.p)
    σ::VF # vector of floats

    # number of nodes in σ (equivalent to g_col.np)
    nσ::I # integer
end

"""
    geom = Geometry(basin_shape, H_func::Function; res=2, nσ=0, chebyshev=false)
"""
function Geometry(basin_shape, H_func::Function; res=2, nσ=0, chebyshev=false)
    if basin_shape ∉ [:circle, :square]
        error("Unsupported basin shape: $basin_shape.")
    end
    g_sfc1 = Grid(Triangle(order=1), "$(@__DIR__)/../../meshes/$(string(basin_shape))/mesh$res.h5")
    
    # second order surface mesh
    g_sfc2 = add_midpoints(g_sfc1)

    # indices of nodes in interior
    in_nodes1 = findall(i -> i ∉ g_sfc1.e["bdy"], 1:g_sfc1.np)
    in_nodes2 = findall(i -> i ∉ g_sfc2.e["bdy"], 1:g_sfc2.np)

    # 3D mesh
    g1, g2, σ = generate_wedge_cols(g_sfc1, g_sfc2, nσ=nσ, chebyshev=chebyshev)

    # 1D grid
    nσ = length(σ)
    p = σ
    t = [i + j - 1 for i=1:nσ-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nσ])
    g_col = Grid(Line(order=1), p, t, e)

    # convert H to FE field
    H = FEField(H_func, g_sfc2)

    # H gradients are DG fields
    Hx = DGField([∂x(H, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    Hy = DGField([∂y(H, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

    return Geometry(H, Hx, Hy, g_sfc1, g_sfc2, in_nodes1, in_nodes2, g1, g2, g_col, σ, nσ)
end

### Model forcing

struct Forcing{F1, F2, F3, F4} 
    # wind stress
    τx::F1 # `FEField` on g_sfc2
    τy::F1

    # wind stress gradients
    τx_x::F2 # `DGField` on g_sfc2
    τx_y::F2
    τy_x::F2
    τy_y::F2

    # viscosity and diffusivity
    ν::F3 # `FEField` on g1
    ν_bot::F4 # `FEField` on g_sfc1
    κ::F3 # `FEField` on g1
end

"""
    forcing = Forcing(geom::Geometry, τx_func::Function, τy_func::Function, ν_func::Function, κ_func::Function)
"""
function Forcing(geom::Geometry, τx_func::Function, τy_func::Function, ν_func::Function, κ_func::Function)
    # unpack
    g_sfc1 = geom.g_sfc1
    g_sfc2 = geom.g_sfc2
    nσ = geom.nσ
    g2 = geom.g2
    H = geom.H

    # convert functions to FE fields
    τx = FEField(τx_func, g_sfc2)
    τy = FEField(τy_func, g_sfc2)
    ν = FEField([ν_func(g2.p[i, 3], H[get_i_sfc(i, nσ)]) for i=1:g2.np], g2)
    ν_bot = FEField(ν[get_i_bot.(1:g_sfc1.np, nσ)], g_sfc1)
    κ = FEField([κ_func(g2.p[i, 3], H[get_i_sfc(i, nσ)]) for i=1:g2.np], g2)

    # gradients as DG fields
    τx_x = DGField([∂x(τx, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τx_y = DGField([∂y(τx, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τy_x = DGField([∂x(τy, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τy_y = DGField([∂y(τy, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

    return Forcing(τx, τy, τx_x, τx_y, τy_x, τy_y, ν, ν_bot, κ)
end

### Model

struct ModelSetup3D{P, G, F, I, E}
    # model parameters
    params::P # `Params`

    # model geometry
    geom::G # `Geometry`

    # model forcing
    forcing::F # `Forcing`

    # components for inversion
    inversion::I # `InversionComponents`

    # components for evolution
    evolution::E # `EvolutionComponents`
end

"""
    m = ModelSetup3D(params::Params, geom::Geometry, forcing::Forcing; advection=true)
"""
function ModelSetup3D(params::Params, geom::Geometry, forcing::Forcing; advection=true)
    quick_plot(geom.H, L"H", "$out_folder/H.png")
    quick_plot(geom.Hx, L"H_x", "$out_folder/Hx.png")
    quick_plot(geom.Hy, L"H_y", "$out_folder/Hy.png")
    f_over_H = FEField(x->params.f + params.β*x[2], geom.g_sfc2)/(geom.H + FEField(1e-5, geom.g_sfc2))
    quick_plot(f_over_H, L"f/H", "$out_folder/f_over_H.png", vmax=6)
    curl = (forcing.τy_x - forcing.τx_y)*geom.H - (forcing.τy*geom.Hx - forcing.τx*geom.Hy)
    quick_plot(curl, L"H^2 \mathbf{z} \cdot \nabla \times (\tau / H)", "$out_folder/curl.png")

    inversion = InversionComponents(params, geom, forcing)
    evolution = EvolutionComponents(params, geom, forcing, advection)

    CUDA.memory_status()

    flush(stdout)
    flush(stderr)

    return ModelSetup3D(params, geom, forcing, inversion, evolution)
end