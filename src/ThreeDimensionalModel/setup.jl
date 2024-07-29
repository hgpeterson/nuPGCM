### Model state

"""
    s = ModelSetup3D(b, ωx, ωy, χx, χy, Ψ, i)

State of 3D model.
"""
struct ModelState3D{F2, F1, FS1, T}
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

    # time
    t::T # vector containing one number
end

### Model parameters

struct Params{FT}
    # Ekman number
    ε²::FT # float

    # Prandtl*Burger number
    μϱ::FT

    # Coriolis parameter
    f::FT

    # meridional gradient of Coriolis
    β::FT

    # Streamline diffusion strength
    δ₀::FT
end

"""
    params = Params(; ε², μϱ, f, β, δ₀)

Set of numerical parameters for 3D model.
"""
function Params(; ε², μϱ, f, β, δ₀)
    return Params(ε², μϱ, f, β, δ₀)
end

### Model geometry

struct Geometry{F1, F2, GS, VI, G, AC, AB, GC, VF, I}
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

    # map from (g_sfc1.nt × g_sfc1.nn × nσ) to (g1.nt × g1.nn)
    g_sfc1_to_g1_map::AC # 3-dim array of CartesianIndices

    # coast mask in (g_sfc1.nt × g_sfc1.nn × nσ) space
    coast_mask::AB # 3-dim array of Bools

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
    if basin_shape ∉ [:circle, :square, :rectangle]
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
    nσ = length(σ)

    # map from (g_sfc1.nt × g_sfc1.nn × nσ) to (g1.nt × g1.nn)
    g_sfc1_to_g1_map = build_g_sfc1_to_g1_map(g_sfc1, g1, nσ)

    # coast mask in (g_sfc1.nt × g_sfc1.nn × nσ) space
    coast_mask = build_coast_mask(g_sfc1, nσ)

    # 1D grid
    p = σ
    t = [i + j - 1 for i=1:nσ-1, j=1:2]
    e = Dict("bot"=>[1], "sfc"=>[nσ])
    g_col = Grid(Line(order=1), p, t, e)

    # convert H to FE field
    H = FEField(H_func, g_sfc2)
    H.values[g_sfc2.e["bdy"]] .= 0 # enforce H = 0 on boundary

    # H gradients are DG fields
    Hx = DGField([∂ξ(H, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    Hy = DGField([∂η(H, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

    return Geometry(H, Hx, Hy, g_sfc1, g_sfc2, in_nodes1, in_nodes2, g1, g2, g_sfc1_to_g1_map, coast_mask, g_col, σ, nσ)
end

"""
    g_sfc1_to_g1_map = build_g_sfc1_to_g1_map(g_sfc1, g1, nσ)

Returns 3-dimensional array of CartesianIndices mapping from (g_sfc1.nt × g_sfc1.nn × nσ) to (g1.nt × g1.nn).
"""
function build_g_sfc1_to_g1_map(g_sfc1, g1, nσ)
    g_sfc1_to_g1_map = Matrix{CartesianIndex{3}}(undef, g1.nt, g1.nn)
    for k ∈ 1:g_sfc1.nt, i ∈ 1:g_sfc1.nn, j ∈ 1:nσ-1
        k_w = get_k_w(k, nσ, j)
        g_sfc1_to_g1_map[k_w, i] = CartesianIndex(k, i, j)
        g_sfc1_to_g1_map[k_w, i+3] = CartesianIndex(k, i, j+1)
    end
    return g_sfc1_to_g1_map
end

"""
    coast_mask = build_coast_mask(g_sfc1, nσ)

Returns 3-dimensional array of ones and zeros in (g_sfc1.nt × g_sfc1.nn × nσ) space (zero for node on coast,
one otherwise).
"""
function build_coast_mask(g_sfc1, nσ)
    coast_mask = zeros(Bool, g_sfc1.nt, g_sfc1.nn, nσ)
    for ig ∈ g_sfc1.e["bdy"], I ∈ g_sfc1.p_to_t[ig]
        coast_mask[I, :] .= 1
    end
    return coast_mask
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
    τx_x = DGField([∂ξ(τx, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τx_y = DGField([∂η(τx, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τy_x = DGField([∂ξ(τy, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)
    τy_y = DGField([∂η(τy, g_sfc1.el.p[i, :], k) for k=1:g_sfc1.nt, i=1:g_sfc1.nn], g_sfc1)

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
    quick_plot(geom.H,  cb_label=L"H",   filename="$out_folder/images/H.png")
    quick_plot(geom.Hx, cb_label=L"H_x", filename="$out_folder/images/Hx.png")
    quick_plot(geom.Hy, cb_label=L"H_y", filename="$out_folder/images/Hy.png")
    f_over_H = FEField(x->params.f + params.β*x[2], geom.g_sfc2)/(geom.H + FEField(1e-5, geom.g_sfc2))
    quick_plot(f_over_H, cb_label=L"f/H", filename="$out_folder/images/f_over_H.png", vmax=6)
    curl = (forcing.τy_x - forcing.τx_y)*geom.H - (forcing.τy*geom.Hx - forcing.τx*geom.Hy)
    quick_plot(curl, cb_label=L"H^2 \mathbf{z} \cdot \nabla \times (\tau / H)", filename="$out_folder/images/curl.png")

    inversion = InversionComponents(params, geom, forcing)
    evolution = EvolutionComponents(params, geom, forcing, advection)

    CUDA.memory_status()

    flush(stdout)
    flush(stderr)

    return ModelSetup3D(params, geom, forcing, inversion, evolution)
end

function advection_off(m::ModelSetup3D)
    evolution_adv_off = advection_off(m.evolution)
    return ModelSetup3D(m.params, m.geom, m.forcing, m.inversion, evolution_adv_off)
end