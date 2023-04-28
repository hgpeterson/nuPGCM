include("baroclinic.jl")
include("utils.jl")

function get_M(g)
    J = g.J
    s = g.sfi
    M = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        Mᵏ = J.dets[k]*s.M 
        for i=1:g.nn, j=1:g.nn
            push!(M, (g.t[k, i], g.t[k, j], Mᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), g.np, g.np)
end

function get_K(g)
    J = g.J
    s = g.sfi
    K = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        Kᵏ = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        for i=1:g.nn, j=1:g.nn
            push!(K, (g.t[k, i], g.t[k, j], Kᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np)
end

function compute_A(sf_χ, sf_b)
    w, ξ = quad_weights_points(deg=max(1, sf_χ.order + 2*sf_b.order - 2), dim=3)
    f(ξ, i, j, k, d1, d2) = ∂φ(sf_χ, k, d1, ξ)*∂φ(sf_b, j, d2, ξ)*φ(sf_b, i, ξ)
    return [nuPGCM.ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), w, ξ) for i=1:sf_b.n, j=1:sf_b.n, k=1:sf_χ.n, d1=1:3, d2=1:3]
end

# function compute_As(A, χx, b)
#     gb = b.g
#     gχ = χx.g
#     J = gb.J
#     tb = gb.t
#     Ax = zeros(gb.np, gb.np, gχ.np)
#     Ay = zeros(gb.np, gb.np, gχ.np)
#     Az1 = zeros(gb.np, gb.np, gχ.np)
#     Az2 = zeros(gb.np, gb.np, gχ.np)
#     for k=1:gb.nt
#         # -∂z(χʸ)*∂x(b)
#         Ax[tb[k, :], :, :] -= sum(A[:, :, :, d1, d2]*J.Js[k, d1, 3]*J.Js[k, d2, 1]*J.dets[k] for d1=1:3, d2=1:3)
#         # ∂z(χˣ)*∂y(b)
#         Ay[tb[k, :], :, :] += sum(A[:, :, :, d1, d2]*J.Js[k, d1, 3]*J.Js[k, d2, 2]*J.dets[k] for d1=1:3, d2=1:3)
#         # [∂x(χʸ) - ∂y(χˣ)]*∂z(b)
#         Az1[tb[k, :], :, :] += sum(A[:, :, :, d1, d2]*J.Js[k, d1, 1]*J.Js[k, d2, 3]*J.dets[k] for d1=1:3, d2=1:3)
#         Az2[tb[k, :], :, :] -= sum(A[:, :, :, d1, d2]*J.Js[k, d1, 2]*J.Js[k, d2, 3]*J.dets[k] for d1=1:3, d2=1:3)
#     end
#     return Ax, Ay, Az1, Az2
# end

# function advection(Ax, Ay, Az1, Az2, χx, χy, b)
#     gb = b.g
#     gχ = χx.g
#     J = gb.J
#     tb = gb.t
#     tχ = gχ.t
#     adv = zeros(gb.np)
#     for k=1:gb.nt
#         # -∂z(χʸ)*∂x(b)
#         adv[tb[k, :]] += sum(Ax[:, i, j]*b[tb[k, i]]*χy[tχ[k, j]] for i=1:gb.nn, j=1:gχ.nn) +
#                          sum(Ay[:, i, j]*b[tb[k, i]]*χx[tχ[k, j]] for i=1:gb.nn, j=1:gχ.nn) +
#                          sum(Az1[:, i, j]*b[tb[k, i]]*χy[tχ[k, j]] for i=1:gb.nn, j=1:gχ.nn) +
#                          sum(Az2[:, i, j]*b[tb[k, i]]*χx[tχ[k, j]] for i=1:gb.nn, j=1:gχ.nn)
#     end
#     return adv
# end

function advection(A, χx, χy, b)
    gb = b.g
    gχ = χx.g
    J = gb.J
    tb = gb.t
    tχ = gχ.t
    Ax = zeros(gb.np)
    Ay = zeros(gb.np)
    Az = zeros(gb.np)
    for k=1:gb.nt
        # -∂z(χʸ)*∂x(b)
        Ax[tb[k, :]] -= sum(A[:, i, j, d1, d2]*b[tb[k, i]]*χy[tχ[k, j]]*J.Js[k, d1, 3]*J.Js[k, d2, 1]*J.dets[k] for i=1:gb.nn, j=1:gχ.nn, d1=1:3, d2=1:3)
        # ∂z(χˣ)*∂y(b)
        Ay[tb[k, :]] += sum(A[:, i, j, d1, d2]*b[tb[k, i]]*χx[tχ[k, j]]*J.Js[k, d1, 3]*J.Js[k, d2, 2]*J.dets[k] for i=1:gb.nn, j=1:gχ.nn, d1=1:3, d2=1:3)
        # [∂x(χʸ) - ∂y(χˣ)]*∂z(b)
        Az[tb[k, :]] += sum(A[:, i, j, d1, d2]*b[tb[k, i]]*χy[tχ[k, j]]*J.Js[k, d1, 1]*J.Js[k, d2, 3]*J.dets[k] -
                            A[:, i, j, d1, d2]*b[tb[k, i]]*χx[tχ[k, j]]*J.Js[k, d1, 2]*J.Js[k, d2, 3]*J.dets[k] for i=1:gb.nn, j=1:gχ.nn, d1=1:3, d2=1:3)
    end
    return Ax + Ay + Az
end

function evolve(; b_order)
    # params
    ε² = 1e-5
    μ = 1
    ϱ = 1e-4
    Δt = 1e-4

    # topo
    H(x) = 1 - x[1]^2 - x[2]^2
    Hx(x) = -2x[1]
    Hy(x) = -2x[2]

    # mesh
    geo = "circle"
    nref = 2
    g_sfc, g, g_cols, z_cols, p_to_tri = gen_3D_valign_mesh(geo, nref, H)
    if b_order == 1
        gb = g
    elseif b_order == 2
        gb = FEGrid(2, g)
    end
    gχ = g

    # matrices
    M = get_M(g)
    K = get_K(g)
    LHS = lu(μ*ϱ*M + ε²*Δt/2*K)
    A = compute_A(gχ.sf, gb.sf)

    # IC
    σ = 0.1
    b = @. exp((-gb.p[:, 1]^2 - gb.p[:, 2]^2 - (gb.p[:, 3] + 0.5)^2)/(2σ^2))
    χx = zeros(gχ.np) # uy = 0
    χy = -gχ.p[:, 3] # ux = 1

    # FEFields
    b = FEField(b, gb)
    χx = FEField(χx, gχ)
    χy = FEField(χy, gχ)

    # pvd file
    pvd = paraview_collection("output/b", append=true)

    # solve
    @showprogress "Evolving..." for i=0:3*360
        if mod(i, 10) == 0
            if b_order == 1
                cell_type = VTKCellTypes.VTK_TETRA
            elseif b_order == 2
                cell_type = VTKCellTypes.VTK_QUADRATIC_TETRA
            end
            cells = [MeshCell(cell_type, gb.t[i, :]) for i ∈ axes(gb.t, 1)]
            vtk_grid("output/t$i", gb.p', cells) do vtk
                vtk["b"] = b.values
                pvd[i*Δt] = vtk
            end
        end

        b_prev = FEField(b.values, gb)
        χx_prev = FEField(χx.values, gχ)
        χy_prev = FEField(χy.values, gχ)

        if i == 1
            # first step: CNAB1
            RHS = μ*ϱ*M*b.values - Δt*(advection(A, χx, χy, b) + ε²/2*K*b.values)
        else
            # other steps: CNAB2
            RHS = μ*ϱ*M*b.values - Δt*(3/2*advection(A, χx, χy, b) - 1/2*advection(A, χx_prev, χy_prev, b_prev) + ε²/2*K*b.values)
        end

        b.values[:] = LHS\RHS
    end

    vtk_save(pvd)
    println("output/b.pvd")

    return b
end

bf = evolve(b_order=1)

println("Done.")