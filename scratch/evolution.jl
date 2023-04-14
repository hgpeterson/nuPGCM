include("baroclinic.jl")
include("utils.jl")

function get_evolution_LHS(g, μ, ϱ, ε², Δt)
    J = g.J
    s = g.sfi
    LHS = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M 
        for i=1:g.nn, j=1:g.nn
            push!(LHS, (g.t[k, i], g.t[k, j], μ*ϱ*(M[i, j] + ε²*Δt/2*K[i, j])))
        end
    end
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), g.np, g.np)
    return lu(LHS)
end

function get_evolution_RHS(g, μ, ϱ, ε², Δt)
    J = g.J
    s = g.sfi
    RHS = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        JJ = J.Js[k, :, end]*J.Js[k, :, end]'
        K = J.dets[k]*sum(s.K.*JJ, dims=(1, 2))[1, 1, :, :]
        M = J.dets[k]*s.M 
        for i=1:g.nn, j=1:g.nn
            push!(RHS, (g.t[k, i], g.t[k, j], μ*ϱ*M[i, j] - ε²*Δt/2*K[i, j]))
        end
    end
    RHS = sparse((x->x[1]).(RHS), (x->x[2]).(RHS), (x->x[3]).(RHS), g.np, g.np)
    return RHS
end

function evolve()
    # params
    ε² = 1e-2
    μ = 1
    ϱ = 1e-1
    Δt = 1e-4

    # topo
    H(x) = 1 - x[1]^2 - x[2]^2
    Hx(x) = -2x[1]
    Hy(x) = -2x[2]

    # mesh
    geo = "circle"
    nref = 2
    g_sfc, g, g_cols, z_cols, p_to_tri = gen_3D_valign_mesh(geo, nref, H)

    # matrices
    LHS = get_evolution_LHS(g, μ, ϱ, ε², Δt)
    RHS = get_evolution_RHS(g, μ, ϱ, ε², Δt)

    # IC: b = z
    b = g.p[:, 3]

    # pvd file
    pvd = paraview_collection("output/b", append=true)

    # solve
    for i=0:3*360
        if mod(i, 30) == 0
            cell_type = VTKCellTypes.VTK_TETRA
            cells = [MeshCell(cell_type, g.t[i, :]) for i ∈ axes(g.t, 1)]
            vtk_grid("output/t$i", g.p', cells) do vtk
                vtk["b"] = b
                pvd[i*Δt] = vtk
            end
        end
        b = LHS\(RHS*b)
    end

    vtk_save(pvd)

    return b
end

b = evolve()

# println("Done.")