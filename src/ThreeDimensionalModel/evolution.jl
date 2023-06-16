using Printf

include("baroclinic.jl")
include("utils.jl")

struct AdvectionArrays{A <: AbstractArray}
    Ax::A
    Ay::A
    Az1::A
    Az2::A
end

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

function get_A(sf_χ, sf_b)
    w, ξ = quad_weights_points(deg=max(1, sf_χ.order + 2*sf_b.order - 2), dim=3)
    f(ξ, i, j, k, d1, d2) = ∂φ(sf_χ, k, d1, ξ)*∂φ(sf_b, j, d2, ξ)*φ(sf_b, i, ξ)
    return [nuPGCM.ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), w, ξ) for i=1:sf_b.n, j=1:sf_b.n, k=1:sf_χ.n, d1=1:3, d2=1:3]
end

function get_As(A, gχ, gb)
    J = gb.J
    Ax  = zeros(gb.nt, gb.nn, gb.nn, gχ.nn)
    Ay  = zeros(gb.nt, gb.nn, gb.nn, gχ.nn)
    Az1 = zeros(gb.nt, gb.nn, gb.nn, gχ.nn)
    Az2 = zeros(gb.nt, gb.nn, gb.nn, gχ.nn)
    for k=1:gb.nt
        # -∂z(χʸ)*∂x(b)
        Ax[k, :, :, :] = -sum(A[:, :, :, d1, d2]*J.Js[k, d1, 3]*J.Js[k, d2, 1]*J.dets[k] for d1=1:3, d2=1:3)
        # ∂z(χˣ)*∂y(b)
        Ay[k, :, :, :] = sum(A[:, :, :, d1, d2]*J.Js[k, d1, 3]*J.Js[k, d2, 2]*J.dets[k] for d1=1:3, d2=1:3)
        # [∂x(χʸ) - ∂y(χˣ)]*∂z(b)
        Az1[k, :, :, :] = sum(A[:, :, :, d1, d2]*J.Js[k, d1, 1]*J.Js[k, d2, 3]*J.dets[k] for d1=1:3, d2=1:3)
        Az2[k, :, :, :] = -sum(A[:, :, :, d1, d2]*J.Js[k, d1, 2]*J.Js[k, d2, 3]*J.dets[k] for d1=1:3, d2=1:3)
    end
    return AdvectionArrays(Ax, Ay, Az1, Az2)
end

function advection(As::AdvectionArrays, χx, χy, b, gχ, gb)
    adv = zeros(gb.np)
    for k=1:gb.nt, i=1:gb.nn
        adv[gb.t[k, i]] += sum(As.Ax[k, i, ib, iχ]*b[gb.t[k, ib]]*χy[gχ.t[k, iχ]]  for ib=1:gb.nn, iχ=1:gχ.nn) +
                           sum(As.Ay[k, i, ib, iχ]*b[gb.t[k, ib]]*χx[gχ.t[k, iχ]]  for ib=1:gb.nn, iχ=1:gχ.nn) +
                           sum(As.Az1[k, i, ib, iχ]*b[gb.t[k, ib]]*χy[gχ.t[k, iχ]] for ib=1:gb.nn, iχ=1:gχ.nn) +
                           sum(As.Az2[k, i, ib, iχ]*b[gb.t[k, ib]]*χx[gχ.t[k, iχ]] for ib=1:gb.nn, iχ=1:gχ.nn)
    end
    return adv
end

function RK2(f, u, Δt)
    return Δt*f(u + Δt/2*f(u))
end

function evolve(; b_order)
    # params
    ε² = 5e-7
    μ = 1
    ϱ = 1e-4
    Δt = 1e-3

    # topo
    H(x) = 1 - x[1]^2 - x[2]^2
    Hx(x) = -2x[1]
    Hy(x) = -2x[2]

    # mesh
    geo = "circle"
    nref = 1
    g_sfc, g, g_cols, z_cols, p_to_tri = gen_3D_valign_mesh(geo, nref, H)
    if b_order == 1
        gb = g
    elseif b_order == 2
        gb = FEGrid(2, g)
    end
    gχ = g

    # IC
    σ = 0.1
    b = @. exp((-gb.p[:, 1]^2 - gb.p[:, 2]^2 - (gb.p[:, 3] + 0.5)^2)/(2σ^2))
    χx = zeros(gχ.np) # uy = 0
    ux = 1
    χy = -ux*gχ.p[:, 3]
    # z = gχ.p[:, 3]
    # HH = [H(gχ.p[i, :]) for i=1:gχ.np]
    # ux = 1
    # χy = @. -1/3*z*(-3 + 3*ux - 6*z/HH - 4*z^2/HH^2) # ux = ux at H/2, zero at top and bot

    # CFL
    println(@sprintf("CFL Δt: %1.1e", min(1/sqrt(g_sfc.np)/ux, 1/cbrt(gb.np)/ux)))
    println(@sprintf("    Δt: %1.1e", Δt))

    # matrices
    M = get_M(gb)
    K = get_K(gb)
    LHS_diff = lu(μ*ϱ*M + ε²*Δt/2*K)
    LHS_adv = cholesky(μ*ϱ*M)
    A = get_A(gχ.sf, gb.sf)
    As = get_As(A, gχ, gb)

    # pvd file
    rm("output/b.pvd", force=true)
    rm("output/b_at_t*.vtu", force=true)
    pvd = paraview_collection("output/b", append=true)

    # solve
    @showprogress "Evolving..." for i=0:300
        if mod(i, 10) == 0
            if b_order == 1
                cell_type = VTKCellTypes.VTK_TETRA
            elseif b_order == 2
                cell_type = VTKCellTypes.VTK_QUADRATIC_TETRA
            end
            cells = [MeshCell(cell_type, gb.t[i, :]) for i ∈ axes(gb.t, 1)]
            vtk_grid("output/b_at_t$i", gb.p', cells) do vtk
                vtk["b"] = b
                vtk["ba"] = @. exp((-(gb.p[:, 1] - ux*i*Δt)^2 - gb.p[:, 2]^2 - (gb.p[:, 3] + 0.5)^2)/(2σ^2))
                pvd[i*Δt] = vtk
            end
        end

        # operator split rhs
        RHS_adv1 = μ*ϱ*M*b - μ*ϱ*RK2(u -> advection(As, χx, χy, u, gχ, gb), b, Δt/2)
        b1 = LHS_adv\RHS_adv1
        RHS_diff = μ*ϱ*M*b1 - Δt*ε²/2*K*b1
        b2 = LHS_diff\RHS_diff
        RHS_adv2 = μ*ϱ*M*b2 - μ*ϱ*RK2(u -> advection(As, χx, χy, u, gχ, gb), b2, Δt/2)
        b = LHS_adv\RHS_adv2

        if any(isnan.(b))
            error("Solution blew up 😢")
        end
    end

    vtk_save(pvd)
    println("output/b.pvd")

    return b
end

bf = evolve(b_order=2)

println("Done.")