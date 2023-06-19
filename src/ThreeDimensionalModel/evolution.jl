struct AdvectionArrays{A<:AbstractArray}
    Ax::A
    Ay::A
    Az1::A
    Az2::A
end

function get_M(g::Grid)
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

function get_K(g::Grid)
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

function get_A(sf_χ::ShapeFunctions, sf_b::ShapeFunctions)
    w, ξ = quad_weights_points(deg=max(1, sf_χ.order + 2*sf_b.order - 2), dim=3)
    f(ξ, i, j, k, d1, d2) = ∂φ(sf_χ, k, d1, ξ)*∂φ(sf_b, j, d2, ξ)*φ(sf_b, i, ξ)
    return [ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), w, ξ) for i=1:sf_b.n, j=1:sf_b.n, k=1:sf_χ.n, d1=1:3, d2=1:3]
end

function AdvectionArrays(A, gχ::Grid, gb::Grid)
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

function advection(As::AdvectionArrays, χx::FEField, χy::FEField, b::FEField, gχ::Grid, gb::Grid)
    adv = zeros(gb.np)
    for k=1:gb.nt, i=1:gb.nn
        adv[gb.t[k, i]] += sum(As.Ax[k, i, ib, iχ]*b[gb.t[k, ib]]*χy[gχ.t[k, iχ]]  for ib=1:gb.nn, iχ=1:gχ.nn) +
                           sum(As.Ay[k, i, ib, iχ]*b[gb.t[k, ib]]*χx[gχ.t[k, iχ]]  for ib=1:gb.nn, iχ=1:gχ.nn) +
                           sum(As.Az1[k, i, ib, iχ]*b[gb.t[k, ib]]*χy[gχ.t[k, iχ]] for ib=1:gb.nn, iχ=1:gχ.nn) +
                           sum(As.Az2[k, i, ib, iχ]*b[gb.t[k, ib]]*χx[gχ.t[k, iχ]] for ib=1:gb.nn, iχ=1:gχ.nn)
    end
    return adv
end

function evolve(m::ModelSetup3D, s::ModelState3D)
    # unpack
    μ = m.μ
    ϱ = m.ϱ
    ε² = m.ε²
    Δt = m.Δt
    g = m.g 

    # second order grid for b
    gb = Grid(2, g)

    # matrices
    M = get_M(gb)
    K = get_K(gb)
    LHS_diff = lu(μ*ϱ*M + ε²*Δt*K)
    LHS_adv = cholesky(μ*ϱ*M)
    A = get_A(g.sf, gb.sf)
    As = AdvectionArrays(A, g, gb)

    # pvd file
    rm("$out_folder/b.pvd", force=true)
    rm("$out_folder/b_at_t*.vtu", force=true)
    pvd = paraview_collection("$out_folder/b", append=true)

    # solve
    @showprogress "Evolving..." for i=0:300
        if mod(i, 10) == 0
            cell_type = VTKCellTypes.VTK_QUADRATIC_TETRA
            cells = [MeshCell(cell_type, gb.t[i, :]) for i ∈ axes(gb.t, 1)]
            vtk_grid("$out_folder/b_at_t$i", gb.p', cells) do vtk
                vtk["b"] = b
                pvd[i*Δt] = vtk
            end

            # CFL
            # println(@sprintf("CFL Δt: %1.1e", min(1/sqrt(g_sfc.np)/ux, 1/cbrt(gb.np)/ux)))
            # println(@sprintf("    Δt: %1.1e", Δt))
        end

        # operator split rhs
        ωx, ωy, χx, χy, Ψ = invert(m, s.b)
        RHS_adv1 = μ*ϱ*M*s.b - μ*ϱ*Δt/2*advection(As, χx, χy, s.b, g, gb)
        b1 = LHS_adv\RHS_adv1
        RHS_diff = μ*ϱ*M*b1 - Δt*ε²/2*K*b1
        b2 = LHS_diff\RHS_diff
        ωx, ωy, χx, χy, Ψ = invert(m, b2)
        RHS_adv2 = μ*ϱ*M*b2 - μ*ϱ*Δt/2*advection(As, χx, χy, b2, g, gb)
        s.b.values[:] = LHS_adv\RHS_adv2

        if any(isnan.(s.b.values))
            error("Solution blew up 😢")
        end
    end

    vtk_save(pvd)
    println("$out_folder/b.pvd")

    return s
end