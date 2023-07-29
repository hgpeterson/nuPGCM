struct AdvectionArrays{A<:AbstractArray}
    Aξ::A
    Aη::A
    Aσξ::A
    Aση::A
end

function AdvectionArrays(m)
    # unpack
    g_sfc2 = m.g_sfc2
    tri2 = g_sfc2.el
    H = m.H
    g1 = m.g1
    g2 = m.g2
    J = g1.J
    el1 = g1.el
    el2 = g2.el
    nσ = m.nσ

    f(ξ, i, j, k, d1, d2) = ∂φ(el1, ξ, k, d1)*∂φ(el2, ξ, j, d2)*φ(el2, ξ, i)
    A = [ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), el1) for i=1:el2.n, j=1:el2.n, k=1:el1.n, d1=1:3, d2=1:3]

    # allocate
    Aξ  = zeros(g1.nt, el2.n, el2.n, el1.n)
    Aη  = zeros(g1.nt, el2.n, el2.n, el1.n)
    Aσξ = zeros(g1.nt, el2.n, el2.n, el1.n)
    Aση = zeros(g1.nt, el2.n, el2.n, el1.n)

    @showprogress "Computing advection arrays..." for k_sfc=1:g_sfc2.nt
        # # integrate ∂φₖ*∂φⱼ*φᵢ/H
        # x(ξ) = transform_from_ref_el(tri2, ξ[1:2], g_sfc2.p[g_sfc2.t[k_sfc, 1:3], :])
        # # f(ξ, i, j, k, d1, d2) = ∂φ(el1, ξ, k, d1)*∂φ(el2, ξ, j, d2)*φ(el2, ξ, i)/H(x(ξ), k_sfc)
        # f(ξ, i, j, k, d1, d2) = ∂φ(el1, ξ, k, d1)*∂φ(el2, ξ, j, d2)*φ(el2, ξ, i)/(1 - norm(x(ξ))^2)*g_sfc2.J.dets[k_sfc]
        # A = [ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), el1) for i=1:el2.n, j=1:el2.n, k=1:el1.n, d1=1:3, d2=1:3]
        # x_pts = el1.quad_pts[:, 1:2]
        # p_sfc = g_sfc2.p[g_sfc2.t[k_sfc, 1:3], :]
        # for i ∈ axes(x_pts, 1)
        #     x_pts[i, :] = transform_from_ref_el(tri2, x_pts[i, :], p_sfc)
        # end
        # wts = el1.quad_wts
        # pts = el1.quad_pts
        # # A = [sum(wts[l]*∂φ(el1, pts[l, :], k, d1)*∂φ(el2, pts[l, :], j, d2)*φ(el2, pts[l, :], i)/H(x_pts[l, :], k_sfc)*g_sfc2.J.dets[k_sfc] for l ∈ eachindex(wts)) for i=1:el2.n, j=1:el2.n, k=1:el1.n, d1=1:3, d2=1:3]
        # A = [sum(wts[l]*∂φ(el1, pts[l, :], k, d1)*∂φ(el2, pts[l, :], j, d2)*φ(el2, pts[l, :], i)/(1 - norm(x_pts[l, :])^2) for l ∈ eachindex(wts)) for i=1:el2.n, j=1:el2.n, k=1:el1.n, d1=1:3, d2=1:3]
        H̄ = sum(H[g_sfc2.t[k_sfc, :]])/g_sfc2.nn
        for k_w=(nσ-1)*(k_sfc-1)+1:(nσ-1)*(k_sfc-1)+nσ-1
            # unpack
            jac = J.Js[k_w, :, :]
            Δ = J.dets[k_w]

            # -∂σ(χη)*∂ξ(b)/H
            Aξ[k_w, :, :, :] = -sum(A[:, :, :, d1, d2]*jac[d1, 3]*jac[d2, 1]*Δ for d1=1:3, d2=1:3)/H̄

            # ∂σ(χξ)*∂η(b)/H
            Aη[k_w, :, :, :] = sum(A[:, :, :, d1, d2]*jac[d1, 3]*jac[d2, 2]*Δ for d1=1:3, d2=1:3)/H̄

            # [∂ξ(χη) - ∂η(χξ)]*∂σ(b)/H
            Aσξ[k_w, :, :, :] = sum(A[:, :, :, d1, d2]*jac[d1, 1]*jac[d2, 3]*Δ for d1=1:3, d2=1:3)/H̄
            Aση[k_w, :, :, :] = -sum(A[:, :, :, d1, d2]*jac[d1, 2]*jac[d2, 3]*Δ for d1=1:3, d2=1:3)/H̄
        end
    end
    return AdvectionArrays(Aξ, Aη, Aσξ, Aση)
end

function advection(As::AdvectionArrays, χξ::DGField, χη::DGField, b::FEField)
    g1 = χξ.g
    g2 = b.g
    adv = zeros(g2.np)
    for k=1:g2.nt, i=1:g2.nn
        adv[g2.t[k, i]] += sum(As.Aξ[k, i, ib, iχ]*b[g2.t[k, ib]]*χη[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(As.Aη[k, i, ib, iχ]*b[g2.t[k, ib]]*χξ[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(As.Aσξ[k, i, ib, iχ]*b[g2.t[k, ib]]*χη[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(As.Aση[k, i, ib, iχ]*b[g2.t[k, ib]]*χξ[k, iχ] for ib=1:g2.nn, iχ=1:g1.nn)
    end
    return adv
end

function evolve!(m::ModelSetup3D, s::ModelState3D)
    # unpack
    μ = m.μ
    ϱ = m.ϱ
    ε² = m.ε²
    Δt = m.Δt
    g1 = m.g1
    g2 = m.g2
    nσ = m.nσ
    H = m.H
    g_sfc1 = m.g_sfc1
    g_col = m.g_col
    in_nodes2 = m.in_nodes2

    # integration time
    T = 5e-2*μ*ϱ/ε²
    n_steps = 50
    Δt = T/n_steps
    # Δt = 4e-3
    # n_steps = 100

    # advection matrices
    M = mass_matrix(g2)
    LHS_adv = cholesky(μ*ϱ*M)
    As = AdvectionArrays(m)
    # # constant vel. (uξ = 1)
    # s.χx.values[:] .= 0.0
    # s.χy.values[:] = @. -g1.p[g1.t, 3]*(1 - g1.p[g1.t, 1]^2 - g1.p[g1.t, 2]^2)
    # # turn down diffusion
    # ε² /= 1e4
    # println(@sprintf("CFL Δt: %1.1e", min(1/sqrt(g_sfc1.np), 1/cbrt(g2.np))))
    # println(@sprintf("    Δt: %1.1e", Δt))

    # diffusion matrices
    M_col = mass_matrix(g_col)
    K_col = stiffness_matrix_zz(g_col)
    LHS_diffs = [lu(μ*ϱ*M_col + ε²/H[i]^2*Δt/2*K_col) for i ∈ in_nodes2]
    RHS_diffs = [μ*ϱ*M_col - ε²/H[i]^2*Δt/2*K_col for i ∈ in_nodes2]

    # pvd file
    rm("$out_folder/state.pvd", force=true)
    # rm("$out_folder/state*.vtu", force=true) # * doesn't work?
    pvd = paraview_collection("$out_folder/state", append=true)

    # for plotting
    pz = copy(g1.p)
    for i=1:g1.np
        pz[i, 3] *= 1 - pz[i, 1]^2 - pz[i, 2]^2
    end

    # solve
    for i=1:n_steps
        println(i)
        if mod(i-1, 10) == 0 || i == n_steps
            # # diffusion solution
            # ba = [b_a(g2.p[k, 3], i*Δt, ε²/μ/ϱ/(1-g2.p[k, 1]^2-g2.p[k, 2]^2)^2, 1-g2.p[k, 1]^2-g2.p[k, 2]^2) for k=1:g2.np]
            # println(@sprintf("Max Error: %1.1e", maximum(abs.(s.b.values - ba))))

            # update state
            invert!(m, s, showplots=true)

            # save state
            cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)]
            vtk_grid("$out_folder/state$i", pz', cells) do vtk
                vtk["b"] = s.b.values[1:g1.np]
                # vtk["ba"] = ba[1:g1.np]
                # vtk["err"] = abs.(s.b.values[1:g1.np] - ba[1:g1.np])
                pvd[(i-1)*Δt] = vtk
            end
            println("$out_folder/state$i.vtu")

            # integrate b
            println(@sprintf("∫b = %1.5e", sum(M*s.b.values)))
        end

        # Δt/2 advection step
        invert!(m, s)
        RHS_adv = μ*ϱ*M*s.b.values - μ*ϱ*Δt/2*advection(As, s.χx, s.χy, s.b)
        s.b.values[:] = LHS_adv\RHS_adv

        # Δt diffusion step
        for j ∈ eachindex(in_nodes2)
            ig = in_nodes2[j]
            inds = (ig-1)*nσ+1:(ig-1)*nσ+nσ
            s.b.values[inds] = LHS_diffs[j]\(RHS_diffs[j]*s.b.values[inds])
        end

        # Δt/2 advection step
        invert!(m, s)
        RHS_adv = μ*ϱ*M*s.b.values - μ*ϱ*Δt/2*advection(As, s.χx, s.χy, s.b)
        s.b.values[:] = LHS_adv\RHS_adv

        if any(isnan.(s.b.values))
            error("Solution blew up 😢")
        end
    end

    vtk_save(pvd)
    println("$out_folder/state.pvd")

    return s
end

"""
    b = b_a(σ, t, α, H; N)

Analytical solution to ∂t(b) = α ∂σσ(b) with ∂σ(b) = 0 at σ = -1, 0
and b(σ, 0) = H*σ (truncated to Nth term in Fourier series).
"""
function b_a(σ, t, α, H; N=1000)
    # b0 = H*σ
    A(n) = 2*H*(1 + (-1)^(n+1))/(n^2*π^2)
    return -H/2 + sum(A(n)*cos(n*π*σ)*exp(-α*(n*π)^2*t) for n=1:2:N)

    # # b0 = H^3*(σ^2 + 2/3*σ^3), nuemann
    # A(n) = 8*H^3*(-1 + (-1)^n)/(n^4*π^4)
    # return H^3/6 + sum(A(n)*cos(n*π*z/H)*exp(-α*(n*π/H)^2*t) for n=1:2:N)
end