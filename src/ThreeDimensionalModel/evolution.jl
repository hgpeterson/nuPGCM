struct AdvectionArrays{A<:AbstractArray}
    Aξ::A
    Aη::A
    Aσξ::A
    Aση::A
end

function get_M(g::Grid)
    J = g.J
    el = g.el
    M_el = mass_matrix(el)
    M = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        Mᵏ = J.dets[k]*M_el 
        for i=1:el.n, j=1:el.n
            push!(M, (g.t[k, i], g.t[k, j], Mᵏ[i, j]))
        end
    end
    return dropzeros!(sparse((x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), g.np, g.np))
end

function get_K(g::Grid)
    J = g.J
    el = g.el
    K_el = stiffness_matrix(el)
    K = Tuple{Int64, Int64, Float64}[]
    for k=1:g.nt 
        Kᵏ = K_el*J.Js[k, 1, 1]^2*g.J.dets[k]
        for i=1:el.n, j=1:el.n
            push!(K, (g.t[k, i], g.t[k, j], -Kᵏ[i, j]))
        end
    end
    return dropzeros!(sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np))
end

function AdvectionArrays(g1::Grid, g2::Grid, H, nσ)
    # unpack
    g_sfc2 = H.g
    J = g1.J
    el1 = g1.el
    el2 = g2.el

    # allocate
    Aξ  = zeros(g1.nt, el2.n, el2.n, el1.n)
    Aη  = zeros(g1.nt, el2.n, el2.n, el1.n)
    Aσξ = zeros(g1.nt, el2.n, el2.n, el1.n)
    Aση = zeros(g1.nt, el2.n, el2.n, el1.n)
    @showprogress "Making advection arrays" for k_w=1:g1.nt
        # unpack
        jac = J.Js[k_w, :, :]
        Δ = J.dets[k_w]

        # surface tri
        k_sfc = div(k_w-1, nσ-1) + 1

        # general integral
        A_from_ref = transformation_matrix(g_sfc2.el, g_sfc2.p[g_sfc2.t[k_sfc, 1:3], :])
        b_from_ref = transformation_vector(g_sfc2.el, g_sfc2.p[g_sfc2.t[k_sfc, 1:3], :])
        x(ξ) = A_from_ref*ξ[1:2] + b_from_ref
        f(ξ, i, j, k, d1, d2) = ∂φ(el1, ξ, k, d1)*∂φ(el2, ξ, j, d2)*φ(el2, ξ, i)/H(x(ξ), k_sfc)
        A = [ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), el2) for i=1:el2.n, j=1:el2.n, k=1:el1.n, d1=1:3, d2=1:3]

        # -∂σ(χη)*∂ξ(b)/H
        Aξ[k_w, :, :, :] = -sum(A[:, :, :, d1, d2]*jac[d1, 3]*jac[d2, 1]*Δ for d1=1:3, d2=1:3)

        # ∂σ(χξ)*∂η(b)/H
        Aη[k_w, :, :, :] = sum(A[:, :, :, d1, d2]*jac[d1, 3]*jac[d2, 2]*Δ for d1=1:3, d2=1:3)

        # [∂ξ(χη) - ∂η(χξ)]*∂σ(b)/H
        Aσξ[k_w, :, :, :] = sum(A[:, :, :, d1, d2]*jac[d1, 1]*jac[d2, 3]*Δ for d1=1:3, d2=1:3)
        Aση[k_w, :, :, :] = -sum(A[:, :, :, d1, d2]*jac[d1, 2]*jac[d2, 3]*Δ for d1=1:3, d2=1:3)
    end
    return AdvectionArrays(Aξ, Aη, Aσξ, Aση)
end

function advection(As::AdvectionArrays, χξ, χη, b, g1::Grid, g2::Grid)
    adv = zeros(g2.np)
    for k=1:g2.nt, i=1:g2.nn
        adv[g2.t[k, i]] += sum(As.Aξ[k, i, ib, iχ]*b[g2.t[k, ib]]*χη[g1.t[k, iχ]]  for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(As.Aη[k, i, ib, iχ]*b[g2.t[k, ib]]*χξ[g1.t[k, iχ]]  for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(As.Aσξ[k, i, ib, iχ]*b[g2.t[k, ib]]*χη[g1.t[k, iχ]] for ib=1:g2.nn, iχ=1:g1.nn) +
                           sum(As.Aση[k, i, ib, iχ]*b[g2.t[k, ib]]*χξ[g1.t[k, iχ]] for ib=1:g2.nn, iχ=1:g1.nn)
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
    g_col = m.g_col
    in_nodes2 = m.in_nodes2

    T = 1e-2*μ*ϱ/ε²
    n_steps = 100
    Δt = T/n_steps

    # # advection matrices
    # M = get_M(g2)
    # LHS_adv = cholesky(μ*ϱ*M)
    # As = AdvectionArrays(g1, g2, H, nσ)

    # diffusion matrices
    M_col = get_M(g_col)
    K_col = get_K(g_col)
    LHS_diffs = [lu(μ*ϱ*M_col - ε²/H[i]^2*Δt/2*K_col) for i ∈ in_nodes2]
    RHS_diffs = [μ*ϱ*M_col + ε²/H[i]^2*Δt/2*K_col for i ∈ in_nodes2]

    # pvd file
    rm("$out_folder/state.pvd", force=true)
    rm("$out_folder/state*.vtu", force=true)
    pvd = paraview_collection("$out_folder/state", append=true)

    # solve
    for i=1:n_steps
        if mod(i, 10) == 0
            # update state
            invert!(m, s, showplots=true)
            # get_u(m, s, showplots=true)

            # save state
            cells = [MeshCell(VTKCellTypes.VTK_WEDGE, g1.t[i, :]) for i ∈ axes(g1.t, 1)]
            vtk_grid("$out_folder/state$i", g1.p', cells) do vtk
                vtk["b"] = s.b.values[1:g1.np]
                # vtk["omega^x"] = s.ωx.values
                # vtk["omega^y"] = s.ωy.values
                # vtk["chi^x"] = s.χx.values
                # vtk["chi^y"] = s.χy.values
                pvd[i*Δt] = vtk
            end
            println("$out_folder/state$i.vtu")

            # CFL
            # println(@sprintf("CFL Δt: %1.1e", min(1/sqrt(g_sfc.np)/ux, 1/cbrt(gb.np)/ux)))
            # println(@sprintf("    Δt: %1.1e", Δt))
        end

        # # operator split rhs
        # ωx, ωy, χx, χy, Ψ = invert(m, s.b)
        # b = merge_cols(s.b, gb, b_cols, pmap)
        # RHS_adv = μ*ϱ*M*b - μ*ϱ*Δt/2*advection(As, χx, χy, b, g, gb)
        # cg!(b, LHS_adv, RHS_adv)
        # RHS_diff = μ*ϱ*M*b + Δt*ε²/2*K*b
        # minres!(b, LHS_diff, RHS_diff)
        # b_split = split_cols(b, b_cols, pmap)
        # ωx, ωy, χx, χy, Ψ = invert(m, b_split)
        # RHS_adv = μ*ϱ*M*b - μ*ϱ*Δt/2*advection(As, χx, χy, b, g, gb)
        # cg!(b, LHS_adv, RHS_adv)
        # s.b[:] = split_cols(b, b_cols, pmap)

        # # operator split rhs
        # ωx, ωy, χx, χy, Ψ = invert(m, s.b)
        # b = merge_cols(s.b, gb, b_cols, pmap)
        # RHS_adv = μ*ϱ*M*b - μ*ϱ*Δt/2*advection(As, χx, χy, b, g, gb)
        # b = LHS_adv\RHS_adv
        # b = LHS_diff\(RHS_diff*b)
        # b_split = split_cols(b, b_cols, pmap)
        # ωx, ωy, χx, χy, Ψ = invert(m, b_split)
        # RHS_adv = μ*ϱ*M*b - μ*ϱ*Δt/2*advection(As, χx, χy, b, g, gb)
        # b = LHS_adv\RHS_adv
        # s.b[:] = split_cols(b, b_cols, pmap)

        # just diffusion
        for i ∈ eachindex(in_nodes2)
            ig = in_nodes2[i]
            inds = (ig-1)*nσ+1:(ig-1)*nσ+nσ
            s.b.values[inds] = LHS_diffs[i]\(RHS_diffs[i]*s.b.values[inds])
        end

        # # analytical solution
        # # ba = [b_a(gb.p[j, 3], i*Δt, ε²/μ/ϱ, 1 - gb.p[j, 1]^2 - gb.p[j, 2]^2) for j=1:gb.np]
        # ba = [FEField([b_a(g.p[j, 3], i*Δt, ε²/μ/ϱ, 1 - g.p[j, 1]^2 - g.p[j, 2]^2) for j=1:g.np], g) for g ∈ b_cols]
        # errs = FVField([maximum(abs(s.b[j] - ba[j])) for j ∈ eachindex(b_cols)], m.g_sfc)
        # # b = ba
        # # s.b[:] = split_cols(ba, b_cols, pmap)
        # # println(@sprintf("Max Error: %1.1e", maximum(abs.(b - ba))))
        # println(@sprintf("Max Error: %1.1e", maximum(errs)))
        # quick_plot(errs, "Error", "$out_folder/error.png")

        if any(isnan.(s.b.values))
            error("Solution blew up 😢")
        end
    end

    vtk_save(pvd)
    println("$out_folder/state.pvd")

    # # save b
    # cell_type = VTKCellTypes.VTK_QUADRATIC_TETRA
    # cells = [MeshCell(cell_type, gb.t[i, :]) for i ∈ axes(gb.t, 1)]
    # vtk_grid("$out_folder/b", gb.p', cells) do vtk
    #     vtk["b"] = b
    #     ba = [b_a(gb.p[i, 3], n_steps*Δt, ε²/μ/ϱ, 1 - gb.p[i, 1]^2 - gb.p[i, 2]^2) for i=1:gb.np]
    #     vtk["ba"] = ba
    #     vtk["error"] = abs.(b - ba)
    # end
    # println("$out_folder/b.vtu")

    # # omega_b's
    # ba = [b_a(gb.p[j, 3], n_steps*Δt, ε²/μ/ϱ, 1 - gb.p[j, 1]^2 - gb.p[j, 2]^2) for j=1:gb.np]
    # ba = split_cols(ba, b_cols, pmap)
    # ωx_b, ωy_b, χx_b, χy_b = get_buoyancy_ω_and_χ(m, ba, showplots=true)
    # ωx_b_bot_a = DGField([ωx_b[k, i][1] for k=1:m.g_sfc.nt, i=1:m.g_sfc.nn], m.g_sfc)
    # ωy_b_bot_a = DGField([ωy_b[k, i][1] for k=1:m.g_sfc.nt, i=1:m.g_sfc.nn], m.g_sfc)
    # quick_plot(ωx_b_bot_a, "analytical", "$out_folder/omegax_b_bot_a.png")
    # quick_plot(ωy_b_bot_a, "analytical", "$out_folder/omegay_b_bot_a.png")
    # ωx_b, ωy_b, χx_b, χy_b = get_buoyancy_ω_and_χ(m, s.b, showplots=true)
    # ωx_b_bot = DGField([ωx_b[k, i][1] for k=1:m.g_sfc.nt, i=1:m.g_sfc.nn], m.g_sfc)
    # ωy_b_bot = DGField([ωy_b[k, i][1] for k=1:m.g_sfc.nt, i=1:m.g_sfc.nn], m.g_sfc)
    # quick_plot(ωx_b_bot, "numerical", "$out_folder/omegax_b_bot_n.png")
    # quick_plot(ωy_b_bot, "numerical", "$out_folder/omegay_b_bot_n.png")
    # quick_plot(abs(ωx_b_bot - ωx_b_bot_a), "Error", "$out_folder/omegax_b_bot_err.png")
    # quick_plot(abs(ωy_b_bot - ωy_b_bot_a), "Error", "$out_folder/omegay_b_bot_err.png")

    return s
end

"""
Analytical solution to ∂t(b) = α ∂zz(b) with ∂z(b) = 0 at z = -H, 0
(truncated to Nth term in Fourier series).
"""
function b_a(z, t, α, H; N=50)
    if H == 0
        return 0
    end
    # A(n) = 2*H*(1 + (-1)^(n+1))/(n^2*π^2)
    # return -H/2 + sum(A(n)*cos(n*π*z/H)*exp(-α*(n*π/H)^2*t) for n=1:2:N)
    A(n) = 8*H^3*(-1 + (-1)^n)/(n^4*π^4)
    return H^3/6 + sum(A(n)*cos(n*π*z/H)*exp(-α*(n*π/H)^2*t) for n=1:2:N)
end