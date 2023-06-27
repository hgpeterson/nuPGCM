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
            push!(K, (g.t[k, i], g.t[k, j], -Kᵏ[i, j]))
        end
    end
    return sparse((x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), g.np, g.np)
end

function get_A(sf_χ::ShapeFunctions, sf_b::ShapeFunctions)
    w, ξ = quad_weights_points(deg=max(1, sf_χ.order + 2*sf_b.order - 2), dim=3)
    f(ξ, i, j, k, d1, d2) = ∂φ(sf_χ, k, d1, ξ)*∂φ(sf_b, j, d2, ξ)*φ(sf_b, i, ξ)
    return [ref_el_quad(ξ -> f(ξ, i, j, k, d1, d2), w, ξ) for i=1:sf_b.n, j=1:sf_b.n, k=1:sf_χ.n, d1=1:3, d2=1:3]
end

function AdvectionArrays(A, g::Grid, gb::Grid)
    J = g.J
    Ax  = zeros(g.nt, gb.nn, gb.nn, g.nn)
    Ay  = zeros(g.nt, gb.nn, gb.nn, g.nn)
    Az1 = zeros(g.nt, gb.nn, gb.nn, g.nn)
    Az2 = zeros(g.nt, gb.nn, gb.nn, g.nn)
    for k=1:g.nt
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

function advection(As::AdvectionArrays, χx, χy, b, g::Grid, gb::Grid)
    adv = zeros(gb.np)
    for k=1:gb.nt, i=1:gb.nn
        adv[gb.t[k, i]] += sum(As.Ax[k, i, ib, iχ]*b[gb.t[k, ib]]*χy[g.t[k, iχ]]  for ib=1:gb.nn, iχ=1:g.nn) +
                           sum(As.Ay[k, i, ib, iχ]*b[gb.t[k, ib]]*χx[g.t[k, iχ]]  for ib=1:gb.nn, iχ=1:g.nn) +
                           sum(As.Az1[k, i, ib, iχ]*b[gb.t[k, ib]]*χy[g.t[k, iχ]] for ib=1:gb.nn, iχ=1:g.nn) +
                           sum(As.Az2[k, i, ib, iχ]*b[gb.t[k, ib]]*χx[g.t[k, iχ]] for ib=1:gb.nn, iχ=1:g.nn)
    end
    return adv
end

function evolve!(m::ModelSetup3D, s::ModelState3D)
    # unpack
    μ = m.μ
    ϱ = m.ϱ
    ε² = m.ε²
    Δt = m.Δt
    g = m.g 
    b_cols = m.b_cols

    # T = 1e-2*μ*ϱ/ε²
    # n_steps = 10
    # Δt = T/n_steps

    # second order grid for b
    gb, pmap = get_gb(m)
    b = merge_cols(s.b, gb, b_cols, pmap)

    # matrices
    M = get_M(gb)
    K = get_K(gb)
    # LHS_diff = lu(μ*ϱ*M - ε²*Δt/2*K)
    LHS_diff = μ*ϱ*M - ε²*Δt/2*K
    RHS_diff = μ*ϱ*M + ε²*Δt/2*K
    # LHS_adv = cholesky(μ*ϱ*M)
    # LHS_adv = μ*ϱ*M
    # A = get_A(g.sf, gb.sf)
    # As = AdvectionArrays(A, g, gb)

    # pvd file
    rm("$out_folder/state.pvd", force=true)
    rm("$out_folder/state*.vtu", force=true)
    pvd = paraview_collection("$out_folder/state", append=true)

    # solve
    n_steps = 10
    for i=1:n_steps
        if mod(i, 10) == 0
            # update state
            invert!(m, s, showplots=true)
            get_u(m, s, showplots=true)

            # save state
            cell_type = VTKCellTypes.VTK_TETRA
            cells = [MeshCell(cell_type, g.t[i, :]) for i ∈ axes(g.t, 1)]
            vtk_grid("$out_folder/state$i", g.p', cells) do vtk
                vtk["omega^x"] = s.ωx.values
                vtk["omega^y"] = s.ωy.values
                vtk["chi^x"] = s.χx.values
                vtk["chi^y"] = s.χy.values
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
        # b = LHS_diff\(RHS_diff*b)
        b = cg!(b, LHS_diff, RHS_diff*b)
        s.b[:] = split_cols(b, b_cols, pmap)

        # # analytical solution
        # ba = [b_a(gb.p[j, 3], i*Δt, ε²/μ/ϱ, 1 - gb.p[j, 1]^2 - gb.p[j, 2]^2) for j=1:gb.np]
        # # b = ba
        # # s.b[:] = split_cols(ba, b_cols, pmap)
        # println(@sprintf("Max Error: %1.1e", maximum(abs.(b - ba))))

        if any(isnan.(b))
            error("Solution blew up 😢")
        end
    end

    vtk_save(pvd)
    println("$out_folder/state.pvd")

    # save b
    cell_type = VTKCellTypes.VTK_QUADRATIC_TETRA
    cells = [MeshCell(cell_type, gb.t[i, :]) for i ∈ axes(gb.t, 1)]
    vtk_grid("$out_folder/b", gb.p', cells) do vtk
        vtk["b"] = b
        ba = [b_a(gb.p[i, 3], n_steps*Δt, ε²/μ/ϱ, 1 - gb.p[i, 1]^2 - gb.p[i, 2]^2) for i=1:gb.np]
        vtk["ba"] = ba
        vtk["error"] = abs.(b - ba)
    end
    println("$out_folder/b.vtu")

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

function get_gb(m::ModelSetup3D)
    # unpack
    b_cols = m.b_cols
    g_sfc = m.g_sfc

    # allocate
    np = sum(g.np for g ∈ b_cols)
    nt = sum(g.nt for g ∈ b_cols)
    p = zeros(Float64, np, 3)
    t = zeros(Int64, nt, 10)
    ebot = zeros(Int64, 6*g_sfc.nt)
    esfc = zeros(Int64, 6*g_sfc.nt)

    # add each column
    i_p = 0
    i_t = 0
    for k=1:g_sfc.nt
        g = b_cols[k]
        np_k = g.np
        nt_k = g.nt
        p[i_p+1:i_p+np_k, :] = g.p
        t[i_t+1:i_t+nt_k, :] = i_p .+ g.t
        ebot[6k-5:6k] = g.e["bot"]
        esfc[6k-5:6k] = g.e["sfc"]
        i_p += np_k
        i_t += nt_k
    end

    # add tag indices to p
    ptag = hcat(p, 1:np)

    # sort rows
    ptag = sortslices(ptag, dims=1)

    # remove duplicate points
    keep = zeros(Bool, np)
    keep[unique(i -> ptag[i, 1:3], 1:np)] .= 1
    p = ptag[keep, 1:3]

    # position of ith point is p[pmap[i], :] 
    pmap = cumsum(keep)
    invpermute!(pmap, Int64.(ptag[:, 4]))

    # apply map to and e
    t = pmap[t]
    ebot = pmap[ebot]
    esfc = pmap[ebot]
    e = Dict("bot" => ebot, "sfc" => esfc)

    # # plot
    # points = p'
    # cell_type = VTKCellTypes.VTK_QUADRATIC_TETRA
    # cells = [MeshCell(cell_type, t[i, :]) for i ∈ axes(t, 1)]
    # vtk_grid("$out_folder/gb.vtu", points, cells) do vtk
    # end

    # make grid
    gb = Grid(2, p, t, e)

    return gb, pmap
end

function merge_cols(b, gb, b_cols, pmap)
    b_merged = zeros(gb.np)
    i_p = 0
    for k ∈ eachindex(b_cols)
        g = b_cols[k]
        b_merged[pmap[i_p+1:i_p+g.np]] = b[k].values
        i_p += g.np
    end
    return b_merged
end

function split_cols(b, b_cols, pmap)
    b_split = [FEField(zeros(g.np), g) for g ∈ b_cols]
    i_p = 0
    for k ∈ eachindex(b_cols)
        g = b_cols[k]
        b_split[k].values[:] = b[pmap[i_p+1:i_p+g.np]]
        i_p += g.np
    end
    return b_split
end