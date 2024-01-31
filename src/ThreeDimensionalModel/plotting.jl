"""
    fig, ax, im = tplot(p, t, u)

Plot filled contour color plot of solution `u` on mesh defined by nodes positions `p` and connectivities `t`.
"""
function tplot(p, t, u; cmap="RdBu_r", vmax=0., contour=false, contour_levels=6, cb_label="", cb_orientation="vertical")
    fig, ax = subplots(1)

    # set vmax
    if vmax == 0.
        vmax = maximum(abs.(u))
        extend = "neither"
    else
        # set extend
        if maximum(u) > vmax && minimum(u) < -vmax
            extend = "both"
        elseif maximum(u) > vmax && minimum(u) > -vmax
            extend = "max"
        elseif maximum(u) < vmax && minimum(u) < -vmax
            extend = "min"
        else
            extend = "neither"
        end
    end

    if size(u, 1) == size(t, 1)
        # `u` represents values on triangle faces
        shading = "flat"
    elseif size(u, 1) == size(p, 1)
        # `u` represents values on triangle vertices
        shading = "gouraud"
    end

    im = ax.tripcolor(p[:, 1], p[:, 2], t[:, 1:3] .- 1, u, cmap=cmap, vmin=-vmax, vmax=vmax, shading=shading, rasterized=true)
    if contour
        spacing = 1/(contour_levels - 2)
        levels = vmax*(spacing:spacing:1-spacing)
        levels = vcat(reverse(-levels), levels)
        ax.tricontour(p[:, 1], p[:, 2], t[:, 1:3] .- 1, u, colors="k", linewidths=0.5, linestyles="-", levels=levels)
    end
    cb = colorbar(im, ax=ax, label=cb_label, extend=extend, orientation=cb_orientation)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)

    # no spines
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end
function tplot(u::AbstractField; kwargs...)
    return tplot(u.g.p, u.g.t, u.values; kwargs...)
end

"""
    fig, ax, im = tplot(p, t)

Plot triangular mesh with nodes `p` and triangles `t`.
"""
function tplot(p, t; lw=0.2, edgecolors="k")
    fig, ax = subplots(1)
    im = ax.tripcolor(p[:, 1], p[:, 2], t[:, 1:3] .- 1, 0*t[:, 1], cmap="Greys", edgecolors=edgecolors, linewidth=lw, rasterized=true)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    return fig, ax, im
end
function tplot(g::Grid; kwargs...)
    return tplot(g.p, g.t; kwargs...)
end

function quick_plot(u::FEField; title="", filename="$out_folder/images/quick_plot.png", kwargs...)
    fig, ax, im = tplot(u, contour=true; kwargs...)
    quick_plot_save(filename, ax, title)
end
function quick_plot(u::FVField; title="", filename="$out_folder/images/quick_plot.png", kwargs...)
    fig, ax, im = tplot(u, contour=false; kwargs...)
    quick_plot_save(filename, ax, title)
end
function quick_plot(u::DGField; kwargs...)
    quick_plot(FEField(u); kwargs...)
end
function quick_plot(f::Function, g::Grid; kwargs...)
    quick_plot(FEField(f, g); kwargs...)
end
function quick_plot_save(filename, ax, title)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Meridional coordinate $y$")
    ax.axis("equal")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:1)
    ax.set_title(title)
    savefig(filename)
    println(filename)
    plt.close()
end

function write_vtk(g, fname, data)
    # define points and cells for vtk
    points = g.p'

    # cells
    if g.el <: Triangle 
        if g.el.n == 3
            cell_type = VTKCellTypes.VTK_TRIANGLE
        elseif g.el.n == 6
            cell_type = VTKCellTypes.VTK_QUADRATIC_TRIANGLE
        end
    elseif g.el <: Wedge
        cell_type = VTKCellTypes.VTK_WEDGE
    end
    cells = [MeshCell(cell_type, g.t[i, :]) for i ∈ axes(g.t, 1)]

    # save as vtu file
    vtk_grid(fname, points, cells) do vtk
        for d ∈ data
            if typeof(d.second) <: AbstractField
                vtk[d.first] = d.second.values
            else
                vtk[d.first] = d.second
            end
        end
    end
    println(fname)
end

function save_u_vtk(m::ModelSetup3D, s::ModelState3D; fname="$out_folder/data/u.vtu")
    # unpack 
    g1 = m.geom.g1
    g_sfc2 = m.geom.g_sfc2
    H = m.geom.H
    Hx = m.geom.Hx
    Hy = m.geom.Hy
    nσ = m.geom.nσ
    χx = s.χx
    χy = s.χy
    b = s.b

    # σ, H, Hx, Hy on g1
    σ1 = FEField(g1.p[:, 3], g1)
    H1 = FEField([H[get_i_sfc(i, nσ)] for i ∈ 1:g1.np], g1)
    Hx1 = DGField([Hx[get_k_sfc(k_w, nσ), mod1(i, 3)] for k_w ∈ 1:g1.nt, i ∈ 1:g1.nn], g1)
    Hy1 = DGField([Hy[get_k_sfc(k_w, nσ), mod1(i, 3)] for k_w ∈ 1:g1.nt, i ∈ 1:g1.nn], g1)

    # compute u
    ux = -∂σ(χy)/H1
    uy = +∂σ(χx)/H1
    uσ = (∂ξ(χy) - ∂η(χx))/H1
    uz = H1*uσ + σ1*Hx1*ux + σ1*Hy1*uy
    uzb = uz*FEField(b[1:g1.np], g1)

    # FE fields
    ux = FEField(ux)
    uy = FEField(uy)
    uz = FEField(uz)
    uzb = FEField(uzb)

    # # DG p, t
    # np = g1.nt*g1.nn
    # p = zeros(Float64, (np, 3))
    # t = zeros(Int64, (g1.nt, 6))

    # # global solutions
    # ux_plot = zeros(np)
    # uy_plot = zeros(np)
    # uz_plot = zeros(np)

    # # map DG fields to p
    # i_p = 0
    # for k_sfc ∈ 1:g_sfc2.nt
    #     for j ∈ 1:nσ-1
    #         k_w = get_k_w(k_sfc, nσ, j)
    #         p[i_p+1:i_p+6, 1:2] = g1.p[g1.t[k_w, :], 1:2]
    #         p[i_p+1:i_p+3, 3] = g1.p[g1.t[k_w, 1:3], 3].*H[g_sfc2.t[k_sfc, 1:3]]
    #         p[i_p+4:i_p+6, 3] = g1.p[g1.t[k_w, 4:6], 3].*H[g_sfc2.t[k_sfc, 1:3]]
    #         t[k_w, :] = i_p+1:i_p+6
    #         ux_plot[i_p+1:i_p+6] = ux[k_w, :]
    #         uy_plot[i_p+1:i_p+6] = uy[k_w, :]
    #         uz_plot[i_p+1:i_p+6] = uz[k_w, :]
    #         i_p += 6
    #     end
    # end

    p = copy(g1.p)
    p[:, 3] = p[:, 3].*H1.values
    t = g1.t

    # save as .vtu
    cells = [MeshCell(VTKCellTypes.VTK_WEDGE, t[i, :]) for i ∈ axes(t, 1)]
    vtk_grid(fname, p', cells) do vtk
        # vtk["ux"] = ux_plot
        # vtk["uy"] = uy_plot
        # vtk["uz"] = uz_plot
        vtk["ux"] = ux.values
        vtk["uy"] = uy.values
        vtk["uz"] = uz.values
        vtk["uzb"] = uzb.values
    end
    println(fname)
end

function plot_xslice(m::ModelSetup3D, b::AbstractField, u::AbstractField, y, cb_label, fname)
    # unpack
    g_sfc1 = m.geom.g_sfc1
    g1 = m.geom.g1
    H = m.geom.H

    # get x slice
    nx = 2^8
    bdy = g_sfc1.p[g_sfc1.e["bdy"], :]
    neary = sort(bdy[sortperm(abs.(bdy[:, 2] .- y)), 1][1:4])
    x = range(neary[2], neary[3], length=nx)

    # vertical grid
    nσ = 2^8
    σ = collect(-(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2)
    
    # get indices of surface tris
    k_sfcs = [get_k([x[i], y], g_sfc1, g_sfc1.el) for i=1:nx]

    # get points in reference tri
    ξ_sfcs = [transform_to_ref_el(g_sfc1.el, [x[i], y], g_sfc1.p[g_sfc1.t[k_sfcs[i], :], :]) for i=1:nx]

    # get indices of wedges
    k_ws = [get_k_w(k_sfcs[j], m.geom.nσ, findfirst(k -> m.geom.σ[k] ≤ σ[i] ≤ m.geom.σ[k+1], 1:m.geom.nσ)) for i ∈ 1:nσ, j ∈ 1:nx] 

    # get points in reference wedge
    ξ_ws = [transform_to_ref_el(g1.el, [x[j], y, σ[i]], g1.p[g1.t[k_ws[i, j], :], :]) for i=1:nσ, j=1:nx]

    # nσ × nx coords
    Hs = [H(ξ_sfcs[i], k_sfcs[i]) for i=1:nx] 
    xx = repeat(x', nσ, 1)
    zz = repeat(σ, 1, nx).*repeat(Hs', nσ, 1)

    # evaluate
    u_fe = FEField(u)
    us = [u_fe(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:nx]
    bs = [b(ξ_ws[i, j], k_ws[i, j])   for i=1:nσ, j=1:nx]

    # plot
    title = latexstring(@sprintf("Slice at \$y = %1.1f\$", y))
    plot_vertical_slice(xx, zz, us, bs, cb_label, fname, title, slice_dir="x")
end

function plot_yslice(m::ModelSetup3D, b::AbstractField, u::AbstractField, x, cb_label, fname)
    # unpack
    g_sfc1 = m.geom.g_sfc1
    g1 = m.geom.g1
    H = m.geom.H

    # get y slice
    ny = 2^8
    bdy = g_sfc1.p[g_sfc1.e["bdy"], :]
    nearx = sort(bdy[sortperm(abs.(bdy[:, 1] .- x)), 2][1:4])
    y = range(nearx[2], nearx[3], length=ny)

    # vertical grid
    nσ = 2^8
    σ = collect(-(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2)

    # get indices of surface tris
    k_sfcs = [get_k([x, y[i]], g_sfc1, g_sfc1.el) for i=1:ny]

    # get points in reference tri
    ξ_sfcs = [transform_to_ref_el(g_sfc1.el, [x, y[i]], g_sfc1.p[g_sfc1.t[k_sfcs[i], :], :]) for i=1:ny]

    # get indices of wedges
    k_ws = [get_k_w(k_sfcs[j], m.geom.nσ, findfirst(k -> m.geom.σ[k] ≤ σ[i] ≤ m.geom.σ[k+1], 1:m.geom.nσ)) for i ∈ 1:nσ, j ∈ 1:ny] 

    # get points in reference wedge
    ξ_ws = [transform_to_ref_el(g1.el, [x, y[j], σ[i]], g1.p[g1.t[k_ws[i, j], :], :]) for i=1:nσ, j=1:ny]

    # nσ × ny coords
    Hs = [H(ξ_sfcs[i], k_sfcs[i]) for i=1:ny] 
    yy = repeat(y', nσ, 1)
    zz = repeat(σ, 1, ny).*repeat(Hs', nσ, 1)

    # evaluate
    u_fe = FEField(u)
    us = [u_fe(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:ny]
    bs = [b(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:ny]

    # plot
    title = latexstring(@sprintf("Slice at \$x = %1.1f\$", x))
    plot_vertical_slice(yy, zz, us, bs, cb_label, fname, title, slice_dir="y")
end

function plot_zslice(m::ModelSetup3D, u::AbstractField, z, cb_label, fname)
    g = m.geom.g_sfc1
    H = m.geom.H
    nσ = m.geom.nσ
    g_col = m.geom.g_col

    u_slice = zeros(g.np)
    for i=1:g.np
        if H[i] < abs(z)
            u_slice[i] = NaN
        else
            u_col = FEField(u[get_col_inds(i, nσ)], g_col)
            u_slice[i] = u_col(z/H[i])
        end
    end

    mask = [any(isnan.(u_slice[g.t[k, :]])) for k=1:g.nt]
    vmax = maximum(i-> isnan(u_slice[i]) ? -Inf : u_slice[i], 1:g.np)

    # plot
    title = latexstring(@sprintf("Slice at \$z = %1.1f\$", z))
    fig, ax = plt.subplots(1)
    img = ax.tripcolor(g.p[:, 1], g.p[:, 2], g.t[:, 1:3] .- 1, u_slice, mask=mask, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud", rasterized=true)
    cb = plt.colorbar(img, ax=ax, label=cb_label)
    cb.ax.ticklabel_format(style="sci", scilimits=(0, 0), useMathText=true)
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xlabel(L"Zonal coordinate $x$")
    ax.set_ylabel(L"Meridional coordinate $y$")
    ax.axis("equal")
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:1)
    ax.set_title(title)
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_u(m::ModelSetup3D, s::ModelState3D, y; i=0)
    if i == 0
        i_str = ""
    else
        i_str = @sprintf("%03d", i)
    end

    # unpack
    g_sfc1 = m.geom.g_sfc1
    g1 = m.geom.g1
    H = m.geom.H
    Hx = m.geom.Hx
    Hy = m.geom.Hy

    # get x slice
    nx = 2^8
    bdy = g_sfc1.p[g_sfc1.e["bdy"], :]
    neary = sort(bdy[sortperm(abs.(bdy[:, 2] .- y)), 1][1:4])
    x = range(0.99neary[2], 0.99neary[3], length=nx)

    # vertical grid
    nσ = 2^8
    σ = collect(-(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2)
    
    # get indices of surface tris
    k_sfcs = [get_k([x[i], y], g_sfc1, g_sfc1.el) for i=1:nx]

    # get points in reference tri
    ξ_sfcs = [transform_to_ref_el(g_sfc1.el, [x[i], y], g_sfc1.p[g_sfc1.t[k_sfcs[i], :], :]) for i=1:nx]

    # get indices of wedges
    k_ws = [get_k_w(k_sfcs[j], m.geom.nσ, findfirst(k -> m.geom.σ[k] ≤ σ[i] ≤ m.geom.σ[k+1], 1:m.geom.nσ)) for i ∈ 1:nσ, j ∈ 1:nx] 

    # get points in reference wedge
    ξ_ws = [transform_to_ref_el(g1.el, [x[j], y, σ[i]], g1.p[g1.t[k_ws[i, j], :], :]) for i ∈ 1:nσ, j ∈ 1:nx]

    # nσ × nx coords
    Hs = [H(ξ_sfcs[i], k_sfcs[i]) for i=1:nx] 
    Hxs = [Hx(ξ_sfcs[i], k_sfcs[i]) for i=1:nx] 
    Hys = [Hy(ξ_sfcs[i], k_sfcs[i]) for i=1:nx] 
    xx = repeat(x', nσ, 1)
    zz = repeat(σ, 1, nx).*repeat(Hs', nσ, 1)

    # evaluate
    χx_fe = FEField(s.χx)
    χy_fe = FEField(s.χy)
    ux = [-∂σ(χy_fe, ξ_ws[i, j], k_ws[i, j])/Hs[j] for i ∈ 1:nσ, j ∈ 1:nx]
    uy = [+∂σ(χx_fe, ξ_ws[i, j], k_ws[i, j])/Hs[j] for i ∈ 1:nσ, j ∈ 1:nx]
    Huσ = [∂ξ(χy_fe, ξ_ws[i, j], k_ws[i, j]) - ∂η(χx_fe, ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:nx]
    uz = [Huσ[i, j] + σ[i]*Hxs[j]*ux[i, j] + σ[i]*Hys[j]*uy[i, j] for i=1:nσ, j=1:nx]
    bs = [s.b(ξ_ws[i, j], k_ws[i, j]) for i=1:nσ, j=1:nx]

    # plot
    title = latexstring(@sprintf("Slice at \$y = %1.1f\$", y))
    plot_vertical_slice(xx, zz, ux,     bs, L"Zonal flow $u^x$",            "$out_folder/images/ux$i_str.png",  title, contour=false, slice_dir="x")
    plot_vertical_slice(xx, zz, uy,     bs, L"Meridional flow $u^y$",       "$out_folder/images/uy$i_str.png",  title, contour=false, slice_dir="x")
    plot_vertical_slice(xx, zz, uz,     bs, L"Vertical flow $u^z$",         "$out_folder/images/uz$i_str.png",  title, contour=false, slice_dir="x")
    plot_vertical_slice(xx, zz, uz.*bs, bs, L"Buoyancy production $u^z b$", "$out_folder/images/uzb$i_str.png", title, contour=false, slice_dir="x")
end

function plot_vertical_slice(xx, zz, u, b, cb_label, fname, title; contour=true, slice_dir, vmax=nothing)
    fig, ax = plt.subplots(1)
    if vmax === nothing
        vmax = maximum(abs.(u))
    end
    img = ax.pcolormesh(xx, zz, u, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=true, shading="nearest")
    if contour
        levels = range(-vmax, vmax, length=8)
        ax.contour(xx, zz, u, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    end
    cb = colorbar(img, ax=ax, label=cb_label, fraction=0.0235)
    cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    levels = range(-1, 0, length=20)
    ax.contour(xx, zz, b, levels=levels[1:end-1], colors="k", alpha=0.3, linestyles="-", linewidths=0.5)
    ax.axis("equal")
    ax.spines["left"].set_visible(false)
    ax.spines["bottom"].set_visible(false)
    ax.set_xticks(-1:0.5:1)
    ax.set_yticks(-1:0.5:0)
    if slice_dir == "x"
        ax.set_xlabel(L"Zonal coordinate $x$")
    elseif slice_dir == "y"
        ax.set_xlabel(L"Meridional coordinate $y$")
    end
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.set_title(title)
    savefig(fname)
    println(fname)
    plt.close()
end

function plot_profiles(m::ModelSetup3D, b, ωx, ωy, χx, χy; x, y, filename="$out_folder/images/profiles.png", m2D=nothing, s2D=nothing)
    g_sfc1 = m.geom.g_sfc1
    g1 = m.geom.g1
    k_sfc = get_k([x, y], g_sfc1, g_sfc1.el)
    ξ_sfc = transform_to_ref_el(g_sfc1.el, [x, y], g_sfc1.p[g_sfc1.t[k_sfc, :], :])

    nσ = 2^8
    σ = collect(-(cos.(π*(0:nσ-1)/(nσ-1)) .+ 1)/2)
    H = m.geom.H(ξ_sfc, k_sfc)
    Hx = m.geom.Hx(ξ_sfc, k_sfc)
    Hy = m.geom.Hy(ξ_sfc, k_sfc)
    z = σ*H
    k_ws = [get_k_w(k_sfc, m.geom.nσ, findfirst(j -> m.geom.σ[j] ≤ σ[i] ≤ m.geom.σ[j+1], 1:m.geom.nσ)) for i=1:nσ] 
    ξ_ws = [transform_to_ref_el(g1.el, [x, y, σ[i]], g1.p[g1.t[k_ws[i], :], :]) for i=1:nσ]

    ωx_fe = FEField(ωx)
    ωy_fe = FEField(ωy)
    χx_fe = FEField(χx)
    χy_fe = FEField(χy)
    ωxs = [ωx_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    ωys = [ωy_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    χxs = [χx_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    χys = [χy_fe(ξ_ws[i], k_ws[i]) for i=1:nσ]
    # bs = [b(ξ_ws[i], k_ws[i]) for i=1:nσ]
    bzs = [∂σ(b, ξ_ws[i], k_ws[i])/H for i=1:nσ]
    uxs = [-∂σ(χy_fe, ξ_ws[i], k_ws[i])/H for i=1:nσ]
    uys = [+∂σ(χx_fe, ξ_ws[i], k_ws[i])/H for i=1:nσ]
    uσs = [(∂ξ(χy_fe, ξ_ws[i], k_ws[i]) - ∂η(χx_fe, ξ_ws[i], k_ws[i]))/H for i=1:nσ]
    uzs = @. H*uσs + σ*Hx*uxs + σ*Hy*uys

    fig, ax = plt.subplots(2, 3, figsize=(6, 6.4), sharey=true)
    ax[1, 1].plot(ωxs, z, label=L"\omega^x")
    ax[1, 1].plot(ωys, z, label=L"\omega^y")
    ax[1, 2].plot(χxs, z, label=L"\chi^x")
    ax[1, 2].plot(χys, z, label=L"\chi^y")
    ax[1, 3].plot(bzs, z)
    ax[2, 1].plot(uxs, z)
    ax[2, 2].plot(uys, z)
    ax[2, 3].plot(uzs, z)
    if m2D !== nothing
        # compare with 2D
        r = √(x^2 + y^2)
        θ = atan(y, x)
        ix = argmin(abs.(m2D.ξ .- r))
        H = m2D.H[ix]
        z = m2D.z[ix, :]
        ωx = -1/H*differentiate(s2D.uη[ix, :]*cos(θ) + s2D.uξ[ix, :]*sin(θ), m2D.σ)
        ωy =  1/H*differentiate(s2D.uξ[ix, :]*cos(θ) - s2D.uη[ix, :]*sin(θ), m2D.σ)
        χx =  H*cumtrapz(s2D.uη[ix, :]*cos(θ) + s2D.uξ[ix, :]*sin(θ), m2D.σ)
        χy = -H*cumtrapz(s2D.uξ[ix, :]*cos(θ) - s2D.uη[ix, :]*sin(θ), m2D.σ)
        ux_full, uy_full, uz_full = transform_from_TF(m2D, s2D)
        ux = ux_full[ix, :]*cos(θ) - uy_full[ix, :]*sin(θ)
        uy = uy_full[ix, :]*cos(θ) + ux_full[ix, :]*sin(θ)
        uz = uz_full[ix, :]
        bz = 1/H*differentiate(s2D.b[ix, :], m2D.σ)
        ax[1, 1].plot(ωx, z, "k--", lw=0.5, label="2D")
        ax[1, 1].plot(ωy, z, "k--", lw=0.5)
        ax[1, 2].plot(χx, z, "k--", lw=0.5)
        ax[1, 2].plot(χy, z, "k--", lw=0.5)
        ax[1, 3].plot(bz, z, "k--", lw=0.5)
        ax[2, 1].plot(ux, z, "k--", lw=0.5)
        ax[2, 2].plot(uy, z, "k--", lw=0.5)
        ax[2, 3].plot(uz, z, "k--", lw=0.5)
    end
    ax[1, 1].set_xlabel("Vorticity")
    ax[1, 2].set_xlabel("Streamfunction")
    ax[1, 3].set_xlabel(L"Stratification $\partial_z b$")
    ax[2, 1].set_xlabel(L"Zonal flow $u^x$")
    ax[2, 2].set_xlabel(L"Meridional flow $u^y$")
    ax[2, 3].set_xlabel(L"Vertical flow $u^z$")
    ax[1, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[2, 1].set_ylabel(L"Vertical coordinate $z$")
    ax[1, 2].set_title(latexstring(@sprintf("\$x = %1.5f \\quad y = %1.5f\$", x, y)))
    ax[1, 1].set_ylim(-H, 0)
    ax[2, 1].set_ylim(-H, 0)
    ax[1, 1].legend()
    ax[1, 2].legend()
    for a ∈ ax
        a.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=true)
    end
    savefig(filename)
    println(filename)
    plt.close()
end
function plot_profiles(m::ModelSetup3D, s::ModelState3D; kwargs...)
    plot_profiles(m, s.b, s.ωx, s.ωy, s.χx, s.χy; kwargs...)
end