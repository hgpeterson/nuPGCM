using nuPGCM
using LinearAlgebra
using SparseArrays
using Printf
using PyPlot
using PyCall
using JLD2

pygui(false)
plt.style.use("../plots.mplstyle")
plt.close("all")

pl = pyimport("matplotlib.pylab")

include("../plots/derivatives.jl")

"""
Build matrix representations of 
    b^(n+1) - a dz(κ (1 + Γ dz(b^(n+1)))) = b^(n) + a dz(κ (1 + Γ dz(b^(n))))
so that LHS*b[n+1] = RHS*b[n] + rhs where a = ε²/μϱ * Δt/2 and Γ = 1 + α^2*tan(θ)^2.
"""
function build_b(z, κ, params)
    # unpack
    μϱ = params.μϱ
    α = params.α
    θ = params.θ
    ε = params.ε
    Δt = params.Δt

    # coeffs
    a = ε^2/μϱ * Δt/2
    if params.horiz_diff 
        Γ = 1 + α^2*tan(θ)^2
    else
        Γ = 1
    end

    # initialize
    N = length(z)
    LHS = Tuple{Int64,Int64,Float64}[] 
    RHS = Tuple{Int64,Int64,Float64}[]
    rhs = zeros(N)

    # interior nodes 
    for j=2:N-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)

        # dz(κ)
        κ_z = fd_z[1]*κ[j-1] + fd_z[2]*κ[j] + fd_z[3]*κ[j+1]

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # LHS: b - a*dz(κ*(1 + Γ*dz(b)))
        #    = b - a*dz(κ + Γ*κ*dz(b))
        #    = b - a*dz(κ) - a*Γ*dz(κ)*dz(b) - a*Γ*κ*dzz(b)
        push!(LHS, (j, j, 1))
        push!(LHS, (j, j-1, (-a*Γ*κ_z*fd_z[1] - a*Γ*κ[j]*fd_zz[1])))
        push!(LHS, (j, j,   (-a*Γ*κ_z*fd_z[2] - a*Γ*κ[j]*fd_zz[2])))
        push!(LHS, (j, j+1, (-a*Γ*κ_z*fd_z[3] - a*Γ*κ[j]*fd_zz[3])))
        rhs[j] += a*κ_z # -α*dz(κ) move to rhs

        # RHS: b + a*dz(κ*(1 + Γ*dz(b)))
        #    = b + a*dz(κ + Γ*κ*dz(b))
        #    = b + a*dz(κ) + a*Γ*dz(κ)*dz(b) + a*Γ*κ*dzz(b)
        push!(RHS, (j, j, 1))
        push!(RHS, (j, j-1, (a*Γ*κ_z*fd_z[1] + a*Γ*κ[j]*fd_zz[1])))
        push!(RHS, (j, j,   (a*Γ*κ_z*fd_z[2] + a*Γ*κ[j]*fd_zz[2])))
        push!(RHS, (j, j+1, (a*Γ*κ_z*fd_z[3] + a*Γ*κ[j]*fd_zz[3])))
        rhs[j] += a*κ_z
    end

    # z = -H: 1 + Γ*dz(b) = 0 -> dz(b) = -1/Γ
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    push!(LHS, (1, 1, fd_z[1]))
    push!(LHS, (1, 2, fd_z[2]))
    push!(LHS, (1, 3, fd_z[3]))
    rhs[1] = -1/Γ

    # z = 0: b = 0
    push!(LHS, (N, N, 1))

    # Create CSC sparse matrices
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), N, N)
    RHS = sparse((x->x[1]).(RHS), (x->x[2]).(RHS), (x->x[3]).(RHS), N, N)

    return LHS, RHS, rhs
end

"""
Build matrix representation of
   -ε²Γ²*dz(ν*dz(u)) - f*v + Px*cos(ϕ) = b*tan(θ)
   -ε²Γ *dz(ν*dz(v)) + f*u + Px*sin(ϕ) = 0
where Γ = 1 + α^2*tan(θ)^2 and ϕ is the angle the zero-transport component makes with x.
Boundary conditions:
    dz(u) = dz(v) = 0 at z = 0
    u = v = 0 at z = -H
    ∫ [u*cos(ϕ) + v*sin(ϕ)] dz = 0 or Px = 0 depending on params.no_Px
"""
function build_LHS_inversion(z, ν, params)
    # unpack
    ε = params.ε
    θ = params.θ
    ϕ = params.ϕ
    α = params.α
    f = params.f

    # setup
    nz = length(z)
    umap = 1:nz
    vmap = nz+1:2nz
    iPx = 2nz + 1
    Γ = 1 + α^2*tan(θ)^2
    LHS = Tuple{Int64,Int64,Float64}[]  

    # interior nodes
    for j ∈ 2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        ν_z = sum(fd_z.*ν[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)
        
        # eq 1: -ε²Γ²*dz(ν*dz(u)) - f*v + Px*cos(ϕ) = b*tan(θ)
        # term 1 = -ε²Γ²*[dz(ν)*dz(u) + ν*dzz(u)] 
        c = ε^2*Γ^2
        push!(LHS, (umap[j], umap[j-1], -c*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(LHS, (umap[j], umap[j],   -c*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(LHS, (umap[j], umap[j+1], -c*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
        # term 2 = -f*v
        push!(LHS, (umap[j], vmap[j], -f))
        # term 3 = Px*cos(ϕ)
        push!(LHS, (umap[j], iPx, cos(ϕ)))

        # eq 2: -ε²Γ *dz(ν*dz(v)) + f*u + Px*sin(ϕ) = 0
        # term 1 = -ε²Γ*[dz(ν)*dz(v) + ν*dzz(v)]
        c = ε^2*Γ
        push!(LHS, (vmap[j], vmap[j-1], -c*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(LHS, (vmap[j], vmap[j],   -c*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(LHS, (vmap[j], vmap[j+1], -c*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
        # term 2 = f*u
        push!(LHS, (vmap[j], umap[j], f))
        # term 3 = Px*sin(ϕ)
        push!(LHS, (vmap[j], iPx, sin(ϕ)))
    end

    # bottom boundary conditions: u = v = 0
    push!(LHS, (umap[1], umap[1], 1))
    push!(LHS, (vmap[1], vmap[1], 1))

    # surface boundary conditions: dz(u) = dz(v) = 0
    fd_z = mkfdstencil(z[end-2:end], z[end], 1)
    push!(LHS, (umap[end], umap[end-2], fd_z[1]))
    push!(LHS, (umap[end], umap[end-1], fd_z[2]))
    push!(LHS, (umap[end], umap[end],   fd_z[3]))
    push!(LHS, (vmap[end], vmap[end-2], fd_z[1]))
    push!(LHS, (vmap[end], vmap[end-1], fd_z[2]))
    push!(LHS, (vmap[end], vmap[end],   fd_z[3]))

    # last degree of freedom: ∫ [u*cos(ϕ) + v*sin(ϕ)] dz = 0 or Px = 0
    if params.no_Px
        push!(LHS, (iPx, iPx, 1))
    else
        for j in 1:nz-1
            # trapezoidal rule
            dz = z[j+1] - z[j]
            push!(LHS, (iPx, umap[j],   cos(ϕ)*dz/2))
            push!(LHS, (iPx, umap[j+1], cos(ϕ)*dz/2))
            push!(LHS, (iPx, vmap[j],   sin(ϕ)*dz/2))
            push!(LHS, (iPx, vmap[j+1], sin(ϕ)*dz/2))
        end
    end

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), 2nz+1, 2nz+1)

    return LHS
end

"""
Update vector for RHS of baroclinic inversion 
   -ε²Γ²*dz(ν*dz(u)) - f*v + Px*cos(ϕ) = b*tan(θ)
   -ε²Γ *dz(ν*dz(v)) + f*u + Px*sin(ϕ) = 0
"""
function update_rhs_inversion!(rhs, z, b, params)
    rhs[2:params.nz-1] .= b[2:params.nz-1]*tan(params.θ)
    return rhs
end

function plot_setup()
    fig, ax = plt.subplots(1, 4, figsize=(8, 3.2), sharey=true)
    ax[1].set_ylabel(L"z")
    ax[1].set_xlabel(L"u")
    ax[2].set_xlabel(L"v")
    ax[3].set_xlabel(L"w")
    ax[4].set_xlabel(L"\partial_z b")
    ax[4].set_xlim(0, 1.5)
    for a ∈ ax[1:3]
        a.spines["left"].set_visible(false)
        a.axvline(0, color="k", lw=0.5)
    end
    for a ∈ ax 
        a.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=true)
    end
    return fig, ax
end

function plot(u, v, w, b, z; fig, ax, label="", color="C0")
    bz = differentiate(b, z)
    ax[1].plot(u,       z, c=color, label=label)
    ax[2].plot(v,       z, c=color, label=label)
    ax[3].plot(w,       z, c=color, label=label)
    ax[4].plot(1 .+ bz, z, c=color, label=label)
    label != "" && ax[1].legend()
end

function plot_finish(; fig, ax, t=nothing, filename="images/1D.png")
    if t !== nothing
        ax[1].set_title(latexstring(@sprintf("\$t = %s\$", sci_notation(t))))
    end
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()
end

function plot_bz(b, z; t=nothing, filename="images/1D.png")
    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.set_ylabel(L"Vertical coordinate $z$")
    ax.set_xlabel(L"Stratification $\partial_z b$")
    ax.set_xlim(0, 1.5)
    bz = differentiate(b, z)
    ax.plot(1 .+ bz, z)
    if t !== nothing
        ax.set_title(latexstring(@sprintf("\$t = %s\$", sci_notation(t))))
    end
    savefig(filename)
    @info "Saved '$filename'"
    plt.close()
end

function save(u, v, w, b, params, t, z; filename="data/1D.jld2")
    jldsave(filename; u, v, w, b, params, t, z) 
    @info "Saved '$filename'"
end

function invert(b, z, ν, params)
    # build matrices
    LHS_inversion = build_LHS_inversion(z, ν, params)
    LHS_inversion = lu(LHS_inversion)

    # initialize
    rhs_inversion = zeros(2*params.nz + 1)
    sol_inversion = zeros(2*params.nz + 1)
    u = @view sol_inversion[1:params.nz]
    v = @view sol_inversion[params.nz+1:2params.nz]

    # invert
    update_rhs_inversion!(rhs_inversion, z, b, params)
    ldiv!(sol_inversion, LHS_inversion, rhs_inversion)

    return u, v, u*tan(params.θ)
end

function sim_setup(params)
    # grid
    z = params.H*chebyshev_nodes(params.nz)

    # forcing
    ν = ones(params.nz)
    κ = @. 1e-2 + exp(-(z + params.H)/0.1)

    # build matrices
    LHS_b, RHS_b, rhs_b = build_b(z, κ, params)
    LHS_b = lu(LHS_b)
    LHS_inversion = build_LHS_inversion(z, ν, params)
    LHS_inversion = lu(LHS_inversion)

    # initial condition
    b = zeros(params.nz)
    rhs_inversion = zeros(2*params.nz + 1)
    sol_inversion = zeros(2*params.nz + 1)
    t = 0

    return z, ν, κ, LHS_b, RHS_b, rhs_b, LHS_inversion, rhs_inversion, b, sol_inversion, t
end

function solve(params)
    # setup 
    z, ν, κ, LHS_b, RHS_b, rhs_b, LHS_inversion, rhs_inversion, b, sol_inversion, t = sim_setup(params)
    u = @view sol_inversion[1:params.nz]
    v = @view sol_inversion[params.nz+1:2params.nz]

    # run
    n_steps = params.T ÷ params.Δt
    for i ∈ 1:n_steps
        # timestep
        ldiv!(b, LHS_b, RHS_b*b + rhs_b - params.Δt*u*tan(params.θ))
        t += params.Δt

        # inversion
        update_rhs_inversion!(rhs_inversion, z, b, params)
        ldiv!(sol_inversion, LHS_inversion, rhs_inversion)
    end

    return u, v, u*tan(params.θ), b, t, z
end

function loop_aspect_ratios()
    # parameters
    μϱ = 1e-4
    θ = π/4
    ε = 2e-2
    Δt = 1e-4*μϱ/ε^2
    no_Px = false
    ϕ = 0
    H = 0.75
    f = 1
    nz = 2^8
    horiz_diff = false
    T = 4e-2*μϱ/ε^2

    αs = [0, 1/2]
    for i ∈ eachindex(αs)
        params = (μϱ=μϱ, α=αs[i], θ=θ, ε=ε, Δt=Δt, no_Px=no_Px, ϕ=ϕ, H=H, f=f, nz=nz, T=T, horiz_diff=horiz_diff)
        u, v, w, b, t, z = solve(params)
        save(u, v, w, b, params, t, z; filename=@sprintf("./data/1D_%0.2f.jld2", αs[i]))
    end
end

function main()
    # parameters
    μϱ = 1e-4
    α = 1/2
    θ = π/4
    ε = 1e-2
    Δt = 1e-4*μϱ/ε^2
    no_Px = false
    H = 0.75
    f = 1
    ϕ = atan(0.5*H/tan(θ)) # tan ϕ = -(β*H - f*Hy)/f*Hx
    println("ϕ = ", ϕ*180/π)
    nz = 2^8
    horiz_diff = false
    T = 3e-3*μϱ/ε^2
    params = (μϱ=μϱ, α=α, θ=θ, ε=ε, Δt=Δt, no_Px=no_Px, ϕ=ϕ, H=H, f=f, nz=nz, T=T, horiz_diff=horiz_diff)

    # solve
    u, v, w, b, t, z = solve(params)

    # # save
    # save(u, v, w, b, params, t, z; filename="../sims/sim048/data/1D_beta1.0.jld2")

    # plot
    fig, ax = plot_setup()
    plot(u, v, w, b, z; fig, ax)
    plot_finish(; fig, ax, t, filename="images/1D.png")

    println(trapz(u*cos(ϕ) + v*sin(ϕ), z))
end

# loop_aspect_ratios()
# main()