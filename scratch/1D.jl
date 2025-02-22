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
    b^(n+1) - α dz(κ (1 + β dz(b^(n+1)))) = b^(n) + α dz(κ (1 + β dz(b^(n))))
so that LHS*b[n+1] = RHS*b[n] + rhs where α = ε²/μϱ * Δt/2 and β = 1 + γ*tan(θ)^2.
"""
function build_b(z, κ, params)
    # unpack
    μϱ = params.μϱ
    γ = params.γ
    θ = params.θ
    ε = params.ε
    Δt = params.Δt

    # coeffs
    α = ε^2/μϱ * Δt/2
    if params.horiz_diff 
        β = 1 + γ*tan(θ)^2
    else
        β = 1
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

        # LHS: b - α*dz(κ*(1 + β*dz(b)))
        #    = b - α*dz(κ + β*κ*dz(b))
        #    = b - α*dz(κ) - α*β*dz(κ)*dz(b) - α*β*κ*dzz(b)
        push!(LHS, (j, j, 1))
        push!(LHS, (j, j-1, (-α*β*κ_z*fd_z[1] - α*β*κ[j]*fd_zz[1])))
        push!(LHS, (j, j,   (-α*β*κ_z*fd_z[2] - α*β*κ[j]*fd_zz[2])))
        push!(LHS, (j, j+1, (-α*β*κ_z*fd_z[3] - α*β*κ[j]*fd_zz[3])))
        rhs[j] += α*κ_z # -α*dz(κ) move to rhs

        # RHS: b + α*dz(κ*(1 + β*dz(b)))
        #    = b + α*dz(κ + β*κ*dz(b))
        #    = b + α*dz(κ) + α*β*dz(κ)*dz(b) + α*β*κ*dzz(b)
        push!(RHS, (j, j, 1))
        push!(RHS, (j, j-1, (α*β*κ_z*fd_z[1] + α*β*κ[j]*fd_zz[1])))
        push!(RHS, (j, j,   (α*β*κ_z*fd_z[2] + α*β*κ[j]*fd_zz[2])))
        push!(RHS, (j, j+1, (α*β*κ_z*fd_z[3] + α*β*κ[j]*fd_zz[3])))
        rhs[j] += α*κ_z
    end

    # z = -H: 1 + β*dz(b) = 0 -> dz(b) = -1/β
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    push!(LHS, (1, 1, fd_z[1]))
    push!(LHS, (1, 2, fd_z[2]))
    push!(LHS, (1, 3, fd_z[3]))
    rhs[1] = -1/β

    # z = 0: b = 0
    push!(LHS, (N, N, 1))

    # Create CSC sparse matrices
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), N, N)
    RHS = sparse((x->x[1]).(RHS), (x->x[2]).(RHS), (x->x[3]).(RHS), N, N)

    return LHS, RHS, rhs
end

"""
Build matrix representation of
   ε²Γ²*dzz(ν*τˣ) + f*τʸ = F
   ε²Γ *dzz(ν*τʸ) - f*τˣ = G
where τˣ = dz(u) and τʸ = dz(v) and Γ = 1 + γ*tan(θ)^2.
Boundary conditions:
    τˣ = τʸ = 0 at z = 0
    -∫ z τˣ dz = U
    -∫ z τʸ dz = V or τʸ(-H) = 0 depending on params.set_V
"""
function build_LHS_τ(z, ν, params)
    # unpack
    ε = params.ε
    θ = params.θ
    γ = params.γ
    f = params.f

    # setup
    nz = length(z)
    umap = 1:nz
    vmap = nz+1:2nz
    Γ = 1 + γ*tan(θ)^2
    LHS = Tuple{Int64,Int64,Float64}[]  

    # interior nodes
    for j ∈ 2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        ν_z = sum(fd_z.*ν[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)
        ν_zz = sum(fd_zz.*ν[j-1:j+1])
        
        # eq 1: ε²Γ²*dzz(ν*τˣ) + f*τʸ = F 
        # term 1 = ε²Γ²*[dzz(ν)*τˣ + 2*dz(ν)*dz(τˣ) + ν*dzz(τˣ)] 
        c = ε^2*Γ^2
        push!(LHS, (umap[j], umap[j],     c*ν_zz))
        push!(LHS, (umap[j], umap[j-1], 2*c*ν_z*fd_z[1]))
        push!(LHS, (umap[j], umap[j],   2*c*ν_z*fd_z[2]))
        push!(LHS, (umap[j], umap[j+1], 2*c*ν_z*fd_z[3]))
        push!(LHS, (umap[j], umap[j-1],   c*ν[j]*fd_zz[1]))
        push!(LHS, (umap[j], umap[j],     c*ν[j]*fd_zz[2]))
        push!(LHS, (umap[j], umap[j+1],   c*ν[j]*fd_zz[3]))
        # term 2 = f*τʸ
        push!(LHS, (umap[j], vmap[j], f))

        # eq 2: ε²Γ *dzz(ν*τʸ) - f*τˣ = G
        # term 1 = ε²Γ*[dzz(ν)*τʸ + 2*dz(ν)*dz(τʸ) + ν*dzz(τʸ)]
        c = ε^2*Γ
        push!(LHS, (vmap[j], vmap[j],     c*ν_zz))
        push!(LHS, (vmap[j], vmap[j-1], 2*c*ν_z*fd_z[1]))
        push!(LHS, (vmap[j], vmap[j],   2*c*ν_z*fd_z[2]))
        push!(LHS, (vmap[j], vmap[j+1], 2*c*ν_z*fd_z[3]))
        push!(LHS, (vmap[j], vmap[j-1],   c*ν[j]*fd_zz[1]))
        push!(LHS, (vmap[j], vmap[j],     c*ν[j]*fd_zz[2]))
        push!(LHS, (vmap[j], vmap[j+1],   c*ν[j]*fd_zz[3]))
        # term 2 = -f*τˣ
        push!(LHS, (vmap[j], umap[j], -f))
    end

    # surface boundary conditions: τˣ = τʸ = 0
    push!(LHS, (umap[nz], umap[nz], 1))
    push!(LHS, (vmap[nz], vmap[nz], 1))

    # integral boundary conditions: -∫ z τˣ dz = U, -∫ z τʸ dz = V
    for j in 1:nz-1
        # trapezoidal rule
        dz = z[j+1] - z[j]
        push!(LHS, (umap[1], umap[j],   -z[j]*dz/2))
        push!(LHS, (umap[1], umap[j+1], -z[j]*dz/2))
        if params.set_V
            push!(LHS, (vmap[1], vmap[j],   -z[j]*dz/2))
            push!(LHS, (vmap[1], vmap[j+1], -z[j]*dz/2))
        end
    end

    # if V is not set, set τʸ = 0 at the bottom
    if !params.set_V
        push!(LHS, (vmap[1], vmap[1], 1))
    end

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), 2nz, 2nz)

    return LHS
end

"""
Update vector for RHS of baroclinic inversion 
   ε²Γ²*dzz(ν*τˣ) + f*τʸ = -dz(b)*tan(θ)
   ε²Γ *dzz(ν*τʸ) - f*τˣ = 0
with integral constraints conditions
    -∫ z τˣ dz = 0
    -∫ z τʸ dz = 0
"""
function update_rhs_τ!(rhs_τ, z, b, params)
    bz = differentiate(b, z)
    rhs_τ[2:params.nz] = -bz[2:params.nz]*tan(params.θ)
    rhs_τ[1] = 0
    rhs_τ[params.nz+1] = 0
    return rhs_τ
end

function uvw(τ, z, params)
    nz = params.nz

    # compute u and v from τ
    u = cumtrapz(τ[1:nz], z)
    v = cumtrapz(τ[nz+1:end], z)

    # transform
    w = u*tan(params.θ)
    return u, v, w
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
end

function plot_finish(; fig, ax, t=nothing, filename="images/1D.png")
    ax[1].legend()
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

function sim_setup(params)
    # grid
    z = params.H*chebyshev_nodes(params.nz)

    # forcing
    ν = ones(params.nz)
    κ = @. 1e-2 + exp(-(z + params.H)/0.1)

    # build matrices
    LHS_b, RHS_b, rhs_b = build_b(z, κ, params)
    LHS_b = lu(LHS_b)
    LHS_τ = build_LHS_τ(z, ν, params)
    LHS_τ = lu(LHS_τ)

    # initial condition
    b = zeros(params.nz)
    rhs_τ = zeros(2*params.nz)
    τ = zeros(2*params.nz)
    t = 0

    return z, ν, κ, LHS_b, RHS_b, rhs_b, LHS_τ, rhs_τ, b, τ, t
end

function solve(params)
    # setup 
    z, ν, κ, LHS_b, RHS_b, rhs_b, LHS_τ, rhs_τ, b, τ, t = sim_setup(params)

    # compute u, v, w
    u, v, w = uvw(τ, z, params)

    # run
    n_steps = Int(round(params.T/params.Δt))
    for i ∈ 1:n_steps
        # timestep
        ldiv!(b, LHS_b, RHS_b*b + rhs_b - params.Δt*u*tan(params.θ))
        t += params.Δt

        # inversion
        update_rhs_τ!(rhs_τ, z, b, params)
        ldiv!(τ, LHS_τ, rhs_τ)
        u, v, w = uvw(τ, z, params)
    end

    return u, v, w, b, t, z
end

function main()
    # parameters
    μϱ = 1e-4
    γ = 1/4
    θ = π/4
    ε = 1e-2
    Δt = 1e-4*μϱ/ε^2
    set_V = true
    H = 0.75
    f = 1
    nz = 2^8
    horiz_diff = false
    T = 3e-3*μϱ/ε^2
    params = (μϱ=μϱ, γ=γ, θ=θ, ε=ε, Δt=Δt, set_V=set_V, H=H, f=f, nz=nz, T=T, horiz_diff=horiz_diff)

    # solve
    u, v, w, b, t, z = solve(params)

    # save
    save(u, v, w, b, params, t, z; filename="../sims/sim048/data/1D_beta1.0.jld2")

    # plot
    fig, ax = plot_setup()
    plot(u, v, w, b, z; fig, ax)
    plot_finish(; fig, ax, t, filename="images/1D.png")
end

main()