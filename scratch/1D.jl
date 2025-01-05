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
function build_b(z, κ, params; horiz_diff=true)
    # unpack
    μϱ = params.μϱ
    γ = params.γ
    θ = params.θ
    ε = params.ε
    Δt = params.Δt

    # coeffs
    α = ε^2/μϱ * Δt/2
    if horiz_diff 
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
Build matrix representations of
    ε⁴*dzz(nu*β*dzz(χ)) + f²*(χ - U)/nu/β = -ε²*dz(b)*tan(θ)
    ε⁴*dzz(nu*β*dzz(χ)) + f²*χ/nu/β = -ε²*dz(b)*tan(θ) + f²*U/nu/β
so that LHS*χ = RHS*b.
"""
function build_χ(z, ν, params)
    # unpack
    ε = params.ε
    θ = params.θ
    γ = params.γ
    f = params.f
    U = params.U

    # setup
    N = length(z)
    β = 1 + γ*tan(θ)^2
    LHS = Tuple{Int64,Int64,Float64}[]  
    RHS = Tuple{Int64,Int64,Float64}[]  
    rhs = zeros(N)

    # interior nodes
    for j ∈ 3:N-2
        row = j 

        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        ν_z = sum(fd_z.*ν[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)
        ν_zz = sum(fd_zz.*ν[j-1:j+1])

        # dzzz stencil
        fd_zzz = mkfdstencil(z[j-2:j+2], z[j], 3)

        # dzzzz stencil
        fd_zzzz = mkfdstencil(z[j-2:j+2], z[j], 4)
        
        # eqtn: ε⁴*dzz(nu*β*dzz(χ)) + f²*(χ - U)/nu/β = -ε²*dz(b)*tan(θ)
        # term 1 (product rule)
        push!(LHS, (row, j-1, ε^4*ν_zz*fd_zz[1]))
        push!(LHS, (row, j,   ε^4*ν_zz*fd_zz[2]))
        push!(LHS, (row, j+1, ε^4*ν_zz*fd_zz[3]))

        push!(LHS, (row, j-2, 2*ε^4*ν_z*fd_zzz[1]))
        push!(LHS, (row, j-1, 2*ε^4*ν_z*fd_zzz[2]))
        push!(LHS, (row, j,   2*ε^4*ν_z*fd_zzz[3]))
        push!(LHS, (row, j+1, 2*ε^4*ν_z*fd_zzz[4]))
        push!(LHS, (row, j+2, 2*ε^4*ν_z*fd_zzz[5]))

        push!(LHS, (row, j-2, ε^4*ν[j]*fd_zzzz[1]))
        push!(LHS, (row, j-1, ε^4*ν[j]*fd_zzzz[2]))
        push!(LHS, (row, j,   ε^4*ν[j]*fd_zzzz[3]))
        push!(LHS, (row, j+1, ε^4*ν[j]*fd_zzzz[4]))
        push!(LHS, (row, j+2, ε^4*ν[j]*fd_zzzz[5]))
        # term 2
        push!(LHS, (row, j,   f^2/(ν[j]*β)))
        rhs[row] = f^2*U/(ν[j]*β)
        # term 3
        push!(RHS, (row, j-1, -ε^2*fd_z[1]*tan(θ)))
        push!(RHS, (row, j,   -ε^2*fd_z[2]*tan(θ)))
        push!(RHS, (row, j+1, -ε^2*fd_z[3]*tan(θ)))
    end

    # for finite difference on the top and bottom boundary
    fd_bot_z =  mkfdstencil(z[1:3], z[1],  1)
    fd_top_zz = mkfdstencil(z[N-3:N], z[N], 2)

    # Lower boundary conditions 
    # b.c. 1: dz(χ) = 0
    push!(LHS, (1, 1, fd_bot_z[1]))
    push!(LHS, (1, 2, fd_bot_z[2]))
    push!(LHS, (1, 3, fd_bot_z[3]))

    # b.c. 2: χ = 0 
    push!(LHS, (2, 1, 1.0))

    # Upper boundary conditions
    # b.c. 1: dzz(χ) = 0 
    push!(LHS, (N, N-3, fd_top_zz[1]))
    push!(LHS, (N, N-2, fd_top_zz[2]))
    push!(LHS, (N, N-1, fd_top_zz[3]))
    push!(LHS, (N, N,   fd_top_zz[4]))
    # b.c. 2: χ = U
    push!(LHS, (N-1, N,  1.0))
    rhs[N-1] = U

    # Create CSC sparse matrix from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), N, N)
    RHS = sparse((x->x[1]).(RHS), (x->x[2]).(RHS), (x->x[3]).(RHS), N, N)

    return LHS, RHS, rhs
end

function step!(b, LHS, rhs)
    ldiv!(b, LHS, rhs)
    return b
end

function invert!(χ, LHS, rhs)
    ldiv!(χ, LHS, rhs)
    return χ
end

function uvw(χ, z, ν, params)
    # unpack
    f = params.f
    β = 1 + params.γ*tan(params.θ)^2
    U = params.U
    ε = params.ε
    θ = params.θ

    # compute u and v from χ
    u = differentiate(χ, z)
    v = cumtrapz((@. f*(χ - U)/ν/β/ε^2), z)

    # transform
    w = u*tan(θ)
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
    println(filename)
    plt.close()
end

function save(u, v, w, b, params, t, z; filename="data/1D.jld2")
    jldsave(filename; u, v, w, b, params.μϱ, params.γ, params.θ, params.ε, params.Δt, t, z) 
    println(filename)
end

function sim_setup(params; horiz_diff=true)
    # grid
    z = chebyshev_grid(params.nz, params.H)

    # forcing
    ν = ones(params.nz)
    # ν = @. 1e-2 + exp(-(z + params.H)/0.1)
    κ = @. 1e-2 + exp(-(z + params.H)/0.1)

    # build matrices
    LHS_b, RHS_b, rhs_b = build_b(z, κ, params; horiz_diff)
    LHS_b = lu(LHS_b)
    LHS_χ, RHS_χ, rhs_χ = build_χ(z, ν, params)
    LHS_χ = lu(LHS_χ)

    # initial condition
    b = zeros(params.nz)
    χ = zeros(params.nz)
    t = 0

    return z, ν, κ, LHS_b, RHS_b, rhs_b, LHS_χ, RHS_χ, rhs_χ, b, χ, t
end

function solve(params; horiz_diff=true)
    # setup 
    z, ν, κ, LHS_b, RHS_b, rhs_b, LHS_χ, RHS_χ, rhs_χ, b, χ, t = sim_setup(params; horiz_diff)

    # run
    n_steps = Int(round(params.T/params.Δt))
    for i ∈ 1:n_steps
        # timestep
        step!(b, LHS_b, RHS_b*b + rhs_b - params.Δt*differentiate(χ, z)*tan(params.θ))
        t += params.Δt

        # inversion
        invert!(χ, LHS_χ, RHS_χ*b + rhs_χ)
    end

    # compute u, v, w
    u, v, w = uvw(χ, z, ν, params)

    return u, v, w, b, t, z
end

function main(; T)
    # parameters
    μϱ = 1e0
    # γ = 1/4
    θ = π/4
    ε = 1e-2
    Δt = 1e-5*μϱ/ε^2
    U = 0
    H = 0.75
    f = 1
    nz = 2^8
    horiz_diff = false
    # T = 1e-3

    # start plot
    fig, ax = plot_setup()

    # loop over γ
    γs = 0:0.1:1
    colors = pl.cm.viridis(range(0, 1, length=length(γs)))
    t = 0
    for i ∈ eachindex(γs)
        γ = γs[i]
        color = colors[i, :]
        label = L"\gamma = "*@sprintf("%.2f", γ)
        println("γ = ", γs[i])

        # parameters
        params = (μϱ=μϱ, γ=γ, θ=θ, ε=ε, Δt=Δt, U=U, H=H, f=f, nz=nz, T=T)

        # solve
        u, v, w, b, t, z = solve(params; horiz_diff)

        # plot
        plot(u, v, w, b, z; fig, ax, label, color)
    end

    # finish plot
    plot_finish(; fig, ax, t, filename=@sprintf("images/1D_gamma_t%0.03f.png", t))

    # # save
    # save(u, v, w, b, params, t, z; filename="data/1D.jld2")
end

# for T ∈ 1e1:1e1:5e2
#     main(; T)
# end

main(T=5e2)