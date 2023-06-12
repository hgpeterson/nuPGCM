using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

include("utils.jl")
include("baroclinic.jl")

plt.style.use("plots.mplstyle")
plt.close("all")
pygui(false)

"""
    -f uʸ = -∂x(P) + b*tan(θ) + ∂z(ν ∂z(uˣ))
     f uˣ = -∂y(P) + ∂z(ν ∂z(uʸ))
with
    uˣ = uʸ = 0 at z = 0
    ∂z(uˣ) = ∂z(uʸ) = 0 at z = H
    ∫ uˣ dz = ∫ uʸ dz = 0
"""
function get_inversion_LHS(z, f, ν)
    nz = size(z, 1)
    N = 2*nz
    umap = reshape(1:N, 2, nz)    
    A = Tuple{Int64,Int64,Float64}[]  
    for j=2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        ν_z = sum(fd_z.*ν[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # 1st eqtn: -f*v + ∂ₓP - (ν u_z)_z = b*tan(θ) 
        row = umap[1, j]
        # first term
        push!(A, (row, umap[2, j], -f))
        # second term
        push!(A, (row, N+1, 1))
        # third term: dz(ν*dz(u))) = dz(ν)*dz(u) + ν*dzz(u)
        push!(A, (row, umap[1, j-1], -ε²*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(A, (row, umap[1, j],   -ε²*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(A, (row, umap[1, j+1], -ε²*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))

        # 2nd eqtn:  f*u + ∂y(P) - (ν*v_z)_z = 0
        row = umap[2, j]
        # first term:
        push!(A, (row, umap[1, j], f))
        # second term:
        push!(A, (row, N+2, 1))
        # third term: dz(ν*dz(v))) = dz(ν)*dz(v) + ν*dzz(v)
        push!(A, (row, umap[2, j-1], -ε²*(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(A, (row, umap[2, j],   -ε²*(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(A, (row, umap[2, j+1], -ε²*(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
    end

    # Boundary Conditions: Bottom
    # u = 0
    row = umap[1, 1] 
    push!(A, (row, row, 1.0))
    # v = 0
    row = umap[2, 1] 
    push!(A, (row, row, 1.0))

    # Boundary Conditions: Top
    fd_z = mkfdstencil(z[nz-2:nz], z[nz], 1)
    # dz(u) = 0
    row = umap[1, nz] 
    push!(A, (row, umap[1, nz-2], fd_z[1]))
    push!(A, (row, umap[1, nz-1], fd_z[2]))
    push!(A, (row, umap[1, nz],   fd_z[3]))
    # dz(v) = 0
    row = umap[2, nz] 
    push!(A, (row, umap[2, nz-2], fd_z[1]))
    push!(A, (row, umap[2, nz-1], fd_z[2]))
    push!(A, (row, umap[2, nz],   fd_z[3]))

    # ∫ u dz = Ux, ∫ v dz = Uy
    for j=1:nz-1
        # trapezoidal rule: (u_j+1 + u_j)/2 * Δz_j
        push!(A, (N+1, umap[1, j],   (z[j+1] - z[j])/2))
        push!(A, (N+1, umap[1, j+1], (z[j+1] - z[j])/2))
        push!(A, (N+2, umap[2, j],   (z[j+1] - z[j])/2))
        push!(A, (N+2, umap[2, j+1], (z[j+1] - z[j])/2))
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N+2, N+2)

    return lu(A)
end

function get_inversion_RHS(z, b, θ, Ux, Uy)
    nz = size(z, 1)
    N = 2*nz
    umap = reshape(1:N, 2, nz)    
    rhs = zeros(N+2)
    for j=2:nz-1
        # 1st eqtn: -f*v + ∂ₓP - (ν u_z)_z = b*tan(θ) 
        rhs[umap[1, j]] = b[j]*tan(θ)
    end
    # boundary conditions
    rhs[N+1] = Ux
    rhs[N+2] = Uy
    return rhs
end

ε² = 1e-4
H = 0.5
nz = 2^8
z = -H*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2
f = 1
ν = ones(nz)
θ = -π/4
δ = 0.1
b = @. δ*exp(-(z + H)/δ)
Ux = 0
Uy = 0
A = get_inversion_LHS(z, f, ν)
rhs = get_inversion_RHS(z, b, θ, Ux, Uy)
sol = A\rhs
umap = reshape(1:2nz, 2, nz)    
ux = sol[umap[1, 1:nz]]
uy = sol[umap[2, 1:nz]]
Px = sol[2nz+1]
Py = sol[2nz+2]

fig, ax = plt.subplots(1, 3, sharey=true)
ax[1].plot(ux, z)
ax[1].axvline(-Py/f, c="k", lw=0.5, ls="--")
ax[2].plot(uy, z)
ax[2].axvline(Px/f, c="k", lw=0.5, ls="--")
ax[3].plot(1 .+ differentiate(b, z), z)
ax[1].set_xlabel(L"u^x")
ax[2].set_xlabel(L"u^y")
ax[3].set_xlabel(L"\partial_z b")
ax[1].set_ylabel(L"z")
println("scratch/images/1dtc.png")
savefig("scratch/images/1dtc.png")
plt.close()
