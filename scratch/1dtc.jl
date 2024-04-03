using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)

if !isdir("../output")
    mkdir("../output")
end
set_out_folder("../output")
if !isdir("$out_folder/data")
    mkdir("$out_folder/data")
end
if !isdir("$out_folder/images")
    mkdir("$out_folder/images")
end

"""
    -f v = -∂x(P) + b*tan(θ) + ∂z(ν ∂z(u))
     f u = -∂y(P) + ∂z(ν ∂z(v))
with
    u = v = 0 at z = -H
    ∂z(u) = ∂z(v) = 0 at z = 0
    ∫ u dz = U
    ∫ v dz = V
"""
function build_inversion_LHS(z, f, ν; set_Px=false, set_Py=false)
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
        push!(A, (row, umap[1, j-1], -(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(A, (row, umap[1, j],   -(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(A, (row, umap[1, j+1], -(ν_z*fd_z[3] + ν[j]*fd_zz[3])))

        # 2nd eqtn:  f*u + ∂y(P) - (ν*v_z)_z = 0
        row = umap[2, j]
        # first term:
        push!(A, (row, umap[1, j], f))
        # second term:
        push!(A, (row, N+2, 1))
        # third term: dz(ν*dz(v))) = dz(ν)*dz(v) + ν*dzz(v)
        push!(A, (row, umap[2, j-1], -(ν_z*fd_z[1] + ν[j]*fd_zz[1])))
        push!(A, (row, umap[2, j],   -(ν_z*fd_z[2] + ν[j]*fd_zz[2])))
        push!(A, (row, umap[2, j+1], -(ν_z*fd_z[3] + ν[j]*fd_zz[3])))
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

    if set_Px
        # ∂x(P) = Px
        push!(A, (N+1, N+1, 1))
    else
        # ∫ u dz = U
        for j=1:nz-1
            # trapezoidal rule: (u_j+1 + u_j)/2 * Δz_j
            push!(A, (N+1, umap[1, j],   (z[j+1] - z[j])/2))
            push!(A, (N+1, umap[1, j+1], (z[j+1] - z[j])/2))
        end
    end

    if set_Py
        # ∂y(P) = Py
        push!(A, (N+2, N+2, 1))
    else
        # ∫ v dz = V
        for j=1:nz-1
            # trapezoidal rule: (v_j+1 + v_j)/2 * Δz_j
            push!(A, (N+2, umap[2, j],   (z[j+1] - z[j])/2))
            push!(A, (N+2, umap[2, j+1], (z[j+1] - z[j])/2))
        end
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), N+2, N+2)

    return lu(A)
end

function build_inversion_RHS(z, b, θ; U=nothing, V=nothing, Px=nothing, Py=nothing)
    nz = size(z, 1)
    N = 2*nz
    umap = reshape(1:N, 2, nz)    
    rhs = zeros(N+2)
    for j=2:nz-1
        # 1st eqtn: -f*v + ∂ₓP - (ν u_z)_z = b*tan(θ) 
        rhs[umap[1, j]] = b[j]*tan(θ)
    end
    # boundary conditions
    if Px !== nothing
        rhs[N+1] = Px
    end
    if U !== nothing
        rhs[N+1] = U
    end
    if Py !== nothing
        rhs[N+2] = Py
    end
    if V !== nothing
        rhs[N+2] = V
    end
    return rhs
end

# domain
H = 2e3
nz = 2^8
z = -H*(cos.(π*(0:nz-1)/(nz-1)) .+ 1)/2

# coriolis
f = -5.5e-5

# ν and κ
κ0 = 6e-5
κ1 = 2e-3
h = 200
κ = @. κ0 + κ1*exp(-(z + H)/h)
μ = 1.0
ν = μ*κ

# slope angle
θ = 2.5e-3

# buoyancy
δ = 100
N² = 1e-6
b = @. N²*δ*exp(-(z + H)/δ)

# set transports or pressure gradients
U = 0; set_Px = false
V = 0; set_Py = false
# Px = -f; set_Px = true
# Py = -f; set_Py = true

# build and solve the system
A = build_inversion_LHS(z, f, ν; set_Px, set_Py)
rhs = build_inversion_RHS(z, b, θ; U, V)
# rhs = build_inversion_RHS(z, b, θ; Px, V)
# rhs = build_inversion_RHS(z, b, θ; U, Py)
# rhs = build_inversion_RHS(z, b, θ; Px, Py)
rhs = build_inversion_RHS(z, b, θ; U, Px)
sol = A\rhs
umap = reshape(1:2nz, 2, nz)    
u = sol[umap[1, 1:nz]]
v = sol[umap[2, 1:nz]]
Px = sol[2nz+1]
Py = sol[2nz+2]

# plot
fig, ax = plt.subplots(1, 3, sharey=true)
ax[1].plot(u, z/1e3)
ax[1].axvline(-Py/f, c="k", lw=0.5, ls="--", label=L"-\partial_y P / f")
# ax[1].legend(fontsize=7)
ax[2].plot(v, z/1e3)
ax[2].axvline(Px/f, c="k", lw=0.5, ls="--", label=L"\partial_x P / f")
# ax[2].legend(fontsize=7)
ax[3].plot(1e6*(N² .+ differentiate(b, z)), z/1e3)
ax[1].set_xlabel(L"$u$ (m s$^{-1}$)")
ax[2].set_xlabel(L"$v$ (m s$^{-1}$)")
ax[3].set_xlabel(L"$\partial_z b$ ($\times 10^{-6}$ s$^{-2}$)")
ax[1].set_ylabel(L"$z$ (km)")
println("$out_folder/images/1dtc.png")
savefig("$out_folder/images/1dtc.png")
plt.close()