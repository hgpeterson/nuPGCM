using nuPGCM
using PyPlot
using SparseArrays
using LinearAlgebra

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)


"""
Baroclinic:
    -ε²∂zz(ωˣ) - ωʸ = 0,
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b)
BC:
    • ωˣ = 0 at z = 0
    • ωˣ = 0 at z = -H
    • ωˣ = 0 at z = 0
    • ∫ zωʸ dz = 0
"""
function solve_baroclinic(z, bx, ε²)
    # indices
    nz = size(z, 1)
    ωxmap = 1:nz
    ωymap = nz+1:2*nz

    # matrix
    A = Tuple{Int64,Int64,Float64}[]  
    r = zeros(2*nz)

    # interior nodes
    for j=2:nz-1 
        # ∂zz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # eqtn 1: -ε²∂zz(ωˣ) - ωʸ = 0
        # term 1
        push!(A, (ωxmap[j], ωxmap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωxmap[j], ωxmap[j],   -ε²*fd_zz[2]))
        push!(A, (ωxmap[j], ωxmap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωxmap[j], ωymap[j], -1))

        # eqtn 2: -ε²∂zz(ωʸ) + ωˣ = -∂x(b)
        # term 1
        push!(A, (ωymap[j], ωymap[j-1], -ε²*fd_zz[1]))
        push!(A, (ωymap[j], ωymap[j],   -ε²*fd_zz[2]))
        push!(A, (ωymap[j], ωymap[j+1], -ε²*fd_zz[3]))
        # term 2
        push!(A, (ωymap[j], ωxmap[j], 1))
        # rhs
        r[ωymap[j]] = -bx[j]
    end

    # ωˣ = ωʸ = 0 at z = 0
    push!(A, (ωxmap[nz], ωxmap[nz], 1))
    push!(A, (ωymap[nz], ωymap[nz], 1))

    # ωˣ = 0 at z = -H
    push!(A, (ωxmap[1], ωxmap[1], 1))

    # ∫ zωʸ dz = 0
    for j=1:nz-1
        # trapezoidal rule
        push!(A, (ωymap[1], ωymap[j],     z[j]*(z[j+1] - z[j])/2))
        push!(A, (ωymap[1], ωymap[j+1], z[j+1]*(z[j+1] - z[j])/2))
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), 2*nz, 2*nz)

    sol = A\r
    return sol[ωxmap], sol[ωymap]
end

nz = 2^8
z = -1:1/(nz - 1):0
bx = ones(nz)
ε² = 0.01
ωx, ωy = solve_baroclinic(z, bx, ε²)

fig, ax = subplots(1, figsize=(2, 3.2))
ax.plot(ωx, z, label=L"\omega^x")
ax.plot(ωy, z, label=L"\omega^y")
ax.legend()
ax.set_xlabel(L"\omega")
ax.set_ylabel(L"z")
savefig("images/omega.png")
println("images/omega.png")