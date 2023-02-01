"""
Baroclinic:
    -ε²∂zz(ωˣ) - ωʸ = 0,
    -ε²∂zz(ωʸ) + ωˣ = -∂x(b)
BC:
    • ωˣ = 0 at z = 0
    • ωˣ = 0 at z = -H
    • ωʸ = 0 at z = 0
    • ∫ zωʸ dz = 0
"""
function get_baroclinic_LHS(z, bx)
    # convention: τξ is variable 1, τη is variable 2
    nσ = size(σ, 1)
    nvar = 2
    imap = reshape(1:nvar*nσ, (nvar, nσ)) 
    A = Tuple{Int64,Int64,FT}[]  

    # Interior nodes
    for j=2:nσ-1 
        # ∂σσ stencil
        fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

        # eqtn 1: ν/f/H² ∂σσ(τξ) + τη = rhs_x
        row = imap[1, j]
        # term 1
        push!(A, (row, imap[1, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(A, (row, imap[1, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(A, (row, imap[1, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(A, (row, imap[2, j], 1))

        # eqtn 2: ν/f/H² ∂σσ(τη) - τξ = rhs_y
        row = imap[2, j]
        # term 1
        push!(A, (row, imap[2, j-1], ν[j]/f/H^2 * fd_σσ[1]))
        push!(A, (row, imap[2, j],   ν[j]/f/H^2 * fd_σσ[2]))
        push!(A, (row, imap[2, j+1], ν[j]/f/H^2 * fd_σσ[3]))
        # term 2
        push!(A, (row, imap[1, j], -1))
    end

    # Upper boundary conditions: wind stress
    # b.c. 1: τξ = τξ₀ at σ = 0
    push!(A, (imap[1, nσ], imap[1, nσ], 1))
    # b.c. 2: τη = τη₀ at σ = 0
    push!(A, (imap[2, nσ], imap[2, nσ], 1))

    # Integral boundary conditions: transport
    # b.c. 1: -H² ∫ σ τξ/ρ₀/ν dσ = Uξ
    for j=1:nσ-1
        # trapezoidal rule
        push!(A, (imap[1, 1], imap[1, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(A, (imap[1, 1], imap[1, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end
    # b.c. 1: -H² ∫ σ τη/ρ₀/ν dσ = Uη
    for j=1:nσ-1
        # trapezoidal rule
        push!(A, (imap[2, 1], imap[2, j],   -H^2/ρ₀/ν[j]   * σ[j]   * (σ[j+1] - σ[j])/2))
        push!(A, (imap[2, 1], imap[2, j+1], -H^2/ρ₀/ν[j+1] * σ[j+1] * (σ[j+1] - σ[j])/2))
    end

    # Create CSC sparse matrix from matrix elements
    A = sparse((x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), nvar*nσ, nvar*nσ)

    return lu(A)
end