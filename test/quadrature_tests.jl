using Test
using nuPGCM.FiniteElements
using LinearAlgebra

function reference_mass_matrix(el::FiniteElements.AbstractElement, space::FiniteElements.AbstractFESpace)
    quad = QuadratureRule(el; deg=4)
    n_dof_per_el = FiniteElements.n_dofs(el, space)
    T = eltype(quad.weights)
    A = zeros(T, n_dof_per_el, n_dof_per_el)
    for q in eachindex(quad.weights)
        φq = φ(el, space, quad.points[q, :])
        for i in 1:n_dof_per_el, j in 1:n_dof_per_el
            A[i, j] += quad.weights[q] * φq[i] * φq[j] # * 1 since |det(J)| = 1 for reference element
        end
    end
    return A
end

function reference_stiffness_matrix(el::FiniteElements.AbstractElement, space::FiniteElements.AbstractFESpace)
    quad = QuadratureRule(el; deg=4)
    n_dof_per_el = FiniteElements.n_dofs(el, space)
    T = eltype(quad.weights)
    A = zeros(T, n_dof_per_el, n_dof_per_el)
    for q in eachindex(quad.weights)
        ∇φq = ∇φ(el, space, quad.points[q, :])
        for i in 1:n_dof_per_el, j in 1:n_dof_per_el
            A[i, j] += quad.weights[q] * dot(∇φq[i], ∇φq[j]) # * 1 since |det(J)| = 1 for reference element
        end
    end
    return A
end

@testset "Mass matrices" begin
    @test reference_mass_matrix(Triangle(), P1()) ≈ [1/12 1/24 1/24; 1/24 1/12 1/24; 1/24 1/24 1/12]
    @test reference_mass_matrix(Triangle(), P2()) ≈ [1/60 -1/360 -1/360 0 -1/90 0; -1/360 1/60 -1/360 0 0 -1/90; -1/360 -1/360 1/60 -1/90 0 0; 0 0 -1/90 4/45 2/45 2/45; -1/90 0 0 2/45 4/45 2/45; 0 -1/90 0 2/45 2/45 4/45]
end

@testset "Stiffness matrices" begin
    @test reference_stiffness_matrix(Triangle(), P1()) ≈ [1 -1/2 -1/2; -1/2 1/2 0; -1/2 0 1/2]
    @test reference_stiffness_matrix(Triangle(), P2()) ≈ [1 1/6 1/6 -(2/3) 0 -(2/3); 1/6 1/2 0 -(2/3) 0 0; 1/6 0 1/2 0 0 -(2/3); -(2/3) -(2/3) 0 8/3 -(4/3) 0; 0 0 0 -(4/3) 8/3 -(4/3); -(2/3) 0 -(2/3) 0 -(4/3) 8/3]
end
