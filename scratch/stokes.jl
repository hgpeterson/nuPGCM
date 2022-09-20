using nuPGCM
using PyPlot
using SparseArrays
using Printf

plt.style.use("../plots.mplstyle")
plt.close("all")
pygui(false)


"""
    u, p = solve_stokes()

Stokes problem:
    -Δu + ∇p = 0 on Ω,
         ∇⋅u = 0 on Ω,
           u = 0 on Γ,
with extra condition
    ∫ p dx = 0.
Here u = (u₁, u₂) is the velocity vector and p is the pressure.
Weak form:
    ∫ ∇u ⋅ ∇v - p (∇⋅v) dx = 0
    ∫ q (∇⋅u) dx = 0
Or just,
    ∫ ∇u ⋅ ∇v - p (∇⋅v) + q (∇⋅u) dx = 0
for all 
    v ∈ V = {(v₁, v₂) | vᵢ ∈ P₂}
    q ∈ Q = {q ∈ P₁ | ∫ q dx = 0}
where Pₙ is the space of continuous polynomials of degree n.
"""
function solve_stokes(g₁::Grid, g₂::Grid, s₁::ShapeFunctionIntegrals, s₂::ShapeFunctionIntegrals, J₁::Jacobians, J₂::Jacobians)
end