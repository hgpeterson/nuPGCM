using Gridap
using GridapGmsh
using LinearAlgebra
using Printf

mesh_name = "channel_basin"
model = GmshDiscreteModel(joinpath(@__DIR__, "../meshes/$mesh_name.msh"))

# dirichlet boundary condition
b₀(x) = x[2] > 0 ? 0.0 : -x[2]^2

reffe_b = ReferenceFE(lagrangian, Float64, 2; space=:P)
B_test = TestFESpace(model, reffe_b, conformity=:H1, dirichlet_tags=["coastline", "surface"])
B_trial = TrialFESpace(B_test, [b₀, b₀])

Ω = Triangulation(model)
dΩ = Measure(Ω, 4)

α = 1/2
H_basin(x) = α*(x[1]*(1 - x[1]))/(0.5*0.5)
H_channel(x) = -α*((x[2] + 1)*(x[2] + 0.5))/(0.25*0.25)
H(x) = x[2] > -0.75 ? max(H_channel(x), H_basin(x)) : H_channel(x)
κ(x) = 1e-2 + exp(-(x[3] + H(x))/(0.5*α))
Δt = 1e-4

a_lhs(b, d) = ∫( b*d + Δt/2*(κ*∇(b)⋅∇(d)) )dΩ
A = assemble_matrix(a_lhs, B_trial, B_test)
A = lu(A)
l_diri_lhs(d) = a_lhs(interpolate(0, B_trial), d) # b₀ on the dirichlet boundary and 0 elsewhere
b_diri_lhs = assemble_vector(l_diri_lhs, B_test)

a_rhs(b, d) = ∫( b*d - Δt/2*(κ*∇(b)⋅∇(d)) )dΩ
B = assemble_matrix(a_rhs, B_trial, B_test)
l_diri_rhs(d) = a_rhs(interpolate(0, B_trial), d) # b₀ on the dirichlet boundary and 0 elsewhere
b_diri_rhs = assemble_vector(l_diri_rhs, B_test)

b_diri = b_diri_rhs - b_diri_lhs

function run()
    b = interpolate(x -> b₀(x) + x[3]/α, B_trial)
    save(b, 0)
    for i in 1:100
        b.free_values .= A \ (B*b.free_values + b_diri)
        save(b, i)
    end
end

function save(b, i)
    @info @sprintf("Iteration %d: %.2e < b < %.2e", i, minimum(b.free_values), maximum(b.free_values))
    writevtk(Ω, @sprintf("%s/data/state_%016d.vtu", @__DIR__, i), cellfields=["b" => b])
    @info @sprintf("%s/data/state_%016d.vtu", @__DIR__, i)
end

run()