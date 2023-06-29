using Gridap

𝒯 = CartesianDiscreteModel((-1,1,-1,0), (20,20))
Ω = Interior(𝒯)
dΩ = Measure(Ω, 2)

refFE = ReferenceFE(lagrangian, Float64, 1)

V = TestFESpace(𝒯, refFE)

g(x,t::Real) = 0.0
g(t::Real) = x -> g(x,t)
U = TransientTrialFESpace(V, g)

α = 100
z = VectorValue(0, 1)
∂z(u) = z⋅∇(u)
m(u,v) = ∫( u*v )dΩ
a(u,v) = ∫( α*(∂z(u)*∂z(v)) )dΩ
b(v) = 0
op = TransientConstantFEOperator(m, a, b, U, V)

linear_solver = LUSolver()

T = 1e-2/α
n_steps = 40
Δt = T/n_steps
θ = 0.5
ode_solver = ThetaMethod(linear_solver, Δt, θ)

u₀ = interpolate_everywhere(x -> x[2]^2 + 2/3*x[2]^3, U(0.0))
t₀ = 0.0
u = solve(ode_solver, op, u₀, t₀, T)

function ua(x, t; N=50)
    A(n) = 8*(-1 + (-1)^n)/(n^4*π^4)
    return 1/6 + sum(A(n)*cos(n*π*x[2])*exp(-α*(n*π)^2*t) for n=1:2:N)
end

createpvd("output/poisson_transient_solution") do pvd
  for (uₕ,t) in u
    err(x) = abs(uₕ(x) - ua(x, t)) 
    pvd[t] = createvtk(Ω,"output/poisson_transient_solution_$t"*".vtu",cellfields=["u"=>uₕ, "error"=>err])
  end
end
