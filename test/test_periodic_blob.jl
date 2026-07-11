# Advect a buoyancy blob around the periodic box with a prescribed uniform
# velocity u = (U, 0, 0) (no inversion). The exact solution is the periodically
# translated blob, so after one full circuit b must return to its initial state
# up to discretization error. Mass is conserved exactly by the Galerkin
# advection operator for this divergence-free, no-normal-flow velocity.
#
# This exercises the time-dependent periodic machinery (RHS condensation and
# constraint recovery of b across the seam, evolution LHS with periodic ch_b,
# and the cached advection kernel) end-to-end: a defect at the seam shows up as
# a localized error jump when the blob crosses x = 0, mass drift, or blow-up.
@testset "Periodic box: blob advection" begin
    h = 0.1
    α = 0.5
    W_box = 1.0
    H₀ = α*W_box
    box_file = joinpath(@__DIR__, @sprintf("../meshes/periodic_box_h%.2e_a%.2e.msh", h, α))
    if !isfile(box_file)
        include(joinpath(@__DIR__, "../meshes/periodic_box.jl"))
        mesh_periodic_box(h, α; W=W_box, L=1.0)
    end

    params = Parameters(; ε=0.5, α, μϱ=1.0, N²=0.0, f=x->1.0, H=x->H₀)
    forcings = Forcings(1.0, x->1e-8, x->1e-8, x->0.0, x->0.0, SurfaceFluxBC(x->0.0))

    mesh = Mesh(box_file)
    fe_data = FEData(mesh;
        u_diri_tags  = ["bottom", "surface", "wall"],
        u_diri_masks = [(true,true,true), (false,false,true), (false,false,true)],
        b_order = 1)   # match production runs

    U = 0.5
    Δt = 0.04
    T_circuit = W_box/U
    nsteps = round(Int, T_circuit/Δt)
    ts = BDF2(; t_start=0.0, t_stop=T_circuit, Δt)

    inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
    evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
    model = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)

    # blob initial condition (periodic in x)
    σ = 0.18
    x₀, y₀, z₀ = 0.75, 0.0, -H₀/2
    blob(x, t) = begin
        dx = mod(x[1] - U*t - x₀ + W_box/2, W_box) - W_box/2
        exp(-(dx^2 + (x[2]-y₀)^2 + (x[3]-z₀)^2)/(2σ^2))
    end
    set_b!(model, x -> blob(x, 0.0))

    # prescribed uniform velocity (bypass the inversion)
    x_up = zeros(ndofs(fe_data.dh_up))
    apply_analytical!(x_up, fe_data.dh_up, :u, x -> Vec{3}((U, 0.0, 0.0)))
    model.state.u .= x_up[fe_data.u_dof_indices]

    M = build_M(fe_data)
    mass0 = sum(M * model.state.b)
    nrm0  = sqrt(dot(model.state.b, M * model.state.b))

    u_prev = copy(model.state.u)
    b_prev = copy(model.state.b)
    errs = Float64[]
    b_ref = zeros(fe_data.nb)
    for i in 1:nsteps
        b_curr = copy(model.state.b)
        evolve!(model, u_prev, b_prev)
        update_t!(model.timestepper)
        b_prev .= b_curr

        apply_analytical!(b_ref, fe_data.dh_b, :b, x -> blob(x, model.timestepper.t[]))
        push!(errs, sqrt(dot(model.state.b - b_ref, M * (model.state.b - b_ref))) / nrm0)

        # discrete mass conservation at every step
        @test abs(sum(M * model.state.b) - mass0) / abs(mass0) < 1e-12
    end

    # after one circuit the blob must be back (discretization-level tolerance,
    # calibrated: ~0.13 at h = 0.1, σ = 0.18, P1 buoyancy)
    @test errs[end] < 0.2

    # smooth error growth: no step (e.g. a seam crossing) may add a large jump
    jumps = diff(errs)
    @test maximum(abs, jumps) < 0.02

    # bounded dispersion overshoot, no blow-up
    @test maximum(abs, model.state.b) < 1.3
end
