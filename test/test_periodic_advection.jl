# Sharper periodic-advection tests than the uniform-velocity blob circuit.
#
# The uniform blob test cannot detect (a) DOF-map or component mixups in the
# cached advection kernel that happen to be invisible for spatially constant
# velocity, (b) seam errors that only appear when u·n varies along the periodic
# boundary, or (c) constraint conflicts at "junction" DOFs where Dirichlet and
# periodic constraints meet (wall∩seam for u, surface∩seam for b with an
# inhomogeneous surface value) -- the production channel configuration.
#
# Testset 2 exploits an exact discrete identity: for the Galerkin advection
# operator with an exactly divergence-free P2 velocity (u·n = 0 on walls,
# periodic traces at the seam) and a periodic buoyancy interpolant,
#
#     ∫ (u·∇b_h) b_h dΩ = ∮_seam u·n b_h²/2 (east) - (west) = 0
#     ∫  u·∇b_h      dΩ = ∮_seam u·n b_h    (east) - (west) = 0
#
# and both integrals are quadrature-exact (degree 3 ≤ QR_ORDER = 4 for P1 b),
# so any mishandling of mirror/image DOFs shows up at O(1), not O(h).
@testset "Periodic box: advection" begin
    h = 0.1
    α = 0.5
    W_box = 1.0
    H₀ = α*W_box
    box_file = joinpath(@__DIR__, @sprintf("../meshes/periodic_box_h%.2e_a%.2e.msh", h, α))
    if !isfile(box_file)
        include(joinpath(@__DIR__, "../meshes/periodic_box.jl"))
        mesh_periodic_box(h, α; W=W_box, L=1.0)
    end
    mesh = Mesh(box_file)

    # fixture without b-Dirichlet (periodic-only ch_b), production b_order = 1
    fe_free = FEData(mesh;
        u_diri_tags  = ["bottom", "surface", "wall"],
        u_diri_masks = [(true,true,true), (false,false,true), (false,false,true)],
        b_order = 1)

    fill_u(fe_data, fn) = begin
        x = zeros(ndofs(fe_data.dh_up))
        apply_analytical!(x, fe_data.dh_up, :u, fn)
        x
    end
    fill_b(fe_data, fn) = begin
        b = zeros(fe_data.nb)
        apply_analytical!(b, fe_data.dh_b, :b, fn)
        b
    end

    # reference advection RHS: independent CellValues-based assembly of the
    # same integral as the cached kernel (same quadrature order)
    function ref_rhs_adv(fe_data, N², Δt, x_up, b_vec, x_up_prev, b_prev; bdf2::Bool)
        cv_u, _, cv_b = make_cell_values(fe_data)
        dh_up = fe_data.dh_up
        u_range = dof_range(dh_up, :u)
        n_b = getnbasefunctions(cv_b)
        f  = zeros(fe_data.nb)
        fₑ = zeros(n_b)
        for (cc_up, cc_b) in zip(CellIterator(dh_up), CellIterator(fe_data.dh_b))
            reinit!(cv_u, cc_up)
            reinit!(cv_b, cc_b)
            u_dofs = celldofs(cc_up)[u_range]
            b_dofs = celldofs(cc_b)
            uₑ  = x_up[u_dofs];      uₑ_prev = x_up_prev[u_dofs]
            bₑ  = b_vec[b_dofs];     bₑ_prev = b_prev[b_dofs]
            fill!(fₑ, 0.0)
            for q in 1:getnquadpoints(cv_b)
                dΩ  = getdetJdV(cv_b, q)
                u   = function_value(cv_u, q, uₑ)
                b   = function_value(cv_b, q, bₑ)
                ∇b  = function_gradient(cv_b, q, bₑ)
                if bdf2
                    u_prev = function_value(cv_u, q, uₑ_prev)
                    b_prev_q = function_value(cv_b, q, bₑ_prev)
                    ∇b_prev  = function_gradient(cv_b, q, bₑ_prev)
                    u_eff  = 2*u - u_prev
                    ∇b_eff = 2*∇b - ∇b_prev
                    b_comb = 4/3*b - 1/3*b_prev_q
                    fac    = 2/3
                else
                    u_eff = u; ∇b_eff = ∇b; b_comb = b; fac = 1.0
                end
                adv = u_eff[1]*∇b_eff[1] + u_eff[2]*∇b_eff[2] + u_eff[3]*(∇b_eff[3] + N²)
                for i in 1:n_b
                    fₑ[i] += (b_comb - fac*Δt*adv) * shape_value(cv_b, q, i) * dΩ
                end
            end
            f[b_dofs] .+= fₑ
        end
        return f
    end

    @testset "cached kernel vs reference assembly" begin
        N² = 1.3
        Δt = 0.07
        params = Parameters(; ε=0.5, α, μϱ=1.0, N², f=x->1.0, H=x->H₀)

        u_fn  = x -> Vec{3}((0.3 + 0.4*x[2]*x[3] + 0.2*sin(2π*x[1]/W_box),
                             0.1*x[3] + 0.3*x[1]*x[2],
                             0.2*x[2] - 0.1*x[1]*x[3]))
        up_fn = x -> Vec{3}((0.1 - 0.2*x[3]^2, 0.4*x[1], 0.1*x[2]*x[3]))
        b_fn  = x -> cos(2π*x[1]/W_box)*(0.5 + x[2])*(x[3] + 0.3) + 0.2*x[2]
        bp_fn = x -> sin(2π*x[1]/W_box)*x[3] + 0.1*x[2]^2

        x_up  = fill_u(fe_free, u_fn);  b_vec  = fill_b(fe_free, b_fn)
        x_upp = fill_u(fe_free, up_fn); b_prev = fill_b(fe_free, bp_fn)

        ts1 = BDF1(; t_start=0.0, t_stop=1.0, Δt)
        f1  = nuPGCM.build_rhs_adv(fe_free, params, x_up, b_vec, ts1)
        r1  = ref_rhs_adv(fe_free, N², Δt, x_up, b_vec, x_up, b_vec; bdf2=false)
        @test norm(f1 - r1) < 1e-12 * norm(r1)

        ts2 = BDF2(; t_start=0.0, t_stop=1.0, Δt)
        f2  = nuPGCM.build_rhs_adv(fe_free, params, x_up, b_vec, x_upp, b_prev, ts2)
        r2  = ref_rhs_adv(fe_free, N², Δt, x_up, b_vec, x_upp, b_prev; bdf2=true)
        @test norm(f2 - r2) < 1e-12 * norm(r2)
    end

    @testset "seam skew-symmetry and mass neutrality" begin
        params0 = Parameters(; ε=0.5, α, μϱ=1.0, N²=0.0, f=x->1.0, H=x->H₀)

        # exactly divergence-free (P2-exact, x-independent) shear with u·n ≠ 0
        # at the seam and u·n = 0 on walls/bottom/surface
        U_fn = x -> Vec{3}((0.4 + 0.3*x[2] + 0.5*x[3] + 0.2*x[2]*x[3], 0.0, 0.0))
        # periodic buoyancy (trace-matched at the seam)
        b_fn = x -> sin(2π*x[1]/W_box)*(0.5 + x[2])*(x[3] + 0.3) + 0.3*x[2]*x[3]

        x_up  = fill_u(fe_free, U_fn)
        b_vec = fill_b(fe_free, b_fn)

        M  = build_M(fe_free)
        ts = BDF1(; t_start=0.0, t_stop=1.0, Δt=1.0)
        adv(x) = M*b_vec - nuPGCM.build_rhs_adv(fe_free, params0, x, b_vec, ts)

        a = adv(x_up)
        scale = dot(abs.(b_vec), abs.(a))
        @test abs(dot(b_vec, a)) < 1e-11 * scale          # energy neutrality
        @test abs(sum(a)) < 1e-11 * sum(abs, a)           # mass neutrality

        # teeth: corrupting the mirror u DOFs must break both identities at O(1),
        # so a recovery bug in the model's velocity path cannot pass silently
        mirror_u = [d for (i, d) in enumerate(fe_free.ch_up.prescribed_dofs)
                    if fe_free.ch_up.dofcoefficients[i] !== nothing &&
                       insorted(d, fe_free.u_dof_indices)]
        @test !isempty(mirror_u)
        x_broken = copy(x_up)
        x_broken[mirror_u] .= 0.0
        a_broken = adv(x_broken)
        @test abs(dot(b_vec, a_broken)) > 1e-4 * scale
        @test abs(sum(a_broken)) > 1e-4 * sum(abs, a_broken)
    end

    @testset "sheared blob advection across the seam" begin
        params   = Parameters(; ε=0.5, α, μϱ=1.0, N²=0.0, f=x->1.0, H=x->H₀)
        forcings = Forcings(1.0, x->1e-8, x->1e-8, x->0.0, x->0.0, SurfaceFluxBC(x->0.0))

        U(z) = 0.3 + 0.8*(z + H₀)   # 0.3 at bottom, 0.7 at surface
        Δt = 0.04
        t_stop = 1.2
        nsteps = round(Int, t_stop/Δt)
        ts = BDF2(; t_start=0.0, t_stop, Δt)

        inv_tk = InversionToolkit(CPU(), fe_free, params, forcings)
        evo_tk = EvolutionToolkit(CPU(), fe_free, params, forcings, ts)
        model  = Model(CPU(), params, forcings, fe_free, inv_tk, evo_tk, ts)

        σ = 0.15
        x₀, y₀, z₀ = 0.75, 0.0, -H₀/2
        blob(x, t) = begin
            dx = mod(x[1] - U(x[3])*t - x₀ + W_box/2, W_box) - W_box/2
            exp(-(dx^2 + (x[2]-y₀)^2 + (x[3]-z₀)^2)/(2σ^2))
        end
        set_b!(model, x -> blob(x, 0.0))
        model.state.u .= fill_u(fe_free, x -> Vec{3}((U(x[3]), 0.0, 0.0)))[fe_free.u_dof_indices]

        M     = build_M(fe_free)
        mass0 = sum(M * model.state.b)
        nrm0  = sqrt(dot(model.state.b, M * model.state.b))
        E0    = nrm0^2

        u_prev = copy(model.state.u)
        b_prev = copy(model.state.b)
        errs  = Float64[]
        b_ref = zeros(fe_free.nb)
        energies = Float64[]
        for i in 1:nsteps
            b_curr = copy(model.state.b)
            evolve!(model, u_prev, b_prev)
            update_t!(model.timestepper)
            b_prev .= b_curr

            apply_analytical!(b_ref, fe_free.dh_b, :b, x -> blob(x, model.timestepper.t[]))
            push!(errs, sqrt(dot(model.state.b - b_ref, M * (model.state.b - b_ref))) / nrm0)

            @test abs(sum(M * model.state.b) - mass0) / abs(mass0) < 1e-12
            push!(energies, dot(model.state.b, M * model.state.b))
        end

        # exact solution is the sheared, periodically wrapped blob
        @test errs[end] < 0.3
        # no localized error jump when the blob crosses the seam
        @test maximum(abs, diff(errs)) < 0.03
        # the advection term is explicit (BDF2 with extrapolated ∇(2b - b_prev)),
        # which is only weakly stable for pure advection: with κ ≈ 0 the energy
        # grows secularly at ~0.05%/step (measured; scheme-intrinsic, seam-
        # independent). A seam defect instead injects energy at O(1) rates (cf.
        # the corrupted-mirror check above) and breaks the per-step mass
        # conservation asserted in the loop, so cap total and per-step growth
        # with an order of magnitude of headroom.
        @test energies[end] < 1.05 * E0
        @test maximum(diff(energies)[3:end]) < 0.005 * E0
        @test maximum(abs, model.state.b) < 1.3
    end

    # production-like channel BCs: full Dirichlet walls for u (wall∩seam
    # junction DOFs) and an inhomogeneous, y-dependent surface value for b
    # (surface∩seam junction DOFs)
    b_surf = x -> 1.0 + 0.5*x[2] - x[2]^2
    fe_diri = FEData(mesh;
        u_diri_tags  = ["bottom", "surface", "wall"],
        u_diri_masks = [(true,true,true), (false,false,true), (true,true,true)],
        b_diri_tags  = ["surface"],
        b_diri_vals  = [b_surf],
        b_order = 1,
        pressure_gauge = :none)

    @testset "junction DOFs: condensed inversion system structure" begin
        ch = fe_diri.ch_up

        # the seam must actually meet Dirichlet boundaries: periodic mirror
        # DOFs whose image is itself prescribed (wall/bottom/surface Dirichlet)
        njunction = count(eachindex(ch.prescribed_dofs)) do i
            ch.dofcoefficients[i] !== nothing &&
                any(haskey(ch.dofmapping, d) for (d, _) in ch.dofcoefficients[i])
        end
        @test njunction > 0

        # condensation must eliminate every prescribed row and column exactly:
        # anything left beyond the placed diagonal couples the reduced system
        # to constrained DOFs whose solve values are meaningless. This is the
        # defect mode of folding a mirror equation into a prescribed image DOF.
        A_raw = allocate_inversion_matrix(fe_diri)
        A_raw.nzval .= sin.(1.0:length(A_raw.nzval))   # deterministic, non-symmetric
        A_c, _ = nuPGCM.condense_system(A_raw, ch, fe_diri.C_up)
        md = sum(abs, diag(A_raw)) / size(A_raw, 1)
        pd = ch.prescribed_dofs

        I1, J1, V1 = findnz(A_c[pd, :])
        off_rows = [abs(V1[k]) for k in eachindex(V1) if J1[k] != pd[I1[k]]]
        @test isempty(off_rows) || maximum(off_rows) < 1e-12 * md

        I2, J2, V2 = findnz(A_c[:, pd])
        off_cols = [abs(V2[k]) for k in eachindex(V2) if I2[k] != pd[J2[k]]]
        @test isempty(off_cols) || maximum(off_cols) < 1e-12 * md

        @test maximum(abs, diag(A_c)[pd] .- md) < 1e-12 * md
    end

    @testset "junction DOFs: Dirichlet + periodic constraints" begin
        # analytic fields satisfying every BC and exactly periodic in x
        g_b = x -> b_surf(x) + x[3]*sin(2π*x[1]/W_box)*cos(x[2])
        s(x) = (0.25 - x[2]^2)*(x[3] + H₀)   # zero on walls and bottom
        g_u = x -> Vec{3}((s(x)*(1.0 + 0.1*cos(2π*x[1]/W_box)),
                           0.5*s(x)*sin(2π*x[1]/W_box),
                           (0.25 - x[2]^2)*x[3]*(x[3] + H₀)*sin(2π*x[1]/W_box)))
        g_p = x -> cos(2π*x[1]/W_box) + x[2]*x[3]

        # b: apply! must be the identity on a BC-satisfying field, and must
        # recover corrupted prescribed DOFs exactly (catches wrong constraint
        # resolution where surface Dirichlet meets the periodic seam)
        v = fill_b(fe_diri, g_b)
        v_id = copy(v)
        apply!(v_id, fe_diri.ch_b)
        @test maximum(abs, v_id - v) < 1e-12

        v_rec = copy(v)
        v_rec[fe_diri.ch_b.prescribed_dofs] .= 999.0
        apply!(v_rec, fe_diri.ch_b)
        @test maximum(abs, v_rec - v) < 1e-12

        # (u, p): same idempotence/recovery checks on the combined handler
        x_up = zeros(ndofs(fe_diri.dh_up))
        apply_analytical!(x_up, fe_diri.dh_up, :u, g_u)
        apply_analytical!(x_up, fe_diri.dh_up, :p, g_p)
        x_id = copy(x_up)
        apply!(x_id, fe_diri.ch_up)
        @test maximum(abs, x_id - x_up) < 1e-12

        x_rec = copy(x_up)
        x_rec[fe_diri.ch_up.prescribed_dofs] .= 999.0
        apply!(x_rec, fe_diri.ch_up)
        @test maximum(abs, x_rec - x_up) < 1e-12

        # smoke test: evolve b with the inhomogeneous surface Dirichlet BC and
        # a shear flow through the seam (the crash configuration minus the
        # inversion). Pure advection + Dirichlet preserves the data bounds, so
        # any junction-DOF corruption shows up as a localized spike.
        params   = Parameters(; ε=0.5, α, μϱ=1.0, N²=0.0, f=x->1.0, H=x->H₀)
        forcings = Forcings(1.0, x->1e-8, x->1e-8, x->0.0, x->0.0,
                            SurfaceDirichletBC(x -> b_surf(x)))
        Δt = 0.04
        ts = BDF2(; t_start=0.0, t_stop=1.0, Δt)
        inv_tk = InversionToolkit(CPU(), fe_diri, params, forcings)
        evo_tk = EvolutionToolkit(CPU(), fe_diri, params, forcings, ts)
        model  = Model(CPU(), params, forcings, fe_diri, inv_tk, evo_tk, ts)

        set_b!(model, g_b)
        U(z) = 0.3 + 0.8*(z + H₀)
        x_shear = zeros(ndofs(fe_diri.dh_up))
        apply_analytical!(x_shear, fe_diri.dh_up, :u, x -> Vec{3}((U(x[3]), 0.0, 0.0)))
        model.state.u .= x_shear[fe_diri.u_dof_indices]

        bound = maximum(abs, model.state.b)
        u_prev = copy(model.state.u)
        b_prev = copy(model.state.b)
        for i in 1:round(Int, 1.0/Δt)
            b_curr = copy(model.state.b)
            evolve!(model, u_prev, b_prev)
            update_t!(model.timestepper)
            b_prev .= b_curr
            @test all(isfinite, model.state.b)
        end
        @test maximum(abs, model.state.b) < 1.3 * bound
    end
end
