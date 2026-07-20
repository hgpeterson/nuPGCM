# Periodic box with uniform wind stress, constant f and ν, b = 0, and a flat
# bottom. BCs: no-slip bottom, w = 0 + wind stress at the surface, and w = 0 on
# the y-walls (natural BCs for the wall-tangential components). The exact
# solution of the full 3D problem is then the 1D Ekman column
#
#     -f v = A u'',  f u = A v'',  A = α²ε²ν,
#     u = v = 0 at z = -H,  A u' = ατˣ, A v' = ατʸ at z = 0,
#     w = 0, p = 0,
#
# which in complex form W = u + iv, γ = √(if/A), τ = τˣ + iτʸ reads
#
#     W(z) = (ατ/(Aγ)) sinh(γ(z + H)) / cosh(γH),
#
# and for f = 0 degenerates to the linear Couette profile u = (ατˣ/A)(z + H).
# One can check that this satisfies the natural BCs exactly: at the walls the
# x- and y-traction components ν(∂y u + ∂x v) and 2ν ∂y v - p vanish
# identically for any u(z), v(z), p = 0.
#
# Any defect in the periodic constraint machinery (facet pairing, mirror/image
# convention, matrix condensation, RHS condensation) breaks the analytic match
# and the x-invariance of the solution along the periodic direction. In
# particular this test catches the Ferrite ≤ 1.4 `apply!` bug that folds
# couplings between pairs of constrained DOFs into transposed positions (fixed
# by `condense_system`), which corrupted periodic inversions by O(1).
#
# Tolerance note: the solution is not exact to machine precision because the
# mean-pressure constraint is redundant here (the wall natural BC already sets
# the pressure gauge), which leaves one near-singular pressure direction in the
# condensed system; velocity errors are ~1e-3 at h = 0.1 and the pressure check
# is correspondingly loose.

function periodic_box_setup(box_file; f₀, ν₀, τˣ₀, τʸ₀, ε, α)
    params = Parameters(; ε, α, μϱ=1.0, N²=0.0, f=x->f₀, H=x->α)
    forcings = Forcings(ν₀, x->1.0, x->1.0,
                        x->τˣ₀, x->τʸ₀,
                        SurfaceDirichletBC(x->0.0))
    mesh = Mesh(box_file)
    fe_data = FEData(mesh;
        u_diri_tags  = ["bottom", "surface", "wall"],
        u_diri_masks = [(true,true,true), (false,false,true), (false,false,true)])
    inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
    model  = Model(CPU(), params, forcings, fe_data, inv_tk)
    invert!(model)   # b = 0
    return model, fe_data
end

@testset "Periodic box wind stress" begin
    h = 0.1
    α = 0.5
    W_box = 1.0   # channel length (periodic in x)
    L_box = 1.0   # channel width
    H₀ = α*W_box
    box_file = ensure_periodic_box_mesh(h, α; W=W_box, L=L_box)

    ε   = 0.5
    ν₀  = 1.0
    τˣ₀ = 0.1
    τʸ₀ = 0.0
    A   = α^2 * ε^2 * ν₀
    tol = 5e-3

    # sample points: columns spanning the domain, including the periodic seam x = 0
    xs = [0.0, 0.13, 0.5, 0.87]
    ys = [-0.31, 0.0, 0.24]
    zs = range(-H₀, 0.0, length=21)
    points = [Vec{3}((x, y, z)) for x in xs for y in ys for z in zs]

    for (f₀, label) in [(0.0, "f = 0 (Couette)"), (1.0, "f = 1 (Ekman spiral)")]
        @testset "$label" begin
            model, fe_data = periodic_box_setup(box_file; f₀, ν₀, τˣ₀, τʸ₀, ε, α)

            # exact solution W = u + iv
            W_exact = if f₀ == 0
                z -> α*(τˣ₀ + im*τʸ₀)/A * (z + H₀)
            else
                γ = sqrt(im*f₀/A)
                z -> α*(τˣ₀ + im*τʸ₀)/(A*γ) * sinh(γ*(z + H₀))/cosh(γ*H₀)
            end

            x_up = nuPGCM._to_up_vec(fe_data, model.state.u)
            u_vals = eval_at_points(model, fe_data.dh_up, x_up, :u, points)
            @test all(v -> v isa Vec{3}, u_vals)   # all points found in the domain

            W_vals = [W_exact(pt[3]) for pt in points]
            u_scale = maximum(abs, W_vals)
            err_u = maximum(abs(u_vals[i][1] - real(W_vals[i])) for i in eachindex(points))
            err_v = maximum(abs(u_vals[i][2] - imag(W_vals[i])) for i in eachindex(points))
            err_w = maximum(abs(u_vals[i][3]) for i in eachindex(points))
            @test err_u / u_scale < tol
            @test err_v / u_scale < tol
            @test err_w / u_scale < tol

            # pressure is zero for the exact solution (loose check, see note above)
            p_up = zeros(ndofs(fe_data.dh_up))
            p_up[fe_data.p_dof_indices] .= model.state.p
            p_vals = eval_at_points(model, fe_data.dh_up, p_up, :p, points)
            @test maximum(abs, p_vals) / (α*τˣ₀*W_box) < 1.0

            # x-invariance: the solution must not vary along the periodic direction
            for y in ys, z in [-0.9H₀, -0.5H₀, -0.1H₀]
                cols = eval_at_points(model, fe_data.dh_up, x_up, :u,
                                      [Vec{3}((x, y, z)) for x in range(0.0, W_box, length=9)])
                spread = maximum(norm(cols[i] - cols[1]) for i in eachindex(cols))
                @test spread / u_scale < tol
            end

            # periodicity: constrained (west) DOFs equal their image (east) DOFs
            ch = fe_data.ch_up
            for (i, pdof) in enumerate(ch.prescribed_dofs)
                dofcoef = ch.dofcoefficients[i]
                dofcoef === nothing && continue
                val = sum(s * x_up[d] for (d, s) in dofcoef)
                @test x_up[pdof] ≈ val atol=1e-12
            end
        end
    end

    @testset "constraint geometry" begin
        # every periodic constraint must link a west DOF to the east DOF of the
        # same field component at the same (y, z), with coefficient +1
        mesh = Mesh(box_file)
        fe_data = FEData(mesh;
            u_diri_tags  = ["bottom", "surface", "wall"],
            u_diri_masks = [(true,true,true), (false,false,true), (false,false,true)])
        dh, ch = fe_data.dh_up, fe_data.ch_up
        N = ndofs(dh)
        cy = zeros(N); cz = zeros(N); cx = zeros(N); comp = zeros(N)
        apply_analytical!(cx, dh, :u, x -> Vec{3}((x[1], x[1], x[1])));  apply_analytical!(cx, dh, :p, x -> x[1])
        apply_analytical!(cy, dh, :u, x -> Vec{3}((x[2], x[2], x[2])));  apply_analytical!(cy, dh, :p, x -> x[2])
        apply_analytical!(cz, dh, :u, x -> Vec{3}((x[3], x[3], x[3])));  apply_analytical!(cz, dh, :p, x -> x[3])
        apply_analytical!(comp, dh, :u, x -> Vec{3}((1.0, 2.0, 3.0)));   apply_analytical!(comp, dh, :p, x -> 4.0)

        n_periodic = 0
        for (i, s) in enumerate(ch.prescribed_dofs)
            dofcoef = ch.dofcoefficients[i]
            (dofcoef === nothing || length(dofcoef) != 1) && continue   # skip Dirichlet + mean pressure
            m, c = dofcoef[1]
            n_periodic += 1
            @test c ≈ 1.0
            @test comp[s] == comp[m]
            @test abs(cy[s] - cy[m]) < 1e-9
            @test abs(cz[s] - cz[m]) < 1e-9
            @test abs(abs(cx[s] - cx[m]) - W_box) < 1e-9
        end
        @test n_periodic > 0
    end
end
