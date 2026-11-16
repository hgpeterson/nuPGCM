# Regression tests on the bowl geometry (CPU): short runs compared against the
# saved states in test/data/. Delete a reference file to regenerate it.
@testset "Bowl" begin
    α = 0.5
    bowl_file = joinpath(@__DIR__, "../meshes/bowl3D_1.000000e-01_5.000000e-01.msh")
    if !isfile(bowl_file)
        include(joinpath(@__DIR__, "../meshes/mesh_bowl3D.jl"))
        Base.invokelatest(generate_bowl_mesh_3D, 0.1, α)
    end
    mesh = Mesh(bowl_file)
    set_out_dir!(mktempdir())

    H(x) = α*(1 - x[1]^2 - x[2]^2)
    κ_bottom(x) = 1e-2 + exp(-(x[3] + H(x))/(0.1*α))

    # b_surface = nothing: no Dirichlet condition on b (surface flux BC)
    function bowl_fe_data(b_surface; kwargs...)
        b_bcs = b_surface === nothing ? (;) :
                (b_diri_tags = ["surface"], b_diri_vals = [b_surface])
        return FEData(mesh;
            u_diri_tags  = ["bottom", "surface"],
            u_diri_masks = [(true,true,true), (false,false,true)],
            b_bcs..., kwargs...)
    end

    function bowl_regression(name, params, forcings, ts; b_surface)
        fe_data = bowl_fe_data(b_surface)
        inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
        evo_tk = EvolutionToolkit(CPU(), fe_data, params, forcings, ts)
        model = Model(CPU(), params, forcings, fe_data, inv_tk, evo_tk, ts)
        run!(model; n_info=typemax(Int), n_save=Inf)

        datafile = joinpath(@__DIR__, "data/bowl_$name.jld2")
        if !isfile(datafile)
            @warn "Reference $datafile not found; saving current state as the reference"
            save_state(model, datafile)
            return
        end
        jldopen(datafile, "r") do d
            @test norm(model.state.u - d["u"]) / norm(d["u"]) < 1e-3
            @test norm(model.state.b - d["b"]) / max(norm(d["b"]), 1e-10) < 1e-3
        end
    end

    @testset "mixing" begin
        ε, μϱ = 2e-1, 1e1
        params = Parameters(; ε, α, μϱ, N²=1/α, f=x->1.0 + 0.5*x[2], H)
        forcings = Forcings(1.0, κ_bottom, κ_bottom, x->0.0, x->0.0, SurfaceDirichletBC(x->0.0))
        Δt = 1e-4 * μϱ / (α*ε)^2
        ts = BDF2(; t_start=0.0, t_stop=50*Δt, Δt)
        bowl_regression("mixing", params, forcings, ts; b_surface = x->0.0)

        @testset "mean pressure gauge" begin
            # with pressure_gauge = :mean, ∫p dΩ must vanish after invert!
            fe_data = bowl_fe_data(x->0.0; pressure_gauge = :mean)
            inv_tk = InversionToolkit(CPU(), fe_data, params, forcings)
            invert!(inv_tk, randn(fe_data.nb))

            # the solve is over free DOFs only: scatter back and recover the rest
            x_up = zeros(ndofs(fe_data.dh_up))
            x_up[fe_data.free_dofs] .= inv_tk.solver.x
            apply!(x_up, fe_data.ch_up)

            _, cv_p, _ = make_cell_values(fe_data)
            p_range = dof_range(fe_data.dh_up, :p)
            p_integral = 0.0
            vol = 0.0
            for cc in CellIterator(fe_data.dh_up)
                reinit!(cv_p, cc)
                local_p = x_up[celldofs(cc)[p_range]]
                for q in 1:getnquadpoints(cv_p)
                    dΩ = getdetJdV(cv_p, q)
                    p_integral += function_value(cv_p, q, local_p) * dΩ
                    vol += dΩ
                end
            end
            # loose: the constraint is built with a lower quadrature order than this check
            @test abs(p_integral) / vol < 1e-3
        end
    end

    @testset "wind stress" begin
        ε, μϱ = sqrt(1e-1), 1.0
        params = Parameters(; ε, α, μϱ, N²=0.0, f=x->0.5*x[2], H)
        forcings = Forcings(1.0, κ_bottom, κ_bottom, x -> -1e-1*cos(π*x[2]/2), x->0.0,
                            SurfaceDirichletBC(x->0.0))
        Δt = 1e-4 * μϱ / (α*ε)^2
        ts = BDF1(; t_start=0.0, t_stop=20*Δt, Δt)
        bowl_regression("wind", params, forcings, ts; b_surface = x->0.0)
    end

    @testset "Dirichlet b" begin
        ε, μϱ = sqrt(1e-1), 1.0
        b_surface(x) = x[2]
        params = Parameters(; ε, α, μϱ, N²=0.0, f=x->0.5*x[2], H)
        forcings = Forcings(1.0, x->1.0, x->1.0, x->0.0, x->0.0, SurfaceDirichletBC(b_surface))
        Δt = 1e-4 * μϱ / (α*ε)^2
        ts = BDF1(; t_start=0.0, t_stop=20*Δt, Δt)
        bowl_regression("dirichlet", params, forcings, ts; b_surface)
    end

    @testset "surface flux" begin
        params = Parameters(; ε=sqrt(1e-1), α, μϱ=1.0, N²=0.0, f=x->1.0, H)
        forcings = Forcings(1.0, x->1e-2, x->1e-2, x->0.0, x->0.0,
                            SurfaceFluxBC(x -> 1e-3*sin(π*x[1])))
        Δt = 1e-1
        ts = BDF2(; t_start=0.0, t_stop=50*Δt, Δt)
        bowl_regression("surface_flux", params, forcings, ts; b_surface = nothing)
    end
end
