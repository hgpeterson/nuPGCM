####
#### Field evaluation helpers (replace Gridap CellField evaluation)
####

"""
    vals = eval_at_points(model, dh, dof_vec, field, points)

Evaluate a scalar FE field at a list of `Vec{3}` points using Ferrite's
`PointEvalHandler`. Returns `NaN` for points outside the domain.
"""
function eval_at_points(model::Model, dh, dof_vec::AbstractVector,
                         field::Symbol, points::Vector{Vec{3, Float64}})
    grid = model.fe_data.mesh.grid
    ph   = PointEvalHandler(grid, points; warn=false)
    vals = evaluate_at_points(ph, dh, dof_vec, field)
    return [isnothing(v) ? NaN : v for v in vals]
end

"""
    val = eval_at_point(model, dh, dof_vec, field, x::Vec{3})

Evaluate a scalar FE field at a single point. Returns `NaN` if outside the domain.
"""
function eval_at_point(model::Model, dh, dof_vec::AbstractVector,
                        field::Symbol, x::Vec{3, Float64})
    v = eval_at_points(model, dh, dof_vec, field, [x])
    return v[1]
end

"""
    H = find_H(model, x, y; tol=1e-8)

Find the depth of the water column at (x, y) by bisection using the buoyancy field.
"""
function find_H(model::Model, x::Real, y::Real; tol=1e-8)
    z_in  = 0.0
    z_out = -1.0
    while abs(z_in - z_out) > tol
        z = (z_in + z_out) / 2
        if isnan(eval_at_point(model, model.fe_data.dh_b, model.state.b,
                                :b, Vec{3}((Float64(x), Float64(y), z))))
            z_out = z
        else
            z_in = z
        end
    end
    return -z_in
end

####
#### Sparsity pattern visualisation
####

"""
    plot_sparsity_pattern(model)

Plot the sparsity pattern of the inversion matrix.
"""
function plot_sparsity_pattern(model::Model)
    A = on_architecture(CPU(), model.inversion.solver.A)
    I, J, _ = findnz(A)
    fig, ax = subplots()
    ax.spy(A, markersize=0.5)
    ax.set_title("Inversion matrix sparsity pattern ($(size(A,1)) × $(size(A,2)), nnz=$(nnz(A)))")
    return fig, ax
end

####
#### Simulation-level plots — stub implementations
####
# These are placeholders; detailed slice/profile plots require postprocessing
# the VTK output in ParaView or a dedicated postprocessing script.

function plot_slice(args...; kwargs...)
    @warn "plot_slice not yet implemented for Ferrite backend; use save_vtk + ParaView"
    return nothing
end

function plot_profiles(args...; kwargs...)
    @warn "plot_profiles not yet implemented for Ferrite backend"
    return nothing
end

function sim_plots(model::Model, t::Real; kwargs...)
    @warn "sim_plots not yet implemented for Ferrite backend"
    return nothing
end

function nan_eval(args...; kwargs...)
    @warn "nan_eval has been replaced by eval_at_point / eval_at_points"
    return NaN
end
