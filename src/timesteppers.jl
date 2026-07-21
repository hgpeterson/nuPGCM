abstract type AbstractTimestepper end

####
#### Backward Difference Formula order 1
####

struct BDF1{RT, T, DT, CF} <: AbstractTimestepper
    t::RT           # Ref to current time
    t_start::T      # start time
    t_stop::T       # stop time
    Δt::DT          # Ref to current timestep
    adaptive::Bool  # whether to use adaptive timestep
    CFL_factor::CF  # factor to multiply by CFL timestep
end

function Base.show(io::IO, ts::BDF1)
    println(io, summary(ts), ":")
    println(io, "├── t: ", ts.t[])
    println(io, "├── t_start: ", ts.t_start)
    println(io, "├── t_stop: ", ts.t_stop)
    println(io, "├── Δt: ", ts.Δt[])
    println(io, "├── adaptive: ", ts.adaptive)
      print(io, "└── CFL_factor: ", ts.CFL_factor)
end

function BDF1(; t_start, t=t_start, t_stop, Δt, adaptive=false, CFL_factor=0.8)
    t_start, t, t_stop, Δt = promote(t_start, t, t_stop, Δt)
   return BDF1(Ref(t), t_start, t_stop, Ref(Δt), adaptive, CFL_factor) 
end

####
#### Backward Difference Formula order 2
####

# NOTE: Adaptive timestepping with BDF2 is not currently implemented, but should be supported in the future
struct BDF2{RT, T, DT} <: AbstractTimestepper
    t::RT           # Ref to current time
    t_start::T      # start time
    t_stop::T       # stop time
    Δt::DT          # Ref to current timestep
end

# TODO: this overload will not be needed once BDF2 supports adaptive timestepping
function Base.getproperty(ts::BDF2, sym::Symbol)
    if sym == :adaptive
        return false
    else
        return getfield(ts, sym)
    end
end

function Base.show(io::IO, ts::BDF2)
    println(io, summary(ts), ":")
    println(io, "├── t: ", ts.t[])
    println(io, "├── t_start: ", ts.t_start)
    println(io, "├── t_stop: ", ts.t_stop)
      print(io, "└── Δt: ", ts.Δt[])
end

function BDF2(; t_start, t=t_start, t_stop, Δt)
    t_start, t, t_stop, Δt = promote(t_start, t, t_stop, Δt)
   return BDF2(Ref(t), t_start, t_stop, Ref(Δt)) 
end

####
#### Generic timestepper functions
####

function Base.summary(ts::AbstractTimestepper)
    t = typeof(ts)
    return "$(parentmodule(t)).$(nameof(t))"
end


"""
    update_t!(ts::AbstractTimestepper)

Advance the time in the timestepper by the current Δt.
"""
function update_t!(ts::AbstractTimestepper)
    ts.t[] += ts.Δt[]
    return ts
end

"""
    status(ts::AbstractTimestepper)

Print status message for timestepper.
"""
function status(ts::AbstractTimestepper)
    @info "t = $(ts.t[]), Δt = $(ts.Δt[])"
end

"""
    update_Δt!(timestepper::AbstractTimestepper, fe_data, u_vec, h_cells; u_min=0.01)

Update Timestepper's Δt based on the CFL condition:

```math
Δt = CFL_factor × min_k ( h_k / max(|u|_k, u_min) )
```

where ``k`` is the cell index. ``h_k`` = h_cells[k]` is some measure of the cell width
(see [compute_h_cells](@ref)). ``|u|_k`` is the maximum speed within a cell, computed
over quadrature points.

`u_min` prevents Δt blowing up for ``u ∼ 0``.
"""
function update_Δt!(timestepper::BDF1, fe_data::FEData, u_vec::AbstractVector, h_cells;
                    u_min=0.01)
    timestepper.adaptive || return timestepper
    Δt = Inf
    for (k, u_k) in enumerate(max_cell_speeds(fe_data, u_vec))
        Δt = min(Δt, h_cells[k] / max(u_k, u_min))
    end
    timestepper.Δt[] = timestepper.CFL_factor * Δt
    return timestepper
end
function update_Δt!(timestepper::BDF2, fe_data::FEData, u_vec::AbstractVector, h_cells;
                    u_min=0.01)
    return timestepper
end

"""
    max_cell_speeds(fe_data::FEData, u_vec) -> Vector

Return the maximum velocity magnitude ``|u|`` per cell, evaluated over the
quadrature points of each cell. `u_vec` is the reduced velocity DOF vector (in
`fe_data.u_dof_indices` ordering, as stored in `State.u`).
"""
function max_cell_speeds(fe_data::FEData, u_vec::AbstractVector)
    dh_up   = fe_data.dh_up
    u_range = dof_range(dh_up, :u)
    ip_u    = Lagrange{RefTetrahedron, fe_data.u_order}()^3
    qr      = QuadratureRule{RefTetrahedron}(2 * fe_data.u_order)
    cv_u    = CellValues(qr, ip_u, IP_GEO)
    nq      = getnquadpoints(cv_u)

    # scatter the reduced velocity vector back into the combined (u, p) ordering
    x_up = zeros(ndofs(dh_up))
    x_up[fe_data.u_dof_indices] .= u_vec

    u_cells = zeros(getncells(Ferrite.get_grid(dh_up)))
    for cc in CellIterator(dh_up)
        reinit!(cv_u, cc)
        ue = x_up[celldofs(cc)[u_range]]
        u_max = 0.0
        for q in 1:nq
            u_max = max(u_max, norm(function_value(cv_u, q, ue)))
        end
        u_cells[cellid(cc)] = u_max
    end
    return u_cells
end