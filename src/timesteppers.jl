abstract type AbstractTimestepper end

"""
    BDF1(; t_start, t=t_start, t_stop, Δt, adaptive=false, CFL_factor=1/(1.7*3*√3))
    BDF2(; t_start, t=t_start, t_stop, Δt, adaptive=false, CFL_factor=1/(1.7*3*√3))

Implicit-explicit backward difference formula of order 1 or 2 with variable timestep.
The first step of `BDF2` is a `BDF1` step.

With `adaptive=true`, `Δt` is the maximum timestep, and the timestep of every step is
the smaller of `Δt` and the CFL limit `CFL_factor / q × h / |u|ₘₐₓ`, where `q` is the
order of the buoyancy elements and `h` is the smallest cell diameter.
"""
struct BDF{order, RT, T, CF} <: AbstractTimestepper
    t::RT           # Ref to current time
    t_start::T      # start time
    t_stop::T       # stop time
    Δt::RT                 # Ref to current timestep
    previous_timestep::RT  # Ref, Inf before the first step
    maximum_timestep::T
    adaptive::Bool         # whether to use adaptive timestep
    CFL_factor::CF         # factor to multiply by CFL timestep
end

const BDF1 = BDF{1}
const BDF2 = BDF{2}

function BDF{order}(; t_start, t=t_start, t_stop, Δt, adaptive=false,
                    CFL_factor=1/(1.7*3*√3)) where order
    t_start, t, t_stop, Δt, previous_timestep = promote(t_start, t, t_stop, Δt, Inf)
    return BDF{order, typeof(Ref(t)), typeof(t), typeof(CFL_factor)}(
        Ref(t), t_start, t_stop, Ref(Δt), Ref(previous_timestep), Δt, adaptive,
        CFL_factor)
end

function Base.show(io::IO, ts::BDF{order}) where order
    println(io, summary(ts), ":")
    println(io, "├── order: ", order)
    println(io, "├── t: ", ts.t[])
    println(io, "├── t_start: ", ts.t_start)
    println(io, "├── t_stop: ", ts.t_stop)
    println(io, "├── Δt: ", ts.Δt[])
    println(io, "├── maximum_timestep: ", ts.maximum_timestep)
    println(io, "├── adaptive: ", ts.adaptive)
      print(io, "└── CFL_factor: ", ts.CFL_factor)
end

function Base.summary(ts::AbstractTimestepper)
    t = typeof(ts)
    return "$(parentmodule(t)).$(nameof(t))"
end

"""
    ω = step_ratio(ts)

Ratio of the current to the previous timestep, zero for `BDF1` and for
the first step of `BDF2`.
"""
step_ratio(ts::BDF{order}) where order = order == 1 ? 0.0 : ts.Δt[] / ts.previous_timestep[]

"""
    effective_timestep(ts)

`Δt (1 + ω) / (1 + 2ω)`, the factor multiplying the right-hand side of `∂ₜb = R` once
the BDF formula is normalized by the coefficient of the new `b`.
"""
function effective_timestep(ts::BDF)
    ω = step_ratio(ts)
    return ts.Δt[] * (1 + ω) / (1 + 2ω)
end

"""
    update_t!(ts::AbstractTimestepper)

Advance the time in the timestepper by the current Δt.
"""
function update_t!(ts::AbstractTimestepper)
    ts.t[] += ts.Δt[]
    ts.previous_timestep[] = ts.Δt[]
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
    update_Δt!(ts, fe_data, u_vec, h_min)

Set the timestep of an adaptive timestepper to
`min(maximum_timestep, CFL_factor / q × h_min / |u|ₘₐₓ)`, where `q` is the order of the buoyancy
elements and `|u|ₘₐₓ` is the maximum speed over the quadrature points.
"""
function update_Δt!(ts::BDF, fe_data::FEData, u_vec::AbstractVector, h_min)
    ts.adaptive || return ts
    u_max = maximum(max_cell_speeds(fe_data, u_vec))
    ts.Δt[] = min(ts.maximum_timestep, ts.CFL_factor / fe_data.b_order * h_min / u_max)
    return ts
end

"""
    max_cell_speeds(fe_data::FEData, u_vec) -> Vector

Return the maximum velocity magnitude ``|u|`` per cell, evaluated over the
quadrature points of each cell. `u_vec` is the reduced velocity DOF vector (in
`fe_data.u_dof_indices` ordering, as stored in `State.u`).
"""
function max_cell_speeds(fe_data::FEData, u_vec::AbstractVector)
    cache = fe_data.cache
    nq, n_nodes = size(cache.phi_u)
    x_up = to_up_vec(fe_data, u_vec)
    u_cells = zeros(size(cache.dofs_u, 2))
    @inbounds for c in eachindex(u_cells), q in 1:nq
        u1 = 0.0; u2 = 0.0; u3 = 0.0
        for j in 1:n_nodes
            φ = cache.phi_u[q, j]
            u1 += φ * x_up[cache.dofs_u[3j - 2, c]]
            u2 += φ * x_up[cache.dofs_u[3j - 1, c]]
            u3 += φ * x_up[cache.dofs_u[3j, c]]
        end
        u_cells[c] = max(u_cells[c], sqrt(u1^2 + u2^2 + u3^2))
    end
    return u_cells
end