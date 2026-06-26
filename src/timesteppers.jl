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
    update_Δt!(timestepper::AbstractTimestepper, u_vec, h_cells; u_min=0.01)

Update Timestepper's Δt based on the CFL condition.

    Δt = CFL_factor × min_K ( h_K / max(|u|_K, u_min) )

`u_vec` is the velocity DOF vector; its element-wise maximum gives a conservative
upper bound on the flow speed. `u_min` prevents Δt blowing up in stagnant regions.
"""
function update_Δt!(timestepper::BDF1, u_vec::AbstractVector, h_cells; u_min=0.01)
    timestepper.adaptive || return timestepper
    u_max = max(maximum(abs, u_vec), u_min)
    timestepper.Δt[] = timestepper.CFL_factor * minimum(h_cells) / u_max
    return timestepper
end
function update_Δt!(timestepper::BDF2, u_vec::AbstractVector, h_cells; u_min=0.01)
    return timestepper
end