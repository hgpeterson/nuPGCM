"""
    log_params(ofile, text)

Write `text` to `ofile` and print it.
"""
function log_params(ofile, text)
    write(ofile, string(text, "\n"))
    println(text)
end

"""
    save_setup(m)

Save .h5 file for parameters.
"""
function save_setup(m::ModelSetup1D)
    save_file = string(out_folder, "setup.h5")
    file = h5open(save_file, "w")
    write(file, "bl", m.bl)
    write(file, "f", m.f)
    write(file, "nz", m.nz)
    write(file, "z", m.z)
    write(file, "H", m.H)
    write(file, "θ", m.θ)
    write(file, "ν", m.ν)
    write(file, "κ", m.κ)
    write(file, "κ_z", m.κ_z)
    write(file, "N2", m.N2)
    write(file, "Δt", m.Δt)
    write(file, "transport_constraint", m.transport_constraint)
    write(file, "U", m.U)
    close(file)
    println(save_file)

    # log 
    ofile = open(string(out_folder, "out.txt"), "w")
    if m.bl
        log_params(ofile, "\n1D BL PG Model with Parameters\n")
    else
        log_params(ofile, "\n1D PG Model with Parameters\n")
    end
    log_params(ofile, @sprintf("nz    = %d", m.nz))
    log_params(ofile, @sprintf("H     = %d km", m.H/1000))
    log_params(ofile, @sprintf("θ     = %1.1e rad", m.θ))
    log_params(ofile, @sprintf("f     = %1.1e s-1", m.f))
    log_params(ofile, @sprintf("N     = %1.1e s-1", sqrt(m.N2)))
    log_params(ofile, @sprintf("S     = %1.1e", m.N2/m.f^2*tan(m.θ)^2))
    log_params(ofile, @sprintf("Δt    = %.2f days", m.Δt/secs_in_day))
    log_params(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*m.ν[1]/abs(m.f))))
    log_params(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", m.z[2] - m.z[1]))
    close(ofile)
end

"""
    m = load_setup_1D(filename)

Load .h5 setup file given by `filename`.
"""
function load_setup_1D(filename)
    file = h5open(filename, "r")
    bl = read(file, "bl")
    f = read(file, "f")
    nz = read(file, "nz")
    z = read(file, "z")
    H = read(file, "H")
    θ = read(file, "θ")
    ν = read(file, "ν")
    κ = read(file, "κ")
    κ_z = read(file, "κ_z")
    N2 = read(file, "N2")
    Δt = read(file, "Δt")
    transport_constraint = read(file, "transport_constraint")
    # transport_constraint = true # for some old files
    U = read(file, "U")
    close(file)
    return ModelSetup1D(bl, f, nz, z, H, θ, ν, κ, κ_z, N2, Δt, transport_constraint, U)
end

"""
    save_state(s, i_save)

Save .h5 checkpoint file for state.
"""
function save_state(s::ModelState1D, i_save)
    save_file = @sprintf("%sstate%d.h5", out_folder, i_save)
    file = h5open(save_file, "w")
    write(file, "b", s.b)
    write(file, "χ", s.χ)
    write(file, "u", s.u)
    write(file, "v", s.v)
    write(file, "i", s.i)
    close(file)
    println(save_file)
end

"""
    s = load_state_1D(filename)

Load .h5 state file given by `filename`.
"""
function load_state_1D(filename)
    file = h5open(filename, "r")
    b = read(file, "b")
    χ = read(file, "χ")
    u = read(file, "u")
    v = read(file, "v")
    i = read(file, "i")
    close(file)
    return ModelState1D(b, χ, u, v, i)
end