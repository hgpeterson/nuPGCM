# """
#     log_params(ofile, text)

# Write `text` to `ofile` and print it.
# """
# function log_params(ofile::IOStream, text::String)
#     write(ofile, string(text, "\n"))
#     println(text)
# end

"""
    save_setup(m, save_file)

Save .h5 file for parameters.
"""
function save_setup(m::ModelSetup2D, save_file)
    save_file = string(out_folder, save_file)
    file = h5open(save_file, "w")
    write(file, "bl", m.bl)
    write(file, "f", m.f)
    write(file, "no_net_transport", m.no_net_transport)
    write(file, "L", m.L)
    write(file, "nξ", m.nξ)
    write(file, "nσ", m.nσ)
    write(file, "coords", m.coords)
    write(file, "periodic", m.periodic)
    write(file, "ξ", m.ξ)
    write(file, "σ", m.σ)
    write(file, "x", m.x)
    write(file, "z", m.z)
    write(file, "H", m.H)
    write(file, "Hx", m.Hx)
    write(file, "ν", m.ν)
    write(file, "κ", m.κ)
    write(file, "N2", m.N2)
    write(file, "Δt", m.Δt)
    close(file)
    println(save_file)

    # log 
    ofile = open(string(out_folder, "out.txt"), "w")
    log_params(ofile, "\n2D νPGCM with Parameters\n")
    log_params(ofile, @sprintf("nξ    = %d", m.nξ))
    log_params(ofile, @sprintf("nσ    = %d\n", m.nσ))
    log_params(ofile, @sprintf("L     = %d km", m.L/1000))
    log_params(ofile, @sprintf("f     = %1.1e s-1", m.f))
    log_params(ofile, @sprintf("Nₘₐₓ  = %1.1e s-1", sqrt(maximum(m.N2))))
    log_params(ofile, @sprintf("Δt    = %.2f days", m.Δt/secs_in_day))
    log_params(ofile, string("\nBL:              ", m.bl))
    log_params(ofile, string("U = 0:           ", m.no_net_transport))
    log_params(ofile, string("Coordinates:     ", m.coords))
    log_params(ofile, string("Periodic:        ", m.periodic))
    log_params(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*m.ν[1, 1]/abs(m.f))))
    log_params(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", m.z[1, 2] - m.z[1, 1]))
    close(ofile)
end
function save_setup(m::ModelSetup2D)
    save_setup(m, "setup.h5")
end

"""
    m = load_setup_2D(filename)

Load .h5 setup file given by `filename`.
"""
function load_setup_2D(filename)
    file = h5open(filename, "r")
    bl = read(file, "bl")
    # bl = false # for some old files
    f = read(file, "f")
    no_net_transport = read(file, "no_net_transport")
    # no_net_transport = true # for some old files
    L = read(file, "L")
    nξ = read(file, "nξ")
    nσ = read(file, "nσ")
    coords = read(file, "coords")
    periodic = read(file, "periodic")
    ξ = read(file, "ξ")
    σ = read(file, "σ")
    x = read(file, "x")
    z = read(file, "z")
    H = read(file, "H")
    Hx = read(file, "Hx")
    ν = read(file, "ν")
    κ = read(file, "κ")
    N2 = read(file, "N2")
    Δt = read(file, "Δt")
    close(file)
    return ModelSetup2D(bl, f, no_net_transport, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt)
end

"""
    save_state(s, save_file)

Save .h5 state file.
"""
function save_state(s::ModelState2D, save_file::String)
    save_file = string(out_folder, save_file)
    file = h5open(save_file, "w")
    write(file, "b", s.b)
    write(file, "χ", s.χ)
    write(file, "uξ", s.uξ)
    write(file, "uη", s.uη)
    write(file, "uσ", s.uσ)
    write(file, "i", s.i)
    close(file)
    println(save_file)
end
function save_state(s::ModelState2D, iSave::Integer)
    save_state(s, "state$iSave.h5")
end

"""
    s = load_state_2D(filename)

Load .h5 state file given by `filename`.
"""
function load_state_2D(filename)
    file = h5open(filename, "r")
    b = read(file, "b")
    χ = read(file, "χ")
    uξ = read(file, "uξ")
    uη = read(file, "uη")
    uσ = read(file, "uσ")
    i = read(file, "i")
    close(file)
    s = ModelState2D(b, χ, uξ, uη, uσ, i)
    return s
end