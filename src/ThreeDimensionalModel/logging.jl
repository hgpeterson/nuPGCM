# """
#     log_params(ofile, text)

# Write `text` to `ofile` and print it.
# """
# function log_params(ofile::IOStream, text::String)
#     write(ofile, string(text, "\n"))
#     println(text)
# end

"""
    save_setup(m)

Save .h5 file for parameters.
"""
function save_setup(m::ModelSetup3DPG)
    save_file = string(out_folder, "setup.h5")
    file = h5open(save_file, "w")
    write(file, "bl", m.bl)
    write(file, "f", m.f)
    write(file, "n_nodes", m.n_nodes)
    write(file, "nσ", m.nσ)
    write(file, "p", m.p)
    write(file, "t", m.t)
    write(file, "σ", m.σ)
    write(file, "H", m.H)
    write(file, "Hx", m.Hx)
    write(file, "Hy", m.Hy)
    write(file, "ν", m.ν)
    write(file, "κ", m.κ)
    write(file, "N2", m.N2)
    write(file, "Δt", m.Δt)
    close(file)
    println(save_file)

    # log 
    ofile = open(string(out_folder, "out.txt"), "w")
    log_params(ofile, "\n3D νPGCM with Parameters\n")
    log_params(ofile, @sprintf("n_nodes = %d", m.n_nodes))
    log_params(ofile, @sprintf("nσ      = %d\n", m.nσ))
    log_params(ofile, @sprintf("f       = %1.1e s-1", m.f))
    log_params(ofile, @sprintf("Nₘₐₓ    = %1.1e s-1", sqrt(maximum(m.N2))))
    log_params(ofile, @sprintf("Δt      = %.2f days", m.Δt/secs_in_day))
    log_params(ofile, string("\nBL:              ", m.bl))
    log_params(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*m.ν[1, 1]/abs(m.f))))
    log_params(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", m.H[1, 1]*m.σ[2] - m.σ[1]))
    close(ofile)
end

"""
    m = load_setup_3D(filename)

Load .h5 setup file given by `filename`.
"""
function load_setup_3D(filename::String)
    file = h5open(filename, "r")
    bl = read(file, "bl")
    f = read(file, "f")
    n_nodes = read(file, "n_nodes")
    nσ = read(file, "nσ")
    p = read(file, "p")
    t = read(file, "t")
    σ = read(file, "σ")
    H = read(file, "H")
    Hx = read(file, "Hx")
    Hy = read(file, "Hy")
    ν = read(file, "ν")
    κ = read(file, "κ")
    N2 = read(file, "N2")
    Δt = read(file, "Δt")
    return ModelSetup3DPG(bl, f, n_nodes, nσ, p, t, σ, H, Hx, Hy, ν, κ, N2, Δt)
end

"""
    save_state(s, iSave)

Save .h5 state file.
"""
function save_state(s::ModelState3DPG, iSave::Int64)
    save_file = @sprintf("%sstate%d.h5", out_folder, iSave)
    file = h5open(save_file, "w")
    write(file, "b", s.b)
    write(file, "Ψ", s.Ψ)
    write(file, "uξ", s.uξ)
    write(file, "uη", s.uη)
    write(file, "uσ", s.uσ)
    write(file, "i", s.i)
    close(file)
    println(save_file)
end

"""
    s = load_state_3D(filename)

Load .h5 state file given by `filename`.
"""
function load_state_3D(filename::String)
    file = h5open(filename, "r")
    b = read(file, "b")
    Ψ = read(file, "Ψ")
    uξ = read(file, "uξ")
    uη = read(file, "uη")
    uσ = read(file, "uσ")
    i = read(file, "i")
    close(file)
    s = ModelState3DPG(b, Ψ, uξ, uη, uσ, i)
    return s
end