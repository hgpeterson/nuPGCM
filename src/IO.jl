"""
    save_state(ux, uy, uz, p, b, t; fname="state.h5")
"""
function save_state(ux, uy, uz, p, b, t; fname="state.h5")
    h5open(fname, "w") do file
        write(file, "ux", ux.free_values)
        write(file, "uy", uy.free_values)
        write(file, "uz", uz.free_values)
        write(file, "p", Vector(p.free_values))
        write(file, "b", b.free_values)
        write(file, "t", t)
    end
    @printf("State saved to '%s'.\n", fname)
end

"""
    save_state_vtu(ux, uy, uz, p, b, Ω; fname="state.vtu")
"""
function save_state_vtu(ux, uy, uz, p, b, Ω; fname="state.vtu")
    writevtk(Ω, fname, cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])
    @printf("State saved to '%s'.\n", fname)
end

"""
    ux, uy, uz, p, b, t = load_state(fname::AbstractString)
"""
function load_state(fname::AbstractString)
    file = h5open(fname, "r")
    ux = read(file, "ux")
    uy = read(file, "uy")
    uz = read(file, "uz")
    p = read(file, "p")
    b = read(file, "b")
    t = read(file, "t")
    close(file)
    @printf("State loaded from '%s'.\n", fname)
    return ux, uy, uz, p, b, t
end