################################################################################
# General utility functions
################################################################################

"""
    Dξ = get_Dξ(ξ, L, periodic)

Compute the ξ derivative matrix.
"""
function get_Dξ(ξ::Array{Float64,1}, L::Float64, periodic::Bool)
    nξ = size(ξ, 1)
    
    Dξ = Tuple{Int64,Int64,Float64}[]

    # Insert stencil in matrices for each node point
    for i=1:nξ
        if i == 1
            # left 
            if periodic
                fd_ξ = mkfdstencil([ξ[nξ] - L, ξ[1], ξ[2]], ξ[1], 1) 
                push!(Dξ, (i, nξ, fd_ξ[1]))
                push!(Dξ, (i, 1,  fd_ξ[2]))
                push!(Dξ, (i, 2,  fd_ξ[3]))
            else
                # ghost point at ξ = 0 where derivative is zero:
                #   fd_ξ0[1]*f[0] = -fd_ξ0[2]*f[1] - fd_ξ0[3]*f[2]
                #   -> dξ(f) at ξ[1] = fd_ξ[1]*f[0] + fd_ξ[2]*f[1] + fd_ξ[3]*f[2]
                #                    = -fd_ξ[1]*(fd_ξ0[2]*f[1] + fd_ξ0[3]*f[2])/fd_ξ0[1] + fd_ξ[2]*f[1] + fd_ξ[3]*f[2]
                #                    = (fd_ξ[2] - fd_ξ[1]*fd_ξ0[2]/fd_ξ0[1]) * f[1] + (fd_ξ[3] - fd_ξ[1]fd_ξ0[3]/fd_ξ0[1]) * f[2]
                fd_ξ0 = mkfdstencil([0, ξ[1], ξ[2]], 0.0, 1) 
                fd_ξ = mkfdstencil([0, ξ[1], ξ[2]], ξ[1], 1) 
                push!(Dξ, (i, 1, fd_ξ[2] - fd_ξ[1]*fd_ξ0[2]/fd_ξ0[1]))
                push!(Dξ, (i, 2, fd_ξ[3] - fd_ξ[1]*fd_ξ0[3]/fd_ξ0[1]))

                # fd_ξ = mkfdstencil(ξ[1:3], ξ[1], 1) 
                # push!(Dξ, (i, 1, fd_ξ[1]))
                # push!(Dξ, (i, 2, fd_ξ[2]))
                # push!(Dξ, (i, 3, fd_ξ[3]))
            end
        elseif i == nξ
            # right
            if periodic
                fd_ξ = mkfdstencil([ξ[nξ-1], ξ[nξ], ξ[1] + L], ξ[nξ], 1)
                push!(Dξ, (i, nξ-1, fd_ξ[1]))
                push!(Dξ, (i, nξ,   fd_ξ[2]))
                push!(Dξ, (i, 1,    fd_ξ[3]))
            else
                fd_ξ = mkfdstencil(ξ[nξ-2:nξ], ξ[nξ], 1)
                push!(Dξ, (i, nξ-2, fd_ξ[1]))
                push!(Dξ, (i, nξ-1, fd_ξ[2]))
                push!(Dξ, (i, nξ,   fd_ξ[3]))
            end
        else
            # interior
            fd_ξ = mkfdstencil(ξ[i-1:i+1], ξ[i], 1)
            push!(Dξ, (i, i-1, fd_ξ[1]))
            push!(Dξ, (i, i,   fd_ξ[2]))
            push!(Dξ, (i, i+1, fd_ξ[3]))
        end
    end

    # Create CSC sparse matrix from matrix elements
    Dξ = sparse((x->x[1]).(Dξ), (x->x[2]).(Dξ), (x->x[3]).(Dξ), nξ, nξ)

    return Dξ
end

"""
    Dσ = get_Dσ(σ)

Compute the σ derivative matrix.
"""
function get_Dσ(σ::Array{Float64,1})
    nσ = size(σ, 1)

    Dσ = Tuple{Int64,Int64,Float64}[]

    # Insert stencil in matrices for each node point
    for j=1:nσ
        if j == 1 
            # bottom 
            fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
            push!(Dσ, (j, 1, fd_σ[1]))
            push!(Dσ, (j, 2, fd_σ[2]))
            push!(Dσ, (j, 3, fd_σ[3]))
        elseif j == nσ
            # top 
            fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
            push!(Dσ, (j, nσ-2, fd_σ[1]))
            push!(Dσ, (j, nσ-1, fd_σ[2]))
            push!(Dσ, (j, nσ,   fd_σ[3]))
        else
            # interior
            fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            push!(Dσ, (j, j-1, fd_σ[1]))
            push!(Dσ, (j, j,   fd_σ[2]))
            push!(Dσ, (j, j+1, fd_σ[3]))
        end
    end

    # Create CSC sparse matrix from matrix elements
    Dσ = sparse((x->x[1]).(Dσ), (x->x[2]).(Dσ), (x->x[3]).(Dσ), nσ, nσ)

    return Dσ
end

"""
    fξ = ξDerivative(m, field)

Compute dξ(`field`) in terrian-following coordinates.
"""
function ξDerivative(m::ModelSetup2DPG, field::Array{Float64})
    return m.Dξ*field
end

"""
    fσ = σDerivative(m, field)

Compute dσ(`field`) in terrian-following coordinates.
"""
function σDerivative(m::ModelSetup2DPG, field::Array{Float64,2})
    return (m.Dσ*field')'
end

"""
    fx = xDerivative(m, field)

Compute dx(`field`) in terrian-following coordinates.
Note: dx() = dξ() - dx(H)*σ*dσ()/H
"""
function xDerivative(m::ModelSetup2DPG, field::Array{Float64,2})
    # dξ(field)
    fx = ξDerivative(m, field)

    # -dx(H)*σ*dσ(field)/H
    fx -= repeat(m.Hx./m.H, 1, m.nσ).*repeat(m.σ', m.nξ, 1).*σDerivative(m, field)

    return fx
end

"""
    fz = zDerivative(m, field)

Compute dz(`field`) in terrian-following coordinates.
Note: dz() = dσ()/H
"""
function zDerivative(m::ModelSetup2DPG, field::Array{Float64,2})
    # dσ(field)/H
    fz = σDerivative(m, field)./repeat(m.H, 1, m.nσ)
    return fz
end

"""
    u, v, w = transform_from_TF(m, s)

Transform from terrain-following coordinates to cartesian coordinates.
"""
function transform_from_TF(m::ModelSetup2DPG, s::ModelState2DPG)
    u = s.uξ
    v = s.uη
    w = s.uσ.*repeat(m.H, 1, m.nσ) + repeat(m.σ', m.nξ, 1).*repeat(m.Hx, 1, m.nσ).*s.uξ
    return u, v, w
end

"""
    log_params(ofile, text)

Write `text` to `ofile` and print it.
"""
function log_params(ofile::IOStream, text::String)
    write(ofile, string(text, "\n"))
    println(text)
end

"""
    save_setup_2DPG(m)

Save .h5 file for parameters.
"""
function save_setup_2DPG(m::ModelSetup2DPG)
    save_file = string(out_folder, "setup.h5")
    file = h5open(save_file, "w")
    write(file, "bl", m.bl)
    write(file, "f", m.f)
    write(file, "ξVariation", m.ξVariation)
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
    log_params(ofile, string("Variations in ξ: ", m.ξVariation))
    log_params(ofile, string("Coordinates:     ", m.coords))
    log_params(ofile, string("Periodic:        ", m.periodic))
    log_params(ofile, @sprintf("\nEkman layer thickness ~ %1.2f m", sqrt(2*m.ν[1, 1]/abs(m.f))))
    log_params(ofile, @sprintf("          z[2] - z[1] ~ %1.2f m\n", m.z[1, 2] - m.z[1, 1]))
    close(ofile)
end

"""
    m = load_setup_2DPG(filename)

Load .h5 setup file given by `filename`.
"""
function load_setup_2DPG(filename::String)
    file = h5open(filename, "r")
    bl = read(file, "bl")
    f = read(file, "f")
    ξVariation = read(file, "ξVariation")
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
    return ModelSetup2DPG(bl, f, ξVariation, L, nξ, nσ, coords, periodic, ξ, σ, x, z, H, Hx, ν, κ, N2, Δt)
end

"""
    save_state_2DPG(s, iSave)

Save .h5 state file.
"""
function save_state_2DPG(s::ModelState2DPG, iSave::Int64)
    save_file = @sprintf("%sstate%d.h5", out_folder, iSave)
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

"""
    s = load_state_2DPG(filename)

Load .h5 state file given by `filename`.
"""
function load_state_2DPG(filename::String)
    file = h5open(filename, "r")
    b = read(file, "b")
    χ = read(file, "χ")
    uξ = read(file, "uξ")
    uη = read(file, "uη")
    uσ = read(file, "uσ")
    i = read(file, "i")
    close(file)
    s = ModelState2DPG(b, χ, uξ, uη, uσ, i)
    return s
end

"""
    χ, b = constructFullSolution(m, s, z, ix)

Construct full solutions `χ` = χI + χB and `b` = bI + bB at ξ = ξ[ix] from BL theory.
The full solutions exist on the new grid `z`.
"""
function get_full_soln(m::ModelSetup2DPG, s::ModelState2DPG, z::Vector{Float64}, ix::Int64)
    # interior vars
    bI = s.b[ix, :]
    χI = s.χ[ix, :]

    # BL thickness 
    bIξ = ξDerivative(m, s.b)
    δ = sqrt(2*m.ν[ix, 1]/abs(m.f))
    μ = m.ν[ix, 1]/m.κ[ix, 1]
    S = -1/m.f^2 * m.Hx[ix]*bIξ[ix, 1]
    q = 1/δ * (1 + μ*S)^(1/4)

    # interpolate onto new grid 
    χI_fine = Spline1D(m.z[ix, :] .- m.z[ix, 1], χI)(z)
    bI_fine = Spline1D(m.z[ix, :] .- m.z[ix, 1], bI)(z)

    # BL correction
    χB = @. -χI[1]*exp(-q*z)*(cos(q*z) + sin(q*z))
    bB = cumtrapz(χB*bIξ[ix, 1]/m.κ[ix, 1], z) .- trapz(χB*bIξ[ix, 1]/m.κ[ix, 1], z) 

    # full sol
    χ = χI_fine + χB
    b = bI_fine + bB
    return χ, b
end