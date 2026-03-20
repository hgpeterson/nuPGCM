function solve(params; eddy_param=false)
    # unpack
    f = params.f
    N² = params.N²
    α = params.α
    ε = params.ε
    μϱ = params.μϱ
    θ = params.θ
    Δt = params.Δt
    T = params.T
    z = params.z
    nz = params.nz
    κ = params.κ

    # initialize ν
    ν = ones(nz)

    # build inversion matrices
    LHS_inversion = build_LHS_inversion(z, ν, params)
    if eddy_param
        # fd_z, fd_zz = make_fd_stencils(z)
    else
        LHS_inversion = lu(LHS_inversion)
    end
    rhs_inversion = zeros(2nz+2)

    # build evolution matrices
    K, rhs_diff = build_diffusion_system(z, κ, N², θ)
    rhs_evolution = zeros(nz)
    rhs_evolution[1] = -N²*cos(θ) # N²cos(θ) + dz(b) = 0 -> dz(b) = -N²cos(θ) at z = -H
    rhs_evolution[nz] = 0 # b = 0 at z = 0

    # BDF1 and BDF2 LHSs
    θ1 = Δt * α^2 * ε^2 / μϱ
    LHS1 = lu(assemble_LHS_evolution(K, θ1))
    θ2 = 2/3 * Δt * α^2 * ε^2 / μϱ
    LHS2 = lu(assemble_LHS_evolution(K, θ2))

    # initial condition
    b = zeros(nz)
    uvp = zeros(2nz+2)
    t = 0
    u = @view uvp[1:nz]
    v = @view uvp[nz+1:2nz]
    Px = @view uvp[2nz+1]
    Py = @view uvp[2nz+2]

    # copies for timesteps
    b_old = zeros(nz)
    b_old_old = zeros(nz)
    uvp_old = zeros(2nz+2)
    uvp_old_old = zeros(2nz+2)
    u_old = @view uvp_old[1:nz]
    u_old_old = @view uvp_old_old[1:nz]

    # run
    n_steps = Int64(T ÷ Δt)
    @showprogress for i in 1:n_steps
        # sync before update
        b_old .= b
        uvp_old .= uvp

        # step forward
        if i == 1
            # BDF1
            rhs_evolution[2:end-1] .= (b_old + θ1*rhs_diff - Δt*u_old*N²*sin(θ))[2:end-1]
            ldiv!(b, LHS1, rhs_evolution)
        else
            # BDF2
            rhs_evolution[2:end-1] .= (4/3*b_old - 1/3*b_old_old + θ2*rhs_diff - 2/3*Δt*(2*u_old - u_old_old)*N²*sin(θ))[2:end-1]
            ldiv!(b, LHS2, rhs_evolution)
        end
        t += Δt

        # invert
        if eddy_param
            ν = abs.(f^2 * cos(θ)^2 ./ ( α * (N² .+ cos(θ)*differentiate(b, z)) ))
            ν[ν .> 1e2 ] .= 1e2
            LHS_inversion = build_LHS_inversion(z, ν, params)
            # LHS_inversion = build_LHS_inversion!(LHS_inversion, fd_z, fd_zz, z, ν, params)
        end
        update_rhs_inversion!(rhs_inversion, b, params)
        uvp .= LHS_inversion\rhs_inversion
        # ldiv!(uvp, LHS_inversion, rhs_inversion)

        # move old -> old old
        b_old_old .= b_old
        uvp_old_old .= uvp_old
    end

    return u, v, Px, Py, b, t, z
end
