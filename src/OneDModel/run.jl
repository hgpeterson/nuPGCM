function solve(params)
    # unpack
    f = params.f
    α = params.α
    ε = params.ε
    μϱ = params.μϱ
    θ = params.θ
    nz = params.nz
    H = params.H
    Δt = params.Δt
    T = params.T

    # grid
    z = params.H*chebyshev_nodes(nz)

    # forcing
    ν = ones(nz)
    κ = @. 1 + (1e2 - 1)*exp(-(z + H)/(α/8))

    # parameter
    Γ = 1 + α^2*tan(θ)^2

    # build inversion matrices
    LHS_inversion = build_LHS_inversion(z, ν, params)
    LHS_inversion = lu(LHS_inversion)
    rhs_inversion = zeros(2nz+2)

    # build evolution matrices
    K, rhs_diff = build_diffusion_system(z, κ, Γ)
    rhs_evolution = zeros(nz)
    rhs_evolution[1] = -1/Γ # dz(b) at z = -H
    rhs_evolution[nz] = 0 # b at z = 0

    # BDF1 and BDF2 LHSs
    # θ1 = Δt * α^2 * ε^2 / μϱ
    θ1 = Δt * ε^2 / μϱ
    LHS1 = lu(assemble_LHS_evolution(K, θ1))
    # θ2 = 2/3 * Δt * α^2 * ε^2 / μϱ
    θ2 = 2/3 * Δt * ε^2 / μϱ
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
    for i in 1:n_steps
        if mod(i, 100) == 0
            @info "$i/$n_steps"
        end
        # sync before update
        b_old .= b
        uvp_old .= uvp

        # step forward
        if i == 1
            # BDF1
            rhs_evolution[2:end-1] .= (b_old + θ1*rhs_diff - Δt*u_old*tan(θ))[2:end-1]
            ldiv!(b, LHS1, rhs_evolution)
        else
            # BDF2
            rhs_evolution[2:end-1] .= (4/3*b_old - 1/3*b_old_old + θ2*rhs_diff - 2/3*Δt*(2*u_old - u_old_old)*tan(θ))[2:end-1]
            ldiv!(b, LHS2, rhs_evolution)
        end
        t += Δt

        # invert
        # ν = f^2 ./ ( α * (1 .+ differentiate(b, z)) )
        # ν = f^2 ./ ( (1 .+ differentiate(b, z)) )
        # LHS_inversion = build_LHS_inversion(z, ν, params)
        update_rhs_inversion!(rhs_inversion, b, params)
        uvp .= LHS_inversion\rhs_inversion
        # ldiv!(uvp, LHS_inversion, rhs_inversion)

        # move old -> old old
        b_old_old .= b_old
        uvp_old_old .= uvp_old
    end

    return u, v, Px, Py, b, t, z
end
