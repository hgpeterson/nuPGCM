################################################################################
# PG evolution functions
################################################################################

"""
    D = get_diffusion_matrix(ξ, σ, κ, H)

Compute the matrices needed for evolution equation integration.
"""
function get_diffusion_matrix(ξ::Array{Float64,1}, σ::Array{Float64,1}, κ::Array{Float64,2},  H::Array{Float64,1})
    nξ = size(ξ, 1)
    nσ = size(σ, 1)
    n_pts = nξ*nσ

    umap = reshape(1:n_pts, nξ, nσ)    
    D = Tuple{Int64,Int64,Float64}[]         # diffusion operator matrix (with boundary flux conditions)

    # Main loop, insert stencil in matrices for each node point
    for i=1:nξ
        # interior nodes only for operators
        for j=2:nσ-1
            row = umap[i, j] 

            # dσ stencil
            fd_σ = mkfdstencil(σ[j-1:j+1], σ[j], 1)
            κ_σ = sum(fd_σ.*κ[i, j-1:j+1])

            # dσσ stencil
            fd_σσ = mkfdstencil(σ[j-1:j+1], σ[j], 2)

            # diffusion term: dσ(κ*dσ(b))/H^2 = 1/H^2*(dσ(κ)*dσ(b) + κ*dσσ(b))
            push!(D, (row, umap[i, j-1], (κ_σ*fd_σ[1] + κ[i, j]*fd_σσ[1])/H[i]^2))
            push!(D, (row, umap[i, j],   (κ_σ*fd_σ[2] + κ[i, j]*fd_σσ[2])/H[i]^2))
            push!(D, (row, umap[i, j+1], (κ_σ*fd_σ[3] + κ[i, j]*fd_σσ[3])/H[i]^2))
        end

        # flux at boundaries: bottom
        row = umap[i, 1] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[1:3], σ[1], 1)
        # flux term: dσ(b)/H = ...
        push!(D, (row, umap[i, 1], fd_σ[1]/H[i]))
        push!(D, (row, umap[i, 2], fd_σ[2]/H[i]))
        push!(D, (row, umap[i, 3], fd_σ[3]/H[i]))

        # flux at boundaries: top
        row = umap[i, nσ] 
        # dσ stencil
        fd_σ = mkfdstencil(σ[nσ-2:nσ], σ[nσ], 1)
        # flux term: dσ(b)/H = ...
        push!(D, (row, umap[i, nσ-2], fd_σ[1]/H[i]))
        push!(D, (row, umap[i, nσ-1], fd_σ[2]/H[i]))
        push!(D, (row, umap[i, nσ],   fd_σ[3]/H[i]))
    end

    # Create CSC sparse matrix from matrix elements
    D = sparse((x->x[1]).(D), (x->x[2]).(D), (x->x[3]).(D), n_pts, n_pts)

    return D
end

"""
    LHS = get_evolution_LHS(m, a)

Generate the left-hand side matrix for the evolution problem with flux boundary conditions on the boundaries.
"""
function get_evolution_LHS(m::ModelSetup2DPG, a::Real)
    # bottom and top boundaries in 1D
    umap = reshape(1:m.nξ*m.nσ, m.nξ, m.nσ)    
    bottomBdy = umap[:, 1]
    topBdy = umap[:, m.nσ]

    # LHS matrix
    LHS = I - a*m.Δt*m.D 

    # no flux boundaries
    LHS[bottomBdy, :] = m.D[bottomBdy, :]
    LHS[topBdy, :] = m.D[topBdy, :]

    return lu(LHS)
end

"""
    stages, c, A_ex, A_im = get_RK_table(order) 

Returns number of `stages`, fractional time steps `c`, explicity coefficients `A_ex`,
and implicit coefficients `A_im` for Runge-Kutta scheme of order `order`. Implemented
options:

    order | description    
    -------------------
    111     1st-order 1-stage DIRK+ERK scheme [Ascher 1997 sec 2.1]
    222     2nd-order 2-stage DIRK+ERK scheme [Ascher 1997 sec 2.6]
    443     3rd-order 4-stage DIRK+ERK scheme [Ascher 1997 sec 2.8]
"""
function get_RK_table(order::String)
    if order == "111"
        # number of stages
        stages = 1

        # time steps between stages
        c = [0., 1]

        # explicit coefficients
        A_ex = [  0.    0
                  1     0]

        # implicit coefficients
        A_im = [0.  0;
                0   1]

    elseif order == "222"
        # number of stages
        stages = 2

        # useful variables
        γ = (2 - sqrt(2)) / 2
        δ = 1 - 1 / γ / 2

        # time steps between stages
        c = [0, γ, 1]

        # explicit coefficients
        A_ex = [0  0  0;
                γ  0  0;
                δ 1-δ 0]

        # implicit coefficients
        A_im = [0  0  0;
                0  γ  0;
                0 1-γ γ]
    elseif order == "443"
        # number of stages
        stages = 4

        # time steps between stages
        c = [0., 1/2, 2/3, 1/2, 1]

        # explicit coefficients
        A_ex = [  0.    0    0    0  0;
                 1/2    0    0    0  0;
                11/18  1/18  0    0  0;
                 5/6  -5/6  1/2   0  0;
                 1/4   7/4  3/4 -7/4 0]

        # implicit coefficients
        A_im = [0.  0    0   0   0 ;
                0  1/2   0   0   0 ;
                0  1/6  1/2  0   0 ;
                0 -1/2  1/2 1/2  0 ;
                0  3/2 -3/2 1/2 1/2]
    else
        error("Order ", order, "not implemented.")
    end

    return stages, c, A_ex, A_im
end

    
function take_RK_step!(m::ModelSetup2DPG, s::ModelState2DPG, stages::Int64, c::Array{Float64,1}, A_ex::Array{Float64,2}, A_im::Array{Float64,2}, advFunc::Function, LHS::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64}; bl=false)
    # timestep 
    k = m.Δt

    # current solution
    b0 = s.b

    # explict and implicit contrbutions at each stage 
    K_ex = zeros(stages+1, m.nξ, m.nσ)
    K_im = zeros(stages, m.nξ, m.nσ)
        
    # start with explicit contribution right now
    K_ex[1, :, :] = advFunc(b0)

    # solve for each stage
    for i=1:stages
        RHS = b0 + k*A_ex[i+1, i]*K_ex[i, :, :]
        for j=1:i-1
            RHS += k*(A_im[i+1, j+1]*K_im[j, :, :] + A_ex[i+1, j]*K_ex[j, :, :])
        end
        reset_BCs!(m, s, RHS; bl)
        bi = reshape(LHS\RHS[:], m.nξ, m.nσ)
        K_im[i, :, :] = reshape(m.D*bi[:], m.nξ, m.nσ)
        K_ex[i+1, :, :] = advFunc(bi)
    end    

    # sum up
    b = b0
    for j=1:stages
        b += k*(A_im[end, j+1]*K_im[j, :, :] + A_ex[end, j]*K_ex[j, :, :])
    end
    b += k*A_ex[end, stages+1]*K_ex[stages+1, :, :]

    # update s.b
    s.b[:, :] = b
end

"""
    reset_BCs!(m, s, RHS; bl)

Modify the right-hand side vector `RHS` to include boundary conditions at the top and bottom.
"""
function reset_BCs!(m::ModelSetup2DPG, s::ModelState2DPG, RHS::Array{Float64,2}; bl=false)
    # boundary fluxes: dσ(b)/H at σ = -1, 0
    if bl
        RHS[:, 1] = s.χ[:, 1].*ξDerivative(m, s.b[:, 1])./m.κ[:, 1]
        RHS[:, m.nσ] = s.χ[:, m.nσ].*ξDerivative(m, s.b[:, m.nσ])./m.κ[:, m.nσ] .+ m.N2[:, m.nσ]
    else
        RHS[:, 1] .= 0
        RHS[:, m.nσ] .= m.N2[:, m.nσ]
    end
end

"""
    evolve!(m, s, t_final, t_plot, t_save; bl=false)

Solve evoluion equation for `b` and update model state.
If `bl` set to `true`, use boundary layer theory inversion and boundary conditions. 
"""
function evolve!(m::ModelSetup2DPG, s::ModelState2DPG, t_final::Real, t_plot::Real, t_save::Real; bl=false)
    # grid points
    n_pts = m.nξ*m.nσ

    # timestep
    n_steps = Int64(t_final/m.Δt)
    n_steps_plot = Int64(t_plot/m.Δt)
    n_steps_save = Int64(t_save/m.Δt)

    # save initial state
    i_save = 0
    save_state_2DPG(s, i_save)
    i_save += 1
    
    # plot initial state
    i_img = 0
    plot_state_2DPG(m, s, i_img)
    i_img += 1

    # store previous buoyancy field for timestepping scheme
    b_prev = zeros(m.nξ, m.nσ)

    # get LHS matrix for CNAB
    LHS = get_evolution_LHS(m, 1/2)

    # stages, c, A_ex, A_im = get_RK_table("111")
    # stages, c, A_ex, A_im = get_RK_table("222")
    # stages, c, A_ex, A_im = get_RK_table("443")
    # LHS = get_evolution_LHS(m, A_im[2, 2])

    # if you want to check CFL
    dξ = m.L/m.nξ
    dσ = zeros(m.nξ, m.nσ)
    σσ = repeat(m.σ', m.nξ, 1)
    dσ[:, 1:end-1] = σσ[:, 2:end] - σσ[:, 1:end-1]
    dσ[:, end] = dσ[:, end-1]

    # main loop
    t = 0
    for i=1:n_steps
        # explicit timestep for advection
        function advection_RHS(b)
            χ, uξ, uη, uσ, U = invert(m, b; bl=bl)
            if m.ξVariation
                return -uξ.*ξDerivative(m, b) .- uσ.*σDerivative(m, b)
            else
                return -uσ.*σDerivative(m, b)
            end
        end

        if i == 1
            # first step: CNAB1
            RHS = s.b + m.Δt*(advection_RHS(s.b) + 1/2*reshape(m.D*s.b[:], m.nξ, m.nσ)) # right-hand-side
            reset_BCs!(m, s, RHS; bl=bl) # modify RHS to implement boundary conditions
            b_prev = s.b # store previoius step for next time
            s.b[:, :] = reshape(LHS\RHS[:], m.nξ, m.nσ)  # solve
            s.i[1] = i + 1 # next step
        else
            # other steps: CNAB2
            RHS = s.b + m.Δt*(3/2*advection_RHS(s.b) - 1/2*advection_RHS(b_prev) + 1/2*reshape(m.D*s.b[:], m.nξ, m.nσ))
            reset_BCs!(m, s, RHS; bl=bl)
            b_prev = s.b 
            s.b[:, :] = reshape(LHS\RHS[:], m.nξ, m.nσ)
            s.i[1] = i + 1
        end
        # take_RK_step!(m, s, stages, c, A_ex, A_im, advFunc, LHS; bl)
        # s.i[1] = i + 1
        t += m.Δt

        # invert buoyancy for flow and save to state
        invert!(m, s; bl=bl)

        # log
        if i % 10 == 0
            println(@sprintf("t = %2.2f yr | i = %d | χₘₐₓ = %.2e m2 s-1", t/secs_in_year, i, maximum(abs.(s.χ))))
            # println(@sprintf("t = %2.2f yr | i = %d | U = %.2e m2 s-1", t/secs_in_year, i, s.χ[1, end]))

            # # CFL stuff
            # uξCFL = minimum(abs.(dξ./s.uξ)) 
            # uσCFL = minimum(abs.(dσ./s.uσ)) 
            # println(@sprintf("CFL: uξ=%.2f days, uσ=%.2f days", uξCFL/secs_in_day, uσCFL/secs_in_day)) 
        end
        
        if i % n_steps_plot == 0
            # plot flow
            plot_state_2DPG(m, s, i_img)
            i_img += 1
        end
        if i % n_steps_save == 0
            save_state_2DPG(s, i_save)
            i_save += 1
        end
    end
end