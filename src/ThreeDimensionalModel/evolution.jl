function reset_BCs!(m::ModelSetup2DPG, s::ModelState2DPG, RHS::Array{Float64,2})
    # boundary fluxes: dσ(b)/H at σ = -1, 0
    if m.bl
        RHS[:, 1] = s.χ[:, 1].*∂ξ(m, s.b[:, 1])./m.κ[:, 1]
        RHS[:, m.nσ] = s.χ[:, m.nσ].*∂ξ(m, s.b[:, m.nσ])./m.κ[:, m.nσ] .+ m.N2[:, m.nσ]
    else
        RHS[:, 1] .= 0
        RHS[:, m.nσ] .= m.N2[:, m.nσ]
    end
end

function evolve!(m::ModelSetup3DPG, s::ModelState3DPG, t_final::Real, t_plot::Real)
    # timestep
    n_steps = Int64(t_final/m.Δt)
    n_steps_plot = Int64(t_plot/m.Δt)

    # get LHS diffusion matrix
    D_LHS = nuPGCM.get_D_LHS(m.κ, m.σ, m.H, m.Δt)

    # get derivative matrix
    Dσ = nuPGCM.get_Dσ(m.σ)

    # main loop
    t = 0
    i_img = 0
    for i=1:n_steps
        diff = (Dσ*(m.κ.*(Dσ*(s.b'))')')'
        RHS = s.b + m.Δt/2*diff
        RHS[:, 1] .= 0
        RHS[:, m.nσ] .= m.N²[:, m.nσ]
        s.b[:, :] = reshape(D_LHS\RHS[:], m.np, m.nσ)
        s.i[1] = i + 1
        t += m.Δt

        if i % n_steps_plot == 0
            ξ_slice = (-m.Lx + 1e4):m.Lx/2^7:(m.Lx - 1e4)
            η₀ = 0
            ax = plot_ξ_slice(m, s, s.b, ξ_slice, η₀; clabel=L"Buoyancy $b$ (m s$^{-2}$)", contours=false)
            ax.set_xlim([-m.Lx/1e3, m.Lx/1e3])
            ax.set_ylim([-maximum(m.H)/1e3, 0])
            savefig("images/b$i_img.png")
            println("images/b$i_img.png")
            plt.close()
            i_img += 1
        end
    end
end
