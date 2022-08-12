function evolve!(m::ModelSetup3DPG, s::ModelState3DPG, t_final::Real, t_plot::Real)
    # timestep
    n_steps = Int64(t_final/m.Δt)
    n_steps_plot = Int64(t_plot/m.Δt)

    # get LHS diffusion matrix
    D_LHS = nuPGCM.get_D_LHS(m.κ, m.σ, m.H, m.Δt)

    # get derivative matrix
    Dσ = get_Dσ(m.σ)

    # main loop
    t = 0
    i_img = 0
    for i=1:n_steps
        # rhs diffusion
        # diff = (Dσ*(m.κ.*(Dσ*s.b')')')'./m.H.^2
        bz = (Dσ*s.b')'./m.H
        bzz = (Dσ*bz')'./m.H
        κz = (Dσ*m.κ')'./m.H
        diff = κz.*bz + m.κ.*bzz

        # I + Δt/2*D
        RHS = s.b + m.Δt/2*diff

        # reset b.c.
        RHS[:, 1] .= 0
        RHS[:, m.nσ] .= m.N²[:, m.nσ]

        # solve
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

            ξ₀ = 2500e3
            η₀ = 0
            b = zeros(m.nσ)
            for j=1:m.nσ
                b[j] = fem_evaluate(m, s.b[:, j], ξ₀, η₀)
            end
            H = fem_evaluate(m, m.H, ξ₀, η₀)
            bz = Dσ*b/H
            fig, ax = subplots()
            ax.plot(bz, m.σ*H/1e3)
            ax.set_xlabel(L"Stratification $\partial_z b$ (s$^{-2}$)")
            ax.set_ylabel(L"Vertical coordinate $z$ (km)")
            savefig("images/bz$i_img.png")
            println("images/bz$i_img.png")
            plt.close()

            i_img += 1
        end
    end
end
