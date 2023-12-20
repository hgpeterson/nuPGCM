function H_quad_pts(m::ModelSetup3D, g::Grid)
    H = m.geom.H
    nσ = m.geom.nσ
    H_qp = zeros(g.nt, length(g.el.quad_wts))
    for k_sfc ∈ 1:H.g.nt
        k_ws = get_k_ws(k_sfc, nσ)
        for i_quad ∈ eachindex(g.el.quad_wts)
            H_qp[k_ws, i_quad] .= H(g.el.quad_pts[i, :], k_sfc)
        end
    end
    return H_qp
end

# integrate
# ∫(m::ModelSetup3D, u::AbstractField) = sum(m.evolution.HM*u.values)
function ∫(m::ModelSetup3D, u::AbstractField) 
    H_qp = H_quad_pts(m, u.g)
    φ_qp = φ_quad_pts(u.g)
    u_qp = sum(u[u.g.t].*φ_qp, dims=2)[:, 1, :]
    return ∫(u_qp.*H_qp, u.g)
end
