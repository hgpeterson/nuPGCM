# derivative shortcuts (TF coordinates)
∂ξ(u::AbstractField, x) = ∂(u, x, 1)
∂η(u::AbstractField, x) = ∂(u, x, 2)
∂σ(u::AbstractField, x) = ∂(u, x, 3)
∂ξ(u::AbstractField, x, k) = ∂(u, x, k, 1)
∂η(u::AbstractField, x, k) = ∂(u, x, k, 2)
∂σ(u::AbstractField, x, k) = ∂(u, x, k, 3)

function H_quad_pts(m::ModelSetup3D, g::Grid)
    H = m.geom.H
    nσ = m.geom.nσ
    H_qp = zeros(g.nt, length(g.el.quad_wts))
    for k_sfc ∈ 1:H.g.nt
        k_ws = get_k_ws(k_sfc, nσ)
        for i_quad ∈ eachindex(g.el.quad_wts)
            H_qp[k_ws, i_quad] .= H(g.el.quad_pts[i_quad, :], k_sfc)
        end
    end
    return H_qp
end

# integrate
# ∫(m::ModelSetup3D, u::AbstractField) = sum(m.evolution.HM*u.values)
function ∫(m::ModelSetup3D, u::AbstractField) 
    H_qp = H_quad_pts(m, u.g)
    u_qp = u[u.g.t]*u.g.φ_qp
    return ∫(u_qp.*H_qp, u.g)
end
