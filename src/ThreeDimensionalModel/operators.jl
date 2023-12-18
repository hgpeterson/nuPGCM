# integrate
# ∫(m::ModelSetup3D, u::AbstractField) = sum(ref_el_quad(ξ -> m.geom.H(ξ, get_k_sfc(k, m.geom.nσ))*u(ξ, k)*u.g.J.dets[k], u.g.el) for k=1:u.g.nt)
∫(m::ModelSetup3D, u::AbstractField) = sum(m.evolution.HM*u.values)