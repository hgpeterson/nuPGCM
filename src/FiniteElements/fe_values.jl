struct FEValues{S <: ShapeFunction, Q <: QuadratureRule, V, G}
    φ::S
    quad_rule::Q
    value::V
    gradient::G
    # generalize to vector-valued problems: divergence, symmetric gradient?
end

function FEValues(φ::ShapeFunction, quad_rule::QuadratureRule)
    nφ = n_dofs(φ)
    nq = length(quad_rule.weights)
    d = size(quad_rule.points, 2)
    T = eltype(quad_rule.weights)

    value = zeros(T, nφ, nq)
    gradient = zeros(T, nφ, nq, d)
    return FEValues(φ, quad_rule, value, gradient)
end

function reinit!(fe_values::FEValues, ∂ξ∂x; compute_gradients=true)
    φ = fe_values.φ
    qr = fe_values.quad_rule

    nφ = n_dofs(φ)
    nq = length(qr.weights)
    for i in 1:nφ
        for q in 1:nq
            fe_values.value[i, q] = φ(qr.points[q, :], Val(i))
            if compute_gradients
                fe_values.gradient[i, q, :] = ∂ξ∂x'*∇(φ)(qr.points[q, :], Val(i))
            end
        end
    end
end