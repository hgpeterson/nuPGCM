"""
    x = cg!(x, A, b; Pinv=I, tol=eps(eltype(b)), debug=false)

Solve `Ax = b` using conjugate gradient method where inverse of preconditioner is `Pinv`.
"""
function cg!(args...; Pinv=I, kwargs...)
    if Pinv == I
        return cg_no_precond!(args...; kwargs...)
    else
        return cg_precond!(args...; Pinv, kwargs...)
    end 
end

function cg_precond!(x, A, b; Pinv, tol=eps(eltype(b)), debug=false)
    # residual
    r = b - A*x 

    # precondition
    z = Pinv*r 
    p = copy(z)

    # iterate
    k = 0
    while norm(r) > tol
        # save for later
        rz = r'z
        Ap = A*p

        # step size
        α = rz / (p'*Ap)

        # update
        @. x = x + α*p
        @. r = r - α*Ap
        z = Pinv*r
        # println("iter=$k    resid=$(norm(r))")

        # use rz from before
        β = r'*z/rz
        @. p = z + β*p

        k += 1

        if k > 10000
            @warn "`cg!` failed to converge within 10000 iterations."
            break
        end
    end

    if debug
        @info "cg: converged in $k iterations"
    end

    return x
end

function cg_no_precond!(x, A, b; tol=eps(eltype(b)), debug=false)
    # residual
    r = b - A*x
    p = copy(r)

    # iterate
    k = 0
    while norm(r) > tol
        # save for later
        rr = r'r
        Ap = A*p

        # step size
        α = rr / (p'*Ap)

        # update
        @. x = x + α*p
        @. r = r - α*Ap
        # println("iter=$k    resid=$(norm(r))")

        # use rr from before
        β = r'*r/rr
        @. p = r + β*p

        k += 1

        if k > 10000
            @warn "`cg!` failed to converge within 10000 iterations."
            break
        end
    end

    if debug
        @info "cg: converged in $k iterations"
    end

    return x
end