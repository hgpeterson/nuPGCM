"""
    x = cg!(x, A, b; P=Identity(), tol=eps(eltype(b)))

Solve `Ax = b` using conjugate gradient method with preconditioner `P`.
"""
# function cg!(x, A, b; P=Identity(), tol=eps(eltype(b)))
# function con_grad!(x, A, b; Pinv=I, tol=eps(eltype(b)))
#     # residual
#     r = b - A*x 

#     # precondition
#     # z = P\r 
#     z = Pinv*r 
#     p = copy(z)

#     # iterate
#     k = 0
#     while norm(r) > tol
#         println("$k $(norm(r))")
#         # save for later
#         rz = r'z
#         Ap = A*p

#         # step size
#         α = rz / (p'*Ap)

#         # update
#         @. x = x + α*p
#         @. r = r - α*Ap
#         # z = P\r 
#         z = Pinv*r

#         # use rz from before
#         β = r'*z/rz
#         @. p = z + β*p

#         k += 1
#     end
#     # println("cg: converged in $k iterations")
#     return x
# end
function cg!(x, A, b; tol=eps(eltype(b)))
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
    end
    # println("cg: converged in $k iterations")
    return x
end