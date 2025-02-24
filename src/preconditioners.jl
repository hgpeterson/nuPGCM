using LinearAlgebra
import LinearAlgebra.mul!

function mul!(y, P::Factorization, x)
    ldiv!(y, P, x)
end

struct MyPreconditioner
    # stuff preconditioner needs
end

function mul!(y, P::MyPreconditioner, x)
    # apply preconditioner
end