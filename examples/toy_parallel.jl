# using Distributed
# using SharedArrays

# @everywhere begin
#     using Pkg; Pkg.activate("../")
#     Pkg.instantiate(); Pkg.precompile()
# end

function get_f(n, m)
    f = SharedArray{Float64,2}((n, m))
    @distributed for k=2:n-1
        fᵏ = zeros(3, m)
        for j=1:m
            fᵏ[:, j] = get_fᵏ(k)
        end
        f[[k-1, k, k+1], :] .+= fᵏ
    end
    return f
end

function get_fᵏ(k)
    fᵏ = zeros(3)
    for i=1:3
        fᵏ[i] = 1
    end
    return fᵏ
end

function get_a(n)
    a = SharedArray{Float64}(n)
    @distributed for i = 1:n
        a[i] = i
    end
    return a
end
