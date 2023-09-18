using Test, BenchmarkTools, CUDA

function gpu_adv!(adv, A, χ, b, t2)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    Is = CartesianIndices((axes(A, 1), axes(A, 2)))
    for i=index:stride:length(Is)
        for ib ∈ axes(b, 2), iχ ∈ axes(χ, 2)
            adv[t2[Is[i]]] += A[Is[i], ib, iχ]*χ[Is[i][1], iχ]*b[t2[Is[i][1], ib]]
        end
    end
    return 
end
function bench_gpu!(adv, A, χ, b, t2)
    kernel = @cuda launch=false gpu_adv!(adv, A, χ, b, t2)
    config = launch_configuration(kernel.fun)
    N = size(A, 1)*size(A, 2)
    threads = min(N, config.threads)
    blocks = cld(N, threads)

    CUDA.@sync begin
        kernel(adv, A, χ, b, t2; threads, blocks)
    end
end
# N = 2^7
A = CUDA.rand(m.g2.nt, m.g2.nn, m.g2.nn, m.g1.nn)
χ = CUDA.rand(m.g2.nt, m.g1.nn)
b = CUDA.rand(m.g2.np)
adv = CUDA.zeros(m.g2.np)
t2 = CuArray(m.g2.t)
CUDA.memory_status()
@btime bench_gpu!($adv, $A, $χ, $b, $t2)

# function bench_cpu!(adv, A, χ, b)
#     for k ∈ axes(A, 1), i ∈ axes(A, 2), ib ∈ axes(b, 2), iχ ∈ axes(χ, 2)
#         adv[k, i] += A[k, i, ib, iχ]*χ[k, iχ]*b[k, ib]
#     end
#     return 
# end
# N = 2^7
# A = rand(Float32, N, N, N, N)
# χ = rand(Float32, N, N)
# b = rand(Float32, N, N)
# adv = zeros(Float32, N, N)
# @btime bench_cpu!($adv, $A, $χ, $b)