using Test, BenchmarkTools, CUDA

# function sum_node(A, χ, b)
#     return b'*A*χ
# end
# function gpu_adv_broadcast!(adv, A, χ, b)
#     CUDA.@sync begin
#        adv .= sum_node.(A, χ, b) 
#     end
#     return
# end
# nt = 2^10
# nn = 12
# A = CuArray[CUDA.rand(nn, nn) for i=1:nt, j=1:nn]
# χ = CuArray[CUDA.rand(nn) for i=1:nt]
# b = CuArray[CUDA.rand(nn) for i=1:nt]
# adv = CUDA.zeros(nt, nn)
# @btime gpu_adv_broadcast!($adv, $A, $χ, $b)

# function cpu_adv_broadcast!(adv, A, χ, b)
#     adv .= sum_node.(A, χ, b) 
#     return
# end
# A = [rand(nn, nn) for i=1:nt, j=1:nn]
# χ = [rand(nn) for i=1:nt]
# b = [rand(nn) for i=1:nt]
# adv = zeros(nt, nn)
# @btime cpu_adv_broadcast!($adv, $A, $χ, $b)



# function gpu_adv!(adv, A, χ, b, t2)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     Is = CartesianIndices((axes(A, 1), axes(A, 2)))
#     for i=index:stride:length(Is)
#         adv[t2[Is[i]]] = b[t2[Is[i][1], :]]'*A[Is[i], :, :]*χ[Is[i][1], :]
#         # for ib ∈ axes(b, 2), iχ ∈ axes(χ, 2)
#         #     adv[t2[Is[i]]] += A[Is[i], ib, iχ]*χ[Is[i][1], iχ]*b[t2[Is[i][1], ib]]
#         # end
#     end
#     return 
# end
# function bench_gpu!(adv, A, χ, b, t2)
#     kernel = @cuda launch=false gpu_adv!(adv, A, χ, b, t2)
#     config = launch_configuration(kernel.fun)
#     N = size(A, 1)*size(A, 2)
#     threads = min(N, config.threads)
#     blocks = cld(N, threads)

#     CUDA.@sync begin
#         kernel(adv, A, χ, b, t2; threads, blocks)
#     end
# end
# # N = 2^7
# A = CUDA.rand(m.g2.nt, m.g2.nn, m.g2.nn, m.g1.nn)
# χ = CUDA.rand(m.g2.nt, m.g1.nn)
# b = CUDA.rand(m.g2.np)
# adv = CUDA.zeros(m.g2.np)
# t2 = CuArray(m.g2.t)
# CUDA.memory_status()
# @btime bench_gpu!($adv, $A, $χ, $b, $t2)

# function bench_cpu!(adv, A, χ, b)
#     for k ∈ axes(A, 1), i ∈ axes(A, 2)
#         adv[k, i] = b[k, :]'*A[k, i, :, :]*χ[k, :]
#     end
#     return 
# end
# N = 2^7
# A = rand(Float32, N, N, N, N)
# χ = rand(Float32, N, N)
# b = rand(Float32, N, N)
# adv = zeros(Float32, N, N)
# @btime bench_cpu!($adv, $A, $χ, $b)


function bench_gpu!(adv, A, χ, b)
    N = size(A, 1)
    @cuda threads=N blocks=N compute_adv(adv, A, χ, b, N)
    return
end

function compute_adv(adv, A, χ, b, N)
    k, i = threadIdx().x, blockIdx().x
    result = 0.0
    for j = 1:N
        result += b[k, j] * A[k, i, j, j] * χ[k, j]
    end
    adv[k, i] = result
    return
end

N = 2^7

# Create CuArrays on the GPU
A = CUDA.rand(Float32, N, N, N, N)
χ = CUDA.rand(Float32, N, N)
b = CUDA.rand(Float32, N, N)
adv = CUDA.zeros(Float32, N, N)

@btime bench_gpu!($adv, $A, $χ, $b)