using LinearAlgebra

"""
    c = mkfdstencil(x, xbar, k)

Compute the coefficients `c` in a finite difference approximation of a function
defined at the grid points `x`, evaluated at `xbar`, of order `k`.
"""
function mkfdstencil(x, xbar, k)
	n = length(x)
	A = @. (x[:]' - xbar) ^ (0:n-1) / factorial(0:n-1)
	b = zeros(n)
	b[k+1] = 1.0
	return A \ b
end

"""
    fz = differentiate_pointwise(f, z, z0, n)

Compute `n`th order derivative of `f` at `z0` given grid `z`.
"""
function differentiate_pointwise(f, z, z0, n)
    fd_z = mkfdstencil(z, z0, n)
    return dot(fd_z, f)
end

"""
    fz = differentiate(f, z)

Compute first derivative of `f` on grid `z`.
"""
function differentiate(f::AbstractVector{T}, z::AbstractVector) where T
    # allocate derivative array, fz
    nz = size(z, 1)
    fz = zeros(T, nz)

    # 2nd order centered difference
    for j=2:nz-1
        fz[j] = differentiate_pointwise(f[j-1:j+1], z[j-1:j+1], z[j], 1)
    end

    # 2nd order off-center difference on top and bottom boundary
    fz[1] = differentiate_pointwise(f[1:3], z[1:3], z[1], 1)
    fz[nz] = differentiate_pointwise(f[nz-2:nz], z[nz-2:nz], z[nz], 1)

    return fz
end

"""
    fz = differentiate(f, dz)

Compute first derivative of `f` on uniformly spaced grid with spacing `dz`.
"""
function differentiate(f::AbstractVector{T}, dz::Real) where T
    # allocate derivative array, fz
    nz = size(f, 1)
    fz = zeros(T, nz)

    # 2nd order centered difference
    for j=2:nz-1
        fz[j] = (f[j+1] - f[j-1])/(2*dz)
    end

    # 2nd order off-center difference on top and bottom boundary
    fz[1] = (-1.5*f[1] + 2*f[2] - 0.5*f[3])/dz
    fz[nz] = (0.5*f[nz-2] - 2*f[nz-1] + 1.5*f[nz])/dz

    return fz
end
function differentiate(f::AbstractVector{T}, z::AbstractRange) where T
    return differentiate(f, Float64(z.step))
end