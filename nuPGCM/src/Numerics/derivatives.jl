"""
    c = mkfdstencil(x, xbar, k)

Compute the coefficients `c` in a finite difference approximation of a function
defined at the grid points `x`, evaluated at `xbar`, of order `k`.
"""
function mkfdstencil(x::Array{Float64,1}, xbar::Float64, k::Int64)
	n = length(x)
	A = @. (x[:]' - xbar) ^ (0:n-1) / factorial(0:n-1)
	b = zeros(n)
	b[k+1] = 1.0
	c = A \ b
end

"""
    fz = differentiate_pointwise(f, z, z0, n)

Compute `n`th order derivative of `f` at `z0` given grid `z`.
"""
function differentiate_pointwise(f::Array{Float64,1}, z::Array{Float64,1}, z0::Float64, n::Int64)
    fd_z = mkfdstencil(z, z0, n)
    return dot(fd_z, f)
end

"""
    fz = differentiate(f, z)

Compute second order first derivative of `f` on grid `z`.
"""
function differentiate(f::Array{Float64,1}, z::Array{Float64,1})
    # allocate derivative array, fz
    nz = size(z, 1)
    fz = zeros(nz)

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

Compute second order first derivative of `f` on uniformly spaced grid with spacing `dz`.
"""
function differentiate(f::Array{Float64,1}, dz::Float64)
    # allocate derivative array, fz
    nz = size(f, 1)
    fz = zeros(nz)

    # 2nd order centered difference
    for j=2:nz-1
        fz[j] = (f[j+1] - f[j-1])/(2*dz)
    end

    # 2nd order off-center difference on top and bottom boundary
    fz[1] = (-1.5*f[1] + 2*f[2] - 0.5*f[3])/dz
    fz[nz] = (0.5*f[nz-2] - 2*f[nz-1] + 1.5*f[nz])/dz

    return fz
end
function differentiate(f::Array{Float64,1}, z::StepRangeLen{Float64})
    dz = convert(Float64, z.step)
    fz = differentiate(f, dz)
    return fz
end
