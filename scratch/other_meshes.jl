using WriteVTK
using LinearAlgebra

### HEXAHEDRON

p_ref = [-1 -1 -1
          1 -1 -1
          1  1 -1
         -1  1 -1
         -1 -1  1
          1 -1  1
          1  1  1
         -1  1  1]

φ(ξ, i) = 1/8*(1 + p_ref[i, 1]*ξ[1])*(1 + p_ref[i, 2]*ξ[2])*(1 + p_ref[i, 3]*ξ[3])
φξ(ξ, i) = 1/8*p_ref[i, 1]*(1 + p_ref[i, 2]*ξ[2])*(1 + p_ref[i, 3]*ξ[3])
φη(ξ, i) = 1/8*p_ref[i, 2]*(1 + p_ref[i, 1]*ξ[1])*(1 + p_ref[i, 3]*ξ[3])
φζ(ξ, i) = 1/8*p_ref[i, 3]*(1 + p_ref[i, 1]*ξ[1])*(1 + p_ref[i, 2]*ξ[2])

vtk_grid("output/hexa_sf.vtu", p_ref', [MeshCell(VTKCellTypes.VTK_HEXAHEDRON, 1:8)]) do vtk 
    for j=1:8
        vtk["phi$j"] = [φ(p_ref[i, :], j) for i=1:8]
    end
end

function gen_col(h)
    p_sfc = [0 0
             h 0
             h h
             0 h]
    z = -1:h:0
    nz = size(z, 1)
    p = zeros(4nz ,3)
    t = zeros(Int64, nz-1 ,8)
    for i=1:nz-1
        j = 4i-3
        p[j:j+3, 3] .= z[i]
        p[j:j+3, 1] = p_sfc[:, 1]
        p[j:j+3, 2] = p_sfc[:, 2]
        p[j+4:j+7, 3] .= z[i+1]
        p[j+4:j+7, 1] = p_sfc[:, 1]
        p[j+4:j+7, 2] = p_sfc[:, 2]
        t[i, :] = j:j+7
    end
    e = [1, 2, 3, 4, 4nz-3, 4nz-2, 4nz-1, 4nz]
    return p, t, e
end

p, t, e = gen_col(0.1)
vtk_grid("output/hexa_col.vtu", p', [MeshCell(VTKCellTypes.VTK_HEXAHEDRON, t[i, :]) for i ∈ axes(t, 1)]) do vtk end

#### WEDGE

p_ref = [ 0  0  0
          1  0  0
          0  1  0
          0  0  1
          1  0  1
          0  1  1]

function φ(ξ, i)
    if i == 1
        return (1 - ξ[1] - ξ[2])*(1 - ξ[3])
    elseif i == 2
        return ξ[1]*(1 - ξ[3])
    elseif i == 3
        return ξ[2]*(1 - ξ[3])
    elseif i == 4
        return (1 - ξ[1] - ξ[2])*ξ[3]
    elseif i == 5
        return ξ[1]*ξ[3]
    elseif i == 6
        return ξ[2]*ξ[3]
    end
end
function φξ(ξ, i)
    if i == 1
        return -(1 - ξ[3])
    elseif i == 2
        return (1 - ξ[3])
    elseif i == 3
        return 0
    elseif i == 4
        return -ξ[3]
    elseif i == 5
        return ξ[3]
    elseif i == 6
        return 0
    end
end
function φη(ξ, i)
    if i == 1
        return -(1 - ξ[3])
    elseif i == 2
        return 0
    elseif i == 3
        return (1 - ξ[3])
    elseif i == 4
        return -ξ[3]
    elseif i == 5
        return 0
    elseif i == 6
        return ξ[3]
    end
end
function φζ(ξ, i)
    if i == 1
        return -(1 - ξ[1] - ξ[2])
    elseif i == 2
        return -ξ[1]
    elseif i == 3
        return -ξ[2]
    elseif i == 4
        return (1 - ξ[1] - ξ[2])
    elseif i == 5
        return ξ[1]
    elseif i == 6
        return ξ[2]
    end
end

xξ(ξ, p) = sum(φξ(ξ, i)*p[i, 1] for i=1:6)
yξ(ξ, p) = sum(φξ(ξ, i)*p[i, 2] for i=1:6)
zξ(ξ, p) = sum(φξ(ξ, i)*p[i, 3] for i=1:6)
xη(ξ, p) = sum(φη(ξ, i)*p[i, 1] for i=1:6)
yη(ξ, p) = sum(φη(ξ, i)*p[i, 2] for i=1:6)
zη(ξ, p) = sum(φη(ξ, i)*p[i, 3] for i=1:6)
xζ(ξ, p) = sum(φζ(ξ, i)*p[i, 1] for i=1:6)
yζ(ξ, p) = sum(φζ(ξ, i)*p[i, 2] for i=1:6)
zζ(ξ, p) = sum(φζ(ξ, i)*p[i, 3] for i=1:6)
J(ξ, p) = inv([xξ(ξ, p) xη(ξ, p) xζ(ξ, p)
               yξ(ξ, p) yη(ξ, p) yζ(ξ, p)
               zξ(ξ, p) zη(ξ, p) zζ(ξ, p)])

vtk_grid("output/wedge_sf.vtu", p_ref', [MeshCell(VTKCellTypes.VTK_WEDGE, 1:6)]) do vtk 
    for j=1:6
        vtk["phi$j"] = [φ(p_ref[i, :], j) for i=1:6]
    end
end

function gen_col(h)
    p_sfc = [0 0
             h 0
             0 h]
    z = -1:h:0
    nz = size(z, 1)
    p = zeros(3nz ,3)
    t = zeros(Int64, nz-1, 6)
    for i=1:nz-1
        j = 3i-2
        p[j:j+2, 3] .= z[i]
        p[j:j+2, 1] = p_sfc[:, 1]
        p[j:j+2, 2] = p_sfc[:, 2]
        p[j+3:j+5, 3] .= z[i+1]
        p[j+3:j+5, 1] = p_sfc[:, 1]
        p[j+3:j+5, 2] = p_sfc[:, 2]
        t[i, :] = j:j+5
    end
    e = [1, 2, 3, 3nz-2, 3nz-1, 3nz]
    return p, t, e
end

p, t, e = gen_col(0.1)
vtk_grid("output/wedge_col.vtu", p', [MeshCell(VTKCellTypes.VTK_WEDGE, t[i, :]) for i ∈ axes(t, 1)]) do vtk end