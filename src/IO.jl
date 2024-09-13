"""
    write_sparse_matrix(file::HDF5.File, A::SparseArrays.AbstractSparseMatrix)

Write a sparse matrix `A` to an HDF5 file `file`. The matrix is stored as three
arrays `I`, `J`, and `V` such that `A[I[k], J[k]] = V[k]`. The dimensions of the
matrix are stored as `m` and `n`.
"""
function write_sparse_matrix(file::HDF5.File, A::SparseArrays.AbstractSparseMatrix)
    I, J, V = findnz(A)
    @printf("Writing sparse matrix of size %.2f GB... ", nnz(A)*(2*sizeof(eltype(I)) + sizeof(eltype(V)))/1e9)
    write(file, "I", I)
    write(file, "J", J)
    write(file, "V", V)
    write(file, "m", size(A, 1))
    write(file, "n", size(A, 2))
end

"""
    write_sparse_matrix(A::SparseArrays.AbstractSparseMatrix; fname="A.h5")

Write a sparse matrix `A` to an HDF5 file `fname`.
"""
function write_sparse_matrix(A::SparseArrays.AbstractSparseMatrix; fname="A.h5")
    h5open(fname, "w") do file
        write_sparse_matrix(file, A)
    end
    println("written to $fname.")
end

"""
    write_sparse_matrix(A::SparseArrays.AbstractSparseMatrix, 
                        perm::AbstractArray, 
                        inv_perm::AbstractArray; fname="A.h5")
            
Write a sparse matrix `A` to an HDF5 file `fname` along with the permutation
`perm` and its inverse `inv_perm`.
"""
function write_sparse_matrix(A::SparseArrays.AbstractSparseMatrix, perm::AbstractArray, inv_perm::AbstractArray; fname="A.h5")
    h5open(fname, "w") do file
        write_sparse_matrix(file, A)
        write(file, "perm", perm)
        write(file, "inv_perm", inv_perm)
    end
    println("written to $fname.")
end

"""
    A = read_sparse_matrix(fname::AbstractString)
    A, perm, inv_perm = read_sparse_matrix(fname::AbstractString)

Read a sparse matrix from an HDF5 file `fname`. If the file contains the permutation
`perm` and its inverse `inv_perm`, return them as well.
"""
function read_sparse_matrix(fname::AbstractString)
    # open file
    file = h5open(fname, "r")

    # read sparse matrix data
    I = read(file, "I")
    J = read(file, "J")
    V = read(file, "V")
    m = read(file, "m")
    n = read(file, "n")

    # if it has permutation data, read it
    perm = inv_perm = nothing
    try 
        perm = read(file, "perm")
        inv_perm = read(file, "inv_perm")
    catch
    end

    # close file
    close(file)

    @printf("Sparse matrix of size %.2f GB loaded from '%s'.\n", length(V)*(2*sizeof(eltype(I)) + sizeof(eltype(V)))/1e9, fname)

    if perm === nothing
        return sparse(I, J, V, m, n)
    end

    return sparse(I, J, V, m, n), perm, inv_perm
end

"""
    save_state(ux, uy, uz, p, b, t; fname="state.h5")
"""
function save_state(ux, uy, uz, p, b, t; fname="state.h5")
    h5open(fname, "w") do file
        write(file, "ux", ux.free_values)
        write(file, "uy", uy.free_values)
        write(file, "uz", uz.free_values)
        write(file, "p", Vector(p.free_values))
        write(file, "b", b.free_values)
        write(file, "t", t)
    end
    @printf("State saved to '%s'.\n", fname)
end

"""
    save_state_vtu(ux, uy, uz, p, b, Ω; fname="state.vtu")
"""
function save_state_vtu(ux, uy, uz, p, b, Ω; fname="state.vtu")
    writevtk(Ω, fname, cellfields=["u"=>ux, "v"=>uy, "w"=>uz, "p"=>p, "b"=>b])
    @printf("State saved to '%s'.\n", fname)
end

"""
    ux, uy, uz, p, b, t = load_state(fname::AbstractString)
"""
function load_state(fname::AbstractString)
    file = h5open(fname, "r")
    ux = read(file, "ux")
    uy = read(file, "uy")
    uz = read(file, "uz")
    p = read(file, "p")
    b = read(file, "b")
    t = read(file, "t")
    close(file)
    @printf("State loaded from '%s'.\n", fname)
    return ux, uy, uz, p, b, t
end