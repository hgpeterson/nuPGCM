# write and read functions for sparse matrices 
function write_sparse_matrix(fname::AbstractString, A::SparseArrays.AbstractSparseMatrix)
    I, J, V = findnz(A)
    @printf("Writing sparse matrix to '%s' (%.2f GB)... ", fname, nnz(A)*(2*sizeof(eltype(I)) + sizeof(eltype(V)))/1e9)
    h5open(fname, "w") do file
        write(file, string("I"), I)
        write(file, string("J"), J)
        write(file, string("V"), V)
        write(file, string("m"), size(A, 1))
        write(file, string("n"), size(A, 2))
    end
    println("Done.")
end
function read_sparse_matrix(fname::AbstractString)
    file = h5open(fname, "r")
    I = read(file, "I")
    J = read(file, "J")
    V = read(file, "V")
    m = read(file, "m")
    n = read(file, "n")
    close(file)
    @printf("Sparse matrix (%.2f GB) loaded from '%s'.\n", length(V)*(2*sizeof(eltype(I)) + sizeof(eltype(V)))/1e9, fname)
    return sparse(I, J, V, m, n)
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