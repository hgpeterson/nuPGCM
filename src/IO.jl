# write and read functions for sparse matrices 
function write_sparse_matrix(filename::AbstractString, A::SparseArrays.AbstractSparseMatrix)
    I, J, V = findnz(A)
    @printf("Writing sparse matrix to '%s' (%.2f GB)... ", filename, nnz(A)*(2*sizeof(eltype(I)) + sizeof(eltype(V)))/1e9)
    h5open(filename, "w") do file
        write(file, string("I"), I)
        write(file, string("J"), J)
        write(file, string("V"), V)
        write(file, string("m"), size(A, 1))
        write(file, string("n"), size(A, 2))
    end
    println("Done.")
end
function read_sparse_matrix(filename::AbstractString)
    file = h5open(filename, "r")
    I = read(file, "I")
    J = read(file, "J")
    V = read(file, "V")
    m = read(file, "m")
    n = read(file, "n")
    close(file)
    @printf("Sparse matrix (%.2f GB) loaded from '%s'.\n", length(V)*(2*sizeof(eltype(I)) + sizeof(eltype(V)))/1e9, filename)
    return sparse(I, J, V, m, n)
end