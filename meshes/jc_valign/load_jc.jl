using DelimitedFiles, HDF5

function load_jc(fbase, savefile)
    p = readdlm(string(fbase, "p.csv"), ',')
    t = readdlm(string(fbase, "t.csv"), ',', Int64)
    e = boundary_nodes(t)

    h5open(savefile, "w") do file
        write(file, "p", p)
        write(file, "t", t)
        write(file, "e", e)
    end

    println(savefile)
end

function all_edges_2d(t)
    etag = vcat(t[:,[1,2]], t[:,[2,3]], t[:,[3,1]])
    etag = hcat(sort(etag, dims=2), 1:3*size(t,1))
    etag = sortslices(etag, dims=1)
    dup = all(etag[2:end,1:2] - etag[1:end-1,1:2] .== 0, dims=2)[:]
    keep = .![false;dup]
    edges = etag[keep,1:2]
    emap = cumsum(keep)
    invpermute!(emap, etag[:,3])
    emap = reshape(emap,:,3)
    dup = [dup;false]
    dup = dup[keep]
    bndix = findall(.!dup)
    return edges, bndix, emap
end
function boundary_nodes(t)
    edges, boundary_indices, _ = all_edges_2d(t)
    return unique(edges[boundary_indices,:][:])
end

for nref=1:4
    fbase = string("meshvaquadbowl", nref)
    savefile = string("mesh", nref, ".h5")
    load_jc(fbase, savefile)
end