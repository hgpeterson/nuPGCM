using Test
using nuPGCM.FiniteElements

nodes = [0.0 0.0 0.0
         0.0 1.0 0.0
         1.0 0.0 0.0
         1.0 1.0 0.0]
elements = [1 2 4
            1 3 4]
boundary_nodes = Dict("left" => [1, 2], 
                      "right" => [3, 4], 
                      "bottom" => [1, 3], 
                      "top" => [2, 4])
mesh = FiniteElements.Mesh(nodes, elements, boundary_nodes)