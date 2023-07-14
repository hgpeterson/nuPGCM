using nuPGCM, Test

function shape_func_test(el)
    @testset "Shape functions for order $(el.order) $(Base.typename(typeof(el)).wrapper)" begin
        for i=1:el.n, j=1:el.n
            @test (φ(el, el.p_ref[i, :], j) == 1) == (i == j)
        end
    end
end
for el_type=[Line, Triangle, Wedge]
    for order=1:2
        shape_func_test(el_type(order=order))
    end
end