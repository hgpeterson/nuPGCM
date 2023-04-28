using nuPGCM, Test

# polynomial evaluation
sf = ShapeFunctions(order=2, dim=3)
@test nuPGCM.eval_poly(sf.C[:, 1], [0.1, 0.1, 0.1], sf.perms) == 0.2799999999999999