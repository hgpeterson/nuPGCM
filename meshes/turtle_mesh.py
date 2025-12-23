import numpy as np
from xml.dom import minidom
from svg.path import parse_path

# read file
doc = minidom.parse('turtle.svg')

# determine path
path = parse_path(doc.getElementsByTagName('path')[0].getAttribute('d'))

# bounding box
bbox = path.boundingbox()
xmin = bbox[0]
ymin = bbox[1]
xmax = bbox[2]
ymax = bbox[3]
Lx = xmax - xmin
Ly = ymax - ymin

# translate points so they all lie in whichever of
#   [-1, 1] x [-Ly/Lx, Ly/Lx]
# or
#   [-Lx/Ly, Lx/Ly] x [-1, 1]
# is smaller.
# (also flip about y = 0)
def translate(z): 
    L = max(Lx, Ly)
    return 2*(z.real - xmin)/L - 1, -(2*(z.imag - ymin)/L - 1)

# gmsh file
gfile = open("turtle_mesh.jl", "w")
gfile.write("""
using Gmsh: gmsh

h = 0.01
gmsh.initialize()
gmsh.model.add("turtle")

"""
)

def add_point(file, pt, tag):
    x, y = translate(pt)
    file.write(f"gmsh.model.geo.addPoint({x:.8f}, {y:.8f}, 0, h, {tag:d})\n")

def add_curve(file, start_pt, tag, curve_type):
    if curve_type == "Line":
        gfile.write(f"gmsh.model.geo.addLine({start_pt:d}, {start_pt+1:d}, {tag:d})\n")
        return start_pt + 1
    elif curve_type == "CubicBezier":
        gfile.write(f"gmsh.model.geo.addBSpline({start_pt:d}:{start_pt+3:d}, {tag:d})\n")
        return start_pt + 3

# first add all points and save curve types
n_pts = 0
curve_types = []
for i, obj in enumerate(path):
    # add start point [and control point(s)] of `obj`
    # (end point is start of next `obj`)
    if type(obj).__name__ == "Line":
        curve_types.append("Line")
        add_point(gfile, obj.start, n_pts+1)
        n_pts += 1
    elif type(obj).__name__ == "CubicBezier":
        curve_types.append("CubicBezier")
        add_point(gfile, obj.start,    n_pts+1)
        add_point(gfile, obj.control1, n_pts+2)
        add_point(gfile, obj.control2, n_pts+3)
        n_pts += 3
gfile.write("\n")

# now add all (but final) curves
n_lines = 0
start_pt = 1
for curve in curve_types[0:-1]:
    start_pt = add_curve(gfile, start_pt, n_lines+1, curve)    
    n_lines += 1

# final curve loops back to first point
if curve_types[-1] == "Line":
    gfile.write(f"gmsh.model.geo.addLine([{n_pts:d}, 1], {n_lines+1:d})\n")
elif curve_types[-1] == "CubicBezier":
    gfile.write(f"gmsh.model.geo.addBSpline([{n_pts-2:d}, {n_pts-1:d}, {n_pts:d}, 1], {n_lines+1:d})\n")
n_lines += 1

# unlink
doc.unlink()

# finish gmsh file
gfile.write(f"""
gmsh.model.geo.addCurveLoop(1:{n_lines:d}, 1)
gmsh.model.geo.addPlaneSurface([1], 1)

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("turtle.msh")
gmsh.finalize()
"""
)
gfile.close()

print( "Mesh generating file 'turtle_mesh.jl' created.")
print(f"  Points: {n_pts:d}")
print(f"  Curves: {n_lines:d}")