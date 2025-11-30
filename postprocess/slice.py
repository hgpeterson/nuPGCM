import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from scipy.integrate import simpson

# params
file_name = "/home/hpeter/Downloads/states/nu/bz_in_A_dt_small_3D/state_0000000000002400.vtu"
field_name = "v"
x_slice = 0.5

# read VTU file
mesh = pv.read(file_name)
print(f"Loaded mesh: {mesh.n_points} points, {mesh.n_cells} cells")
print(f"Available fields: {list(mesh.point_data.keys())}")

# slice with plane
slice_mesh = mesh.slice(normal=[1, 0, 0], origin=[x_slice, 0, 0])
coords = slice_mesh.points
values = slice_mesh.point_data[field_name]
print(f"Slice contains {slice_mesh.n_points} points")

# y-z coordinates
y = coords[:, 1]
z = coords[:, 2]
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()
    
# regular grid
n_points = 100
yi = np.linspace(y_min, y_max, n_points)
zi = np.linspace(z_min, z_max, n_points)
Y, Z = np.meshgrid(yi, zi)
    
# interpolate onto regular grid
U = griddata((y, z), values, (Y, Z), method='linear', fill_value=0.0)
    
# Integrate using 2D Simpson's rule
dy = yi[1] - yi[0]
dz = zi[1] - zi[0]
integral = simpson(simpson(U, dx=dz, axis=0), dx=dy)
print(f"Detected 3D mesh: y ∈ [{y_min:.3f}, {y_max:.3f}], z ∈ [{z_min:.3f}, {z_max:.3f}]")
print(f"Integral of u at x = 0: {integral:.6e}")

# visualize
plotter = pv.Plotter()
vmax = np.max(np.abs(values))
plotter.add_mesh(slice_mesh, scalars=field_name, cmap='RdBu_r', clim=[-vmax, vmax])
plotter.add_axes()
plotter.show_bounds()
plotter.show()