import numpy as np
import pyvista as pv
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")

class SlicePlotter:

    def __init__(self, file_name):
        self.file_name = file_name
        self.dataset = pv.read(file_name)

    def set_slice(self, direction, location):
        self.direction = direction.lower()
        if self.direction == "x":
            self.normal = [0, 1, 0]
            self.origin = [location, 0, 0]
            self.xlabel = r"$y$"
            self.ylabel = r"$z$"
        elif self.direction == "y":
            self.normal = [1, 0, 0]
            self.origin = [0, location, 0]
            self.xlabel = r"$x$"
            self.ylabel = r"$z$"
        elif self.direction == "z":
            self.normal = [0, 0, 1]
            self.origin = [0, 0, location]
            self.xlabel = r"$x$"
            self.ylabel = r"$y$"
        else:
            ValueError("'direction' must be one of 'x', 'y', or 'z'")

    def plot(self, field_name, label=None, n=2**8, interactive=False, output_file="image.png"):
        # # slice with plane
        # ds_slice = self.dataset.slice(normal=self.normal, origin=self.origin)
        # coords = ds_slice.points
        # values = ds_slice.point_data[field_name]
        # if self.direction == "x":
        #     x1 = coords[:, 1]
        #     x2 = coords[:, 2]
        # elif self.direction == "y":
        #     x1 = coords[:, 0]
        #     x2 = coords[:, 2]
        # elif self.direction == "z":
        #     x1 = coords[:, 0]
        #     x2 = coords[:, 1]
            
        # # regular grid
        # x1_grid = np.linspace(x1.min(), x1.max(), n)
        # x2_grid = np.linspace(x2.min(), x2.max(), n)
        # X1, X2 = np.meshgrid(x1_grid, x2_grid)
            
        # # interpolate onto regular grid
        # field = griddata((x1, x2), values, (X1, X2), method='linear', fill_value=np.nan)
        plane = pv.Plane(center=self.origin, direction=self.normal, i_resolution=n, j_resolution=n)
        x = plane.points[:, 0]
        y = plane.points[:, 1]
        z = plane.points[:, 2]
        samples = plane.sample(self.dataset)
        field = samples[field_name]
            
        vmax = np.max(np.abs(field))

        # plot
        fig, ax = plt.subplots(1)
        im = ax.pcolormesh(x, y, field, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        if label is None: 
            label = field_name
        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        plt.savefig(output_file) 
        print(output_file)
        plt.close()

        # if interactive:
        #     plotter = pv.Plotter()
        #     vmax = np.max(np.abs(values))
        #     plotter.add_mesh(slice_mesh, scalars=field_name, cmap='RdBu_r', clim=[-vmax, vmax])
        #     plotter.add_axes()
        #     plotter.show_bounds()
        #     plotter.show()

if __name__ == "__main__":
    file_name = f"{wd}/../../nuPGCM/docs/src/literated/data/state.vtu"
    # file_name = f"{wd}/../../nuPGCM/docs/src/literated/data/state_{25:016d}.vtu"
    # sp = SlicePlotter(file_name)
    # sp.set_slice("x", 0)
    # sp.plot("v", label=r"$v$")


    # mesh = pv.read(file_name)
    # sliced_mesh = mesh.slice(normal='y', origin=mesh.center)
    # contours_b = sliced_mesh.contour(isosurfaces=10, scalars='b')
    # plotter = pv.Plotter()
    # vmax_abs = np.max(np.abs(sliced_mesh['v']))
    # plotter.add_mesh(
    #     sliced_mesh, 
    #     scalars='v', 
    #     clim=[-vmax_abs, vmax_abs],
    #     cmap='RdBu_r',
    #     lighting=False,
    #     scalar_bar_args={'title': 'Along-slope flow v'}
    # )
    # plotter.add_mesh(
    #     contours_b, 
    #     color='black',
    #     opacity=0.25,
    #     line_width=2, 
    # )
    # plotter.enable_2d_style()
    # plotter.show(cpos="xz")
    # plotter.screenshot("v.png")

    mesh = pv.read(file_name)
    origin = mesh.center
    normal = [0, 1, 0]
    # slice_plane = pv.Plane(center=origin, normal=normal, i_size=mesh.bounds[1]-mesh.bounds[0], j_size=mesh.bounds[5]-mesh.bounds[4])
    sliced_polydata = mesh.slice(normal=normal, origin=origin)

    x_unstructured = sliced_polydata.points[:, 0]
    z_unstructured = sliced_polydata.points[:, 2]
    v_unstructured = sliced_polydata['v']

    bounds = sliced_polydata.bounds
    n = 2**8
    spacing = (bounds[1] - bounds[0]) / (n-1), 0, (bounds[5] - bounds[4]) / (n-1)
    grid_2d = pv.ImageData(dimensions=(n, 1, n), spacing=spacing, origin=(bounds[0], 0, bounds[4]))

    # gridded_data = grid_2d.sample(sliced_polydata)
    gridded_data = grid_2d.sample(mesh)

    # nan mask
    nan_mask = gridded_data['vtkValidPointMask'] == 0
    u = gridded_data['u']
    v = gridded_data['v']
    w = gridded_data['w']
    b = gridded_data['b']
    u[nan_mask] = np.nan
    v[nan_mask] = np.nan
    w[nan_mask] = np.nan
    b[nan_mask] = np.nan

    # maxima
    umax = np.nanmax(np.abs(u))
    vmax = np.nanmax(np.abs(v))
    wmax = np.nanmax(np.abs(w))

    x_2d = gridded_data.points[:, 0].reshape((n, n))
    z_2d = gridded_data.points[:, 2].reshape((n, n))
    u_2d = u.reshape((n, n))
    v_2d = v.reshape((n, n))
    w_2d = w.reshape((n, n))
    b_2d = b.reshape((n, n))

    fig, ax = plt.subplots(3, 1, figsize=(3.2, 3*1.1))
    c = ax[0].pcolormesh(x_2d, z_2d, u_2d, cmap='RdBu_r', vmin=-umax, vmax=umax, shading='gouraud')
    cbar = fig.colorbar(c, ax=ax[0], label=r'$u$', shrink=0.8)
    cbar.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    ax[0].contour(x_2d, z_2d, b_2d, levels=np.arange(-0.9, 0.0, 0.1), 
              colors='k', linestyles='-', linewidths=0.5, alpha=0.25)
    c = ax[1].pcolormesh(x_2d, z_2d, v_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
    cbar = fig.colorbar(c, ax=ax[1], label=r'$v$', shrink=0.8)
    cbar.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    ax[1].contour(x_2d, z_2d, b_2d, levels=np.arange(-0.9, 0.0, 0.1), 
              colors='k', linestyles='-', linewidths=0.5, alpha=0.25)
    c = ax[2].pcolormesh(x_2d, z_2d, w_2d, cmap='RdBu_r', vmin=-wmax, vmax=wmax, shading='gouraud')
    cbar = fig.colorbar(c, ax=ax[2], label=r'$w$', shrink=0.8)
    cbar.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    ax[2].contour(x_2d, z_2d, b_2d, levels=np.arange(-0.9, 0.0, 0.1), 
              colors='k', linestyles='-', linewidths=0.5, alpha=0.25)
    for a in ax:
        a.axis('equal')
        a.set_xticks([])
        a.set_yticks([-0.5, 0])
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
        a.set_ylabel(r'$z$')
    ax[2].set_xticks(np.arange(-1, 1.1, 0.5))
    ax[2].set_xlabel(r'$x$')
    plt.savefig('example1a.png')
    plt.close()