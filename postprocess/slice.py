import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid
import utils
import shutil

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")

class SlicePlotter:

    def __init__(self, file_name):
        self.file_name = file_name
        self.dataset = pv.read(file_name)

    def set_slice(self, direction, location):
        self.direction = direction.lower()
        self.location = location
        if self.direction == "x":
            self.normal = [1, 0, 0]
            self.origin = [location, 0, 0]
            self.xlabel = r"Meridional coordinate $y$"
            self.ylabel = r"Vertical coordinate $z$"
        elif self.direction == "y":
            self.normal = [0, 1, 0]
            self.origin = [0, location, 0]
            self.xlabel = r"Zonal coordinate $x$"
            self.ylabel = r"Vertical coordinate $z$"
        elif self.direction == "z":
            self.normal = [0, 0, 1]
            self.origin = [0, 0, location]
            self.xlabel = r"Zonal coordinate $x$"
            self.ylabel = r"Meridional coordinate $y$"
        else:
            ValueError("'direction' must be one of 'x', 'y', or 'z'")

    def plot(self, field_name, label=None, output_file="image.png", bmin=None, bmax=None, vmax=None):
        # slice with plane
        ds_slice = self.dataset.slice(normal=self.normal, origin=self.origin)
        p = ds_slice.points
        if self.direction == "x":
            x1 = p[:, 1]
            x2 = p[:, 2]
        elif self.direction == "y":
            x1 = p[:, 0]
            x2 = p[:, 2]
        elif self.direction == "z":
            x1 = p[:, 0]
            x2 = p[:, 1]
            
        if field_name == "u":
            field = ds_slice["u"][:, 0]
        elif field_name == "v":
            field = ds_slice["u"][:, 1]
        elif field_name == "w":
            field = ds_slice["u"][:, 2]
        else:
            field = ds_slice[field_name]
        if vmax is None:
            vmax = np.max(np.abs(field))
            extend = "neither"
        else:
            if vmax < np.max(np.abs(field)):
                extend = "both"
        b = ds_slice["b"]
        if bmax is None:
            bmax = b.max()
        if bmin is None:
            bmin = b.min()

        # plot
        fig, ax = plt.subplots(1)
        im = ax.tripcolor(x1, x2, field, vmin=-vmax, vmax=vmax, cmap="RdBu_r", shading="gouraud")
        if label is None: 
            label = field_name
        # plt.colorbar(im, ax=ax, label=label, shrink=0.8)
        cb = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[-vmax, 0, vmax], extend=extend)
        ax.tricontour(x1, x2, b, levels=np.linspace(bmin, bmax, 20), linestyles="-", colors="k", alpha=0.3, linewidths=0.5)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.direction in ["x", "y"]:
            ax.set_yticks([x2.min(), 0])
            ax.set_yticklabels([r"$\dfrac{H}{L}$", r"$0$"])
        if self.direction == "y":
            ax.set_xticks([0, x1.max()])
            ax.set_xticklabels([r"$0$", r"$L$"])
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.set_title(rf"${self.direction} = {self.location:0.2f}$")
        ax.set_title(label)
        plt.savefig(output_file) 
        print(output_file)
        plt.close()

def circulation_plot(vtu_file, direction, location, n=2**8, output_file="image.png"):
    dataset = pv.read(vtu_file)
    p = dataset.points
    x_min, x_max = p[:, 0].min(), p[:, 0].max()
    y_min, y_max = p[:, 1].min(), p[:, 1].max()
    z_min, z_max = p[:, 2].min(), p[:, 2].max()
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    z = np.linspace(z_min, z_max, n)
    if direction == "x":
        x = location
        x1 = y
        x2 = z
        flow_comp = 'w'
        xlabel = r"$y$"
        ylabel = r"$z$"
    elif direction == "y":
        y = location
        x1 = x
        x2 = z
        flow_comp = 'w'
        xlabel = r"$x$"
        ylabel = r"$z$"
    elif direction == "z":
        z = location
        x1 = x
        x2 = y
        flow_comp = 'v'
        xlabel = r"$x$"
        ylabel = r"$y$"
    else:
        ValueError("'direction' must be one of 'x', 'y', or 'z'")

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = pv.PointSet(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
    samples = points.sample(dataset)
    flow = samples[flow_comp].reshape(n, n)

    # compute circulation as \phi = \int_{x_E}^x v dx = \int_{x_W}^x v dx - \int_{x_W}^{x_E} v dx
    circ = np.zeros_like(flow)
    for i in range(n):
        circ[:, i] = cumulative_trapezoid(flow[:, i], x1, initial=0) - trapezoid(flow[:, i], x1)
    circ[:, np.where(x2 < -0.5)] = 0

    # plot
    aspect_ratio = (x2.max() - x2.min())/(x1.max() - x1.min())
    width = 19/6
    vmax = np.nanmax(np.abs(circ))
    fig, ax = plt.subplots(1, figsize=(width, width*aspect_ratio))
    im = ax.pcolormesh(x1, x2, circ.T, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.contour(x1, x2, circ.T, levels=np.linspace(-0.9*vmax, 0.9*vmax, 10), colors="k", linestyles="-")
    plt.colorbar(im, ax=ax, label=r"$\phi$", shrink=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis('equal')
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(rf"$z = {z:0.2f}$")
    plt.savefig(output_file) 
    print(output_file)
    plt.close()

def diapycnal_vel_plot(vtu_file, y, bmin=None, bmax=None, n=2**8, output_file="image.png"):
    dataset = pv.read(vtu_file)
    p = dataset.points
    x_min, x_max = p[:, 0].min(), p[:, 0].max()
    z_min, z_max = p[:, 2].min(), p[:, 2].max()
    x = np.linspace(x_min, x_max, n)
    z = np.linspace(z_min, z_max, n)
    alpha = -z_min

    # params: sim034/35
    varepsilon = 6.9e-2
    muvarrho = 18

    # sample
    xx, zz = np.meshgrid(x, z, indexing="ij")
    points = pv.PointSet(np.column_stack([xx.ravel(), y*np.ones(n**2), zz.ravel()]))
    samples = points.sample(dataset)
    b = samples["b"].reshape(n, n)
    kappa = samples["kappa_v"].reshape(n, n)
    kappa *= alpha**2 * varepsilon**2 / muvarrho
    nanmask = (samples["vtkValidPointMask"] == 0).reshape(n, n)
    b[nanmask] = np.nan

    # diapycnal velocity e = -(kappa * b_z)_z / |b_z|
    bz = np.gradient(b, z, axis=1)
    e = -np.gradient(kappa*bz, z, axis=1) / (np.abs(bz) + 1e-16)

    # mask errors
    e[np.where(np.abs(e) > 0.1)] = 0.1

    # plot
    # vmax = np.nanmax(np.abs(e))
    vmax = 0.1
    if bmin is None:
        bmin = b.min()
    if bmax is None:
        bmax = b.max()
    fig, ax = plt.subplots(1)
    im = ax.pcolormesh(x, z, e.T, vmin=-vmax, vmax=vmax, cmap="RdBu_r", rasterized=True)
    plt.colorbar(im, ax=ax, label=r"$\tilde{e}$", shrink=0.5)
    ax.contour(x, z, b.T, levels=np.linspace(bmin, bmax, 20), colors="k", linestyles="-", alpha=0.3)
    ax.set_xlabel(r"Zonal coordinate $x$")
    ax.set_ylabel(r"Vertical coordinate")
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(rf"$y = {y:0.2f}$")
    plt.savefig(output_file) 
    print(output_file)
    plt.close()


if __name__ == "__main__":
    # for sim in range(33, 45):
    #     if sim in [34, 35, 36, 37]:
    #         dir = f"../sims/sim{sim:03d}b"
    #     elif sim in [27, 28, 33, 38, 39, 40, 41, 44]:
    #         dir = f"../sims/sim{sim:03d}a"
    #     else:
    #         dir = f"../sims/sim{sim:03d}"
    #     vtu_files = sorted(Path(f"{dir}/data/").glob("state_*.vtu"))
    #     vtu_file = vtu_files[-1]
    #     print(f"Creating slices from: {vtu_file}")

    #     sp = SlicePlotter(vtu_file)
    #     # for y in np.arange(-0.95, 1, 0.05):
    #     for y in [-0.25, 0.25, 0.5]:
    #         sp.set_slice("y", y)
    #         sp.plot("u", vmax=0.1, bmin=-15, bmax=-10, label=r"$u$", output_file=f"{dir}/images/u_slice_y{y:0.2f}.png")
    #         sp.plot("v", vmax=0.1, bmin=-15, bmax=-10, label=r"$v$", output_file=f"{dir}/images/v_slice_y{y:0.2f}.png")
    #         sp.plot("w", vmax=0.1, bmin=-15, bmax=-10, label=r"$w$", output_file=f"{dir}/images/w_slice_y{y:0.2f}.png")

    # for sim in [34, 35]:
    #     if sim in [27, 28, 34, 35, 36, 37]:
    #         dir = f"{wd}/../sims/sim{sim:03d}a"
    #     else:
    #         dir = f"{wd}/../sims/sim{sim:03d}"
    #     vtu_files = sorted(Path(f"{dir}/data/").glob("state_*.vtu"))
    #     vtu_file = vtu_files[-1]
    #     print(f"Creating slices from: {vtu_file}")
    #     for y in [-0.25]:
    #         diapycnal_vel_plot(vtu_file, y, bmin=-15, bmax=-10, output_file=f"{dir}/images/e_slice_y{y:0.2f}.png")

    # e.vtu
    # for sim in [27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]:
    for sim in [42, 43, 44, 45]:
        if sim in [34, 35, 36, 37]:
            dir = f"../sims/sim{sim:03d}b"
        elif sim in [27, 28, 33, 38, 39, 40, 41, 44, 48, 49]:
            dir = f"../sims/sim{sim:03d}a"
        else:
            dir = f"../sims/sim{sim:03d}"
        vtu_file = Path(f"{dir}/data/e.vtu")
        shutil.copy(f'/resnick/scratch/hppeters/{dir[8:]}/data/e.vtu', vtu_file)
        print(f"Creating slices from: {vtu_file}")

        sp = SlicePlotter(vtu_file)
        # for y in [-0.25, 0.25, 0.5]:
        for y in [-0.25]:
            sp.set_slice("y", y)
            sp.plot("e", bmin=-15, bmax=0, label=r"Diapycnal flow $\tilde{e}$", 
                    vmax=0.5,
                    # output_file=f"{dir}/images/e_slice_y{y:0.2f}.png")
                    output_file=f"{dir}/images/e_slice_y{y:0.2f}.pdf")

    ################################################################################

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

    ################################################################################

    # mesh = pv.read(file_name)
    # origin = mesh.center
    # normal = [0, 1, 0]
    # # slice_plane = pv.Plane(center=origin, normal=normal, i_size=mesh.bounds[1]-mesh.bounds[0], j_size=mesh.bounds[5]-mesh.bounds[4])
    # sliced_polydata = mesh.slice(normal=normal, origin=origin)

    # x_unstructured = sliced_polydata.points[:, 0]
    # z_unstructured = sliced_polydata.points[:, 2]
    # v_unstructured = sliced_polydata['v']

    # bounds = sliced_polydata.bounds
    # n = 2**8
    # spacing = (bounds[1] - bounds[0]) / (n-1), 0, (bounds[5] - bounds[4]) / (n-1)
    # grid_2d = pv.ImageData(dimensions=(n, 1, n), spacing=spacing, origin=(bounds[0], 0, bounds[4]))

    # # gridded_data = grid_2d.sample(sliced_polydata)
    # gridded_data = grid_2d.sample(mesh)

    # # nan mask
    # nan_mask = gridded_data['vtkValidPointMask'] == 0
    # u = gridded_data['u']
    # v = gridded_data['v']
    # w = gridded_data['w']
    # b = gridded_data['b']
    # u[nan_mask] = np.nan
    # v[nan_mask] = np.nan
    # w[nan_mask] = np.nan
    # b[nan_mask] = np.nan

    # # maxima
    # umax = np.nanmax(np.abs(u))
    # vmax = np.nanmax(np.abs(v))
    # wmax = np.nanmax(np.abs(w))

    # x_2d = gridded_data.points[:, 0].reshape((n, n))
    # z_2d = gridded_data.points[:, 2].reshape((n, n))
    # u_2d = u.reshape((n, n))
    # v_2d = v.reshape((n, n))
    # w_2d = w.reshape((n, n))
    # b_2d = b.reshape((n, n))

    # fig, ax = plt.subplots(3, 1, figsize=(3.2, 3*1.1))
    # c = ax[0].pcolormesh(x_2d, z_2d, u_2d, cmap='RdBu_r', vmin=-umax, vmax=umax, shading='gouraud')
    # cbar = fig.colorbar(c, ax=ax[0], label=r'$u$', shrink=0.8)
    # cbar.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    # ax[0].contour(x_2d, z_2d, b_2d, levels=np.arange(-0.9, 0.0, 0.1), 
    #           colors='k', linestyles='-', linewidths=0.5, alpha=0.25)
    # c = ax[1].pcolormesh(x_2d, z_2d, v_2d, cmap='RdBu_r', vmin=-vmax, vmax=vmax, shading='gouraud')
    # cbar = fig.colorbar(c, ax=ax[1], label=r'$v$', shrink=0.8)
    # cbar.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    # ax[1].contour(x_2d, z_2d, b_2d, levels=np.arange(-0.9, 0.0, 0.1), 
    #           colors='k', linestyles='-', linewidths=0.5, alpha=0.25)
    # c = ax[2].pcolormesh(x_2d, z_2d, w_2d, cmap='RdBu_r', vmin=-wmax, vmax=wmax, shading='gouraud')
    # cbar = fig.colorbar(c, ax=ax[2], label=r'$w$', shrink=0.8)
    # cbar.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    # ax[2].contour(x_2d, z_2d, b_2d, levels=np.arange(-0.9, 0.0, 0.1), 
    #           colors='k', linestyles='-', linewidths=0.5, alpha=0.25)
    # for a in ax:
    #     a.axis('equal')
    #     a.set_xticks([])
    #     a.set_yticks([-0.5, 0])
    #     a.spines['bottom'].set_visible(False)
    #     a.spines['left'].set_visible(False)
    #     a.set_ylabel(r'$z$')
    # ax[2].set_xticks(np.arange(-1, 1.1, 0.5))
    # ax[2].set_xlabel(r'$x$')
    # plt.savefig('example1a.png')
    # plt.close()