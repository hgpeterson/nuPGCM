import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


class SlicePlotter:
    def __init__(self, file_name):
        if not Path(file_name).exists():
            raise FileNotFoundError(f"VTU file not found: {file_name}")
        self.file_name = file_name
        self.dataset = pv.read(file_name)
        self.alpha = -self.dataset.points[:, 2].min()  # aspect ratio

        print(f"SlicePlotter initialized for VTU file: {vtu_file}")

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

    def plot(self, field_name, title=None, output_file="image.png", bmin=None, bmax=None, vmax=None):
        # slice with plane
        ds_slice = self.dataset.slice(normal=self.normal, origin=self.origin)

        p = ds_slice.points
        if self.direction == "x":
            x1 = p[:, 1]
            x2 = p[:, 2]
            figsize = (33 / 6, 33 / 6 / 1.62 / 2)
        elif self.direction == "y":
            x1 = p[:, 0]
            x2 = p[:, 2]
            figsize = (19 / 6, 19 / 6 / 1.62)
        elif self.direction == "z":
            x1 = p[:, 0]
            x2 = p[:, 1]
            figsize = (19 / 6 / 1.62, 19 / 6)

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
            elif vmax < field.max():
                extend = "max"
            elif -vmax < field.min():
                extend = "min"
            else:
                extend = "neither"

        b = ds_slice["b"]
        if bmax is None:
            bmax = b.max()
        if bmin is None:
            bmin = b.min()

        # plot
        fig, ax = plt.subplots(1, figsize=figsize)
        im = ax.tripcolor(x1, x2, field, vmin=-vmax, vmax=vmax, cmap="RdBu_r", shading="gouraud")
        plt.colorbar(im, ax=ax, shrink=0.5, ticks=[-vmax, 0, vmax], extend=extend)
        ax.tricontour(
            x1, x2, b, levels=np.linspace(bmin, bmax, 20), linestyles="-", colors="k", alpha=0.3, linewidths=0.5
        )
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.direction == "x":
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([x2.min(), 0])
        if self.direction == "y":
            ax.set_xticks([0, 1])
            ax.set_yticks([x2.min(), 0])
        if self.direction == "z":
            ax.axis("equal")
            ax.set_xticks([0, 1])
            ax.set_yticks([-1, 0, 1])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if title is None:
            title = rf"${field_name}$ at ${self.direction} = {self.location:0.2f}$"
        ax.set_title(title)
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
        flow_comp = "w"
        xlabel = r"$y$"
        ylabel = r"$z$"
    elif direction == "y":
        y = location
        x1 = x
        x2 = z
        flow_comp = "w"
        xlabel = r"$x$"
        ylabel = r"$z$"
    elif direction == "z":
        z = location
        x1 = x
        x2 = y
        flow_comp = "v"
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
    aspect_ratio = (x2.max() - x2.min()) / (x1.max() - x1.min())
    width = 19 / 6
    vmax = np.nanmax(np.abs(circ))
    fig, ax = plt.subplots(1, figsize=(width, width * aspect_ratio))
    im = ax.pcolormesh(x1, x2, circ.T, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    ax.contour(x1, x2, circ.T, levels=np.linspace(-0.9 * vmax, 0.9 * vmax, 10), colors="k", linestyles="-")
    plt.colorbar(im, ax=ax, label=r"$\phi$", shrink=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis("equal")
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(rf"$z = {z:0.2f}$")
    plt.savefig(output_file)
    print(output_file)
    plt.close()


if __name__ == "__main__":
    sims_dir = "/resnick/scratch/hppeters"
    sims = ["050b", "051e", "052", "053", "054", "055", "056"]
    xvals = [0.25, 0.5, 0.75]
    yvals = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    zvals = [-0.75, -0.5, -0.25]  # scaled by 1/alpha

    for sim in sims:
        dir = f"{sims_dir}/sim{sim}"

        # flow/isopycnal slices
        vtu_file = sorted(Path(f"{dir}/data/").glob("state_*.vtu"))[-1]
        sp = SlicePlotter(vtu_file)
        for x in xvals:
            sp.set_slice("x", x)
            sp.plot("u", bmin=-15, bmax=-10, output_file=f"{dir}/images/u_slice_x{x:0.2f}.png")
            sp.plot("v", bmin=-15, bmax=-10, output_file=f"{dir}/images/v_slice_x{x:0.2f}.png")
            sp.plot("w", bmin=-15, bmax=-10, output_file=f"{dir}/images/w_slice_x{x:0.2f}.png")
        for y in yvals:
            sp.set_slice("y", y)
            sp.plot("u", bmin=-15, bmax=-10, output_file=f"{dir}/images/u_slice_y{y:0.2f}.png")
            sp.plot("v", bmin=-15, bmax=-10, output_file=f"{dir}/images/v_slice_y{y:0.2f}.png")
            sp.plot("w", bmin=-15, bmax=-10, output_file=f"{dir}/images/w_slice_y{y:0.2f}.png")
        for z in zvals:
            sp.set_slice("z", z * sp.alpha)  # note the alpha scaling
            sp.plot("u", bmin=-15, bmax=-10, output_file=f"{dir}/images/u_slice_z{z:0.2f}a.png")
            sp.plot("v", bmin=-15, bmax=-10, output_file=f"{dir}/images/v_slice_z{z:0.2f}a.png")
            sp.plot("w", bmin=-15, bmax=-10, output_file=f"{dir}/images/w_slice_z{z:0.2f}a.png")

        # diapycnal flow slices
        vtu_file = Path(f"{dir}/data/e.vtu")
        if vtu_file.exists():
            sp = SlicePlotter(vtu_file)
            for y in yvals:
                sp.set_slice("y", y)
                sp.plot(
                    "e",
                    bmin=-15,
                    bmax=0,
                    title=rf"Diapycnal flow $\tilde{{e}}$ at $y = {y:0.2f}$",
                    output_file=f"{dir}/images/e_slice_y{y:0.2f}.png",
                )
        print()