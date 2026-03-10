import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from tqdm import tqdm
from time import time
from scipy.integrate import trapezoid, cumulative_trapezoid
from pathlib import Path
import os
import utils

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


def calculate_barotropic_streamfunction(vtu_file, n_grid=2**6, n_z_samples=2**5):
    # read the VTU file
    dataset = pv.read(vtu_file)

    if "u" not in dataset.array_names:
        raise ValueError(f"Velocity field 'u' not found. Available: {dataset.array_names}")

    # create 2D grid for (x, y) evaluation
    points = dataset.points
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    x_1d = np.linspace(x_min, x_max, n_grid)
    y_1d = np.linspace(y_min, y_max, n_grid)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d)
    z_grid = np.linspace(z_min, z_max, n_z_samples)

    # for each (x, y) point, integrate u(x, y, z) from z = -H to z = 0
    U_grid = np.zeros_like(x_grid)
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            x_ij = x_grid[j, i]
            y_ij = y_grid[j, i]

            # sample velocity along this vertical line
            line = pv.PointSet(np.array([[x_ij, y_ij, z] for z in z_grid]))
            samples = line.sample(dataset)

            # check which points are inside the mesh (points outside will be NaNs)
            valid_mask = samples["vtkValidPointMask"] == 1

            if any(valid_mask):
                # integrate U = ∫ u dz from -H to 0
                z_valid = z_grid[valid_mask]
                u_valid = samples["u"][valid_mask]
                U_grid[j, i] = trapezoid(u_valid, z_valid)

    # calculate streamfunction as Ψ(x,y) = ∫_y^L U(x, y') dy' from y = y to y = L
    # or equivalently, ∫_{-L}^L U(x, y') dy' - ∫_0^y U(x, y') dy'
    psi_grid = np.zeros_like(U_grid)
    for i in range(n_grid):
        psi_grid[:, i] = trapezoid(U_grid[:, i], y_1d) - cumulative_trapezoid(U_grid[:, i], y_1d, initial=0)

    return psi_grid, x_grid, y_grid, U_grid


def calculate_overturning_streamfunction(vtu_file, nx=2**8, ny=2**8, nz=2**8, printtime=False):
    if printtime:
        t0 = time()

    # read the VTU file
    dataset = pv.read(vtu_file)

    # time
    t = dataset["t"][0]

    # evenly-spaced grid
    grid = utils.Grid(dataset, nx, ny, nz)

    # sample
    samples = utils.sample_to_grid(dataset, grid)
    # v = samples["v"].reshape(nx, ny, nz)
    v = samples["u"][:, 1].reshape(nx, ny, nz)
    b = samples["b"].reshape(nx, ny, nz)

    # zonal means
    width = utils.zonal_width(samples, grid)
    v_bar = utils.zonal_mean(v, grid, width)
    b_bar = utils.zonal_mean(b, grid, width)

    # calculate streamfunction as Ψ(y,z) = -∫_-H^z v(y, z') dz'
    v_bar_filled = np.nan_to_num(v_bar, nan=0.0)  # replaces NaNs with 0
    psi_bar = -cumulative_trapezoid(v_bar_filled, grid.z, axis=1, initial=0)
    nan_mask = np.isnan(v_bar)
    psi_bar[nan_mask] = np.nan

    if printtime:
        print(f"psi computed in {time()-t0:.3e} s")

    return psi_bar, v_bar, b_bar, grid, t


def plot_barotropic_streamfunction(psi_grid, x_grid, y_grid, U_grid=None):
    fig, axes = plt.subplots(1, 2 if U_grid is not None else 1, figsize=(14, 5))

    if U_grid is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # Plot streamfunction
    levels = 20
    vmax = np.max(np.abs(psi_grid))
    cf1 = ax1.contourf(x_grid, y_grid, psi_grid, levels=levels, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.contour(x_grid, y_grid, psi_grid, levels=levels, colors="k", linewidths=0.5, alpha=0.3)
    plt.colorbar(cf1, ax=ax1, label="Streamfunction Ψ")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Barotropic Streamfunction")
    ax1.set_aspect("equal")

    # Plot depth-averaged velocity if provided
    if U_grid is not None:
        vmax = np.max(np.abs(U_grid))
        cf2 = ax2.contourf(x_grid, y_grid, U_grid, levels=levels, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(cf2, ax=ax2, label="U (depth-averaged velocity)")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Depth-Averaged Velocity U")
        ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()


def plot_zonal_mean(field, grid, b, label="", cb_label="", rescale_z=True, t=None, i=None, cmap="RdBu_r", cb_sym=True):
    y = grid.y
    z = grid.z

    if rescale_z:
        alpha = -np.min(z)
        z = z / alpha / 2

    fig, ax = plt.subplots(1, figsize=(33/6, 33/6/1.62))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    vmax = np.nanmax(np.abs(field))
    if cb_sym:
        vmin = -vmax
    else:
        vmin = np.nanmin(np.abs(field))
    cf = ax.pcolormesh(y, z, field.T, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cf, label=cb_label, shrink=0.5)
    levels = np.linspace(-15, 15, 40)
    ax.contour(y, z, -b.T, levels=levels, colors="k", linewidths=0.5, linestyles="-", alpha=0.3)
    ax.set_xlabel(r"$y$")
    if rescale_z:
        ax.set_ylabel(rf"$z$ (rescaled, $\alpha = {alpha:0.3f}$)")
    else:
        ax.set_ylabel(r"$z$")
    ax.set_aspect("equal")
    if t is not None:
        ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    if i is not None:
        filename = f"{label}{i:016d}.png"
    else:
        filename = f"{label}.png"
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_overturning_streamfunction(psi, b_bar, grid, t=None, filename="psi.png", bmax=10):
    y = grid.y
    z = grid.z

    fig, ax = plt.subplots(1, figsize=(33/6, 33/6/1.62/2))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    vmax = np.nanmax(np.abs(psi))
    cf1 = ax.pcolormesh(y, z, psi.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    levels = np.linspace(-0.9 * vmax, 0.9 * vmax, 8)
    ax.contour(y, z, psi.T, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    cb = plt.colorbar(cf1, label=r"Streamfunction $\bar{\psi}$")
    cb.ax.set_yticks([-vmax, 0, vmax])
    cb.ax.set_yticklabels([r"$-$Max", r"$0$", r"Max"])
    ax.text(0.65, 0.02, rf"Max = {utils.to_latex_sci(vmax)}")
    levels = np.linspace(-bmax, bmax, 40)
    ax.contour(y, z, -b_bar.T, levels=levels, colors="k", linewidths=0.5, alpha=0.3)
    y0 = -0.6875
    y1 = -0.5
    y_c = np.linspace(y0, y1, 100)
    ax.plot(y_c, z.min()*(1 - ((y_c - y0)/(y1 - y0))**2), "k--", lw=0.5, alpha=0.4)
    # ax.axvline(-0.5, c="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([z.min(), 0])
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$z$")
    if t is not None:
        ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print(filename)
    plt.close()


if __name__ == "__main__":

    # psi_bar, v_bar, b_bar, grid, t = calculate_overturning_streamfunction("../velocity_linear.vtu", printtime=True)
    # plot_overturning_streamfunction(
    #     psi_bar, b_bar, grid, t=t, filename=f"../sims/sim024/images/psi{15100:016d}.png", bmax=15
    # )

    overwrite = False
    sim = 24
    dir = f"../sims/sim{sim:03d}"
    # dir = "../scratch/h0.10_wind_bim"
    vtu_files = sorted(Path(f"{dir}/data/").glob("*.vtu"))

          
    for vtu_file in vtu_files:
        i = int(vtu_file.stem.split("_")[1])  # assuming file is of the form "/foo/bar/state_{i:016d}.vtu"
        img_file = f"{dir}/images/psi{i:016d}.png"
        if os.path.exists(img_file) and not overwrite:
            print("Skipping " + img_file)
            continue

        psi_bar, v_bar, b_bar, grid, t = calculate_overturning_streamfunction(vtu_file)
        plot_overturning_streamfunction(
            psi_bar, b_bar, grid, t=t, filename=img_file, bmax=15
        )

        # dataset = pv.read(vtu_file)
        # t = dataset["t"][0]
        # n = 2**8
        # grid = utils.Grid(dataset, n, n, n)
        # samples = utils.sample_to_grid(dataset, grid)
        # b = samples["b"].reshape(n, n, n)
        # # nu = samples["nu"].reshape(n, n, n)
        # # kappa_v = samples["kappa_v"].reshape(n, n, n)
        # # w = samples["w"].reshape(n, n, n)
        # adv = samples["advection"].reshape(n, n, n)

        # width = utils.zonal_width(samples, grid)
        # b = utils.zonal_mean(b, grid, width)
        # # nu = utils.zonal_mean(nu, grid, width)
        # # kappa_v = utils.zonal_mean(kappa_v, grid, width)
        # # w = utils.zonal_mean(w, grid, width)
        # adv = utils.zonal_mean(adv, grid, width)

        # # plot_zonal_mean(nu, grid, b, label="nu", cb_label=r"$\bar \nu$", t=t, i=i, cmap="viridis", cb_sym=False)
        # # plot_zonal_mean(
        # #     kappa_v, grid, b, label="kappa_v", cb_label=r"$\bar \kappa_v$", t=t, i=i, cmap="viridis", cb_sym=False
        # # )
        # # plot_zonal_mean(w, grid, b, label="w", cb_label=r"$\bar w$", t=t, i=i, cmap="RdBu_r", cb_sym=True)
        # plot_zonal_mean(adv, grid, b, label="adv", cb_label=r"$\overline{\vec{u} \cdot \nabla b}$", t=t, i=i, cmap="RdBu_r", cb_sym=True)