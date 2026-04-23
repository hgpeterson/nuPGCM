import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from time import time
from scipy.integrate import trapezoid
from pathlib import Path
import os
import utils

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


def calculate_stratification(vtu_file, nx=2**8, ny=2**8, nz=2**8, printtime=False):
    """Calculate stratification from a VTU file"""

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
    b = samples["b"].reshape(nx, ny, nz)
    N2 = samples["alpha*b_z"].reshape(nx, ny, nz)
    mask = samples["vtkValidPointMask"].reshape(nx, ny, nz)

    # if there's no "alpha*b_z" field, compute it from "b"
    # alpha = -grid.z.min()  # aspect ratio
    # N2 = alpha * np.gradient(b, grid.z, axis=2)

    if printtime:
        print(f"stratification computed in {time() - t0:.3f} s")

    return b, N2, mask, grid, t


def average_stratification(N2, mask, grid, ymin=-1, ymax=1):
    """Calculate average stratification between ymin and ymax"""

    # take subset
    iymin = np.searchsorted(grid.y, ymin)
    iymax = np.searchsorted(grid.y, ymax)
    N2 = N2[:, iymin : iymax + 1, :]
    mask = mask[:, iymin : iymax + 1, :]

    # clean up before integrating
    N2[np.where(N2 < 0)] = 0
    N2[np.where(mask == 0)] = 0

    # 2D horizontal average
    area = trapezoid(trapezoid(mask, x=grid.x, axis=0), x=grid.y[iymin : iymax + 1], axis=0)
    N2_bar = trapezoid(trapezoid(N2, x=grid.x, axis=0), x=grid.y[iymin : iymax + 1], axis=0) / area

    return N2_bar


def plot_stratification_slice(N2, b, mask, grid, y0, t=None, filename="strat.png", vmin=-2, bmin=-15, bmax=-10):
    """Plot stratification and isopycnals at y0"""

    x = grid.x
    y = grid.y
    z = grid.z
    iy = np.searchsorted(y, y0)
    fig, ax = plt.subplots(1, figsize=(19 / 6, 19 / 6 / 1.62))
    N2_log = np.copy(N2)
    N2_log[np.where(N2 > 0)] = np.log10(N2[np.where(N2 > 0)])
    N2_log[np.where(N2 <= 0)] = vmin
    N2_log[np.where(mask == 0)] = np.nan
    b[np.where(mask == 0)] = np.nan
    im = ax.pcolormesh(x, z, N2_log[:, iy, :].T, shading="auto", cmap="viridis", vmin=vmin, vmax=1)
    plt.colorbar(im, ax=ax, label=r"Stratification $\log \alpha \partial_z b$", extend="both")
    blevels = np.linspace(bmin, bmax, 20)
    ax.contour(x, z, b[:, iy, :].T, levels=blevels, colors="w", linewidths=0.5, linestyles="-", alpha=0.5)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([z.min(), 0])
    ax.set_xlabel(r"Zonal coordinate $x$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    if t is not None:
        ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_stratification(N2_bar, grid, t=None, filename="strat.png"):
    """Plot average stratification profile"""

    z = grid.z
    fig, ax = plt.subplots(1, figsize=(19 / 6 / 1.62, 19 / 6))
    ax.semilogx(N2_bar, z, "k-")
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(z.min(), 0)
    ax.set_xlabel(r"Average stratification $\overline{\alpha \partial_z b}$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    if t is not None:
        ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_stratification_dict(dict, filename="strat.png"):
    """Plot average stratification profiles from a label-to-profile dictionary"""

    fig, ax = plt.subplots(1, figsize=(19 / 6 / 1.62, 19 / 6))
    for label in dict:
        N2_bar, z = dict[label]
        ax.semilogx(N2_bar, z, label=label)
    ax.legend()
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(z.min(), 0)
    ax.set_xlabel(r"Average stratification $\overline{\alpha \partial_z b}$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    plt.savefig(filename)
    print(filename)
    plt.close()


if __name__ == "__main__":
    # overwrite = False
    overwrite = True
    sims = ["050b", "051e"]
    # sims = ["050b"]
    sims_dir = "/resnick/scratch/hppeters"
    N2_basin_dict = {}
    N2_channel_dict = {}
    for sim in sims:
        dir = f"{sims_dir}/sim{sim}"
        vtu_file = sorted(Path(f"{dir}/data/").glob("state_*.vtu"))[-1]
        i = int(vtu_file.stem.split("_")[1])  # assuming file is of the form "/foo/bar/state_{i:016d}.vtu"

        img_file = f"{dir}/images/strat_basin{i:016d}.png"
        if os.path.exists(img_file) and not overwrite:
            print("Skipping " + img_file)
            continue
        n = 2**7
        b, N2, mask, grid, t = calculate_stratification(vtu_file, nx=n, ny=n, nz=n, printtime=True)
        N2_bar = average_stratification(N2, mask, grid, ymin=-0.5, ymax=1)
        N2_basin_dict[sim] = (N2_bar, grid.z)
        plot_stratification(N2_bar, grid, t=t, filename=img_file)

        img_file = f"{dir}/images/strat_basin_y0_{i:016d}.png"
        plot_stratification_slice(N2, b, mask, grid, 0, t=t, filename=img_file)

        img_file = f"{dir}/images/strat_channel{i:016d}.png"
        N2_bar = average_stratification(N2, mask, grid, ymin=-1, ymax=-0.5)
        N2_channel_dict[sim] = (N2_bar, grid.z)
        plot_stratification(N2_bar, grid, t=t, filename=img_file)

    sims_str = "_".join(sims)
    plot_stratification_dict(N2_basin_dict, filename=f"images/strat_basin_{sims_str}.png")
    plot_stratification_dict(N2_channel_dict, filename=f"images/strat_channel_{sims_str}.png")
