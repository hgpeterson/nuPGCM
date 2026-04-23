import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from time import time
from scipy.integrate import trapezoid, cumulative_trapezoid
from pathlib import Path
import os
import utils

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


def calculate_barotropic_streamfunction(vtu_file, nx=2**8, ny=2**8, nz=2**8, printtime=False):
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
    u = samples["u"][:, 0].reshape(nx, ny, nz)

    # vertical integral
    U = trapezoid(u, x=grid.z, axis=2)
    H = utils.depth(samples, grid)

    # calculate streamfunction as Psi(x, y) = -∫_0^y U(x, y') dx
    Psi = trapezoid(U, grid.y, axis=1) - cumulative_trapezoid(U, grid.y, axis=1, initial=0)
    nan_mask = np.where(H == 0)
    U[nan_mask] = np.nan
    Psi[nan_mask] = np.nan

    if printtime:
        print(f"barotropic streamfunction computed in {time() - t0:.3e} s")

    return Psi, U, grid, t


def calculate_overturning_streamfunction(vtu_file, nx=2**8, ny=2**8, nz=2**8, printtime=False):
    if printtime:
        t0 = time()

    # read the VTU file
    dataset = pv.read(vtu_file)

    # time
    t = dataset["t"][0]

    # evenly-spaced grid
    grid = utils.Grid(dataset, nx, ny, nz)
    alpha = -grid.z.min()  # aspect ratio

    # sample
    samples = utils.sample_to_grid(dataset, grid)
    v = samples["u"][:, 1].reshape(nx, ny, nz)
    b = samples["b"].reshape(nx, ny, nz)

    # zonal means
    width = utils.zonal_width(samples, grid)
    v_int = trapezoid(v, x=grid.x, axis=0)
    b_bar = utils.zonal_mean(b, grid, width)

    # calculate streamfunction as psi(y,z) = -1/α * ∫_-H^z v(y, z') dz'
    psi_bar = -1 / alpha * cumulative_trapezoid(v_int, grid.z, axis=1, initial=0)
    nan_mask = np.where(width == 0)
    v_int[nan_mask] = np.nan
    psi_bar[nan_mask] = np.nan

    if printtime:
        print(f"psi computed in {time() - t0:.3e} s")

    return psi_bar, v_int, b_bar, grid, t


def plot_barotropic_streamfunction(Psi, grid, t=None, filename="psi_baro.png", Psimax=None, maskchannel=False):
    x = grid.x
    y = grid.y
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots(1, figsize=(19 / 6, 19 / 6 * 1.62))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if maskchannel:
        nan_mask = np.where((yy < -0.5))
        Psi[nan_mask] = np.nan
    if Psimax is None:
        Psimax = np.nanmax(np.abs(Psi))

    cf1 = ax.pcolormesh(x, y, Psi.T, cmap="RdBu_r", vmin=-Psimax, vmax=Psimax, rasterized=True)
    levels = np.linspace(-0.9 * Psimax, 0.9 * Psimax, 8)
    ax.contour(x, y, Psi.T, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    cb = plt.colorbar(cf1, label=r"Barotropic streamfunction $\Psi$", shrink=0.5)
    cb.ax.set_yticks([-Psimax, 0, Psimax])
    cb.ax.set_yticklabels([r"$-$Max", r"$0$", r"Max"])
    ax.text(0.8, 1.02, rf"Max = {utils.to_latex_sci(Psimax)}", transform=ax.transAxes, size=7)
    if maskchannel:
        ax.fill_between(x, -0.5, y.min(), color="k", alpha=0.1, ec="none")
    ax.axhline(-0.5, c="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xticks([0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"Zonal coordinate $x$")
    ax.set_ylabel(r"Meridional coordinate $y$")
    if t is not None:
        ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_zonal_mean(field, grid, b, label="", cb_label="", rescale_z=True, t=None, i=None, cmap="RdBu_r", cb_sym=True):
    y = grid.y
    z = grid.z

    if rescale_z:
        alpha = -np.min(z)
        z = z / alpha / 2

    fig, ax = plt.subplots(1, figsize=(33 / 6, 33 / 6 / 1.62))
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


def plot_overturning_streamfunction(psi, b_bar, grid, t=None, filename="psi.png", bmin=None, bmax=None, geometry="", 
                                    psimax=None):
    y = grid.y
    z = grid.z

    alpha = -z.min()  # aspect ratio

    if bmin is None:
        bmin = b_bar.min()
    if bmax is None:
        bmax = b_bar.max()

    fig, ax = plt.subplots(1, figsize=(33 / 6, 33 / 6 / 1.62 / 2))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if psimax is None:
        psimax = np.nanmax(np.abs(psi))
    cf1 = ax.pcolormesh(y, z, psi.T, cmap="RdBu_r", vmin=-psimax, vmax=psimax, rasterized=True)
    levels = np.linspace(-0.9 * psimax, 0.9 * psimax, 8)
    ax.contour(y, z, psi.T, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    cb = plt.colorbar(cf1, label=r"Streamfunction $\psi$")
    cb.ax.set_yticks([-psimax, 0, psimax])
    cb.ax.set_yticklabels([r"$-$Max", r"$0$", r"Max"])
    ax.text(0.8, 1.02, rf"Max = {utils.to_latex_sci(psimax)}", transform=ax.transAxes, size=7)
    levels = np.linspace(bmin, bmax, 20)
    ax.contour(y, z, b_bar.T, levels=levels, colors="k", linestyles="-", linewidths=0.5, alpha=0.3)
    if geometry == "slope":
        y0 = -0.6875
        y1 = -0.5
        y_c = np.linspace(y0, y1, 100)
        ax.plot(y_c, z.min() * (1 - ((y_c - y0) / (y1 - y0)) ** 2), "k--", lw=0.5, alpha=0.4)
    elif geometry == "flat":
        ax.axvline(-0.5, c="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels([r"$-L$", r"$0$", r"$L$"])
    ax.set_yticks([-alpha, 0])
    ax.set_yticklabels([r"$-\dfrac{H}{L}$", "0"])
    ax.set_xlabel(r"Meridional coordinate $y$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    if t is not None:
        ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print(filename)
    plt.close()


if __name__ == "__main__":
    overwrite = False
    # overwrite = True
    sims = ["050b", "051e"]
    geoms = ["slope", "flat"]
    # sims_dir = "../sims"
    sims_dir = "/resnick/scratch/hppeters"
    for i in range(len(sims)):
        sim = sims[i]
        geom = geoms[i]

        dir = f"{sims_dir}/sim{sim}"
        vtu_files = sorted(Path(f"{dir}/data/").glob("state_*.vtu"))

        for vtu_file in vtu_files:
        # for vtu_file in [vtu_files[-1]]:
            i = int(vtu_file.stem.split("_")[1])  # assuming file is of the form "/foo/bar/state_{i:016d}.vtu"

            # img_file = f"{dir}/images/psi{i:016d}.png"
            # if os.path.exists(img_file) and not overwrite:
            #     print("Skipping " + img_file)
            #     continue
            # n = 2**7
            # psi_bar, v_bar, b_bar, grid, t = calculate_overturning_streamfunction(
            #     vtu_file, nx=n, ny=n, nz=n, printtime=True
            # )
            # plot_overturning_streamfunction(psi_bar, b_bar, grid, 
            #                                 t=t, 
            #                                 filename=img_file, 
            #                                 bmin=-15, 
            #                                 # bmax=0, 
            #                                 bmax=-10, 
            #                                 geometry=geom)

            img_file = f"{dir}/images/psi_baro{i:016d}.png"
            if os.path.exists(img_file) and not overwrite:
                print("Skipping " + img_file)
                continue
            n = 2**7
            Psi, U, grid, t = calculate_barotropic_streamfunction(vtu_file, nx=n, ny=n, nz=n, printtime=True)
            plot_barotropic_streamfunction(Psi, grid, t=t, filename=img_file)

            img_file = f"{dir}/images/psi_baro_mask{i:016d}.png"
            if os.path.exists(img_file) and not overwrite:
                print("Skipping " + img_file)
                continue
            plot_barotropic_streamfunction(Psi, grid, t=t, filename=img_file, maskchannel=True)

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
