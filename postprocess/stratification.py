import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import utils

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


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
    ax.contour(
        x,
        z,
        b[:, iy, :].T,
        levels=blevels,
        colors="w",
        linewidths=0.5,
        linestyles="-",
        alpha=0.5,
    )
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([z.min(), 0])
    ax.set_xlabel(r"Zonal coordinate $x$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    if t is not None:
        ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print("Saved", filename)
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
    print("Saved", filename)
    plt.close()


def plot_stratifications(N2_bars, grid, labels, filename="strats.png"):
    """Plot average stratification profiles"""

    z = grid.z
    fig, ax = plt.subplots(1, figsize=(19 / 6 / 1.62, 19 / 6))
    for i, N2_bar in enumerate(N2_bars):
        ax.semilogx(N2_bar, z, label=labels[i])
    ax.legend()
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(z.min(), 0)
    ax.set_xlabel(r"Average stratification $\overline{\alpha \partial_z b}$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    plt.savefig(filename)
    print("Saved", filename)
    plt.close()


def plot_b_flux_slice(F, b, mask, grid, y0, t=None, filename="b_flux.png", bmin=-15, bmax=-10):
    """Plot buoyancy flux and isopycnals at y0"""

    x = grid.x
    y = grid.y
    z = grid.z
    iy = np.searchsorted(y, y0)
    fig, ax = plt.subplots(1, figsize=(19 / 6, 19 / 6 / 1.62))
    F_log = np.copy(F)
    F_log[np.where(F < 0)] = np.log10(-F[np.where(F < 0)])
    F_log[np.where(F >= 0)] = np.nan
    F_log[np.where(mask == 0)] = np.nan
    b[np.where(mask == 0)] = np.nan
    im = ax.pcolormesh(x, z, F_log[:, iy, :].T, shading="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label=r"Buoyancy flux $-\log \alpha \kappa \partial_z b$")
    blevels = np.linspace(bmin, bmax, 20)
    ax.contour(
        x,
        z,
        b[:, iy, :].T,
        levels=blevels,
        colors="w",
        linewidths=0.5,
        linestyles="-",
        alpha=0.5,
    )
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([z.min(), 0])
    ax.set_xlabel(r"Zonal coordinate $x$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    if t is not None:
        ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print("Saved", filename)
    plt.close()


def plot_b_fluxes(F_bars, grid, labels, filename="bfluxes.png"):
    """Plot buoyancy flux profiles"""

    z = grid.z
    fig, ax = plt.subplots(1, figsize=(19 / 6 / 1.62, 19 / 6))
    ax.spines["left"].set_visible(False)
    ax.axvline(3, lw=0.5, c="k", ls="-")
    for i, F_bar in enumerate(F_bars):
        ax.plot(-np.log10(-F_bar), z, label=labels[i])
    ax.legend()
    ax.set_xlim(-3, 3.1)
    ax.set_xticks([-3, -1, 1, 3])
    ax.set_xticklabels([r"$-10^{3}$", r"$-10^{1}$", r"$-10^{-1}$", r"$-10^{-3}$"])
    ax.set_ylim(z.min(), 0)
    ax.set_xlabel(r"Integrated buoyancy flux $-\iint \alpha \kappa \partial_z b \; \mathrm{d}x\mathrm{d}y$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    plt.savefig(filename)
    print("Saved", filename)
    plt.close()


if __name__ == "__main__":
    # sims = ["050b", "051e"]
    # sims = ["052", "053"]
    # sims = ["055", "056"]
    # sims = ["057", "058"]
    # sims = ["061", "063"]
    sims = ["062", "064"]
    sims_dir = Path("/resnick/scratch/hppeters")
    N2_bars_channel = []
    N2_bars_basin = []
    F_ints_channel = []
    F_ints_basin = []
    for sim in sims:
        dir = sims_dir / f"sim{sim}"
        vtu_file = sorted((dir / "data").glob("state_*.vtu"))[-1]
        i = int(vtu_file.stem.split("_")[1])  # assuming file is of the form "/foo/bar/state_{i:016d}.vtu"

        n = 2**7
        fields_dict, grid, t = utils.sample_fields(
            vtu_file, ["b", "alpha*b_z", "kappa_v"], nx=n, ny=n, nz=n, printtime=True
        )
        N2 = fields_dict["alpha*b_z"]
        b = fields_dict["b"]
        kappa = fields_dict["kappa_v"]
        F = -kappa * N2
        mask = fields_dict["mask"]

        # slices
        img_file = dir / f"images/strat_basin_y0_{i:016d}.png"
        plot_stratification_slice(N2, b, mask, grid, 0, t=t, filename=img_file)

        img_file = dir / f"images/b_flux_basin_y0_{i:016d}.png"
        plot_b_flux_slice(F, b, mask, grid, 0, t=t, filename=img_file)

        # basin avg
        N2_bar = utils.horizontal_integral(N2, mask, grid, ymin=-0.5, ymax=1, area_weighted=True)
        F_int = utils.horizontal_integral(F, mask, grid, ymin=-0.5, ymax=1)
        N2_bars_basin.append(N2_bar)
        F_ints_basin.append(F_int)
        img_file = dir / f"images/strat_basin{i:016d}.png"
        plot_stratification(N2_bar, grid, t=t, filename=img_file)

        # channel avg
        N2_bar = utils.horizontal_integral(N2, mask, grid, ymin=-1.0, ymax=-0.5, area_weighted=True)
        F_int = utils.horizontal_integral(F, mask, grid, ymin=-1.0, ymax=-0.5)
        N2_bars_channel.append(N2_bar)
        F_ints_channel.append(F_int)
        img_file = dir / f"images/strat_channel{i:016d}.png"
        plot_stratification(N2_bar, grid, t=t, filename=img_file)

    sims_str = "_".join(sims)
    plot_stratifications(N2_bars_basin, grid, sims, filename=f"images/strat_basin_{sims_str}.png")
    plot_stratifications(N2_bars_channel, grid, sims, filename=f"images/strat_channel_{sims_str}.png")
    plot_b_fluxes(F_ints_basin, grid, sims, filename=f"images/b_fluxes_basin_{sims_str}.png")
    plot_b_fluxes(F_ints_channel, grid, sims, filename=f"images/b_fluxes_channel_{sims_str}.png")
