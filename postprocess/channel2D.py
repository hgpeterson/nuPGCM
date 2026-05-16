import numpy as np
import pyvista as pv
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from pathlib import Path
import utils

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


def plot_fieldb(vtu_file, field, label=None, vmax=None, filename="field.png"):
    # read VTU file
    dataset = pv.read(vtu_file)

    # prep data
    coords = dataset.points
    if field == "u":
        f = dataset["u"][:, 0]
    elif field == "v":
        f = dataset["u"][:, 1]
    elif field == "w":
        f = dataset["u"][:, 2]
    else:
        f = dataset[field]
    b = dataset["b"]
    y = coords[:, 1]
    z = coords[:, 2]
    t = dataset["t"][0]
    tri = dataset.cells_dict[22]  # 22 = quadratic triangle

    # vmax for colorbar
    if vmax is None:
        print(f"max({field}) = {np.max(np.abs(f)):.3e}")
        vmax = np.max(np.abs(f))

    # data is 2D, create a tri-plot
    fig, ax = plt.subplots(1, figsize=(19 / 6, 19 / 6))
    im = ax.tripcolor(y, z, f, triangles=tri, vmin=-vmax, vmax=vmax, shading="gouraud", cmap="RdBu_r")
    ax.tricontour(y, z, b, levels=20, colors="k", alpha=0.25, linestyles="-", linewidths=0.5)
    if label is None:
        label = field
    plt.colorbar(im, label=label, fraction=0.03)
    ax.triplot(y, z, tri[:, 0:3], linestyle="-", color="k", linewidth=0.25, alpha=0.1)
    ax.set_xlabel(r"Meridional coordinate $y$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_psib(
    vtu_file,
    n=2**8,
    bmin=None,
    bmax=None,
    vmax=None,
    filename="psi.png",
):
    # read VTU file
    dataset = pv.read(vtu_file)

    # load data
    x = dataset.points[0, 0]  # should all be the same
    y = dataset.points[:, 1]
    z = dataset.points[:, 2]
    t = dataset["t"][0]
    alpha = np.max(np.abs(z))

    # create 2D grid for (y, z) evaluation
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    y_1d = np.linspace(y_min, y_max, n)
    z_1d = np.linspace(z_min, z_max, n)
    y_grid, z_grid = np.meshgrid(y_1d, z_1d, indexing="ij")

    # evaluate v and b on the grid
    points = pv.PointSet(np.column_stack([x * np.ones(n**2), y_grid.ravel(), z_grid.ravel()]))
    samples = points.sample(dataset)
    v_grid = samples["u"][:, 1].reshape(n, n)
    b_grid = samples["b"].reshape(n, n)

    # integrate: -alpha*dz(psi) = v
    psi = -1 / alpha * cumulative_trapezoid(v_grid, z_1d, initial=0)

    # mask points outside the domain
    nan_mask = (samples["vtkValidPointMask"] == 0).reshape(n, n)
    b_grid[nan_mask] = np.nan
    psi[nan_mask] = np.nan

    # max value for colorbar
    if vmax is None:
        vmax = np.nanmax(np.abs(psi))
        print(f"max(psi) = {vmax:0.3e}")

    # plotting
    fig, ax = plt.subplots(1, figsize=(19 / 6, 19 / 6))
    im = ax.pcolormesh(y_grid, z_grid, psi, vmin=-vmax, vmax=vmax, shading="gouraud", cmap="RdBu_r")
    ax.contour(
        y_grid,
        z_grid,
        psi,
        levels=np.linspace(-0.9 * vmax, 0.9 * vmax, 8),
        colors="k",
        linestyles="-",
        linewidths=0.25,
    )
    if bmin is None:
        bmin = np.nanmin(b_grid)
    if bmax is None:
        bmax = np.nanmax(b_grid)
    levels = np.linspace(bmin, bmax, 20)
    ax.contour(
        y_grid,
        z_grid,
        b_grid,
        levels=levels,
        colors="k",
        alpha=0.25,
        linestyles="-",
        linewidths=0.5,
    )
    cb = plt.colorbar(im, label=r"Streamfunction $\psi$", fraction=0.03)
    cb.ax.set_yticks([-vmax, 0, vmax])
    cb.ax.set_yticklabels([r"$-$Max", r"$0$", r"$+$Max"])
    ax.annotate(f"Max = {utils.to_latex_sci(vmax)}", xy=(0.92, 1.02), xycoords="axes fraction")
    tri = dataset.cells_dict[22]
    ax.triplot(y, z, tri[:, 0:3], "k-", linewidth=0.25, alpha=0.1)
    ax.set_xlabel(r"Meridional coordinate $y$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_psi_profile(vtu_file, y, n=2**8):
    # read VTU file
    dataset = pv.read(vtu_file)

    # load data
    x = dataset.points[0, 0]  # should all be the same
    z = dataset.points[:, 2]
    t = dataset["t"][0]
    alpha = np.max(np.abs(z))

    # create 1D grid for evaluation
    z_min, z_max = z.min(), z.max()
    z_1d = np.linspace(z_min, z_max, n)

    # evaluate v on the grid and compute psi
    line = pv.PointSet(np.array([[x, y, z] for z in z_1d]))
    samples = line.sample(dataset)
    v = samples["v"]
    psi = -1 / alpha * cumulative_trapezoid(v, z_1d, initial=0)

    # mask points outside the domain
    nan_mask = samples["vtkValidPointMask"] == 0
    psi[nan_mask] = np.nan

    # plotting
    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.spines["left"].set_visible(False)
    ax.axvline(0, lw=0.5, c="k", ls="-")
    ax.plot(psi, z_1d)
    ax.set_xlabel(r"$\Psi$")
    ax.set_ylabel(r"$z$")
    ax.set_title(rf"$y = {y:0.2f}$")
    img_file = "images/psi_profile.png"
    plt.savefig(img_file)
    print(img_file)
    plt.close()


def plot_surface_b_flux(vtu_file, n=2**8, show_progress=False):
    # hardcode parameters for now:
    Ek = np.sqrt(1e-1)
    PrBu = 1

    dataset = pv.read(vtu_file)
    x = dataset.points[0, 0]
    y = dataset.points[:, 1]
    z = dataset.points[:, 2]
    t = dataset["t"][0]
    alpha = np.max(np.abs(z))
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    y_1d = np.linspace(y_min, y_max, n)
    dz = (z_max - z_min) / (n - 1)
    z_1d = z_max - np.array([2 * dz, dz, 0])
    surface_b_flux = np.zeros(n)
    for i in tqdm(range(n), disable=(not show_progress)):
        y_i = y_1d[i]

        line = pv.PointSet(np.array([[x, y_i, z] for z in z_1d]))
        samples = line.sample(dataset)

        b = samples["b"]
        kappa_v = samples["kappa_v"]
        bz = 1 / dz * (1 / 2 * b[-3] - 2 * b[-2] + 3 / 2 * b[-1])
        surface_b_flux[i] = alpha * Ek**2 / PrBu * kappa_v[-1] * bz
    surface_b_flux[0] = 0  # H = 0 here

    fig, ax = plt.subplots(1)
    ax.plot(y_1d, surface_b_flux)
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$F$")
    ax.set_ylim(-1e-1, 1e-1)
    ax.spines["bottom"].set_visible(False)
    ax.axhline(0, lw=0.5, c="k", ls="-")
    plt.savefig("images/sfc_b_flux.png")
    print("images/sfc_b_flux.png")
    plt.close()


def plot_isopycnal_depth(vtu_files, bs=np.arange(-14.5, -12.5, 0.5), y0=-0.75, n=2**8, filename="isopycnal_depths.png"):
    # same profile for every file
    dataset0 = pv.read(vtu_files[0])
    x0 = dataset0.points[0, 0]
    zp = dataset0.points[:, 2]
    z = np.linspace(zp.min(), zp.max(), n)
    profile = pv.PointSet(np.array([[x0, y0, z] for z in z]))

    # loop over files
    depths = np.zeros((len(vtu_files), len(bs)))
    ts = np.zeros(len(vtu_files))
    for i, vtu_file in enumerate(vtu_files):
        dataset = pv.read(vtu_file)

        samples = profile.sample(dataset)
        b = samples["b"]
        ts[i] = dataset["t"][0]

        # find depth where b crosses the specified value
        for j, b0 in enumerate(bs):
            if (b0 < b.min()) or (b0 > b.max()):
                depths[i, j] = np.nan
            else:
                depths[i, j] = np.interp(b0, b, z)

    fig, ax = plt.subplots(1)
    for j, b0 in enumerate(bs):
        ax.plot(ts, depths[:, j], label=rf"$b = {b0:.1f}$")
    ax.legend()
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(z[0], z[-1])
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    ax.set_title(rf"Depth of isopycnal at $y = {y0:.2f}$")
    plt.savefig(filename)
    print(filename)
    plt.close()


if __name__ == "__main__":
    # overwrite = True
    overwrite = False
    sims_dir = Path("/home/hppeters/group_dir/nuPGCM/scratch/channel2D")
    # sims = ["000", "001", "002"]
    sims = ["002", "003"]
    for sim in sims:
        dir = sims_dir / f"sim{sim}"
        print(f"Processing files in {dir}")
        vtu_files = sorted((dir / "data").glob("state_*.vtu"))

        # plot_isopycnal_depth(vtu_files, filename=dir / "images/isopycnal_depths.png")
        plot_isopycnal_depth(vtu_files, bs=np.arange(-11.5, -9.5, 0.5), filename=dir / "images/isopycnal_depths.png")

        for vtu_file in vtu_files:
        # for vtu_file in [vtu_files[-1]]:
            print(f"Processing {vtu_file}")
            i = int(vtu_file.stem.split("_")[1])  # assuming file is of the form "/foo/bar/state_{i:016d}.vtu"
            if (dir / f"images/psi_{i:016d}.png").exists() and not overwrite:
                print(f"Skipping {vtu_file}")
                continue
            # plot_fieldb(vtu_file, "v", label=r"Meridional flow $v$", filename=dir/f"images/v_{i:016d}.png")
            # plot_fieldb(vtu_file, "w", label=r"Vertical flow $w$", filename=dir/f"images/w_{i:016d}.png")
            plot_fieldb(vtu_file, "nu", label=r"Turbulent viscosity $\nu$", filename=dir / f"images/nu_{i:016d}.png")
            plot_fieldb(
                vtu_file,
                "kappa_v",
                vmax=100,
                label=r"Turbulent diffusivity $\kappa_v$",
                filename=dir / f"images/kappa_v_{i:016d}.png",
            )
            # plot_psib(vtu_file, bmin=-15, bmax=-13, filename=dir / f"images/psi_{i:016d}.png")
            plot_psib(vtu_file, bmin=-15, bmax=0, filename=dir / f"images/psi_{i:016d}.png")
            # plot_psib(vtu_file, filename=dir / f"images/psi_{i:016d}.png")
            # plot_surface_b_flux(vtu_file, n=2**10)
            # plot_psi_profile(vtu_file, -0.51)

        print()
