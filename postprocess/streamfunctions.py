import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from time import time
from scipy.integrate import trapezoid, cumulative_trapezoid
from pathlib import Path
import utils

matplotlib.use("Agg")  # non-interactive backend
FILE_DIR = Path(__file__).parent.resolve()
plt.style.use(Path(FILE_DIR, "../plots.mplstyle"))


def calculate_barotropic_streamfunction(
    vtu_file=None, nx=2**8, ny=2**8, nz=2**8, printtime=False, *, dataset=None, grid=None, samples=None
):
    if printtime:
        t0 = time()

    if dataset is None:
        dataset = pv.read(vtu_file)

    t = dataset["t"][0]

    if grid is None:
        grid = utils.Grid(dataset, nx, ny, nz)
    if samples is None:
        samples = utils.sample_to_grid(dataset, grid)

    nx, ny, nz = len(grid.x), len(grid.y), len(grid.z)
    u = samples["u"][:, 0].reshape(nx, ny, nz)

    # vertical integral
    U = trapezoid(u, x=grid.z, axis=2)
    H = utils.depth(samples, grid)

    # calculate streamfunction as Psi(x, y) = -∫_0^y U(x, y') dy'
    Psi = trapezoid(U, grid.y, axis=1) - cumulative_trapezoid(U, grid.y, axis=1, initial=0)
    nan_mask = np.where(H == 0)
    U[nan_mask] = np.nan
    Psi[nan_mask] = np.nan

    if printtime:
        print(f"barotropic streamfunction computed in {time() - t0:.3e} s")

    return Psi, U, grid, t


def _overturning_streamfunction_from_means(v, b, grid, width):
    """Compute psi(y, z), v_int(y, z), and b_bar(y, z) from (possibly time-averaged) v, b, and zonal width."""
    alpha = -grid.z.min()  # aspect ratio

    v_int = trapezoid(v, x=grid.x, axis=0)
    b_bar = utils.zonal_mean(b, grid, width)

    # calculate streamfunction as psi(y,z) = -1/α * ∫_-H^z v(y, z') dz'
    psi_bar = -1 / alpha * cumulative_trapezoid(v_int, grid.z, axis=1, initial=0)
    nan_mask = np.where(width == 0)
    v_int[nan_mask] = np.nan
    psi_bar[nan_mask] = np.nan

    return psi_bar, v_int, b_bar


def calculate_overturning_streamfunction(
    vtu_file=None, nx=2**8, ny=2**8, nz=2**8, printtime=False, *, dataset=None, grid=None, samples=None
):
    if printtime:
        t0 = time()

    if dataset is None:
        dataset = pv.read(vtu_file)

    t = dataset["t"][0]

    if grid is None:
        grid = utils.Grid(dataset, nx, ny, nz)
    if samples is None:
        samples = utils.sample_to_grid(dataset, grid)

    nx, ny, nz = len(grid.x), len(grid.y), len(grid.z)

    v = samples["u"][:, 1].reshape(nx, ny, nz)
    b = samples["b"].reshape(nx, ny, nz)
    width = utils.zonal_width(samples, grid)

    psi_bar, v_int, b_bar = _overturning_streamfunction_from_means(v, b, grid, width)

    if printtime:
        print(f"psi computed in {time() - t0:.3e} s")

    return psi_bar, v_int, b_bar, grid, t


def calculate_overturning_streamfunction_time_mean(dir, n, nx=2**8, ny=2**8, nz=2**8, printtime=False):
    """
    Compute the overturning streamfunction from the time mean of v and b over
    the last `n` VTU output files in a simulation directory.

    Parameters
    ----------
    dir : Path
        Simulation directory containing a "data" subdirectory of
        "state_{i:016d}.vtu" files.
    n : int
        Number of output files (from the end) to average over.
    nx, ny, nz : int, optional
        Grid resolution for sampling.
    printtime : bool, optional
        Print timing information.

    Returns
    -------
    psi_bar : ndarray, shape (ny, nz)
        Time-mean overturning streamfunction.
    v_int : ndarray, shape (ny, nz)
        Time-mean, zonally-integrated meridional velocity.
    b_bar : ndarray, shape (ny, nz)
        Time-mean, zonally-averaged buoyancy.
    grid : utils.Grid
    times : ndarray, shape (n,)
        Time of each output file included in the average.
    """
    if printtime:
        t0 = time()

    vtu_files = sorted((Path(dir) / "data").glob("state_*.vtu"))[-n:]

    grid = None
    v_sum = b_sum = width_sum = None
    times = np.empty(len(vtu_files))
    for i, vtu_file in enumerate(vtu_files):
        dataset = pv.read(vtu_file)
        times[i] = dataset["t"][0]

        if grid is None:
            grid = utils.Grid(dataset, nx, ny, nz)
        samples = utils.sample_to_grid(dataset, grid)

        v = samples["u"][:, 1].reshape(grid.nx, grid.ny, grid.nz)
        b = samples["b"].reshape(grid.nx, grid.ny, grid.nz)
        width = utils.zonal_width(samples, grid)

        if v_sum is None:
            v_sum, b_sum, width_sum = v.copy(), b.copy(), width.copy()
        else:
            v_sum += v
            b_sum += b
            width_sum += width

    v_mean = v_sum / len(vtu_files)
    b_mean = b_sum / len(vtu_files)
    width_mean = width_sum / len(vtu_files)

    psi_bar, v_int, b_bar = _overturning_streamfunction_from_means(v_mean, b_mean, grid, width_mean)

    if printtime:
        print(f"time-mean psi computed from {len(vtu_files)} files in {time() - t0:.3e} s")

    return psi_bar, v_int, b_bar, grid, times


def calculate_overturning_streamfunction_isopycnal(
    vtu_file=None, nx=2**8, ny=2**8, nz=2**8, nb=None, printtime=False, *, dataset=None, grid=None, samples=None
):
    if printtime:
        t0 = time()

    if dataset is None:
        dataset = pv.read(vtu_file)

    t = dataset["t"][0]

    if grid is None:
        grid = utils.Grid(dataset, nx, ny, nz)
    if samples is None:
        samples = utils.sample_to_grid(dataset, grid)

    nx_g, ny_g, nz_g = len(grid.x), len(grid.y), len(grid.z)
    if nb is None:
        nb = nz_g
    alpha = -grid.z.min()

    mask = samples["vtkValidPointMask"].reshape(nx_g, ny_g, nz_g).astype(bool)
    v = samples["u"][:, 1].reshape(nx_g, ny_g, nz_g)
    b = samples["b"].reshape(nx_g, ny_g, nz_g)

    # pyvista fills b=0 outside the mesh; mask to NaN
    b_masked = np.where(mask, b, np.nan)

    # zonal mean buoyancy for back-transform to z-space
    width = utils.zonal_width(samples, grid)
    b_bar = utils.zonal_mean(b, grid, width)

    # V(x, y, z) = ∫_{-H}^{z} v dz'  (land → 0 before integrating, then re-mask)
    V = cumulative_trapezoid(np.where(mask, v, 0.0), grid.z, axis=2, initial=0)
    V_masked = np.where(mask, V, np.nan)

    # remap V to buoyancy coordinates: V_b(x, y, b) = V(x, y, z(x,y,b))
    V_b, b_coords = utils.to_buoyancy_coords(V_masked, b_masked, nb)

    # psi(y, b) = -1/α ∫_x V_b dx
    psi_b = -1 / alpha * trapezoid(np.where(np.isnan(V_b), 0.0, V_b), x=grid.x, axis=0)
    mask_b = np.all(np.isnan(V_b), axis=0)
    psi_b[mask_b] = np.nan

    # zonal mean isopycnal depth z_bar_b(y, b): transform the z-coordinate
    # into b-space, then average over x
    z_b, _ = utils.to_buoyancy_coords(
        np.where(mask, grid.zz, np.nan), b_masked, nb, b_range=(b_coords[0], b_coords[-1])
    )
    z_bar_b = np.nanmean(z_b, axis=0)  # (ny, nb)

    # remap psi_b to z-space: interpolate psi_b(y, b) onto the regular z-grid
    # using z_bar_b(y, b) as the z-coordinate of each b-level
    psi = np.full((ny_g, nz_g), np.nan)
    for iy in range(ny_g):
        z_col = z_bar_b[iy, :]
        p_col = psi_b[iy, :]
        valid = ~np.isnan(z_col) & ~np.isnan(p_col)
        if valid.sum() < 2:
            continue
        sort_idx = np.argsort(z_col[valid])
        psi[iy, :] = np.interp(grid.z, z_col[valid][sort_idx], p_col[valid][sort_idx], left=np.nan, right=np.nan)

    if printtime:
        print(f"isopycnal psi computed in {time() - t0:.3e} s")

    return psi_b, b_coords, psi, b_bar, grid, t


def plot_barotropic_streamfunction(
    Psi, grid, t=None, filename="psi_baro.png", Psimax=None, maskchannel=False, channel_y=-0.5
):
    x = grid.x
    y = grid.y
    xx, yy = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots(1, figsize=(19 / 6, 19 / 6 * 1.62))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if maskchannel:
        nan_mask = np.where((yy < channel_y))
        Psi[nan_mask] = np.nan
    if Psimax is None:
        Psimax = np.nanmax(np.abs(Psi))

    cf1 = ax.pcolormesh(x, y, Psi.T, cmap="RdBu_r", vmin=-Psimax, vmax=Psimax, rasterized=True)
    levels = np.linspace(-0.9 * Psimax, 0.9 * Psimax, 8)
    ax.contour(x, y, Psi.T, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    cb = plt.colorbar(cf1, label=r"Barotropic streamfunction $\Psi$", shrink=0.5)
    cb.ax.set_yticks([-Psimax, 0, Psimax])
    cb.ax.set_yticklabels([r"$-$Max", r"$0$", r"Max"])
    ax.text(
        0.8,
        1.02,
        rf"Max = ${utils.to_latex_sci(Psimax)}$",
        transform=ax.transAxes,
        size=7,
    )
    if maskchannel:
        ax.fill_between(x, channel_y, y.min(), color="k", alpha=0.1, ec="none")
    ax.axhline(channel_y, c="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xticks([0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(r"Zonal coordinate $x$")
    ax.set_ylabel(r"Meridional coordinate $y$")
    if t is not None:
        ax.set_title(r"$t = " + utils.to_latex_sci(t) + r"$")
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_zonal_mean(
    field,
    grid,
    b,
    label="",
    cb_label="",
    rescale_z=True,
    t=None,
    i=None,
    cmap="RdBu_r",
    cb_sym=True,
    b_levels=None,
):
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
    if b_levels is None:
        b_levels = np.linspace(-15, 15, 40)
    ax.contour(y, z, -b.T, levels=b_levels, colors="k", linewidths=0.5, linestyles="-", alpha=0.3)
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


def plot_overturning_streamfunction(
    psi,
    b_bar,
    grid,
    t=None,
    filename="psi.png",
    bmin=None,
    bmax=None,
    geometry="",
    psimax=None,
    tub_y0=-0.6875,
    tub_y1=-0.5,
    box_y=-0.5,
):
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
        extend = "neither"
    else:
        extend_max = np.nanmax(psi) > psimax
        extend_min = np.nanmin(psi) < -psimax
        if extend_max and extend_min:
            extend = "both"
        elif extend_max:
            extend = "max"
        elif extend_min:
            extend = "min"
        else:
            extend = "neither"
    cf1 = ax.pcolormesh(y, z, psi.T, cmap="RdBu_r", vmin=-psimax, vmax=psimax, rasterized=True)
    levels = np.linspace(-0.9 * psimax, 0.9 * psimax, 8)
    ax.contour(y, z, psi.T, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    cb = plt.colorbar(cf1, label=r"Streamfunction $\psi$", extend=extend)
    cb.ax.set_yticks([-psimax, 0, psimax])
    cb.ax.set_yticklabels([r"$-$Max", r"$0$", r"Max"])
    ax.text(
        0.8,
        1.02,
        rf"Max = ${utils.to_latex_sci(psimax)}$",
        transform=ax.transAxes,
        size=7,
    )
    levels = np.linspace(bmin, bmax, 20)
    ax.contour(
        y,
        z,
        b_bar.T,
        levels=levels,
        colors="k",
        linestyles="-",
        linewidths=0.5,
        alpha=0.3,
    )
    if geometry == "tub":
        y_c = np.linspace(tub_y0, tub_y1, 100)
        ax.plot(y_c, z.min() * (1 - ((y_c - tub_y0) / (tub_y1 - tub_y0)) ** 2), "k--", lw=0.5, alpha=0.4)
    elif geometry == "box":
        ax.axvline(box_y, c="k", ls="--", lw=0.5, alpha=0.4)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-alpha, 0])
    ax.set_xlabel(r"Meridional coordinate $y$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    if t is not None:
        ax.set_title(r"$t = " + utils.to_latex_sci(t) + r"$")
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_overturning_streamfunction_isopycnal(
    psi_b,
    b_coords,
    grid,
    t=None,
    filename="psi_iso.png",
    psimax=None,
):
    y = grid.y

    fig, ax = plt.subplots(1, figsize=(33 / 6, 33 / 6 / 1.62 / 2))
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if psimax is None:
        psimax = np.nanmax(np.abs(psi_b))
        extend = "neither"
    else:
        extend_max = np.nanmax(psi_b) > psimax
        extend_min = np.nanmin(psi_b) < -psimax
        if extend_max and extend_min:
            extend = "both"
        elif extend_max:
            extend = "max"
        elif extend_min:
            extend = "min"
        else:
            extend = "neither"
    cf1 = ax.pcolormesh(y, b_coords, psi_b.T, cmap="RdBu_r", vmin=-psimax, vmax=psimax, rasterized=True)
    levels = np.linspace(-0.9 * psimax, 0.9 * psimax, 8)
    ax.contour(y, b_coords, psi_b.T, levels=levels, colors="k", linestyles="-", linewidths=0.25)
    cb = plt.colorbar(cf1, label=r"Streamfunction $\psi$", extend=extend)
    cb.ax.set_yticks([-psimax, 0, psimax])
    cb.ax.set_yticklabels([r"$-$Max", r"$0$", r"Max"])
    ax.text(
        0.8,
        1.02,
        rf"Max = ${utils.to_latex_sci(psimax)}$",
        transform=ax.transAxes,
        size=7,
    )
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([b_coords.min(), b_coords.max()])
    ax.set_xlabel(r"Meridional coordinate $y$")
    ax.set_ylabel(r"Buoyancy $b$")
    if t is not None:
        ax.set_title(r"$t = " + utils.to_latex_sci(t) + r"$")
    plt.savefig(filename)
    print(filename)
    plt.close()


def process_vtu(vtu_file, dir, geom, overwrite, n=2**8):
    """Process a single VTU file: calculate and plot overturning and barotropic streamfunctions."""
    i = int(vtu_file.stem.split("_")[1])  # assuming file is of the form "/foo/bar/state_{i:016d}.vtu"

    # image files
    psi_file = dir / f"images/psi{i:016d}.png"
    psi_iso_file = dir / f"images/psi_iso{i:016d}.png"
    psi_iso_remap_file = dir / f"images/psi_iso_remap{i:016d}.png"
    baro_file = dir / f"images/psi_baro{i:016d}.png"
    baro_mask_file = dir / f"images/psi_baro_mask{i:016d}.png"
    psi_needed = not psi_file.exists() or overwrite
    psi_iso_needed = not psi_iso_file.exists() or overwrite
    psi_iso_remap_needed = not psi_iso_remap_file.exists() or overwrite
    baro_needed = not baro_file.exists() or overwrite
    baro_mask_needed = not baro_mask_file.exists() or overwrite
    if not (psi_needed or psi_iso_needed or baro_needed or baro_mask_needed):
        return

    # read and sample once, shared by all streamfunction calculations
    t0 = time()
    dataset = pv.read(vtu_file)
    grid = utils.Grid(dataset, n, n, n)
    samples = utils.sample_to_grid(dataset, grid)
    print(f"Data read and sampled in {time() - t0:.3e} s")

    # Cartesian overturning streamfunction
    if psi_needed:
        psi_bar, v_bar, b_bar, grid, t = calculate_overturning_streamfunction(
            dataset=dataset,
            grid=grid,
            samples=samples,
        )
        plot_overturning_streamfunction(
            psi_bar,
            b_bar,
            grid,
            t=t,
            filename=psi_file,
            bmin=-15,
            bmax=-10,
            geometry=geom,
        )

    # isopycnal overturning streamfunction
    if psi_iso_needed or psi_iso_remap_needed:
        psi_b, b_coords, psi_iso, b_bar, grid, t = calculate_overturning_streamfunction_isopycnal(
            dataset=dataset,
            grid=grid,
            samples=samples,
            nb=20 * n,
            printtime=True,
        )
        if psi_iso_needed:
            plot_overturning_streamfunction_isopycnal(
                psi_b,
                b_coords,
                grid,
                t=t,
                filename=psi_iso_file,
            )
        if psi_iso_remap_needed:
            plot_overturning_streamfunction(
                psi_iso,
                b_bar,
                grid,
                t=t,
                filename=psi_iso_remap_file,
                bmin=-15,
                bmax=-10,
                geometry=geom,
            )

    # barotropic streamfunction
    if baro_needed or baro_mask_needed:
        Psi, U, grid, t = calculate_barotropic_streamfunction(dataset=dataset, grid=grid, samples=samples)
        if baro_needed:
            plot_barotropic_streamfunction(Psi, grid, t=t, filename=baro_file)
        if baro_mask_needed:
            plot_barotropic_streamfunction(Psi.copy(), grid, t=t, filename=baro_mask_file, maskchannel=True)


if __name__ == "__main__":
    overwrite = False
    # overwrite = True
    sims = [
        # ["052", "tub"],
        # ["053", "box"],
        # ["054", "tub"],
        # ["055", "tub"],
        # ["056", "box"],
        # ["057", "tub"],
        # ["058", "box"],
        # ["059", "box"],
        # ["060", "tub"],
        # ["061", "tub"],
        # ["062", "tub"],
        # ["063", "box"],
        # ["064", "box"],
        # ["065", "tub"],
        # ["065a", "tub"],
        # ["065b", "tub"],
        # ["065c", "tub"],
        # ["065d", "tub"],
        # ["066", "box"],
        # ["066a", "box"],
        # ["066b", "box"],
        # ["066c", "box"],
        # ["066d", "box"],
        # ["067", "tub"],
        ["068", "tub"],
        ["068a", "tub"],
        ["069", "box"],
        ["069a", "box"],
        ["070", "tub"],
        ["070a", "tub"],
        ["071", "box"],
        ["071a", "box"],
    ]
    sims_dir = Path("/resnick/scratch/hppeters")
    for sim, geom in sims:
        dir = sims_dir / f"sim{sim}"
        print(f"\nProcessing files in {dir}")

        vtu_files = sorted((dir / "data").glob("state_*.vtu"))
        for vtu_file in vtu_files:
            # for vtu_file in [vtu_files[-1]]:
            process_vtu(vtu_file, dir, geom, overwrite, n=2**7)

        # psi_bar, v_bar, b_bar, grid, times = calculate_overturning_streamfunction_time_mean(dir, n=10, nx=2**7, ny=2**7, nz=2**7)
        # plot_overturning_streamfunction(
        #     psi_bar,
        #     b_bar,
        #     grid,
        #     filename=dir / "images/psi_mean.png",
        #     bmin=-15,
        #     bmax=-10,
        #     geometry=geom,
        # )
