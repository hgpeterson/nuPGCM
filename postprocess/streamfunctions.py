import warnings

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


def _twa_snapshot_fields(samples, grid, b_coords):
    """
    Compute σ, σv, and ζ on the buoyancy-coordinate grid for one snapshot.

    σ = 1/(α b_z) = ζ_b̃/α is the isopycnal thickness (the aspect ratio α
    enters through the nondimensionalization). It is computed as ζ_b̃/α from
    the isopycnal depth ζ(x̃, ỹ, b̃), with ζ clamped to the bottom/surface
    where an isopycnal outcrops so that σ = 0 in "vacuum" regions
    (Young 2012). σ and σv are set to 0 over land. The returned ζ is NaN
    over land and in vacuum, so that averages of ζ are taken only over
    columns where the isopycnal actually exists (otherwise clamped bottom
    values would bias ζ̄ shallow over varying bathymetry).

    Returns
    -------
    sigma, sigma_v : ndarray, shape (nx, ny, nb)
    zeta : ndarray, shape (nx, ny, nb)
    """
    alpha = -grid.z.min()
    nb = len(b_coords)
    b_range = (b_coords[0], b_coords[-1])

    mask = samples["vtkValidPointMask"].reshape(grid.nx, grid.ny, grid.nz).astype(bool)
    v = samples["u"][:, 1].reshape(grid.nx, grid.ny, grid.nz)
    b = samples["b"].reshape(grid.nx, grid.ny, grid.nz)

    # pyvista fills fields with 0 outside the mesh; mask to NaN
    b_masked = np.where(mask, b, np.nan)

    # isopycnal depth ζ(x, y, b) and v in buoyancy coordinates
    zeta, _ = utils.to_buoyancy_coords(np.where(mask, grid.zz, np.nan), b_masked, nb, b_range=b_range, clamp=True)
    v_b, _ = utils.to_buoyancy_coords(np.where(mask, v, np.nan), b_masked, nb, b_range=b_range)

    # thickness σ = ζ_b̃/α; vacuum (isopycnal outside the column's buoyancy
    # range) and land carry zero thickness
    sigma = np.gradient(zeta, b_coords, axis=2) / alpha
    vacuum = np.isnan(v_b)
    sigma = np.where(vacuum, 0.0, sigma)
    sigma_v = sigma * np.where(vacuum, 0.0, v_b)

    # ζ is only meaningful where the isopycnal exists
    zeta = np.where(vacuum, np.nan, zeta)

    return sigma, sigma_v, zeta


def _twa_streamfunction_from_means(sigma, sigma_v, zeta, grid, b_coords):
    """
    Compute the TWA overturning streamfunction from (possibly time-averaged)
    σ, σv, and ζ on the (x, y, b) grid.

    The TWA meridional velocity is v̂ = ⟨σv⟩/⟨σ⟩ with ⟨·⟩ the average in
    buoyancy coordinates (here zonal; any time averaging is done by the
    caller before passing the fields in). The streamfunction follows from
    ψ(ỹ, b̃) = -∫∫_{b̃_min}^{b̃} ⟨σv⟩ db̃' dx, equivalent to the Cartesian
    ψ(y, z) = -1/α ∫∫_{-H}^z v dz' dx since dz = α σ db̃. To return to
    Cartesian coordinates, the mean isopycnal depth ζ̄(ỹ, b̃) is inverted to
    obtain b♯(y, z) such that b̃ = b♯(y, ζ̄(ỹ, b̃)), and ψ is evaluated at
    each z via ψ_TWA(y, z) = ψ(y, b♯(y, z)).

    Returns
    -------
    psi_b : ndarray, shape (ny, nb)
        Overturning streamfunction in buoyancy coordinates.
    v_hat : ndarray, shape (ny, nb)
        TWA meridional velocity.
    zeta_bar : ndarray, shape (ny, nb)
        Zonal-mean isopycnal depth.
    psi : ndarray, shape (ny, nz)
        psi_b remapped to Cartesian coordinates via b♯.
    b_sharp : ndarray, shape (ny, nz)
        Mean buoyancy field b♯(y, z) from inverting ζ̄.
    """
    # zonal integrals in buoyancy coordinates (σ = 0 over land and in vacuum)
    sigma_int = trapezoid(sigma, x=grid.x, axis=0)  # (ny, nb)
    sigma_v_int = trapezoid(sigma_v, x=grid.x, axis=0)

    # TWA meridional velocity v̂ = ⟨σv⟩/⟨σ⟩
    v_hat = np.divide(sigma_v_int, sigma_int, where=sigma_int != 0, out=np.full_like(sigma_int, np.nan))

    # ψ(y, b) = -∫_x ∫_{b_min}^{b} σv db' dx
    psi_b = -cumulative_trapezoid(sigma_v_int, b_coords, axis=1, initial=0)
    psi_b[sigma_int == 0] = np.nan

    # zonal-mean isopycnal depth ζ̄(y, b) (land columns are NaN)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows over land
        zeta_bar = np.nanmean(zeta, axis=0)  # (ny, nb)

    # invert ζ̄ to get b♯(y, z) and remap ψ to Cartesian coordinates
    psi = np.full((grid.ny, grid.nz), np.nan)
    b_sharp = np.full((grid.ny, grid.nz), np.nan)
    for iy in range(grid.ny):
        zeta_col = zeta_bar[iy, :]
        valid = ~np.isnan(zeta_col)
        if valid.sum() >= 2:
            sort_idx = np.argsort(zeta_col[valid])
            b_sharp[iy, :] = np.interp(
                grid.z, zeta_col[valid][sort_idx], b_coords[valid][sort_idx], left=np.nan, right=np.nan
            )
        valid_p = valid & ~np.isnan(psi_b[iy, :])
        if valid_p.sum() >= 2:
            sort_idx = np.argsort(zeta_col[valid_p])
            psi[iy, :] = np.interp(
                grid.z, zeta_col[valid_p][sort_idx], psi_b[iy, valid_p][sort_idx], left=np.nan, right=np.nan
            )

    return psi_b, v_hat, zeta_bar, psi, b_sharp


def calculate_overturning_streamfunction_TWA(
    vtu_file=None,
    nx=2**8,
    ny=2**8,
    nz=2**8,
    nb=None,
    b_range=None,
    printtime=False,
    *,
    dataset=None,
    grid=None,
    samples=None,
):
    """
    Thickness-weighted-average (Young 2012) overturning streamfunction for a
    single snapshot.

    Returns
    -------
    psi_b : ndarray, shape (ny, nb)
        Streamfunction in buoyancy coordinates.
    b_coords : ndarray, shape (nb,)
        Buoyancy coordinate values.
    psi : ndarray, shape (ny, nz)
        Streamfunction remapped to Cartesian coordinates via b♯.
    b_sharp : ndarray, shape (ny, nz)
        Mean buoyancy b♯(y, z) from inverting the mean isopycnal depth.
    v_hat : ndarray, shape (ny, nb)
        TWA meridional velocity.
    zeta_bar : ndarray, shape (ny, nb)
        Zonal-mean isopycnal depth.
    grid : utils.Grid
    t : float
    """
    if printtime:
        t0 = time()

    if dataset is None:
        dataset = pv.read(vtu_file)

    t = dataset["t"][0]

    if grid is None:
        grid = utils.Grid(dataset, nx, ny, nz)
    if samples is None:
        samples = utils.sample_to_grid(dataset, grid)

    if nb is None:
        nb = grid.nz
    if b_range is None:
        mask = samples["vtkValidPointMask"].astype(bool)
        b_range = (samples["b"][mask].min(), samples["b"][mask].max())
    b_coords = np.linspace(b_range[0], b_range[1], nb)

    sigma, sigma_v, zeta = _twa_snapshot_fields(samples, grid, b_coords)
    psi_b, v_hat, zeta_bar, psi, b_sharp = _twa_streamfunction_from_means(sigma, sigma_v, zeta, grid, b_coords)

    if printtime:
        print(f"TWA psi computed in {time() - t0:.3e} s")

    return psi_b, b_coords, psi, b_sharp, v_hat, zeta_bar, grid, t


def calculate_overturning_streamfunction_TWA_time_mean(
    dir, n, nx=2**8, ny=2**8, nz=2**8, nb=None, b_range=None, printtime=False
):
    """
    Thickness-weighted-average (Young 2012) overturning streamfunction from
    the time mean over the last `n` VTU output files in a simulation
    directory. σ, σv, and ζ are averaged in buoyancy coordinates
    (x̃, ỹ, b̃, t̃) before forming v̂ = ⟨σv⟩/⟨σ⟩ and ψ.

    Parameters
    ----------
    dir : Path
        Simulation directory containing a "data" subdirectory of
        "state_{i:016d}.vtu" files.
    n : int
        Number of output files (from the end) to average over.
    nx, ny, nz : int, optional
        Grid resolution for sampling.
    nb : int, optional
        Number of buoyancy levels (default: nz).
    b_range : tuple (b_min, b_max), optional
        Buoyancy coordinate range (default: range of b in the first file).
    printtime : bool, optional
        Print timing information.

    Returns
    -------
    Same as calculate_overturning_streamfunction_TWA, but with `times`
    (ndarray, shape (n,)) in place of `t`.
    """
    if printtime:
        t0 = time()

    vtu_files = sorted((Path(dir) / "data").glob("state_*.vtu"))[-n:]

    grid = None
    b_coords = None
    sigma_sum = sigma_v_sum = zeta_sum = None
    times = np.empty(len(vtu_files))
    for i, vtu_file in enumerate(vtu_files):
        dataset = pv.read(vtu_file)
        times[i] = dataset["t"][0]

        if grid is None:
            grid = utils.Grid(dataset, nx, ny, nz)
        samples = utils.sample_to_grid(dataset, grid)

        if b_coords is None:
            if nb is None:
                nb = grid.nz
            if b_range is None:
                mask = samples["vtkValidPointMask"].astype(bool)
                b_range = (samples["b"][mask].min(), samples["b"][mask].max())
            b_coords = np.linspace(b_range[0], b_range[1], nb)

        sigma, sigma_v, zeta = _twa_snapshot_fields(samples, grid, b_coords)

        # ζ is NaN where the isopycnal doesn't exist (vacuum/land), so its
        # time mean is taken only over the snapshots where it does
        if sigma_sum is None:
            sigma_sum, sigma_v_sum = sigma, sigma_v
            zeta_sum = np.nan_to_num(zeta)
            zeta_count = np.isfinite(zeta).astype(np.int64)
        else:
            sigma_sum += sigma
            sigma_v_sum += sigma_v
            zeta_sum += np.nan_to_num(zeta)
            zeta_count += np.isfinite(zeta)

    sigma_mean = sigma_sum / len(vtu_files)
    sigma_v_mean = sigma_v_sum / len(vtu_files)
    zeta_mean = np.divide(zeta_sum, zeta_count, where=zeta_count > 0, out=np.full_like(zeta_sum, np.nan))

    psi_b, v_hat, zeta_bar, psi, b_sharp = _twa_streamfunction_from_means(
        sigma_mean, sigma_v_mean, zeta_mean, grid, b_coords
    )

    if printtime:
        print(f"time-mean TWA psi computed from {len(vtu_files)} files in {time() - t0:.3e} s")

    return psi_b, b_coords, psi, b_sharp, v_hat, zeta_bar, grid, times


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
    psi_twa_b_file = dir / f"images/psi_twa_b{i:016d}.png"
    psi_twa_file = dir / f"images/psi_twa{i:016d}.png"
    baro_file = dir / f"images/psi_baro{i:016d}.png"
    baro_mask_file = dir / f"images/psi_baro_mask{i:016d}.png"
    psi_needed = not psi_file.exists() or overwrite
    psi_twa_b_needed = not psi_twa_b_file.exists() or overwrite
    psi_twa_needed = not psi_twa_file.exists() or overwrite
    baro_needed = not baro_file.exists() or overwrite
    baro_mask_needed = not baro_mask_file.exists() or overwrite
    if not (psi_needed or psi_twa_b_needed or psi_twa_needed or baro_needed or baro_mask_needed):
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

    # thickness-weighted-average overturning streamfunction
    if psi_twa_b_needed or psi_twa_needed:
        psi_b, b_coords, psi_twa, b_sharp, v_hat, zeta_bar, grid, t = calculate_overturning_streamfunction_TWA(
            dataset=dataset,
            grid=grid,
            samples=samples,
            nb=20 * n,
            printtime=True,
        )
        if psi_twa_b_needed:
            plot_overturning_streamfunction_isopycnal(
                psi_b,
                b_coords,
                grid,
                t=t,
                filename=psi_twa_b_file,
            )
        if psi_twa_needed:
            plot_overturning_streamfunction(
                psi_twa,
                b_sharp,
                grid,
                t=t,
                filename=psi_twa_file,
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
        # ["068", "tub"],
        # ["068a", "tub"],
        # ["069", "box"],
        # ["069a", "box"],
        ["070", "tub"],
        # ["070a", "tub"],
        ["071", "box"],
        # ["071a", "box"],
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

        # psi_b, b_coords, psi_twa, b_sharp, v_hat, zeta_bar, grid, times = (
        #     calculate_overturning_streamfunction_TWA_time_mean(dir, n=10, nx=2**7, ny=2**7, nz=2**7, nb=20 * 2**7)
        # )
        # plot_overturning_streamfunction_isopycnal(psi_b, b_coords, grid, filename=dir / "images/psi_twa_b_mean.png")
        # plot_overturning_streamfunction(
        #     psi_twa,
        #     b_sharp,
        #     grid,
        #     filename=dir / "images/psi_twa_mean.png",
        #     bmin=-15,
        #     bmax=-10,
        #     geometry=geom,
        # )
