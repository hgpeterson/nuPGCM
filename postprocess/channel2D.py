import numpy as np
import pyvista as pv
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import utils

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


def plot_tri_field(vtu_file, field, label=None, vmax=None, filename="field.png"):
    # prep data
    dataset = pv.read(vtu_file)
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
    fields_dict,
    grid,
    t,
    bmin=None,
    bmax=None,
    vmax=None,
    filename="psi.png",
):
    # unpack
    z = grid.z
    yy = grid.yy[0, :, :]
    zz = grid.zz[0, :, :]
    v = fields_dict["v"][0, :, :]
    b = fields_dict["b"][0, :, :]
    mask = fields_dict["mask"][0, :, :]
    alpha = np.max(np.abs(z))

    # integrate: -alpha*dz(psi) = v
    psi = -1 / alpha * cumulative_trapezoid(v, z, axis=1, initial=0)

    # mask points outside the domain
    b[mask == 0] = np.nan
    psi[mask == 0] = np.nan

    # max value for colorbar
    if vmax is None:
        vmax = np.nanmax(np.abs(psi))
        print(f"max(psi) = {vmax:0.3e}")

    # plotting
    fig, ax = plt.subplots(1, figsize=(19 / 6, 19 / 6))
    im = ax.pcolormesh(yy, zz, psi, vmin=-vmax, vmax=vmax, shading="gouraud", cmap="RdBu_r")
    ax.contour(
        yy,
        zz,
        psi,
        levels=np.linspace(-0.9 * vmax, 0.9 * vmax, 8),
        colors="k",
        linestyles="-",
        linewidths=0.25,
    )
    if bmin is None:
        bmin = np.nanmin(b)
    if bmax is None:
        bmax = np.nanmax(b)
    levels = np.linspace(bmin, bmax, 20)
    ax.contour(
        yy,
        zz,
        b,
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
    ax.set_xlabel(r"Meridional coordinate $y$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_thermal_wind(
    fields_dict,
    grid,
    t,
    filename=Path("thermal_wind.png"),
):
    # unpack
    y = grid.y
    z = grid.z
    yy = grid.yy[0, :, :]
    zz = grid.zz[0, :, :]

    alpha = np.max(np.abs(z))
    f = yy  # hardcode f = y

    u = fields_dict["u"][0, :, :]
    b = fields_dict["b"][0, :, :]
    mask = fields_dict["mask"][0, :, :]

    # thermal wind terms: f*u_z = -b_y/alpha
    u_z = np.gradient(u, z, axis=1)
    b_y = np.gradient(b, y, axis=0)

    # mask points outside the domain
    u_z[mask == 0] = np.nan
    b_y[mask == 0] = np.nan

    # plot some profiles
    fig, ax = plt.subplots(1, figsize=(19 / 6, 19 / 6 * 1.62))
    ax.spines["left"].set_visible(False)
    ax.axvline(0, lw=0.5, c="k", ls="-")
    ys = [-0.9, -0.8, -0.7, -0.6]
    colors = ["C0", "C1", "C2", "C3"]
    custom_lines = [
        Line2D([0], [0], color="k", ls="-", lw=1),
        Line2D([0], [0], color="k", ls="--", lw=0.5),
    ]
    custom_handles = [r"$f \partial_z u$", r"$-\partial_y b / \alpha$"]
    for i, y0 in enumerate(ys):
        iy = np.argmin(np.abs(y - y0))
        ax.plot((f * u_z)[iy, :], z, c=colors[i], ls="-")
        ax.plot((-b_y / alpha)[iy, :], z, c=colors[i], ls="--", lw=0.5)
        custom_lines.append(
            Line2D([0], [0], color=colors[i], ls="-", lw=1),
        )
        custom_handles.append(rf"$y = {y0:0.2f}$")
    ax.legend(custom_lines, custom_handles)
    ax.set_ylabel(r"Vertical coordinate $z$")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-alpha, 0)
    ax.set_title(r"$t = $" + utils.to_latex_sci(t))
    profiles_file = filename.parent / filename.name.replace("thermal_wind", "thermal_wind_profiles")
    plt.savefig(profiles_file)
    print(profiles_file)
    plt.close()

    # 2d plot
    fig, ax = plt.subplots(1, 3, figsize=(3 * 19 / 6, 19 / 6), constrained_layout=True)
    # vmax = np.max([np.nanmax(np.abs(f * u_z)), np.nanmax(np.abs(b_y / alpha))])
    vmax = 10
    ax[0].pcolormesh(yy, zz, f * u_z, vmin=-vmax, vmax=vmax, shading="gouraud", cmap="RdBu_r")
    ax[1].pcolormesh(yy, zz, -b_y / alpha, vmin=-vmax, vmax=vmax, shading="gouraud", cmap="RdBu_r")
    im = ax[2].pcolormesh(yy, zz, f * u_z + b_y / alpha, vmin=-vmax, vmax=vmax, shading="gouraud", cmap="RdBu_r")
    levels = np.linspace(np.nanmin(b), np.nanmax(b), 20)
    for a in ax:
        a.contour(
            yy,
            zz,
            b,
            levels=levels,
            colors="k",
            alpha=0.25,
            linestyles="-",
            linewidths=0.5,
        )
    cb = plt.colorbar(im, extend="both", shrink=0.5)
    cb.ax.set_yticks([-vmax, 0, vmax])
    ax[1].set_xlabel(r"Meridional coordinate $y$")
    ax[0].set_ylabel(r"Vertical coordinate $z$")
    ax[1].set_yticks([])
    ax[2].set_yticks([])
    for a in ax:
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)
    ax[0].set_title(r"$f \partial_z u$")
    ax[1].set_title(r"$-\partial_y b / \alpha$")
    ax[2].set_title(r"$f \partial_z u + \partial_y b / \alpha$")
    plt.savefig(filename)
    print(filename)
    plt.close()


def plot_psi_profile(fields_dict, grid, y):
    # unpack
    z = grid.z
    v = fields_dict["u"][0, :, :, 1]
    alpha = np.max(np.abs(z))

    # find nearest y index
    iy = np.argmin(np.abs(grid.y - y))
    v_profile = v[iy, :]

    # compute psi
    psi = -1 / alpha * cumulative_trapezoid(v_profile, z, initial=0)

    # plotting
    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.spines["left"].set_visible(False)
    ax.axvline(0, lw=0.5, c="k", ls="-")
    ax.plot(psi, z)
    ax.set_xlabel(r"$\Psi$")
    ax.set_ylabel(r"$z$")
    ax.set_title(rf"$y = {y:0.2f}$")
    img_file = "images/psi_profile.png"
    plt.savefig(img_file)
    print(img_file)
    plt.close()


def plot_surface_b_flux(fields_dict, grid, show_progress=False):
    # hardcode parameters for now:
    Ek = np.sqrt(1e-1)
    PrBu = 1

    z = grid.z
    y = grid.y
    b = fields_dict["b"][0, :, :]
    kappa_v = fields_dict["kappa_v"][0, :, :]
    alpha = np.max(np.abs(z))

    # compute surface b flux using finite difference near surface
    surface_b_flux = np.zeros_like(y)
    dz = np.diff(z[:3])
    for i in tqdm(range(len(y)), disable=(not show_progress)):
        if dz[1] > 0:
            bz = 1 / dz[1] * (1 / 2 * b[i, -3] - 2 * b[i, -2] + 3 / 2 * b[i, -1])
            surface_b_flux[i] = alpha * Ek**2 / PrBu * kappa_v[i, -1] * bz
    surface_b_flux[0] = 0  # H = 0 here

    fig, ax = plt.subplots(1)
    ax.plot(y, surface_b_flux)
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
    # sims = ["000", "001", "002", "003", "004", "004a", "005", "006"]
    sims = ["004", "004a", "005", "006"]
    # sims = ["006"]
    for sim in sims:
        dir = sims_dir / f"sim{sim}"
        print(f"Processing files in {dir}")
        vtu_files = sorted((dir / "data").glob("state_*.vtu"))

        # plot_isopycnal_depth(vtu_files, filename=dir / "images/isopycnal_depths.png")
        plot_isopycnal_depth(
            vtu_files, bs=[-14, -12, -10, -8, -6, -4, -2], filename=dir / "images/isopycnal_depths.png"
        )

        for vtu_file in vtu_files:
            # for vtu_file in [vtu_files[-1]]:
            print(f"Processing {vtu_file}")
            i = int(vtu_file.stem.split("_")[1])  # assuming file is of the form "/foo/bar/state_{i:016d}.vtu"

            if not (dir / f"images/nu_{i:016d}.png").exists() or overwrite:
                plot_tri_field(
                    vtu_file, "nu", label=r"Turbulent viscosity $\nu$", filename=dir / f"images/nu_{i:016d}.png"
                )
            if not (dir / f"images/kappa_v_{i:016d}.png").exists() or overwrite:
                plot_tri_field(
                    vtu_file,
                    "kappa_v",
                    vmax=100,
                    label=r"Turbulent diffusivity $\kappa_v$",
                    filename=dir / f"images/kappa_v_{i:016d}.png",
                )
            if not (dir / f"images/u_{i:016d}.png").exists() or overwrite:
                plot_tri_field(vtu_file, "u", label=r"Zonal flow $u$", filename=dir / f"images/u_{i:016d}.png")

            if not (dir / f"images/psi_{i:016d}.png").exists() or overwrite:
                fields_dict, grid, t = utils.sample_fields(vtu_file, ["u", "v", "b"], printtime=True)
                plot_psib(fields_dict, grid, t, bmin=-15, bmax=0, filename=dir / f"images/psi_{i:016d}.png")
            # if not (dir / f"images/thermal_wind_{i:016d}.png").exists() or overwrite:
            #     plot_thermal_wind(fields_dict, grid, t, filename=dir / f"images/thermal_wind_{i:016d}.png")

        print()
