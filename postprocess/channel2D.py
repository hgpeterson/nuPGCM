import numpy as np
import pyvista as pv
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import os
from pathlib import Path
from utils import to_latex_sci

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")

def plot_fieldb(file_name, field, label=None, i=None, rescale_z=False, vmax=None):
    # read VTU file
    dataset = pv.read(file_name)

    # prep data
    coords = dataset.points
    f = dataset[field]
    b = dataset['b']
    y = coords[:, 1]
    z = coords[:, 2]
    alpha = np.max(np.abs(z))
    tri = dataset.cells_dict[5]

    # vmax for colorbar
    if vmax is None:
        print(f"max({field}) = {np.max(np.abs(f)):.3e}")
        vmax = np.max(np.abs(f))

    if rescale_z:
        z = z/(2*alpha)

    # data is 2D, create a tri-plot
    fig, ax = plt.subplots(1, figsize=(19/6, 19/6))
    im = ax.tripcolor(y, z, f, triangles=tri, vmin=-vmax, vmax=vmax, shading='gouraud', cmap='RdBu_r')
    ax.tricontour(y, z, b, levels=20, colors='k', alpha=0.25, linestyles='-', linewidths=0.5)
    if label is None:
        label = field
    plt.colorbar(im, label=label, fraction=0.03)
    tri = dataset.cells_dict[5]
    ax.triplot(y, z, tri, "k-", linewidth=0.25, alpha=0.1)
    ax.set_xlabel(r'$y$')
    if rescale_z:
        ax.set_ylabel(r'$z$ (rescaled)')
    else:
        ax.set_ylabel(r'$z$')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axis('equal')
    ax.set_title(r'$\alpha = $' + f'{alpha:0.3f}')
    if i is None:
        img_file = f"images/{field}.png"
    else:
        img_file = f"images/{field}{i:05d}.png"
    plt.savefig(img_file)
    print(img_file)
    plt.close()

def plot_uvwb(file_name, i=None, rescale_z=False):
    # read VTU file
    dataset = pv.read(file_name)

    # prep data
    coords = dataset.points
    u = dataset['u']
    v = dataset['v']
    w = dataset['w']
    b = dataset['b']
    y = coords[:, 1]
    z = coords[:, 2]
    alpha = np.max(np.abs(z))
    tri = dataset.cells_dict[5]
    speed = np.sqrt(v**2 + w**2)

    # vmax for colorbar
    print(f"max speed: {np.max(speed):.3e}")
    vmax = np.max(np.abs(u))
    # vmax = 1

    if rescale_z:
        # rescale for plotting only
        z = z/(2*alpha)

    # data is 2D, create a tri-plot
    fig, ax = plt.subplots(1, figsize=(19/6, 19/6))
    im = ax.tripcolor(y, z, u, triangles=tri, vmin=-vmax, vmax=vmax, shading='gouraud', cmap='RdBu_r')
    ax.tricontour(y, z, b, levels=20, colors='k', alpha=0.25, linestyles='-', linewidths=0.5)
    ax.quiver(y, z, v, w)
    plt.colorbar(im, label=r"$u$", fraction=0.03)
    ax.set_xlabel(r'$y$')
    if rescale_z:
        ax.set_ylabel(r'$z$ (rescaled)')
    else:
        ax.set_ylabel(r'$z$')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axis('equal')
    ax.set_title(r'$\alpha = $' + f'{alpha:0.3f}')
    if i is None:
        img_file = "images/uvwb.png"
    else:
        img_file = f"images/uvwb{i:05d}.png"
    plt.savefig(img_file)
    print(img_file)
    plt.close()

def plot_psib(file_name, i=None, n=2**8, rescale_z=False, vmax=None, subdir="", show_progress=False):
    # read VTU file
    dataset = pv.read(file_name)

    # load data
    x = dataset.points[0, 0]  # should all be the same
    # x = 0.5
    y = dataset.points[:, 1]
    z = dataset.points[:, 2]
    t = dataset['t'][0]
    alpha = np.max(np.abs(z))

    # create 2D grid for (y, z) evaluation
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    y_1d = np.linspace(y_min, y_max, n)
    z_1d = np.linspace(z_min, z_max, n)
    y_grid, z_grid = np.meshgrid(y_1d, z_1d)
    
    # evaluate v and b on the grid and compute psi
    psi = np.zeros_like(y_grid)
    v_grid = np.zeros_like(y_grid)
    b_grid = np.zeros_like(y_grid)
    for j in tqdm(range(n), disable=(not show_progress)):
        y_j = y_1d[j]

        line = pv.PointSet(np.array([[x, y_j, z] for z in z_1d]))
        samples = line.sample(dataset)

        v_grid[:, j] = samples['v']
        b_grid[:, j] = samples['b']

        # integrate: -alpha*dz(psi) = v
        psi[:, j] = -1/alpha*cumulative_trapezoid(v_grid[:, j], z_1d, initial=0)

        # mask points outside the domain
        nan_mask = samples['vtkValidPointMask'] == 0
        b_grid[nan_mask, j] = np.nan
        psi[nan_mask, j] = np.nan

    if rescale_z:
        # rescale for plotting only
        z_grid = z_grid/(2*alpha)
        z = z/(2*alpha)

    # max value for colorbar
    if vmax is None:
        vmax = np.nanmax(np.abs(psi))
        print(f"max(psi) = {vmax:0.3e}")

    # plotting
    fig, ax = plt.subplots(1, figsize=(19/6, 19/6))
    im = ax.pcolormesh(y_grid, z_grid, psi, vmin=-vmax, vmax=vmax, shading="gouraud", cmap='RdBu_r')
    ax.contour(y_grid, z_grid, psi, levels=np.linspace(-0.9*vmax, 0.9*vmax, 8), colors='k', linestyles='-', linewidths=0.25)
    levels = np.linspace(-9.5, -0.5, 20)
    ax.contour(y_grid, z_grid, b_grid, levels=levels, colors='k', alpha=0.25, linestyles='-', linewidths=0.5)
    levels = np.linspace(0, 9, 20)
    ax.contour(y_grid, z_grid, b_grid, levels=levels, colors='k', alpha=0.25, linestyles='--', linewidths=0.5)
    cb = plt.colorbar(im, label=r"$\Psi$", fraction=0.03)
    cb.ax.set_yticks([-vmax, 0, vmax])
    cb.ax.set_yticklabels([r"$-$Max", r"$0$", r"$+$Max"])
    ax.annotate(f"Max = {to_latex_sci(vmax)}", xy=(0.92, 0.98), xycoords="axes fraction")
    # cb.ax.ticklabel_format(style="sci", scilimits=(-2, 2), useMathText=True)
    # tri = dataset.cells_dict[5]
    # ax.triplot(y, z, tri, "k-", linewidth=0.25, alpha=0.1)
    ax.set_xlabel(r'$y$')
    if rescale_z:
        ax.set_ylabel(rf'$z$ (rescaled, $\alpha = {alpha:0.3f}$)')
    else:
        ax.set_ylabel(r'$z$')
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axis('equal')
    ax.set_xticks([-1, -3/4, -1/2])
    ax.set_yticks([z_grid.min(), z_grid.min()/2, 0])
    ax.set_title(r'$t = $' + to_latex_sci(t))
    if i is None:
        img_file = f"images/{subdir}psi.png"
    else:
        img_file = f"images/{subdir}psi{i:05d}.png"
    plt.savefig(img_file)
    print(img_file)
    plt.close()

def plot_psi_profile(file_name, y, n=2**8):
    # read VTU file
    dataset = pv.read(file_name)

    # load data
    x = dataset.points[0, 0]  # should all be the same
    z = dataset.points[:, 2]
    t = dataset['t'][0]
    alpha = np.max(np.abs(z))

    # create 1D grid for evaluation
    z_min, z_max = z.min(), z.max()
    z_1d = np.linspace(z_min, z_max, n)
    
    # evaluate v on the grid and compute psi
    line = pv.PointSet(np.array([[x, y, z] for z in z_1d]))
    samples = line.sample(dataset)
    v = samples['v']
    psi = -1/alpha*cumulative_trapezoid(v, z_1d, initial=0)

    # mask points outside the domain
    nan_mask = samples['vtkValidPointMask'] == 0
    psi[nan_mask] = np.nan

    # plotting
    fig, ax = plt.subplots(1, figsize=(2, 3.2))
    ax.spines['left'].set_visible(False)
    ax.axvline(0, lw=0.5, c='k', ls='-')
    ax.plot(psi, z_1d)
    ax.set_xlabel(r'$\Psi$')
    ax.set_ylabel(r'$z$')
    ax.set_title(rf'$y = {y:0.2f}$')
    img_file = "images/psi_profile.png"
    plt.savefig(img_file)
    print(img_file)
    plt.close()

def plot_surface_b_flux(file_name, n=2**8, show_progress=False):
    # hardcode parameters for now:
    Ek = np.sqrt(1e-1)
    PrBu = 1

    dataset = pv.read(file_name)
    x = dataset.points[0, 0]
    y = dataset.points[:, 1]
    z = dataset.points[:, 2]
    t = dataset['t'][0]
    alpha = np.max(np.abs(z))
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    y_1d = np.linspace(y_min, y_max, n)
    dz = (z_max - z_min)/(n - 1)
    z_1d = z_max - np.array([2*dz, dz, 0])
    surface_b_flux = np.zeros(n)
    for i in tqdm(range(n), disable=(not show_progress)):
        y_i = y_1d[i]

        line = pv.PointSet(np.array([[x, y_i, z] for z in z_1d]))
        samples = line.sample(dataset)

        b = samples['b']
        kappa_v = samples['kappa_v']
        bz = 1/dz*(1/2*b[-3] - 2*b[-2] + 3/2*b[-1])
        surface_b_flux[i] = alpha*Ek**2/PrBu*kappa_v[-1]*bz
    surface_b_flux[0] = 0 # H = 0 here

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

def make_plots(subdir, overwrite=False, i_start=0, i_stop=np.inf, inc=1):
    if subdir[-1] != '/':
        subdir += '/'

    if not os.path.exists(f"images/{subdir}"):
        os.mkdir(f"images/{subdir}")

    for file in sorted(Path(f"/home/hpeter/Downloads/states/nu/channel2D/{subdir}data/").glob("*.vtu")):
        i = int(file.stem.split("_")[1])  # assuming file is of the form "/foo/bar/state_{i:016d}.vtu"
        if (i < i_start) or (i % inc != 0):
            print(f"Skipping {file}")
            continue
        if i > i_stop:
            return
        if os.path.exists(f"images/{subdir}psi{i:05d}.png"):
            if overwrite:
                print(f"WARNING: Overwriting images/{subdir}psi{i:05d}.png")
            else:
                print(f"Skipping {file}")
                continue
        plot_psib(file, i, n=2**8, rescale_z=True, subdir=subdir)

if __name__ == "__main__":
    # i_state = 44900
    # i_sim = 10
    # file_name = f"/home/hpeter/Downloads/states/nu/channel2D/sim{i_sim:03d}/data/state_{i_state:016d}.vtu"

    # plot_fieldb(file_name, 'lambda', label=r"$\lambda$", rescale_z=True, i=i_state)
    # plot_fieldb(file_name, 'v', label=r"$v$", rescale_z=True, i=i)
    # plot_fieldb(file_name, 'w', label=r"$w$", rescale_z=True, i=i)
    # plot_fieldb(file_name, 'nu', rescale_z=True)
    # plot_fieldb(file_name, 'kappa_v', rescale_z=True, vmax=1)
    # plot_uvwb(file_name, rescale_z=True)
    # plot_psib(file_name, n=2**9, rescale_z=True)
    # plot_surface_b_flux(file_name, n=2**10)
    # plot_psi_profile(file_name, -0.51)

    # make_plots("sim008", overwrite=False, inc=100)
    make_plots("test", overwrite=True, i_start=1200, inc=100, i_stop=4500)
    # make_plots("sim009", overwrite=False, inc=100)
    # make_plots("sim010", overwrite=False, inc=100)
    # make_plots("sim011", overwrite=False, inc=100)
    # make_plots("sim012", overwrite=False, inc=100)
