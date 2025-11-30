import numpy as np
import pyvista as pv
from tqdm import tqdm
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt
import os
from pathlib import Path

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")

def to_latex_sci(x, decimals=2):
    s = f"{x:.{decimals}e}"
    mantissa, exp = s.split("e")
    exp = int(exp)  # removes leading zeros and '+' sign
    return rf"${mantissa} \times 10^{{{exp}}}$"

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
    # print(dataset['t'][0])

    # load data
    x = dataset.points[0, 0]  # should all be the same
    y = dataset.points[:, 1]
    z = dataset.points[:, 2]
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

        # print(samples['v'].shape)
        # samples = dataset.sample_over_line([x, y_j, z_min], 
        #                                    [x, y_j, z_max], 
        #                                    resolution=n-1)  # 'resolution' is number of pieces to divide line into

        v_grid[:, j] = samples['v']
        b_grid[:, j] = samples['b']

        # integrate: -alpha*dz(psi) = v
        psi[:, j] = -1/alpha*cumulative_trapezoid(v_grid[:, j], z_1d, initial=0)

        # # theoretical
        # f = y_j
        # tau = -1e-1*(y_j +1)*(y_j + 0.5)/0.25**2
        # psi[:, j] = -tau/f

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
    ax.set_xticks([-1, -3/4, -1/2])
    ax.set_yticks([z_grid.min(), z_grid.min()/2, 0])
    ax.set_title(r'$\alpha = $' + f'{alpha:0.3f}')
    if i is None:
        img_file = f"images/{subdir}psi.png"
    else:
        img_file = f"images/{subdir}psi{i:05d}.png"
    plt.savefig(img_file)
    print(img_file)
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
    i = 31000
    file_name = f"/home/hpeter/Downloads/states/nu/channel2D/sim008/data/state_{i:016d}.vtu"
    plot_fieldb(file_name, 'v', label=r"$v$", rescale_z=True, i=i)
    plot_fieldb(file_name, 'w', label=r"$w$", rescale_z=True, i=i)
    # plot_fieldb(file_name, 'nu', rescale_z=True)
    # plot_fieldb(file_name, 'kappa_v', rescale_z=True, vmax=1)
    # plot_uvwb(file_name, rescale_z=True)
    # plot_psib(file_name, n=2**9, rescale_z=True)

    # make_plots("sim008", overwrite=False, inc=100)