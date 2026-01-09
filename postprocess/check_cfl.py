import numpy as np
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")

def get_h(triangle, direction):
    if direction == 'x':
        i = 0
    elif direction == 'y':
        i = 1
    elif direction == 'z':
        i = 2
    else:
        ValueError("`direction` must be one of 'x', 'y', or 'z'")

    return np.max(triangle[i, :]) - np.min(triangle[i, :])

def plot_cfl(file_name, i=None, rescale_z=False):
    # read VTU file
    dataset = pv.read(file_name)

    # prep data
    coords = dataset.points
    v = dataset['v']
    w = dataset['w']
    y = coords[:, 1]
    z = coords[:, 2]
    alpha = np.max(np.abs(z))
    tri = dataset.cells_dict[5]

    tri_y = y[tri]
    tri_z = z[tri]
    hys = np.max(tri_y, axis=1) - np.min(tri_y, axis=1)
    hzs = np.max(tri_z, axis=1) - np.min(tri_z, axis=1)
    tri_v = np.max(np.abs(v[tri]), axis=1)
    tri_w = np.max(np.abs(w[tri]), axis=1)
    dt_y = hys/(tri_v + 1e-16)
    dt_z = hzs/(tri_w + 1e-16)
    print(f"min(hy) = {hys.min():.1e}")
    print(f"min(hz) = {hzs.min():.1e}")
    print(f"max(v)  = {tri_v.max():.1e}")
    print(f"max(w)  = {tri_w.max():.1e}")
    print(f"CFL dt (y): {dt_y.min():.1e}")
    print(f"CFL dt (z): {dt_z.min():.1e}")

    # for i, t in enumerate(tri):
    #     hy = get_h(coords[t, :], 'y')
    #     hz = get_h(coords[t, :], 'z')
    #     vmax = np.max(np.abs(v[t]))
    #     wmax = np.max(np.abs(w[t]))
    #     # print(f"Triangle {i}: hy = {hy:.1e}, hz = {hz:.1e}, vmax = {vmax:.1e}, wmax = {wmax:.1e}")
    #     print(f"CFL dt = {np.min([hy/vmax, hz/wmax])}")

    if rescale_z:
        z = z/(2*alpha)

    # data is 2D, create a tri-plot
    vmax = 1
    fig, ax = plt.subplots(1, 2, figsize=(5.5, 2.2), sharey=True)
    im = ax[0].tripcolor(y, z, dt_y, triangles=tri, shading='flat', cmap='inferno_r', norm=LogNorm(vmax=vmax))
    im = ax[1].tripcolor(y, z, dt_z, triangles=tri, shading='flat', cmap='inferno_r', norm=LogNorm(vmax=vmax))
    plt.colorbar(im, label=r"CFL $\Delta t$", fraction=0.03)
    for a in ax:
        a.triplot(y, z, tri, "k-", linewidth=0.25, alpha=0.1)
        a.set_xlabel(r'$y$')
        a.set_xticks([-1, -0.75, -0.5])
        a.spines['left'].set_visible(False)
        a.spines['bottom'].set_visible(False)
        a.axis('equal')
    if rescale_z:
        ax[0].set_ylabel(r'$z$ (rescaled)')
        ax[0].set_yticks([-0.5, -0.25, -0.0])
    else:
        ax[0].set_ylabel(r'$z$')
    ax[0].set_title(r'$y$-direction')
    ax[1].set_title(r'$z$-direction')
    # plt.subplots_adjust(wspace=0.1)
    if i is None:
        img_file = "images/cfl.png"
    else:
        img_file = f"images/cfl{i:05d}.png"
    plt.savefig(img_file)
    print(img_file)
    plt.close()

if __name__ == "__main__":
    i = 1700
    # file_name = f"/home/hpeter/Downloads/states/nu/channel2D/sim008/data/state_{i:016d}.vtu"
    # file_name = f"/home/hpeter/Downloads/states/nu/channel2D/sim009/data/state_{i:016d}.vtu"
    file_name = f"/home/hpeter/Downloads/states/nu/channel2D/test/data/state_{i:016d}.vtu"
    plot_cfl(file_name, rescale_z=True, i=i)