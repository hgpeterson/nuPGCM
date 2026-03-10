import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from tqdm import tqdm
from scipy.integrate import trapezoid, cumulative_trapezoid
from pathlib import Path
from utils import to_latex_sci

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")

def calculate_streamfunction(vtu_file, n=2**8):
    # read the VTU file
    dataset = pv.read(vtu_file)
    t = dataset['t'][0]
    
    if 'u' not in dataset.array_names:
        raise ValueError(f"Velocity field 'u' not found. Available: {dataset.array_names}")
    if 'b' not in dataset.array_names:
        raise ValueError(f"Buoyancy field 'b' not found. Available: {dataset.array_names}")

    # create 2D grid for (x, z) evaluation
    points = dataset.points
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    if y_min != y_max:
        raise ValueError(f"y_min ({y_min}) != y_max ({y_max})")
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    x_1d = np.linspace(x_min, x_max, n)
    z_1d = np.linspace(z_min, z_max, n)
    x_grid, z_grid = np.meshgrid(x_1d, z_1d)
    
    # sample u, b at each (x, z) point
    points = pv.PointSet(np.array([[x_grid[i, j], y_min, z_grid[i, j]] for i in range(n) for j in range(n)]))
    samples = points.sample(dataset)
    u_grid = samples['u'].reshape(n, n)
    b_grid = samples['b'].reshape(n, n)
    
    # calculate streamfunction as Ψ(x,z) = -∫_-H^z u(x, z') dz'
    psi_grid = np.zeros_like(u_grid)
    for i in range(n):
        psi_grid[:, i] = -cumulative_trapezoid(u_grid[:, i], z_1d, initial=0)
    
    return psi_grid, x_grid, z_grid, u_grid, b_grid, t


def plot_streamfunction(psi, x, z, b, t=None, i=None):

    fig, ax = plt.subplots(1)

    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    vmax = np.nanmax(np.abs(psi))
    cf1 = ax.pcolormesh(x, z, psi, cmap="RdBu_r", vmin=-vmax, vmax=vmax, rasterized=True, shading="nearest")
    levels = np.linspace(-0.9*vmax, 0.9*vmax, 8)
    ax.contour(x, z, psi, levels=levels, colors='k', linestyles="-", linewidths=0.5)
    plt.colorbar(cf1, label=r'Streamfunction $\Psi$')
    bmin = np.nanmin(b)
    bmax = np.nanmax(b)
    db = (bmax - bmin)/10
    levels = np.linspace(bmin + db, bmax - db, 20)
    ax.contour(x, z, b, levels=levels, colors="k", linewidths=0.5, linestyles="-", alpha=0.3)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$z$')
    ax.set_aspect('equal')
    if t is not None:
        ax.set_title(r'$t = $' + to_latex_sci(t))
    
    if i is not None:
        filename = f"psi{i:016d}.png"
    else:
        filename = "psi.png"
    plt.savefig(filename)
    print(filename)
    plt.close()

if __name__ == "__main__":
    i = 1000
    vtu_file = f"/home/hpeter/Downloads/states/circle/data/state_{i:016d}.vtu"

    psi, x, z, u, b, t = calculate_streamfunction(vtu_file, n=2**8) 
    plot_streamfunction(psi, x, z, b, t=t, i=i)