import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from tqdm import tqdm
from scipy.integrate import trapezoid, cumulative_trapezoid
from pathlib import Path
from utils import to_latex_sci

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")

def calculate_zonal_mean(vtu_file, field, n_grid=2**6, n_x_samples=2**5):
    """
    field_grid, y_grid, z_grid, t = calculate_zonal_mean(vtu_file, field, n_grid=2**6, n_x_samples=2**5)
    """

    # read the VTU file
    dataset = pv.read(vtu_file)
    t = dataset['t'][0]
    
    if field not in dataset.array_names:
        raise ValueError(f"Velocity field '{field}' not found. Available: {dataset.array_names}")

    # create 2D grid for (y, z) evaluation
    points = dataset.points
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    x_grid = np.linspace(x_min, x_max, n_x_samples)
    y_1d = np.linspace(y_min, y_max, n_grid)
    z_1d = np.linspace(z_min, z_max, n_grid)
    y_grid, z_grid = np.meshgrid(y_1d, z_1d)
    
    # for each (y, z) point, compute zonal mean of field(x, y, z)
    field_grid = np.zeros_like(y_grid)
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            y_ij = y_grid[j, i]
            z_ij = z_grid[j, i]

            # sample along this horizontal line
            line = pv.PointSet(np.array([[x, y_ij, z_ij] for x in x_grid]))
            samples = line.sample(dataset)
            
            # check which points are inside the mesh (points outside will be NaNs)
            valid_mask = samples['vtkValidPointMask'] == 1
            
            if any(valid_mask):
                # width of integral
                x_valid = x_grid[valid_mask]
                W = trapezoid(np.ones(len(x_valid)), x_valid)
                
                # zonal mean
                field_valid = samples[field][valid_mask]
                field_grid[j, i] = trapezoid(field_valid, x_valid)/W
    
    return field_grid, y_grid, z_grid, t

def calculate_barotropic_streamfunction(vtu_file, n_grid=2**6, n_z_samples=2**5):
    # read the VTU file
    dataset = pv.read(vtu_file)
    
    if 'u' not in dataset.array_names:
        raise ValueError(f"Velocity field 'u' not found. Available: {dataset.array_names}")

    # create 2D grid for (x, y) evaluation
    points = dataset.points
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    x_1d = np.linspace(x_min, x_max, n_grid)
    y_1d = np.linspace(y_min, y_max, n_grid)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d)
    z_grid = np.linspace(z_min, z_max, n_z_samples)
    
    # for each (x, y) point, integrate u(x, y, z) from z = -H to z = 0
    U_grid = np.zeros_like(x_grid)
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            x_ij = x_grid[j, i]
            y_ij = y_grid[j, i]

            # sample velocity along this vertical line
            line = pv.PointSet(np.array([[x_ij, y_ij, z] for z in z_grid]))
            samples = line.sample(dataset)
            
            # check which points are inside the mesh (points outside will be NaNs)
            valid_mask = samples['vtkValidPointMask'] == 1
            
            if any(valid_mask):
                # integrate U = ∫ u dz from -H to 0
                z_valid = z_grid[valid_mask]
                u_valid = samples['u'][valid_mask]
                U_grid[j, i] = trapezoid(u_valid, z_valid)
    
    # calculate streamfunction as Ψ(x,y) = ∫_y^L U(x, y') dy' from y = y to y = L
    # or equivalently, ∫_{-L}^L U(x, y') dy' - ∫_0^y U(x, y') dy' 
    psi_grid = np.zeros_like(U_grid)
    for i in range(n_grid):
        psi_grid[:, i] = trapezoid(U_grid[:, i], y_1d) - cumulative_trapezoid(U_grid[:, i], y_1d, initial=0)
    
    return psi_grid, x_grid, y_grid, U_grid

def calculate_overturning_streamfunction(vtu_file, nx=2**8, ny=2**8, nz=2**8):
    # read the VTU file
    dataset = pv.read(vtu_file)
    
    for field in ['v', 'b', 't']:
        if field not in dataset.array_names:
            raise ValueError(f"Velocity field '{field}' not found. Available: {dataset.array_names}")

    # time
    t = dataset['t'][0]

    # grid
    p = dataset.points
    x_min, x_max = p[:, 0].min(), p[:, 0].max()
    y_min, y_max = p[:, 1].min(), p[:, 1].max()
    z_min, z_max = p[:, 2].min(), p[:, 2].max()
    
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    xx, yy, zz = np.meshgrid(x, y ,z, indexing='ij')

    points = pv.PointSet(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
    samples = points.sample(dataset)
    v = samples['v'].reshape(nx, ny, nz)
    b = samples['b'].reshape(nx, ny, nz)
    valid_mask = samples['vtkValidPointMask'].reshape(nx, ny, nz)

    # zonal mean
    v_bar = trapezoid(v, x=x, axis=0)
    b_bar = trapezoid(b, x=x, axis=0)
    width = trapezoid(valid_mask, x=x, axis=0)
    width[width == 0] = np.nan
    v_bar /= width
    b_bar /= width
    
    # calculate streamfunction as Ψ(y,z) = -∫_-H^z v(y, z') dz'
    v_bar_filled = np.nan_to_num(v_bar, nan=0.0)  # replaces NaNs with 0
    psi_bar = -cumulative_trapezoid(v_bar_filled, z, axis=1, initial=0)
    nan_mask = np.isnan(v_bar)
    psi_bar[nan_mask] = np.nan
    
    return psi_bar, y, z, v_bar, b_bar, t


def plot_barotropic_streamfunction(psi_grid, x_grid, y_grid, U_grid=None):
    fig, axes = plt.subplots(1, 2 if U_grid is not None else 1, 
                             figsize=(14, 5))
    
    if U_grid is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes
    
    # Plot streamfunction
    levels = 20
    vmax = np.max(np.abs(psi_grid))
    cf1 = ax1.contourf(x_grid, y_grid, psi_grid, levels=levels, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.contour(x_grid, y_grid, psi_grid, levels=levels, colors='k', 
                linewidths=0.5, alpha=0.3)
    plt.colorbar(cf1, ax=ax1, label='Streamfunction Ψ')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Barotropic Streamfunction')
    ax1.set_aspect('equal')
    
    # Plot depth-averaged velocity if provided
    if U_grid is not None:
        vmax = np.max(np.abs(U_grid))
        cf2 = ax2.contourf(x_grid, y_grid, U_grid, levels=levels, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(cf2, ax=ax2, label='U (depth-averaged velocity)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Depth-Averaged Velocity U')
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def plot_zonal_mean(field, y, z, b=None, label="", cb_label="", rescale_z=True, t=None, i=None, cmap='RdBu_r', cb_sym=True):
    if rescale_z:
        alpha = -np.min(z)
        z = z/alpha/2

    fig, ax = plt.subplots(1)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    vmax = np.nanmax(np.abs(field))
    if cb_sym:
        vmin = -vmax
    else:
        vmin = np.nanmin(np.abs(field))
    cf = ax.pcolormesh(y, z, field, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(cf, label=cb_label, shrink=0.5)
    if b is not None:
        levels = np.linspace(-9.5, -0.5, 20)
        ax.contour(y, z, b, levels=levels, colors="k", linewidths=0.5, linestyles="-", alpha=0.3)
        levels = np.linspace(0.5, 9.5, 20)
        ax.contour(y, z, b, levels=levels, colors="k", linewidths=0.5, linestyles="--", alpha=0.3)
    ax.set_xlabel(r'$y$')
    if rescale_z:
        ax.set_ylabel(rf'$z$ (rescaled, $\alpha = {alpha:0.3f}$)')
    else:
        ax.set_ylabel(r'$z$')
    ax.set_aspect('equal')
    if t is not None:
        ax.set_title(r'$t = $' + to_latex_sci(t))
    if i is not None:
        filename = f"{label}{i:016d}.png"
    else:
        filename = f"{label}.png"
    plt.savefig(filename)
    print(filename)
    plt.close()

def plot_overturning_streamfunction(psi, y, z, v_bar=None, b_bar=None, 
                                    rescale_z=True, t=None, filename="psi.png"):

    fig, axes = plt.subplots(2, 1 if v_bar is not None else 1, 
                             figsize=(6.5, 4))
    for a in axes:
        a.spines['bottom'].set_visible(False)
        a.spines['left'].set_visible(False)
    
    if v_bar is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes

    if rescale_z:
        alpha = -np.min(z)
        z = z/alpha/2
    
    # Plot streamfunction
    vmax = np.nanmax(np.abs(psi))
    cf1 = ax1.pcolormesh(y, z, psi.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    levels = np.linspace(-0.9*vmax, 0.9*vmax, 8)
    ax1.contour(y, z, psi.T, levels=levels, colors='k', linestyles="-", linewidths=0.5)
    plt.colorbar(cf1, label=r'Streamfunction $\Psi$')
    if b_bar is not None:
        levels = np.linspace(-10, 10, 40)
        ax1.contour(y, z, -b_bar.T, levels=levels, colors="k", linewidths=0.5, alpha=0.3)
    ax1.set_xlabel(r'$y$')
    if rescale_z:
        ax1.set_ylabel(rf'$z$ (rescaled, $\alpha = {alpha:0.3f}$)')
    else:
        ax1.set_ylabel(r'$z$')
    ax1.set_aspect('equal')
    if t is not None:
        ax1.set_title(r'$t = $' + to_latex_sci(t))
    
    # Plot depth-averaged velocity if provided
    if v_bar is not None:
        vmax = np.nanmax(np.abs(v_bar))
        cf2 = ax2.pcolormesh(y, z, v_bar.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(cf2, label='Zonal mean\n'+r'meridional flow $\bar{v}$')
        if b_bar is not None:
            levels = np.linspace(-10, 10, 40)
            ax2.contour(y, z, -b_bar.T, levels=levels, colors="k", linewidths=0.5, alpha=0.3)
        ax2.set_xlabel(r'$y$')
        if rescale_z:
            ax2.set_ylabel(rf'$z$ (rescaled, $\alpha = {alpha:0.3f}$)')
        else:
            ax2.set_ylabel(r'$z$')
        ax2.set_aspect('equal')

    plt.savefig(filename)
    print(filename)
    plt.close()

if __name__ == "__main__":
    for i in range(7200, 7201, 100):
        sim = 15
        vtu_file = f"/home/hpeter/Downloads/states/sim{sim:03d}/data/state_{i:016d}.vtu"

        # n = 2**8
        # b, y, z, t = calculate_zonal_mean(vtu_file, "b", n_grid=n, n_x_samples=n)
        # nu, _, _, _ = calculate_zonal_mean(vtu_file, "nu", n_grid=n, n_x_samples=n)
        # kappa_v, _, _, _ = calculate_zonal_mean(vtu_file, "kappa_v", n_grid=n, n_x_samples=n)
        # plot_zonal_mean(nu, y, z, b, label="nu", cb_label=r"$\nu$", t=t, i=i, cmap="viridis", cb_sym=False)
        # plot_zonal_mean(kappa_v, y, z, b, label="kappa_v", cb_label=r"$\kappa_v$", t=t, i=i, cmap="viridis", cb_sym=False)
        
        # psi, x_grid, y_grid, U = calculate_barotropic_streamfunction(vtu_file) 
        # plot_barotropic_streamfunction(psi, x_grid, y_grid, U)
        
        psi, y, z, v_bar, b_bar, t = calculate_overturning_streamfunction(vtu_file) 
        np.savez(f"images/sim{sim:03d}/psi{i:016d}.npz", psi=psi, y=y, z=z, v_bar=v_bar, b_bar=b_bar, t=t)
        # d = np.load(f"psi{i:016d}.npz")
        # psi = d["psi"]
        # y = d["y"]
        # z = d["z"]
        # v_bar = d["v_bar"]
        # b_bar = d["b_bar"]
        # t = d["t"]
        plot_overturning_streamfunction(psi, y, z, v_bar, b_bar, t=t, filename=f"images/sim{sim:03d}/psi{i:016d}.png")

        # grid = pv.StructuredGrid(x_grid, y_grid, np.zeros_like(x_grid))
        # grid['streamfunction'] = psi.flatten(order='F')
        # grid['depth_avg_velocity'] = U.flatten(order='F')
        # grid.save('barotropic_streamfunction.vts')
        # print("Saved streamfunction to barotropic_streamfunction.vts")