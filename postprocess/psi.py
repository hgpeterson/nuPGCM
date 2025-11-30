import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from tqdm import tqdm
from scipy.integrate import trapezoid, cumulative_trapezoid

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
            samples = dataset.sample_over_line([x_ij, y_ij, z_min], 
                                               [x_ij, y_ij, z_max], 
                                               resolution=n_z_samples-1)  # 'resolution' is number of pieces to divide line into
            u_samples = samples['u']
            
            # check which points are inside the mesh (points outside will be NaNs)
            valid_mask = ~np.isnan(u_samples)  # FIXME: there are never any NaNs...
            
            if valid_mask.sum() > 1:
                # integrate U = ∫ u dz from -H to 0
                z_valid = z_grid[valid_mask]
                u_valid = u_samples[valid_mask]
                U_grid[j, i] = trapezoid(u_valid, z_valid)
    
    # calculate streamfunction as Ψ(x,y) = ∫_y^L U(x, y') dy' from y = y to y = L
    # or equivalently, ∫_{-L}^L U(x, y') dy' - ∫_0^y U(x, y') dy' 
    psi_grid = np.zeros_like(U_grid)
    for i in range(n_grid):
        psi_grid[:, i] = trapezoid(U_grid[:, i], y_1d) - cumulative_trapezoid(U_grid[:, i], y_1d, initial=0)
    
    return psi_grid, x_grid, y_grid, U_grid

def calculate_overturning_streamfunction(vtu_file, n_grid=2**6, n_x_samples=2**5):
    # read the VTU file
    dataset = pv.read(vtu_file)
    
    if 'v' not in dataset.array_names:
        raise ValueError(f"Velocity field 'v' not found. Available: {dataset.array_names}")
    if 'b' not in dataset.array_names:
        raise ValueError(f"Buoyancy field 'b' not found. Available: {dataset.array_names}")

    # create 2D grid for (y, z) evaluation
    points = dataset.points
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    x_1d = np.linspace(x_min, x_max, n_x_samples)
    y_1d = np.linspace(y_min, y_max, n_grid)
    z_1d = np.linspace(z_min, z_max, n_grid)
    y_grid, z_grid = np.meshgrid(y_1d, z_1d)
    
    # for each (y, z) point, compute zonal mean of v(x, y, z) and b(x, y, z)
    v_bar_grid = np.zeros_like(y_grid)
    b_bar_grid = np.zeros_like(y_grid)
    for i in tqdm(range(n_grid)):
        for j in range(n_grid):
            y_ij = y_grid[j, i]
            z_ij = z_grid[j, i]

            # sample velocity along this horizontal line
            samples = dataset.sample_over_line([x_min, y_ij, z_ij], 
                                               [x_max, y_ij, z_ij], 
                                               resolution=n_x_samples-1)  # 'resolution' is number of pieces to divide line into
            v_samples = samples['v']
            b_samples = samples['b']
            
            # check which points are inside the mesh (points outside will be NaNs)
            valid_mask = ~np.isnan(v_samples)  # FIXME: there are never any NaNs...
            
            if valid_mask.sum() > 1:
                # width of integral
                x_valid = x_1d[valid_mask]
                W = trapezoid(np.ones(len(valid_mask)), x_valid)
                
                # zonal mean of v
                v_valid = v_samples[valid_mask]
                v_bar_grid[j, i] = trapezoid(v_valid, x_valid)/W

                # zonal mean of b
                b_valid = b_samples[valid_mask]
                b_bar_grid[j, i] = trapezoid(b_valid, x_valid)/W
    
    # calculate streamfunction as Ψ(y,z) = -∫_-H^z v(y, z') dz'
    psi_grid = np.zeros_like(v_bar_grid)
    for i in range(n_grid):
        psi_grid[:, i] = -cumulative_trapezoid(v_bar_grid[:, i], z_1d, initial=0)
    
    return psi_grid, y_grid, z_grid, v_bar_grid, b_bar_grid


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

def plot_overturning_streamfunction(psi, y, z, v_bar=None, b_bar=None):
    fig, axes = plt.subplots(2, 1 if v_bar is not None else 1, 
                             figsize=(6.5, 4))
    
    if v_bar is not None:
        ax1, ax2 = axes
    else:
        ax1 = axes
    
    # Plot streamfunction
    vmax = np.max(np.abs(psi))
    cf1 = ax1.pcolormesh(y, z, psi, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax1.contour(y, z, psi, levels=10, colors='k', linestyles="-", linewidths=0.5)
    plt.colorbar(cf1, label=r'Streamfunction $\Psi$')
    if b_bar is not None:
        ax1.contour(y, z, b_bar, levels=20, colors="k", linewidths=0.5, linestyles="-", alpha=0.3)
    ax1.set_xlabel(r'$y$')
    ax1.set_ylabel(r'$z$')
    ax1.set_aspect('equal')
    
    # Plot depth-averaged velocity if provided
    if v_bar is not None:
        vmax = np.max(np.abs(v_bar))
        cf2 = ax2.pcolormesh(y, z, v_bar, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        plt.colorbar(cf2, label='Zonal mean\n'+r'meridional flow $\bar{v}$')
        if b_bar is not None:
            ax2.contour(y, z, b_bar, levels=20, colors="k", linewidths=0.5, linestyles="-", alpha=0.3)
        ax2.set_xlabel(r'$y$')
        ax2.set_ylabel(r'$z$')
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("psi.png", dpi=200)

# Example usage
if __name__ == "__main__":
    # vtu_file = "/home/hpeter/Downloads/states/convection/rounded/tau1e-4/state_0000000000007200.vtu"
    # vtu_file = "/home/hpeter/Downloads/states/convection/rounded/tau1e-1/state_0000000000007200.vtu"
    # vtu_file = "/home/hpeter/Downloads/states/nu/bz_in_A_dt_small_3D/state_0000000000002400.vtu"
    vtu_file = "/home/hpeter/Downloads/states/nu/quad_kappa_bzmin5_dt_half/state_0000000000005960.vtu"
    
    # psi, x_grid, y_grid, U = calculate_barotropic_streamfunction(vtu_file) 
    # plot_barotropic_streamfunction(psi, x_grid, y_grid, U)
    
    psi, y, z, v_bar, b_bar = calculate_overturning_streamfunction(vtu_file, n_grid=2**7, n_x_samples=2**7) 
    np.savez("psi.npz", psi=psi, y=y, z=z, v_bar=v_bar, b_bar=b_bar)
    # d = np.load("psi.npz")
    # psi = d["psi"]
    # y = d["y"]
    # z = d["z"]
    # v_bar = d["v_bar"]
    # b_bar = d["b_bar"]
    plot_overturning_streamfunction(psi, y, z, v_bar, b_bar)

    # grid = pv.StructuredGrid(x_grid, y_grid, np.zeros_like(x_grid))
    # grid['streamfunction'] = psi.flatten(order='F')
    # grid['depth_avg_velocity'] = U.flatten(order='F')
    # grid.save('barotropic_streamfunction.vts')
    # print("Saved streamfunction to barotropic_streamfunction.vts")