import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import utils

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


def plot_isopycnal_depth(vtu_files, bs=np.arange(-14.5, -12.5, 0.5), y0=0, n=2**8, filename="isopycnal_depths.png"):
    # same profile for every file
    dataset0 = pv.read(vtu_files[0])
    x0 = (dataset0.points[:, 0].max() - dataset0.points[:, 0].min()) / 2
    print(f"x, y = {x0:.2f}, {y0:.2f}")
    zp = dataset0.points[:, 2]
    z = np.linspace(zp.min(), zp.max(), n)
    profile = pv.PointSet(np.array([[x0, y0, z] for z in z]))

    # loop over files
    depths = np.zeros((len(vtu_files), len(bs)))
    ts = np.zeros(len(vtu_files))
    for i, vtu_file in enumerate(tqdm(vtu_files)):
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
                if depths[i, j] == z[0]:
                    # interp doesn't work if b is non-monotonic
                    depths[i, j] = z[np.argmin(np.abs(b - b0))]

    # plot
    fig, ax = plt.subplots(1)
    dt_end = ts[-1] - ts[0]
    for j, b0 in enumerate(bs):
        dzdt = (depths[-1, j] - depths[-2, j]) / dt_end
        if np.isnan(dzdt):
            label = rf"$b = {b0:.1f}$"
        else:
            label = rf"$b = {b0:.1f}$ ($z_t = {utils.to_latex_sci(dzdt)}$)"
        ax.plot(ts, depths[:, j], label=label)
    ax.legend(loc=(1.05, 0.5), fontsize=6)
    ax.set_xlim(ts[0], ts[-1])
    ax.set_ylim(z[0], z[-1])
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"Vertical coordinate $z$")
    ax.set_title(rf"Depth of isopycnal at $y = {y0:.2f}$")
    plt.savefig(filename)
    print(filename)
    plt.close()

    return ts, depths


if __name__ == "__main__":
    overwrite = True
    # overwrite = False
    sims_dir = Path("/resnick/scratch/hppeters")
    sims = [
        "065a", 
        "065c", 
        "065d", 
        "066c", 
        "066d", 
        "067"
        ]
    for sim in sims:
        dir = sims_dir / f"sim{sim}"
        print(f"Processing files in {dir}")
        vtu_files = sorted((dir / "data").glob("state_*.vtu"))

        filename = dir / "images/isopycnal_depths.png"
        if filename.exists() and (not overwrite):
            print(f"Skipping {sim}")
            continue

        plot_isopycnal_depth(vtu_files, bs=[-5, -10, -13, -13.5, -14], filename=filename)

        print()
