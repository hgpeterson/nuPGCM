import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.integrate import trapezoid, cumulative_trapezoid

wd = Path(__file__).parent.resolve()
plt.style.use(f"{wd}/../plots.mplstyle")


class SlicePlotter:
    def __init__(self, file_name):
        if not Path(file_name).exists():
            raise FileNotFoundError(f"VTU file not found: {file_name}")
        self.file_name = file_name
        self.dataset = pv.read(file_name)
        self.alpha = -self.dataset.points[:, 2].min()  # aspect ratio

        print(f"SlicePlotter initialized for VTU file: {vtu_file}")

    def set_slice(self, direction, location):
        self.direction = direction.lower()
        self.location = location
        if self.direction == "x":
            self.normal = [1, 0, 0]
            self.origin = [location, 0, 0]
            self.xlabel = r"Meridional coordinate $y$"
            self.ylabel = r"Vertical coordinate $z$"
        elif self.direction == "y":
            self.normal = [0, 1, 0]
            self.origin = [0, location, 0]
            self.xlabel = r"Zonal coordinate $x$"
            self.ylabel = r"Vertical coordinate $z$"
        elif self.direction == "z":
            self.normal = [0, 0, 1]
            self.origin = [0, 0, location]
            self.xlabel = r"Zonal coordinate $x$"
            self.ylabel = r"Meridional coordinate $y$"
        else:
            ValueError("'direction' must be one of 'x', 'y', or 'z'")

    def plot(
        self,
        field_name,
        vmin=None,
        vmax=None,
        bmin=None,
        bmax=None,
        n_isopycnals=10,
        label_isopycnals=False,
        cmap="RdBu_r",
        isopycnal_color="k",
        title=None,
        output_file="image.png",
    ):
        # slice with plane and extract triangles from actual mesh connectivity
        ds_slice = self.dataset.slice(normal=self.normal, origin=self.origin).triangulate()
        faces = ds_slice.faces.reshape(-1, 4)  # [n_tris, [3, i, j, k]]
        triangles = faces[:, 1:]

        p = ds_slice.points
        if self.direction == "x":
            x1 = p[:, 1]
            x2 = p[:, 2]
            figsize = (33 / 6, 33 / 6 / 1.62 / 2)
        elif self.direction == "y":
            x1 = p[:, 0]
            x2 = p[:, 2]
            figsize = (19 / 6, 19 / 6 / 1.62)
        elif self.direction == "z":
            x1 = p[:, 0]
            x2 = p[:, 1]
            figsize = (19 / 6 / 1.62, 19 / 6)

        triang = mtri.Triangulation(x1, x2, triangles)

        if field_name == "u":
            field = ds_slice["u"][:, 0]
        elif field_name == "v":
            field = ds_slice["u"][:, 1]
        elif field_name == "w":
            field = ds_slice["u"][:, 2]
        else:
            field = ds_slice[field_name]

        if vmax is None:
            vmax = field.max() if vmin is not None else np.max(np.abs(field))
            extend = "neither"
        else:
            if vmin is None:
                vmin_eff = -vmax
            else:
                vmin_eff = vmin
            if vmin_eff > field.min() and vmax < field.max():
                extend = "both"
            elif vmax < field.max():
                extend = "max"
            elif vmin_eff > field.min():
                extend = "min"
            else:
                extend = "neither"
        if vmin is None:
            vmin = -vmax

        b = ds_slice["b"]
        if bmax is None:
            bmax = b.max()
        if bmin is None:
            bmin = b.min()

        # plot
        fig, ax = plt.subplots(1, figsize=figsize)
        im = ax.tripcolor(triang, field, vmin=vmin, vmax=vmax, cmap=cmap, shading="gouraud")
        plt.colorbar(im, ax=ax, shrink=0.5, ticks=[vmin, (vmin + vmax) / 2, vmax], extend=extend)
        if bmin != bmax:
            isopycnals = ax.tricontour(
                triang,
                b,
                levels=np.linspace(bmin, bmax, n_isopycnals),
                linestyles="-",
                colors=isopycnal_color,
                alpha=0.3,
                linewidths=0.5,
            )
            if label_isopycnals:
                ax.clabel(isopycnals, fontsize=4)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.direction == "x":
            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([x2.min(), 0])
        if self.direction == "y":
            ax.set_xticks([0, 1])
            ax.set_yticks([x2.min(), 0])
        if self.direction == "z":
            ax.axis("equal")
            ax.set_xticks([0, 1])
            ax.set_yticks([-1, 0, 1])
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        if title is None:
            _labels = {"kappa_v": r"\kappa_v", "kappa_h": r"\kappa_h", "nu": r"\nu"}
            label = _labels.get(field_name, field_name)
            title = rf"${label}$ at ${self.direction} = {self.location:0.2f}$"
        ax.set_title(title)
        plt.savefig(output_file)
        print(output_file)
        plt.close()


if __name__ == "__main__":
    sims_dir = Path("/resnick/scratch/hppeters")
    sims = [
        # "052",
        # "053",
        # "054",
        # "055",
        # "056",
        # "057",
        # "058",
        # "059",
        # "060",
        # "061",
        # "062",
        # "063",
        # "064",
        # "065a",
        # "065b",
        # "065c",
        # "065d",
        # "066a",
        # "066b",
        # "066c",
        # "066d",
        # "067",
        "068",
        "068a",
        "069",
        "069a",
        "070",
        "070a",
        "071",
        "071a",
    ]
    xvals = [0.25, 0.5, 0.75]
    yvals = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    zvals = [-0.75, -0.5, -0.25]  # scaled by 1/alpha

    for sim in sims:
        dir = sims_dir / f"sim{sim}"
        print(f"Processing {dir}")

        # latest snapshot
        vtu_file = sorted((dir / "data").glob("state_*.vtu"))[-1]
        print(f"Latest VTU file: {vtu_file}")
        slices_state = (
            dir / "images/slices_state.txt"
        )  # text file containing name of VTU file last used to makes slices

        if slices_state.exists():
            prev_vtu_file = slices_state.read_text()
            print(f"{slices_state} found. Contents: {prev_vtu_file}")
            if prev_vtu_file == str(vtu_file):
                # skip if no new states have been saved
                print(f"Skipping {dir}\n")
                continue
        else:
            print(f"No {slices_state} found.")

        # flow/isopycnal slices
        sp = SlicePlotter(vtu_file)
        for i, x in enumerate(xvals):
            sp.set_slice("x", x)
            sp.plot("u", label_isopycnals=True, output_file=dir / f"images/u_slice_x{i}.png")
            sp.plot("v", label_isopycnals=True, output_file=dir / f"images/v_slice_x{i}.png")
            sp.plot("w", label_isopycnals=True, output_file=dir / f"images/w_slice_x{i}.png")
            sp.plot(
                "kappa_v",
                vmin=0,
                vmax=100,
                cmap="magma",
                isopycnal_color="w",
                label_isopycnals=True,
                output_file=dir / f"images/kappa_v_slice_x{i}.png",
            )
        for i, y in enumerate(yvals):
            sp.set_slice("y", y)
            sp.plot("u", label_isopycnals=True, output_file=dir / f"images/u_slice_y{i}.png")
            sp.plot("v", label_isopycnals=True, output_file=dir / f"images/v_slice_y{i}.png")
            sp.plot("w", label_isopycnals=True, output_file=dir / f"images/w_slice_y{i}.png")
            sp.plot(
                "kappa_v",
                vmin=0,
                vmax=100,
                cmap="magma",
                isopycnal_color="w",
                label_isopycnals=True,
                output_file=dir / f"images/kappa_v_slice_y{i}.png",
            )
        for i, z in enumerate(zvals):
            sp.set_slice("z", z * sp.alpha)  # note the alpha scaling
            sp.plot("u", label_isopycnals=True, output_file=dir / f"images/u_slice_z{i}.png")
            sp.plot("v", label_isopycnals=True, output_file=dir / f"images/v_slice_z{i}.png")
            sp.plot("w", label_isopycnals=True, output_file=dir / f"images/w_slice_z{i}.png")
            sp.plot(
                "kappa_v",
                vmin=0,
                vmax=100,
                cmap="magma",
                isopycnal_color="w",
                label_isopycnals=True,
                output_file=dir / f"images/kappa_v_slice_z{i}.png",
            )

        # diapycnal flow slices
        e_vtu_file = dir / "data/e.vtu"
        if e_vtu_file.exists():
            sp = SlicePlotter(e_vtu_file)
            for i, x in enumerate(xvals):
                sp.set_slice("x", x)
                sp.plot(
                    "e",
                    label_isopycnals=True,
                    output_file=dir / f"images/e_slice_x{i}.png",
                )

            for i, y in enumerate(yvals):
                sp.set_slice("y", y)
                sp.plot(
                    "e",
                    label_isopycnals=True,
                    output_file=dir / f"images/e_slice_y{i}.png",
                )

            for i, z in enumerate(zvals):
                sp.set_slice("z", z * sp.alpha)  # note the alpha scaling
                sp.plot(
                    "e",
                    label_isopycnals=True,
                    output_file=dir / f"images/e_slice_z{i}.png",
                )

        # create or overwrite "slices_state.txt" file with last used VTU file name
        slices_state.write_text(str(vtu_file))
        print(f"Contents of {slices_state} set to {vtu_file}\n")

    print("Done.")
