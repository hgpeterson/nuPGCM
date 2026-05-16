import numpy as np
import pyvista as pv
from scipy.integrate import trapezoid
from time import time


def to_latex_sci(x: float, decimals=2):
    """
    Convert number to a latex string in scientific notation

    Parameters
    ----------
    x : float
        Input number.
    decimals: int, optional
        Number of digits of precision in the mantissa. Default: 2.

    Returns
    -------
    s : string
        Latex string of the form '$mantissa \times 10^{exp}$'

    Examples
    --------
    >>> to_latex_sci(0.098765, decimals=3)
    '$9.88 \\times 10^{-2}$'
    """
    s = f"{x:.{decimals}e}"
    mantissa, exp = s.split("e")
    exp = int(exp)  # removes leading zeros and '+' sign
    return rf"${mantissa} \times 10^{{{exp}}}$"


class Grid:
    def __init__(self, dataset: pv.DataSet, nx: int, ny: int, nz: int):
        p = dataset.points
        x_min, x_max = p[:, 0].min(), p[:, 0].max()
        y_min, y_max = p[:, 1].min(), p[:, 1].max()
        z_min, z_max = p[:, 2].min(), p[:, 2].max()
        self.x = np.linspace(x_min, x_max, nx)
        self.y = np.linspace(y_min, y_max, ny)
        self.z = np.linspace(z_min, z_max, nz)
        # self.xx, self.yy, self.zz = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        self.nx = nx
        self.ny = ny
        self.nz = nz


def sample_to_grid(dataset: pv.DataSet, grid: Grid):
    """
    Sample pv.DataSet to an evenly-spaced 3D grid

    Parameters
    ----------
    dataset : pyvista.DataSet
        Input VTU data.
    nx : int
        Number of grid points in the x direction. Default 2**8.
    ny : int
        Number of grid points in the y direction. Default 2**8.
    nz : int
        Number of grid points in the z direction. Default 2**8.

    Returns
    -------
    samples : pyvista.PointSet
        Sampled dataset.
    x : numpy array
        Grid in x direction.
    y : numpy array
        Grid in y direction.
    z : numpy array
        Grid in z direction.
    """
    xx, yy, zz = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    points = pv.PointSet(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))
    samples = points.sample(dataset)
    return samples


def sample_fields(vtu_file, fields, nx=2**8, ny=2**8, nz=2**8, printtime=False):
    """Sample fields from a VTU file"""

    if printtime:
        t0 = time()

    # read the VTU file
    dataset = pv.read(vtu_file)

    # time
    t = dataset["t"][0]

    # evenly-spaced grid
    grid = Grid(dataset, nx, ny, nz)

    # sample
    samples = sample_to_grid(dataset, grid)
    fields_dict = {}
    for field in fields:
        fields_dict[field] = samples[field].reshape(nx, ny, nz)
    fields_dict["mask"] = samples["vtkValidPointMask"].reshape(nx, ny, nz)

    if printtime:
        print(f"sampled fields {fields} from {vtu_file} in {time() - t0:.3f} s")

    return fields_dict, grid, t


def horizontal_average(field, mask, grid, xmin=0, xmax=1, ymin=-1, ymax=1):
    """Calculate horizontal average of field in [xmin, xmax] x [ymin, ymax]"""

    # take subset
    ixmin = np.searchsorted(grid.x, xmin)
    ixmax = np.searchsorted(grid.x, xmax)
    iymin = np.searchsorted(grid.y, ymin)
    iymax = np.searchsorted(grid.y, ymax)
    field = field[ixmin : ixmax + 1, iymin : iymax + 1, :]
    mask = mask[ixmin : ixmax + 1, iymin : iymax + 1, :]

    # clean up before integrating
    field[np.where(mask == 0)] = 0

    # 2D horizontal average: (\int_xmin^xmax \int_ymin^ymax field(x, y, z) dx dy) / (\int_xmin^xmax \int_ymin^ymax 1 dx dy)
    area = trapezoid(
        trapezoid(mask, x=grid.x[ixmin : ixmax + 1], axis=0),
        x=grid.y[iymin : iymax + 1],
        axis=0,
    )
    return (
        trapezoid(
            trapezoid(field, x=grid.x[ixmin : ixmax + 1], axis=0),
            x=grid.y[iymin : iymax + 1],
            axis=0,
        )
        / area
    )


def horizontal_integral(field, mask, grid, xmin=0, xmax=1, ymin=-1, ymax=1, area_weighted=False):
    """Calculate horizontal integral of field in [xmin, xmax] x [ymin, ymax]"""

    # take subset
    ixmin = np.searchsorted(grid.x, xmin)
    ixmax = np.searchsorted(grid.x, xmax)
    iymin = np.searchsorted(grid.y, ymin)
    iymax = np.searchsorted(grid.y, ymax)
    field = field[ixmin : ixmax + 1, iymin : iymax + 1, :]
    mask = mask[ixmin : ixmax + 1, iymin : iymax + 1, :]

    # clean up before integrating
    field[np.where(mask == 0)] = 0

    # 2D horizontal integral: (\int_xmin^xmax \int_ymin^ymax field(x, y, z) dx dy)
    field_int = trapezoid(
        trapezoid(field, x=grid.x[ixmin : ixmax + 1], axis=0),
        x=grid.y[iymin : iymax + 1],
        axis=0,
    )
    if area_weighted:
        area = trapezoid(
            trapezoid(mask, x=grid.x[ixmin : ixmax + 1], axis=0),
            x=grid.y[iymin : iymax + 1],
            axis=0,
        )
        return field_int / area
    else:
        return field_int


def depth(samples: pv.PointSet, grid: Grid):
    mask = samples["vtkValidPointMask"].reshape(grid.nx, grid.ny, grid.nz)
    return trapezoid(mask, x=grid.z, axis=2)


def zonal_width(samples: pv.PointSet, grid: Grid):
    mask = samples["vtkValidPointMask"].reshape(grid.nx, grid.ny, grid.nz)
    return trapezoid(mask, x=grid.x, axis=0)


def zonal_mean(field, grid: Grid, width):
    field_bar = trapezoid(field, x=grid.x, axis=0)
    return np.divide(field_bar, width, where=width != 0, out=np.full_like(field_bar, np.nan))
