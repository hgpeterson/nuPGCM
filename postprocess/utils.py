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
    return f"{mantissa} \\times 10^{{{exp}}}"


class Grid:
    def __init__(self, dataset: pv.DataSet, nx: int = 2**8, ny: int = 2**8, nz: int = 2**8):
        p = dataset.points
        x_min, x_max = p[:, 0].min(), p[:, 0].max()
        y_min, y_max = p[:, 1].min(), p[:, 1].max()
        z_min, z_max = p[:, 2].min(), p[:, 2].max()
        if x_min == x_max:
            self.x = np.array([x_min])
            self.nx = 1
        else:
            self.x = np.linspace(x_min, x_max, nx)
            self.nx = nx

        if y_min == y_max:
            self.y = np.array([y_min])
            self.ny = 1
        else:
            self.y = np.linspace(y_min, y_max, ny)
            self.ny = ny

        if z_min == z_max:
            self.z = np.array([z_min])
            self.nz = 1
        else:
            self.z = np.linspace(z_min, z_max, nz)
            self.nz = nz

        self.xx, self.yy, self.zz = np.meshgrid(self.x, self.y, self.z, indexing="ij")


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


def sample_field(samples, field, nx, ny, nz):
    if field == "u":
        return samples["u"][:, 0].reshape(nx, ny, nz)
    elif field == "v":
        return samples["u"][:, 1].reshape(nx, ny, nz)
    elif field == "w":
        return samples["u"][:, 2].reshape(nx, ny, nz)
    else:
        return samples[field].reshape(nx, ny, nz)


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
        fields_dict[field] = sample_field(samples, field, grid.nx, grid.ny, grid.nz)
    fields_dict["mask"] = sample_field(samples, "vtkValidPointMask", grid.nx, grid.ny, grid.nz)

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


def to_buoyancy_coords(field, b, nb, b_range=None):
    """
    Transform a field from Cartesian (x, y, z) to buoyancy (x, y, b) coordinates.

    For each (x, y) column, linearly interpolates field(z) onto a uniform buoyancy
    grid using the local buoyancy profile b(z) as the vertical coordinate. Handles
    non-monotonic b by sorting before interpolation.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz)
        Field to transform.
    b : ndarray, shape (nx, ny, nz)
        Buoyancy field on the Cartesian grid.
    nb : int
        Number of buoyancy levels in the output grid.
    b_range : tuple (b_min, b_max), optional
        Range of buoyancy values for the output grid. Defaults to the global
        min/max of b.

    Returns
    -------
    field_b : ndarray, shape (nx, ny, nb)
        Field on the buoyancy coordinate grid. NaN outside b_range or where
        the column has fewer than 2 valid points.
    b_coords : ndarray, shape (nb,)
        Uniform buoyancy coordinate values from b_min to b_max.
    """
    if b_range is None:
        b_min, b_max = np.nanmin(b), np.nanmax(b)
    else:
        b_min, b_max = b_range

    b_coords = np.linspace(b_min, b_max, nb)
    nx, ny, _ = field.shape
    field_b = np.full((nx, ny, nb), np.nan)

    for ix in range(nx):
        for iy in range(ny):
            b_col = b[ix, iy, :]
            f_col = field[ix, iy, :]
            valid = ~np.isnan(b_col) & ~np.isnan(f_col)
            if valid.sum() < 2:
                continue
            sort_idx = np.argsort(b_col[valid])
            field_b[ix, iy, :] = np.interp(
                b_coords,
                b_col[valid][sort_idx],
                f_col[valid][sort_idx],
                left=np.nan,
                right=np.nan,
            )

    return field_b, b_coords


def to_cartesian_coords(field_b, b_coords, b):
    """
    Transform a field from buoyancy (x, y, b) back to Cartesian (x, y, z) coordinates.

    For each (x, y) column, evaluates field_b at the local buoyancy values b(x, y, z),
    inverting the mapping done by to_buoyancy_coords.

    Parameters
    ----------
    field_b : ndarray, shape (nx, ny, nb)
        Field on the buoyancy coordinate grid.
    b_coords : ndarray, shape (nb,)
        Uniform buoyancy coordinate values (monotonically increasing).
    b : ndarray, shape (nx, ny, nz)
        Buoyancy field on the Cartesian grid, defining the b -> z mapping.

    Returns
    -------
    field : ndarray, shape (nx, ny, nz)
        Field on the Cartesian grid. NaN where b is outside the range of
        b_coords or where the column has no valid data.
    """
    nx, ny, nz = b.shape
    field = np.full((nx, ny, nz), np.nan)

    for ix in range(nx):
        for iy in range(ny):
            b_col = b[ix, iy, :]
            fb_col = field_b[ix, iy, :]
            valid_z = ~np.isnan(b_col)
            valid_b = ~np.isnan(fb_col)
            if valid_z.sum() < 1 or valid_b.sum() < 2:
                continue
            field[ix, iy, valid_z] = np.interp(
                b_col[valid_z],
                b_coords[valid_b],
                fb_col[valid_b],
                left=np.nan,
                right=np.nan,
            )

    return field
