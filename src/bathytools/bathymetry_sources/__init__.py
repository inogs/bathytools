from collections import deque
from itertools import product as cart_prod
from logging import getLogger
from os import PathLike

import numpy as np
import xarray as xr
from bitsea.commons.geodistances import compute_great_circle_distance
from bitsea.commons.utils import search_closest_sorted

from bathytools.bathymetry_config import BathyInterpolationMethod
from bathytools.bathymetry_config import BathymetryConfig
from bathytools.bathymetry_config import DomainGeometry
from bathytools.bathymetry_config import InvalidBathymetrySourceError
from bathytools.bathymetry_sources.emodnet import EMODnetBathymetryDownloader


LOGGER = getLogger(__name__)


def download_bathymetry_data(
    bathymetry_config: BathymetryConfig,
    cache_path: PathLike,
    volatile_cache: bool,
) -> xr.Dataset:
    """
    Downloads bathymetry data based on the provided configuration.

    This function determines the appropriate downloader to use based on the
    `bathymetry_source` type specified in the configuration. If the specified
    bathymetry source is invalid or not supported, an error is raised. The
    downloaded data is stored or cached at the specified `cache_path`.

    Args:
        bathymetry_config: Configuration object containing details about the
            bathymetry source and domain for the data to be downloaded.
        cache_path: Path where the downloaded bathymetry data will be cached
            or stored.
        volatile_cache: Flag to determine whether the cache should be treated
            as volatile or persistent.

    Return:
        Path to the downloaded bathymetry data.

    Raises:
        InvalidBathymetrySourceError: Raised if the bathymetry source
            specified in the configuration is unrecognized.
    """
    if bathymetry_config.bathymetry_source.kind.lower() == "emodnet":
        downloader = EMODnetBathymetryDownloader(
            bathymetry_config.domain, cache_path, volatile_cache
        )
        return downloader.download_data()
    else:
        raise InvalidBathymetrySourceError(
            f"Unknown bathymetry source: {bathymetry_config.bathymetry_source.kind}"
        )


def interpolate_raw_bathymetry_on_domain(
    raw_bathymetry: xr.Dataset,
    domain: DomainGeometry,
    algorithm: BathyInterpolationMethod,
):
    """
    Interpolates raw bathymetric data onto a specified domain using the
    provided interpolation method. The function supports different
    interpolation algorithms to process the raw bathymetry data appropriately
    and map it onto the target domain geometry.

    Args:
        raw_bathymetry: Dataset containing raw bathymetric data to be
            interpolated.
        domain: Geometry of the domain onto which the bathymetric data will be
            interpolated.
        algorithm: Method used for interpolation of the bathymetry data.
            Supported methods include linear interpolation and integration.

    Returns:
        The interpolated bathymetry data as a result of applying the chosen
        interpolation method on the raw bathymetric dataset.

    Raises:
        ValueError: If the specified interpolation algorithm is not recognized.
    """
    if algorithm == BathyInterpolationMethod.LINEAR:
        return interpolate_raw_bathymetry_on_domain_linearly(
            raw_bathymetry, domain
        )
    elif algorithm == BathyInterpolationMethod.INTEGRAL_AVERAGE:
        return integrate_raw_bathymetry_on_domain(raw_bathymetry, domain)
    else:
        raise ValueError(f"Unknown interpolation method: {algorithm}")


def interpolate_raw_bathymetry_on_domain_linearly(
    raw_bathymetry: xr.Dataset, domain: DomainGeometry
) -> xr.Dataset:
    """
    Interpolates the raw bathymetry data onto a given domain geometry.

    This function interpolates the input bathymetry dataset to match the
    coordinates of the specified domain geometry, ensuring consistent depth
    values. Positive elevation values in the dataset are capped at zero to
    represent valid underwater depths. It assumes compatibility between the
    raw bathymetry data and the provided domain geometry.

    Args:
        raw_bathymetry: The raw bathymetry dataset containing depth values
            over a certain region. It must include longitude and latitude
            dimensions to enable interpolation.
        domain: An object containing the domain geometry description,
            including longitude and latitude coordinates for the interpolation
            domain.

    Returns:
        xr.Dataset: A bathymetry dataset interpolated onto the given domain
            geometry. Positive bathymetry values are replaced with zero to
            ensure valid underwater depth values.
    """
    ds_dom = raw_bathymetry.interp(
        longitude=domain.longitude, latitude=domain.latitude, method="linear"
    )
    ds_dom = xr.where(ds_dom.elevation > 0.0, 0, ds_dom)

    # Remove all NaNs and put 0 instead
    ds_dom = xr.where(ds_dom.elevation.isnull(), 0, ds_dom)
    return ds_dom


def refine_array(
    new_points: np.ndarray,
    original_array: np.ndarray,
    tolerance: float = 1e-10,
):
    """
    Inserts values from `new_points` into `original_array` while maintaining
    ascending order.

    Given two ordered arrays, `original_array` and `new_points`, this function
    merges the two into a single sorted array. The values in `new_points` are
    expected to fall within the range of `original_array` (between its first
    and last values). Any duplicate values common to both arrays are included
    only once in the output array.

    Args:
        new_points: a 1D array of values to be inserted into `original_array`
        original_array: a 1D array of values to be refined
        tolerance: a tolerance value to determine if two values are equal
            between `new_points` and `original_array`

    Returns:
        A tuple with two elements. A new ordered array with the elements of the
        two inputs array, and a list of indices `v` such that v[i] is the index
        of the greater element of `original_array` that is smaller or equal
        than the i-th element of the output array.
    """
    for b, b_name in (
        (new_points, "new_points"),
        (original_array, "original_array"),
    ):
        if len(b.shape) != 1:
            raise ValueError(f"{b_name} must be a 1D array")
        if np.any(b[1:] <= b[:-1]):
            raise ValueError(f"{b_name} must be strictly increasing")

    if new_points[0] < original_array[0]:
        raise ValueError(
            "new_points[0] must be less than or equal to original_array[0]"
        )
    if new_points[-1] > original_array[-1]:
        raise ValueError(
            "new_points[-1] must be greater than or equal to "
            "original_array[-1]"
        )

    new_points = deque(new_points)
    original_array = deque(original_array)

    previous_indices_map = [0]
    new_values = [original_array.popleft()]
    while len(original_array) > 0:
        if len(new_points) > 0:
            if abs(new_points[0] - original_array[0]) < tolerance:
                previous_indices_map.append(previous_indices_map[-1] + 1)
                new_values.append(original_array.popleft())
                new_points.popleft()
                continue
            elif new_points[0] <= original_array[0]:
                previous_indices_map.append(previous_indices_map[-1])
                new_values.append(new_points.popleft())
                continue

        previous_indices_map.append(previous_indices_map[-1] + 1)
        new_values.append(original_array.popleft())

    new_values = np.array(new_values)
    assert np.all(new_values[1:] >= new_values[:-1])

    return new_values, previous_indices_map


def compute_slices(pivots, values):
    """
    Computes slices of a sorted array corresponding to the ranges defined by
    pivots.

    Given a sorted array `values` and a sorted array `pivots` (which must be
    a subset of `values`), this function generates a list of slices. Each slice
    extracts the section of `values` between successive `pivots`, such that
    `values[slices[i]]` contains elements between `pivots[i]` (inclusive) and
    `pivots[i+1]` (exclusive).

    Returns:
        list[slice]: A list of slice objects representing value ranges for
        each pivot interval.
    """
    indices = search_closest_sorted(values, pivots)

    slices = []
    for i in range(len(indices) - 1):
        slices.append(slice(int(indices[i]), int(indices[i + 1])))
    return slices


def _compute_distance(args):
    """
    Wrapper for `compute_great_circle_distance` to facilitate parallel
    distance computation.

    This wrapper simplifies the interface by requiring only a single tuple of
    arguments, making it compatible with tools like `Pool.map` for parallel
    processing. It calculates the great circle distance between two sets of
    geographical coordinates.
    """
    lon1, lat1, lon2, lat2 = args
    return compute_great_circle_distance(
        lat1=lat1, lon1=lon1, lat2=lat2, lon2=lon2
    )


def compute_grid_area(
    *, longitudes: np.ndarray, latitudes: np.ndarray
) -> np.ndarray:
    """
    Approximates the area of cells in a rectangular geographical grid.

    This function calculates the area of grid cells by multiplying the
    longitudinal and latitudinal distances between the midpoints of opposite
    cell sides. The result is an approximation of the cell area, suitable for
    use in geographic computations.

    Args:
        longitudes: 1D array containing the longitude coordinates of the grid;
            the `i`-th cell extends from `longitudes[i]` to `longitudes[i+1]`.
        latitudes: 1D array containing the latitude coordinates of the grid; it
            follows the same convention as `longitudes`.

    Returns:
        A 2D array containing the area of the cells of the grid.
    """
    n_x = len(longitudes) - 1
    n_y = len(latitudes) - 1

    # The length to cross the cell passing through the center and moving
    # along a fixed latitude
    x_sides = np.empty(shape=(n_x, n_y), dtype=np.float32)

    y_sides = np.empty(shape=(n_x, n_y), dtype=np.float32)
    LOGGER.debug("%s distances must be computed", n_x * n_y * 2)
    LOGGER.debug("Computing longitudinal distances between cells")
    # This has been written in this way to make this part of the algorithm
    # easy to parallelize with multiprocessing (if it will be needed in the
    # future). We first produce a list of arguments for the `compute_distance`
    # function. Then we compute the distances using a `map`, and finally we put
    # the values in the proper place of the array.
    arg_list = []
    for j in range(n_y):
        fixed_lat = (latitudes[j] + latitudes[j + 1]) / 2
        arg_list.append(
            (longitudes[:-1], fixed_lat, longitudes[1:], fixed_lat)
        )

    distances = map(_compute_distance, arg_list)
    for j, v in zip(range(n_y), distances):
        x_sides[:, j] = v

    LOGGER.debug("Computing latitudinal distances between cells")
    arg_list = []
    for i in range(n_x):
        fixed_lon = (longitudes[i] + longitudes[i + 1]) / 2
        arg_list.append((fixed_lon, latitudes[:-1], fixed_lon, latitudes[1:]))

    distances = map(_compute_distance, arg_list)
    for i, v in zip(range(n_x), distances):
        y_sides[i, :] = v

    return x_sides * y_sides


def integrate_raw_bathymetry_on_domain(
    raw_bathymetry: xr.Dataset, domain: DomainGeometry
) -> xr.Dataset:
    """
    Integrates raw bathymetry data over cells of a specified domain geometry.

    This function analyzes a raw bathymetry dataset and calculates depth values
    for a new domain by averaging portions of the raw dataset that correspond
    to each cell. A rectangular region around each known depth point is used
    for this computation. It ensures consistency, replaces positive elevation
    values with zero, and removes NaN values from the resulting dataset.

    This function also caps positive values to zero, removing the `NaN`s. This
    function assumes that the raw bathymetry data and domain geometry are
    compatible for interpolation.

    Args:
        raw_bathymetry: The raw bathymetry dataset containing depth values
            over a certain region. It must include longitude and latitude
            dimensions to enable interpolation.
        domain: An object containing the domain geometry description,
            including longitude and latitude coordinates for the interpolation
            domain.

    Returns:
        xr.Dataset: A bathymetry dataset interpolated onto the given domain
            geometry. Positive bathymetry values are replaced with zero to
            ensure valid underwater depth values.
    """
    # Compute the boundary of the cells for our current domain
    domain_lon_boundaries = np.linspace(
        domain.minimum_longitude, domain.maximum_longitude, domain.n_x + 1
    )
    domain_lat_boundaries = np.linspace(
        domain.minimum_latitude, domain.maximum_latitude, domain.n_y + 1
    )

    # Prepare the output dataset
    new_bathymetry = xr.Dataset(
        coords={
            "latitude": ("latitude", domain.latitude),
            "longitude": ("longitude", domain.longitude),
        },
        data_vars={
            "elevation": (
                ("longitude", "latitude"),
                np.zeros((domain.n_x, domain.n_y)),
            )
        },
    )

    # Read the position of the points for which we know the bathymetry
    raw_lon_points = raw_bathymetry.longitude.values
    raw_lat_points = raw_bathymetry.latitude.values
    if np.any(raw_lon_points[1:] <= raw_lon_points[:-1]):
        raise ValueError(
            "Longitudes of the bathymetry must be strictly increasing"
        )
    if np.any(raw_lat_points[1:] <= raw_lat_points[:-1]):
        raise ValueError(
            "Latitudes of the bathymetry must be strictly increasing"
        )

    # Now we compute a region around each point. We work independently on both
    # the axes
    bathy_lon_boundaries = np.empty(
        (raw_lon_points.size + 1), dtype=raw_lon_points.dtype
    )
    bathy_lat_boundaries = np.empty(
        (raw_lat_points.size + 1), dtype=raw_lat_points.dtype
    )
    bathy_lon_boundaries[0] = raw_lon_points[0]
    bathy_lon_boundaries[1:-1] = (raw_lon_points[1:] + raw_lon_points[:-1]) / 2
    bathy_lon_boundaries[-1] = raw_lon_points[-1]
    bathy_lat_boundaries[0] = raw_lat_points[0]
    bathy_lat_boundaries[1:-1] = (raw_lat_points[1:] + raw_lat_points[:-1]) / 2
    bathy_lat_boundaries[-1] = raw_lat_points[-1]

    # Now we cut the raw bathymetry over the domain that we need; first we
    # compute the first and the last useful index for each axis
    left_cut = (
        np.searchsorted(
            bathy_lon_boundaries, domain_lon_boundaries[0], side="left"
        )
        - 1
    )
    right_cut = (
        np.searchsorted(
            bathy_lon_boundaries, domain_lon_boundaries[-1], side="right"
        )
        + 1
    )
    bottom_cut = (
        np.searchsorted(
            bathy_lat_boundaries, domain_lat_boundaries[0], side="left"
        )
        - 1
    )
    top_cut = (
        np.searchsorted(
            bathy_lat_boundaries, domain_lat_boundaries[-1], side="right"
        )
        + 1
    )

    # Now we want to "refine" the grid of the old bathymetry so that it has
    # also some points that are perfectly aligned with our domain. In this way,
    # each cell of the new grid will be totally inside or outside each cell.
    grid_lon_boundaries, lon_indices = refine_array(
        domain_lon_boundaries, bathy_lon_boundaries[left_cut:right_cut]
    )
    grid_lat_boundaries, lat_indices = refine_array(
        domain_lat_boundaries, bathy_lat_boundaries[bottom_cut:top_cut]
    )
    # We do not need the last value, because the last point is only the end of
    # a cell, and not the beginning of the next.
    lon_indices = lon_indices[:-1]
    lat_indices = lat_indices[:-1]

    # If we need to check which value is associated to each new cell, we must
    # use the lon_indices and lat_indices arrays, that associate to each cell of
    # the refined grid the previous cell from where the new cell derives
    lon_indices += left_cut
    lat_indices += bottom_cut

    # Here we explicitly create the values for the new grid
    grid_values = np.empty(
        shape=(len(grid_lon_boundaries) - 1, len(grid_lat_boundaries) - 1),
        dtype=raw_bathymetry.elevation.dtype,
    )

    grid_values[:] = raw_bathymetry.elevation.transpose(
        "longitude", "latitude"
    ).values[lon_indices[:, np.newaxis], lat_indices[np.newaxis, :]]

    # Put nan values in the cells that are outside the water; this values
    # will become 0 later
    grid_values = np.where(grid_values >= 0, np.nan, grid_values)

    # These slices associate the domain grid to the new one, by giving the slice
    # of the values of the refined grid that are inside each cell.
    lat_slices = compute_slices(domain_lat_boundaries, grid_lat_boundaries)
    lon_slices = compute_slices(domain_lon_boundaries, grid_lon_boundaries)

    LOGGER.debug("Computing area of the cells of the grid")
    grid_area = compute_grid_area(
        longitudes=grid_lon_boundaries, latitudes=grid_lat_boundaries
    )

    # For every cell, we compute the integral now
    LOGGER.debug(
        "Computing the integral of the bathymetry on each cell of the grid"
    )
    for i, j in cart_prod(range(domain.n_x), range(domain.n_y)):
        # Get the area of the pieces that decompose the cell
        cell_decomposition = grid_area[lon_slices[i], lat_slices[j]]

        # Get the values on the current cell
        cell_values = grid_values[lon_slices[i], lat_slices[j]]

        # Total area of the cell
        cell_area = np.sum(cell_decomposition)

        # Area of the cell that is filled with water
        water_area = np.sum(cell_decomposition, where=~np.isnan(cell_values))

        # If more than 50% of the cell is on the land, this cell is
        # a land cell
        if water_area / cell_area < 0.5:
            new_bathymetry.elevation.values[i, j] = 0
            continue

        # Here we sum the cells that are filled with water, averaging on their
        # area
        average_value = (
            np.sum(
                grid_values[lon_slices[i], lat_slices[j]] * cell_decomposition,
                where=~np.isnan(cell_values),
            )
            / water_area
        )

        # Finally, we save the result in the bathymetry array
        new_bathymetry.elevation.values[i, j] = average_value

    LOGGER.debug("Bathymetry integration completed")
    return new_bathymetry
