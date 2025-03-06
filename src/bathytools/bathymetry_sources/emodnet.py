from os import PathLike
from pathlib import Path

import xarray as xr

from bathytools.bathymetry_config import DomainGeometry


class EMODnetBathymetryDownloader:
    """
    A class to facilitate downloading bathymetry data from the EMODnet ERDDAP
    server.

    This class allows downloading and managing bathymetry data using a
    specified geographical domain. The data can be cached for efficient
    reuse in subsequent operations.

    Attributes:
        URL (str): The URL of the EMODnet bathymetry dataset.

    Args:
        domain: The geographical domain for which bathymetry data is required.
        cache: The directory path where cached files will be saved.
        volatile_cache: If True, only uses the cache when absolutely necessary.
            When False, saves all downloaded results to the cache for reuse in
            subsequent runs. Defaults to `True`.
        bathymetry_file_name: The name of the file where the cached bathymetry
            data will be stored. Defaults to "bathymetry_raw.nc".
    """

    URL: str = (
        "https://erddap.emodnet.eu/erddap/griddap/dtm_2020_v2_e0bf_e7e4_5b8f"
    )

    def __init__(
        self,
        domain: DomainGeometry,
        cache: PathLike,
        volatile_cache: bool = True,
        bathymetry_file_name: str = "bathymetry_raw.nc",
    ):
        self._domain = domain
        self._cache = Path(cache)
        self._volatile_cache = volatile_cache
        self._bathymetry_file_name = bathymetry_file_name

    def download_data(self) -> xr.Dataset:
        """
        Downloads bathymetry data for the specified geographical domain and
        returns it as an xarray dataset.
        """
        lon_slice = slice(
            self._domain.minimum_longitude - self._domain.resolution,
            self._domain.maximum_longitude + self._domain.resolution,
        )
        lat_slice = slice(
            self._domain.minimum_latitude - self._domain.resolution,
            self._domain.maximum_latitude + self._domain.resolution,
        )
        ds = xr.open_dataset(self.URL).sel(
            longitude=lon_slice, latitude=lat_slice
        )

        if not self._volatile_cache:
            ds.to_netcdf(self._cache / self._bathymetry_file_name)

        return ds
