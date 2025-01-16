from os import PathLike

import xarray as xr

from bathytools.bathymetry_config import BathymetryConfig
from bathytools.bathymetry_config import InvalidBathymetrySourceError
from bathytools.bathymetry_sources.emodnet import EMODnetBathymetryDownloader


def download_bathymetry_data(
    bathymetry_config: BathymetryConfig,
    cache_path: PathLike,
    volatile_cache: bool,
) -> xr.Dataset:
    """
    Downloads bathymetry data based on the specified configuration. This
    function utilizes a downloader based on the `bathymetry_source` type
    provided in the configuration. If the `bathymetry_source` is not
    recognized, an error will be raised. The data is cached at the specified
    `cache_path`.

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
