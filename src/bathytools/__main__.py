import argparse
import logging
import tempfile
from os import PathLike
from pathlib import Path
from sys import exit as sys_exit

import xarray as xr
from bitsea.utilities.argparse_types import existing_file_path
from bitsea.utilities.argparse_types import path_inside_an_existing_dir

from bathytools.actions import Action
from bathytools.bathymetry_config import BathymetryConfig
from bathytools.bathymetry_config import DomainGeometry
from bathytools.bathymetry_sources import download_bathymetry_data
from bathytools.depth_levels import generate_level_heights
from bathytools.geoarrays import GeoArrays
from bathytools.utilities.logtools import LoggingNanMax
from bathytools.utilities.logtools import LoggingNanMin
from bathytools.water_fractions import WaterFractions


if __name__ == "__main__":
    LOGGER = logging.getLogger()
else:
    LOGGER = logging.getLogger(__name__)


def configure_logger():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Disable logging from numba
    logging.getLogger("numba").setLevel(logging.INFO)

    LOGGER.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)

    LOGGER.addHandler(handler)


def argument():
    parser = argparse.ArgumentParser(
        description="""
        This tool generates a mesh for a model starting from a set of
        instructions contained inside a YAML file
        """
    )
    parser.add_argument(
        "--config",
        "-c",
        type=existing_file_path,
        required=True,
        help="""
        The YAML config file that describes how the bathymetry must be
        generated
        """,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=path_inside_an_existing_dir,
        required=True,
        help="""
        The YAML config file that describes how the bathymetry must be
        generated
        """,
    )

    parser.add_argument(
        "--cache",
        "-k",
        type=path_inside_an_existing_dir,
        required=False,
        default=None,
        help="""
        A path to a directory that will be used to store temporary files; if
        this is not submitted, a generic temporary directory will be used and
        it will be deleted after the mesh generation is complete
        """,
    )

    return parser.parse_args()


def _check_cache_validity(
    bathymetry_config: BathymetryConfig, cache_path: Path, volatile_cache: bool
) -> bool:
    """
    Checks whether the cache is valid or if we need to download the data from
    the bathymetry source again.

    Args:
        bathymetry_config: The overall configuration of this script.
        cache_path: The Path of the directory where the cache is stored.
        volatile_cache: will this cache be used also for other runs or should
            it be deleted after use?

    Returns:
        `True` if the data must be downloaded again (and the cache is invalid),
        `False` otherwise.
    """
    invalid_cache = True
    if not volatile_cache:
        current_config_hash = bathymetry_config.source_stable_hash().hex()
        LOGGER.debug("Current configuration hash: %s", current_config_hash)

        config_hash_file = cache_path / "config_hash.txt"
        if config_hash_file.exists():
            LOGGER.debug(
                "Reading previous configuration hash from %s", config_hash_file
            )
            with open(config_hash_file, "r") as f:
                previous_config_hash = f.read().strip("\n")
                LOGGER.debug(
                    "Previous configuration hash: %s", previous_config_hash
                )
        else:
            LOGGER.debug("No previous configuration hash found")
            previous_config_hash = ""

        invalid_cache = previous_config_hash != current_config_hash
        LOGGER.debug(
            "Checking if the current cache is invalid: %s", invalid_cache
        )

        with open(config_hash_file, "w") as f:
            f.write(current_config_hash + "\n")

    return invalid_cache


def interpolate_raw_bathymetry_on_domain(
    raw_bathymetry: xr.Dataset, domain: DomainGeometry
) -> xr.Dataset:
    """
    Interpolates a raw bathymetry dataset onto a given domain geometry and
    ensures valid bathymetry values by capping positive values to zero. This
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
    ds_dom = raw_bathymetry.interp(
        longitude=domain.longitude, latitude=domain.latitude, method="linear"
    )
    ds_dom = xr.where(ds_dom.elevation > 0.0, 0, ds_dom)

    # Remove all NaNs and put 0 instead
    ds_dom = xr.where(ds_dom.elevation.isnull(), 0, ds_dom)
    return ds_dom


def apply_actions(bathymetry, actions):
    for action_config in actions:
        LOGGER.info('Applying action "%s"', action_config["name"])
        action = Action.build(action_config)
        bathymetry = action(bathymetry)
    return bathymetry


def write_output_files(bathymetry, domain_geometry, output_dir: PathLike):
    output_dir = Path(output_dir)

    # Compression level
    compression = dict(zlib=True, complevel=9)

    first_layer_height = domain_geometry.vertical_levels.first_layer_thickness
    max_depth = domain_geometry.vertical_levels.maximum_depth
    depth_levels = generate_level_heights(first_layer_height, max_depth)

    bathymetry_values = bathymetry.elevation.transpose(
        "latitude", "longitude"
    ).values

    water_fractions = WaterFractions.build(
        depth_levels=depth_levels,
        bathymetry_data=bathymetry_values,
        domain_geometry=domain_geometry,
    )
    geo_arrays = GeoArrays.build(
        domain_geometry=domain_geometry, depth_levels=depth_levels
    )

    bathy_file = output_dir / "bathy.bin"
    LOGGER.info('Writing bathymetry to "%s"', bathy_file)
    bathymetry_values.astype("float32", copy=False).tofile(bathy_file)
    LOGGER.info("Done!")

    water_fraction_file = output_dir / "hFacC.bin"
    LOGGER.info('Writing water cell fractions to "%s"', water_fraction_file)
    water_fractions.on_cells.astype("float32", copy=False).tofile(
        water_fraction_file
    )
    LOGGER.info("Done!")

    LOGGER.debug("Building meshmask arrays")
    mesh_mask = geo_arrays.build_mesh_mask(water_fractions)

    mesh_mask_file = output_dir / "meshmask.nc"
    LOGGER.info('Writing meshmask to "%s"', mesh_mask_file)
    mesh_mask.to_netcdf(
        mesh_mask_file, encoding={v: compression for v in mesh_mask.data_vars}
    )
    LOGGER.info("Done!")

    LOGGER.debug("Building MitGCM static files")
    mitgcm_statics = geo_arrays.build_mit_static_data(water_fractions)

    statics_file = output_dir / "MIT_static.nc"
    LOGGER.info('Writing static arrays to "%s"', statics_file)
    mitgcm_statics.to_netcdf(
        statics_file,
        encoding={v: compression for v in mitgcm_statics.data_vars},
    )
    LOGGER.info("Done!")


def generate_bathymetry(
    bathymetry_config: BathymetryConfig,
    cache_path: PathLike,
    output_dir: PathLike,
    volatile_cache: bool = True,
) -> int:
    cache_path = Path(cache_path)
    invalid_cache = _check_cache_validity(
        bathymetry_config, cache_path, volatile_cache
    )

    bathymetry_file_name = "bathymetry_raw.nc"
    bathymetry_file_path = cache_path / bathymetry_file_name

    if bathymetry_file_path.exists() and not invalid_cache:
        raw_bathymetry_data = xr.load_dataset(bathymetry_file_path)
    else:
        raw_bathymetry_data = download_bathymetry_data(
            bathymetry_config, cache_path, volatile_cache
        )

    LOGGER.debug(
        "Raw bathymetry has values between %s and %s",
        LoggingNanMin(raw_bathymetry_data.elevation),
        LoggingNanMax(raw_bathymetry_data.elevation),
    )

    domain_bathymetry = interpolate_raw_bathymetry_on_domain(
        raw_bathymetry_data, bathymetry_config.domain
    )

    LOGGER.debug(
        "Interpolated bathymetry has values between %s and %s",
        LoggingNanMin(domain_bathymetry.elevation),
        LoggingNanMax(domain_bathymetry.elevation),
    )

    domain_bathymetry = apply_actions(
        domain_bathymetry, bathymetry_config.actions
    )

    LOGGER.debug(
        "After having applied the actions, bathymetry has values between %s and %s",
        LoggingNanMin(domain_bathymetry.elevation),
        LoggingNanMax(domain_bathymetry.elevation),
    )

    write_output_files(
        domain_bathymetry,
        domain_geometry=bathymetry_config.domain,
        output_dir=output_dir,
    )

    return 0


def main():
    args = argument()
    configure_logger()

    config_path = args.config

    LOGGER.debug("Reading config file from %s", config_path)
    bathy_config = BathymetryConfig.from_yaml(config_path)

    output_dir = args.output_dir
    LOGGER.debug("Creating directory %s", output_dir)
    output_dir.mkdir(exist_ok=True)

    LOGGER.info('Generating mesh for domain "%s"', bathy_config.name)
    if args.cache is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            LOGGER.debug("Using temporary directory %s", tmpdir)
            output_status = generate_bathymetry(
                bathy_config, Path(tmpdir), output_dir, volatile_cache=True
            )
    else:
        LOGGER.debug("Using temporary directory %s", args.cache)
        args.cache.mkdir(exist_ok=True)
        output_status = generate_bathymetry(
            bathy_config, Path(args.cache), output_dir, volatile_cache=False
        )

    sys_exit(output_status)


if __name__ == "__main__":
    main()
