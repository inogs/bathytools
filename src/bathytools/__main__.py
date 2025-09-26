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
from bathytools.bathymetry_sources import download_bathymetry_data
from bathytools.bathymetry_sources import interpolate_raw_bathymetry_on_domain
from bathytools.domain_discretization import DomainDiscretization
from bathytools.filters import Filter
from bathytools.output_appendix import OutputAppendix
from bathytools.utilities.logtools import LoggingNanMax
from bathytools.utilities.logtools import LoggingNanMin


if __name__ == "__main__" or __name__ == "bathytools.__main__":
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
        Bathytools is a suite of tools designed to create a suitable bathymetry
        file for use with the MITgcm model, enabling the discretization of a
        marine domain.
        """
    )
    parser.add_argument(
        "--config",
        "-c",
        type=existing_file_path,
        required=True,
        help="""
        The YAML config file that describes how the bathymetry must be
        generated;
        """,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=path_inside_an_existing_dir,
        required=True,
        help="""
        The path of the directory where the output files must be written;
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
        it will be deleted after the generation of the outputs is complete
        """,
    )

    parser.add_argument(
        "--compression-level",
        "-l",
        type=int,
        required=False,
        default=9,
        help="""
        The level of compression to be used for the netCDF output files; it
        must be an integer number between 0 and 9; if 0, no compression will
        be used". If it is not submitted, the default value of 9 will be used.
        """,
    )

    parser.add_argument(
        "--mer",
        "-m",
        action="store_true",
        help="""
        If this flag is set, this script will generate the meshmask in the
        format used by the MER project. This format is CF-1.4 compliant.
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


def apply_actions(
    bathymetry, actions, output_appendix: OutputAppendix
) -> xr.Dataset:
    action_classes = Action.get_subclasses()
    LOGGER.debug("Action classes: %s", sorted(action_classes.keys()))
    for action_config in actions:
        LOGGER.info('Applying action "%s"', action_config["name"])
        action = Action.build(
            action_config,
            output_appendix=output_appendix,
            subclasses=action_classes,
        )
        bathymetry = action(bathymetry)
    return bathymetry


def apply_filters(
    domain_discretization, filters, output_appendix: OutputAppendix
) -> DomainDiscretization:
    filter_classes = Filter.get_subclasses()
    LOGGER.debug("Filter classes: %s", sorted(filter_classes.keys()))
    for filter_config in filters:
        LOGGER.info('Applying filter "%s"', filter_config["name"])
        current_filter = Filter.build(
            filter_config,
            output_appendix=output_appendix,
            subclasses=filter_classes,
        )
        domain_discretization = current_filter(domain_discretization)
    return domain_discretization


def write_output_files(
    domain_discretization: DomainDiscretization,
    output_appendix: OutputAppendix,
    compression_level: int = 9,
    use_mer_format: bool = False,
):
    output_dir = output_appendix.output_dir

    # Prepare the compression value for the encoding of the netCDF file
    if compression_level == 0:
        compression = dict(zlib=False)
    else:
        compression = dict(zlib=True, complevel=compression_level)

    bathy_file = output_dir / "bathy.bin"
    LOGGER.info('Writing bathymetry to "%s"', bathy_file)
    # Swap signs to have negative values where there is water (this is the
    # convention used by MitGCM)
    bathy_content = -domain_discretization.bathymetry.astype(
        "float32", copy=False
    )
    bathy_content.tofile(bathy_file)
    LOGGER.info("Done!")

    water_fraction_file = output_dir / "hFacC.bin"
    LOGGER.info('Writing water cell fractions to "%s"', water_fraction_file)
    h_fac_content = domain_discretization.water_fractions.on_cells.astype(
        "float32", copy=False
    )
    h_fac_content.tofile(water_fraction_file)
    LOGGER.info("Done!")

    LOGGER.debug("Building meshmask arrays")
    mesh_mask = domain_discretization.build_mesh_mask(
        output_appendix=output_appendix, use_mer_format=use_mer_format
    )

    mesh_mask_file = output_dir / "meshmask.nc"
    LOGGER.info('Writing meshmask to "%s"', mesh_mask_file)
    mesh_mask.to_netcdf(
        mesh_mask_file, encoding={v: compression for v in mesh_mask.data_vars}
    )
    LOGGER.info("Done!")

    LOGGER.debug("Building MitGCM static files")
    mitgcm_statics = domain_discretization.build_mit_static_data()

    statics_file = output_dir / "MIT_static.nc"
    LOGGER.info('Writing static arrays to "%s"', statics_file)
    mitgcm_statics.to_netcdf(
        statics_file,
        encoding={v: compression for v in mitgcm_statics.data_vars},
    )
    LOGGER.info("Done!")

    if use_mer_format and output_appendix.get_n_additional_variables() > 0:
        output_file = output_dir / "additional_variables.nc"
        additional_variables = xr.Dataset(
            output_appendix.get_additional_variables()
        )
        LOGGER.info("Writing additional variables to %s", output_file)
        additional_variables.to_netcdf(
            output_file,
            encoding={v: compression for v in additional_variables.data_vars},
        )
        LOGGER.info("Done!")


def generate_bathymetry(
    bathymetry_config: BathymetryConfig,
    cache_path: PathLike,
    output_dir: PathLike,
    volatile_cache: bool = True,
    compression_level: int = 9,
    use_mer_format: bool = True,
) -> int:
    cache_path = Path(cache_path)
    invalid_cache = _check_cache_validity(
        bathymetry_config, cache_path, volatile_cache
    )

    bathymetry_file_name = "bathymetry_raw.nc"
    bathymetry_file_path = cache_path / bathymetry_file_name

    if bathymetry_file_path.exists() and not invalid_cache:
        LOGGER.info("Loading bathymetry data from cache")
        raw_bathymetry_data = xr.load_dataset(bathymetry_file_path)
    else:
        LOGGER.info(
            "Downloading bathymetry data from %s",
            bathymetry_config.bathymetry_source.kind,
        )
        raw_bathymetry_data = download_bathymetry_data(
            bathymetry_config, cache_path, volatile_cache
        )

    LOGGER.debug(
        "Raw bathymetry has values between %s and %s",
        LoggingNanMin(raw_bathymetry_data.elevation),
        LoggingNanMax(raw_bathymetry_data.elevation),
    )

    LOGGER.info(
        "Interpolating bathymetry on domain %s", bathymetry_config.name
    )
    domain_bathymetry = interpolate_raw_bathymetry_on_domain(
        raw_bathymetry_data,
        bathymetry_config.domain,
        bathymetry_config.bathymetry_source.interpolation_method,
    )

    LOGGER.debug(
        "Interpolated bathymetry has values between %s and %s",
        LoggingNanMin(domain_bathymetry.elevation),
        LoggingNanMax(domain_bathymetry.elevation),
    )

    output_appendix = OutputAppendix(output_dir=output_dir)
    domain_bathymetry = apply_actions(
        domain_bathymetry, bathymetry_config.actions, output_appendix
    )

    LOGGER.debug(
        "After having applied the actions, bathymetry has values between %s and %s",
        LoggingNanMin(domain_bathymetry.elevation),
        LoggingNanMax(domain_bathymetry.elevation),
    )

    domain_discretization = DomainDiscretization.build(
        bathymetry=domain_bathymetry, domain_geometry=bathymetry_config.domain
    )

    domain_discretization = apply_filters(
        domain_discretization, bathymetry_config.filters, output_appendix
    )

    write_output_files(
        domain_discretization=domain_discretization,
        output_appendix=output_appendix,
        compression_level=compression_level,
        use_mer_format=use_mer_format,
    )
    LOGGER.info("Execution completed successfully!")

    return 0


def main():
    args = argument()
    configure_logger()

    config_path = args.config

    compression_level = args.compression_level
    if compression_level < 0 or compression_level > 9:
        raise ValueError(
            "Compression level must be an integer between 0 and 9. Received "
            f"{compression_level}."
        )

    use_mer_format = args.mer

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
                bathy_config,
                Path(tmpdir),
                output_dir,
                volatile_cache=True,
                compression_level=compression_level,
                use_mer_format=use_mer_format,
            )
    else:
        LOGGER.debug("Using temporary directory %s", args.cache)
        args.cache.mkdir(exist_ok=True)
        output_status = generate_bathymetry(
            bathy_config,
            Path(args.cache),
            output_dir,
            volatile_cache=False,
            compression_level=compression_level,
            use_mer_format=use_mer_format,
        )

    sys_exit(output_status)


if __name__ == "__main__":
    main()
