import argparse
import logging
import tempfile
from os import PathLike
from pathlib import Path
from sys import exit as sys_exit

import xarray as xr
from bitsea.utilities.argparse_types import existing_dir_path
from bitsea.utilities.argparse_types import existing_file_path

from bathytools.bathymetry_config import BathymetryConfig
from bathytools.bathymetry_sources import download_bathymetry_data


if __name__ == "__main__":
    LOGGER = logging.getLogger()
else:
    LOGGER = logging.getLogger(__name__)


def configure_logger():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    LOGGER.setLevel(logging.DEBUG)

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
        "--cache",
        "-k",
        type=existing_dir_path,
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


def generate_bathymetry(
    bathymetry_config: BathymetryConfig,
    cache_path: PathLike,
    volatile_cache: bool = True,
) -> int:
    cache_path = Path(cache_path)
    invalid_cache = _check_cache_validity(
        bathymetry_config, cache_path, volatile_cache
    )

    bathymetry_file_name = "bathymetry_raw.nc"
    bathymetry_file_path = cache_path / bathymetry_file_name

    if bathymetry_file_path.exists() and not invalid_cache:
        xr.load_dataset(bathymetry_file_path)
    else:
        download_bathymetry_data(bathymetry_config, cache_path, volatile_cache)

    return 0


def main():
    args = argument()
    configure_logger()

    config_path = args.config

    LOGGER.debug("Reading config file from %s", config_path)
    bathy_config = BathymetryConfig.from_yaml(config_path)

    LOGGER.info('Generating mesh for domain "%s"', bathy_config.name)
    if args.cache is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            LOGGER.debug("Using temporary directory %s", tmpdir)
            output_status = generate_bathymetry(
                bathy_config, Path(tmpdir), volatile_cache=True
            )
    else:
        LOGGER.debug("Using temporary directory %s", args.cache)
        output_status = generate_bathymetry(
            bathy_config, Path(args.cache), volatile_cache=False
        )

    sys_exit(output_status)


if __name__ == "__main__":
    main()
