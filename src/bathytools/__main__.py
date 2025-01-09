import argparse
import logging
from sys import exit as sys_exit

from bitsea.utilities.argparse_types import existing_file_path

from bathytools.mesh_config import MeshConfig


if __name__ == "__main__":
    LOGGER = logging.getLogger()
else:
    LOGGER = logging.getLogger(__name__)


def configure_logger():
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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

    return parser.parse_args()


def main():
    args = argument()
    configure_logger()

    config_path = args.config

    LOGGER.debug("Reading config file from %s", config_path)
    mesh_config = MeshConfig.from_yaml(config_path)

    LOGGER.info('Generating mesh for domain "%s"', mesh_config.name)

    sys_exit()


if __name__ == "__main__":
    main()
