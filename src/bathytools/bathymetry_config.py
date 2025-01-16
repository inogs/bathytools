import hashlib
from dataclasses import dataclass
from os import PathLike

import yaml


class ConfigFieldMissingError(Exception):
    """
    An error that is raised when a mandatory field is missing in the YAML
    config file
    """

    pass


class InvalidBathymetrySourceError(Exception):
    """
    An error that is raised when an invalid bathymetry source is specified
    """

    pass


@dataclass
class DomainGeometry:
    """
    Represents the geographical description of a domain for grid construction.

    Attributes:
        minimum_latitude: The minimum latitude of the domain.
        maximum_latitude: The maximum latitude of the domain.
        minimum_longitude: The minimum longitude of the domain.
        maximum_longitude: The maximum longitude of the domain.
        resolution: The resolution of the grid in the domain.
        minimum_h_factor: The minimum h-factor, describing grid quality.
        maximum_depth: The maximum depth in the domain.
        minimum_depth: The minimum depth in the domain. Defaults to 0.
    """

    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float
    resolution: float
    minimum_h_factor: float
    maximum_depth: float
    minimum_depth: float = 0

    def stable_hash(self) -> bytes:
        """
        Generates a stable hash for the DomainGeometry object.

        This method generates a hash by serializing the float attributes as
        strings with 8 decimal places of precision. Two `DomainGeometry`
        objects will produce identical hashes if and only if the attributes
        have the same values, rounded to 8 decimal places.

        Returns:
            bytes: A SHA-256 hash representing the object.
        """
        attributes = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("_")
        ]
        values_str = "___".join(
            [f"{getattr(self, f):.8e}" for f in attributes]
        )

        hasher = hashlib.new("sha256", values_str.encode("utf-8"))
        return hasher.digest()


@dataclass
class BathymetrySource:
    """
    Represents the source of bathymetry data, including the raw data source
    and smoothing options.

    Attributes:
        kind: The type or identifier of the bathymetry source.
        smoother: Whether to apply smoothing to the data.

    """

    kind: str
    smoother: bool

    def source_stable_hash(self) -> bytes:
        """
        Computes a stable SHA-256 hash for the bathymetry source.

        This hash changes if a different download source is specified,
        but remains the same if smoothing options are adjusted. It ensures
        that subsequent script executions can determine whether to reuse
        previously downloaded data or perform a new download.

        Returns:
            bytes: A SHA-256 hash based on the source type.
        """
        hasher = hashlib.new("sha256", self.kind.lower().encode("utf-8"))
        return hasher.digest()


class BathymetryConfig:
    """
    Describes the process for creating bathymetry data for the model. Includes
    details on downloading raw data, processing it, and performing required
    operations.

    Attributes:
        name (str): The name of the bathymetry configuration.
        domain (DomainGeometry): The geographical description of the domain.
        bathymetry_source (BathymetrySource): The source of bathymetry data.
    """

    def __init__(
        self,
        name: str,
        domain: DomainGeometry,
        bathymetry_source: BathymetrySource,
    ):
        self.name = name
        self.domain = domain
        self.bathymetry_source = bathymetry_source

    @staticmethod
    def from_yaml(file_path: PathLike):
        """
        Parses a YAML file to create a BathymetryConfig object.

        Args:
            file_path: The path to the YAML config file.

        Returns:
            An initialized BathymetryConfig instance.

        Raises:
            ConfigFieldMissingError: If a required field is missing in the
            YAML file.
        """
        with open(file_path, "r") as f:
            yaml_content = yaml.safe_load(f)

        for mandatory_field in ("name", "domain", "bathymetry"):
            if mandatory_field not in yaml_content:
                raise ConfigFieldMissingError(
                    f'Field "{mandatory_field}" is missing in config file'
                )

        name = yaml_content["name"]
        domain = DomainGeometry(**yaml_content["domain"])
        bathymetry_source = BathymetrySource(**yaml_content["bathymetry"])

        return BathymetryConfig(name, domain, bathymetry_source)

    def source_stable_hash(self) -> bytes:
        """
        Computes a SHA-256 hash summarizing the bathymetry configuration.

        It is used to ensure that subsequent script executions can determine
        whether to reuse previously downloaded data or perform a new download.

        Returns:
            A SHA-256 hash that combines stable hashes of the name,
            domain, and bathymetry source.
        """
        description_strings = []
        for field in ("name", "domain", "bathymetry_source"):
            if hasattr(getattr(self, field), "source_stable_hash"):
                description_strings.append(
                    f"{field}={getattr(self, field).source_stable_hash().hex()}"
                )
            elif hasattr(getattr(self, field), "stable_hash"):
                description_strings.append(
                    f"{field}={getattr(self, field).stable_hash().hex()}"
                )
            else:
                description_strings.append(f"{field}={getattr(self, field)}")

        hash_str = "___".join(description_strings)
        hasher = hashlib.new("sha256", hash_str.encode("utf-8"))
        return hasher.digest()
