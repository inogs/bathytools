import hashlib
from dataclasses import dataclass
from dataclasses import is_dataclass
from os import PathLike

import numpy as np
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
class DomainLevels:
    """
    All the properties used to define the depth of the vertical levels

    Attributes:
        maximum_depth: The maximum depth in the domain.
        first_layer_thickness: The thickness of the first layer. Defaults to 1.
        minimum_depth: The minimum depth in the domain. Defaults to 0.
    """

    maximum_depth: float
    first_layer_thickness: float = 1.0
    minimum_depth: float = 0.0


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
        minimum_h_factor: The minimum h-factor; the minimum percentage of
            water of each cell on the bottom.
        vertical_levels: a DomainLevels object
    """

    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float
    resolution: float
    minimum_h_factor: float
    vertical_levels: DomainLevels

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

        # A function that checks if a name of an attribute represents some data
        # or a method of the target. Usually the target is the object itself
        # but, if some attributes of this object are instances of another
        # dataclass, the target can be used to specify this new object.
        def is_data_attribute(attr_name: str, target=self) -> bool:
            if attr_name.startswith("_"):
                return False
            if callable(getattr(target, attr_name)):
                return False
            # Avoid properties
            if isinstance(getattr(type(target), attr_name, None), property):
                return False

            return True

        attributes = [attr for attr in dir(self) if is_data_attribute(attr)]

        values_str_list = []
        for attr in attributes:
            # If an attribute points to a dataclass, add all the data
            # attributes of this data to the hash
            if is_dataclass(getattr(self, attr)):
                for sub_attr in dir(getattr(self, attr)):
                    if not is_data_attribute(sub_attr, getattr(self, attr)):
                        continue
                    sub_attr_value = getattr(getattr(self, attr), sub_attr)
                    if isinstance(sub_attr_value, float):
                        values_str_list.append(
                            f"{attr}__{sub_attr}={sub_attr_value:.8e}"
                        )
                    else:
                        values_str_list.append(
                            f"{attr}__{sub_attr}={sub_attr_value}"
                        )
            else:
                values_str_list.append(f"{attr}={getattr(self, attr):.8e}")

        values_str = "___".join(values_str_list)

        hasher = hashlib.new("sha256", values_str.encode("utf-8"))
        return hasher.digest()

    @property
    def n_x(self):
        """
        Calculates and returns the number of grid cells along the longitudinal
        direction based on the specified resolution and the range of
        longitudes. The resolution defines the size of each grid cell in
        degrees, while the longitudinal range is determined by the maximum and
        minimum longitude values.

        Returns:
            int: The number of grid cells along the longitudinal direction calculated by
            dividing the longitudinal range by the resolution.
        """
        return int(
            (self.maximum_longitude - self.minimum_longitude) / self.resolution
        )

    @property
    def n_y(self):
        """
        Calculates the number of latitude grid points (n_y) based on the resolution
        and given latitudinal boundaries. The value is computed by dividing the
        difference between the maximum and minimum latitude by the spatial
        resolution and rounding it to the nearest integer.

        Returns:
            int: The number of latitude grid points based on the input parameters.
        """
        return int(
            (self.maximum_latitude - self.minimum_latitude) / self.resolution
        )

    @property
    def longitude(self):
        """
        Gets the longitude values as a NumPy array based on provided minimum
        longitude, maximum longitude, resolution, and the number of points
        in the x-dimension (n_x). The resulting array is spaced linearly
        between the adjusted bounds, accounting for the specified resolution.

        Returns:
            numpy.ndarray: A NumPy array containing the linearly spaced
            longitude values.
        """
        return np.linspace(
            self.minimum_longitude + self.resolution * 0.5,
            self.maximum_longitude - self.resolution * 0.5,
            self.n_x,
        )

    @property
    def latitude(self):
        """
        Gets the latitude values as a NumPy array based on provided minimum
        latitude, maximum latitude, resolution, and the number of points
        in the y-dimension (n_y). The resulting array is spaced linearly
        between the adjusted bounds, accounting for the specified resolution.

        Returns:
            numpy.ndarray: A NumPy array containing the linearly spaced
            longitude values.
        """
        return np.linspace(
            self.minimum_latitude + self.resolution * 0.5,
            self.maximum_latitude - self.resolution * 0.5,
            self.n_y,
        )


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

        domain_args = yaml_content["domain"]
        if "vertical_levels" in domain_args:
            domain_args["vertical_levels"] = DomainLevels(
                **domain_args["vertical_levels"]
            )
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
