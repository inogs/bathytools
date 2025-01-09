from dataclasses import dataclass
from os import PathLike

import yaml


class ConfigFieldMissingError(Exception):
    pass


@dataclass
class DomainGeometry:
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float
    resolution: float
    minimum_h_factor: float
    maximum_depth: float
    minimum_depth: float = 0


class MeshConfig:
    def __init__(self, name: str, domain: DomainGeometry):
        self.name = name
        self.domain = domain

    @staticmethod
    def from_yaml(file_path: PathLike):
        with open(file_path, "r") as f:
            yaml_content = yaml.safe_load(f)

        for mandatory_field in ("name", "domain"):
            if mandatory_field not in yaml_content:
                raise ConfigFieldMissingError(
                    f'Field "{mandatory_field}" is missing in config file'
                )

        name = yaml_content["name"]
        domain = DomainGeometry(**yaml_content["domain"])

        return MeshConfig(name, domain)
