import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from numbers import Real
from operator import attrgetter
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Tuple

from bitsea.utilities.argparse_types import dir_to_be_created_if_not_exists
from bitsea.utilities.argparse_types import existing_dir_path
from bitsea.utilities.argparse_types import existing_file_path


@dataclass
class Domain:
    name: str
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float

    @staticmethod
    def from_json(domain_file: Path):
        data = json.loads(domain_file.read_text())
        return Domain(
            name=data["name"],
            minimum_latitude=data["minimum_latitude"],
            maximum_latitude=data["maximum_latitude"],
            minimum_longitude=data["minimum_longitude"],
            maximum_longitude=data["maximum_longitude"],
        )

    def contains(self, *, lat, lon):
        lat_inside = self.minimum_latitude <= lat <= self.maximum_latitude
        lon_inside = self.minimum_longitude <= lon <= self.maximum_longitude
        return lat_inside and lon_inside


@dataclass
class Variable:
    name: str
    unit: str
    BFM_conversion_factor: float | None = None

    def to_json(self) -> OrderedDict[str, str | float]:
        json_data: OrderedDict[str, str | float] = OrderedDict(
            [("unit", self.unit)]
        )
        if self.BFM_conversion_factor is not None:
            json_data["BFM_conversion_factor"] = self.BFM_conversion_factor
        return json_data

    def __lt__(self, other):
        return self.name < other.name

    def __le__(self, other):
        return self.name <= other.name


@dataclass
class RiverGeometry:
    mouth_longitude: float
    mouth_latitude: float
    side: Literal["N", "S", "E", "W"] | None = None
    width: int | None = None
    depth: int | None = None
    stem: List[str] | None = None
    stem_length: Real | None = None

    unique_side: bool = True
    unique_stem_desc: bool = True

    @staticmethod
    def from_dict(raw_dict):
        return RiverGeometry(**raw_dict)

    def to_json(self) -> OrderedDict[str, float | Literal["N", "S", "E", "W"]]:
        data: List[Tuple[str, float | Literal["N", "S", "E", "W"]]] = [
            ("mouth_longitude", self.mouth_longitude),
            ("mouth_latitude", self.mouth_latitude),
        ]
        if self.side is not None:
            data.append(("side", self.side))
        if self.width is not None:
            data.append(("width", self.width))
        if self.depth is not None:
            data.append(("depth", self.depth))
        if self.stem is not None and self.unique_side is True:
            data.append(("stem", self.stem))
        if self.stem_length is not None and self.unique_side is True:
            data.append(("stem_length", self.stem_length))
        return OrderedDict(data)

    def _update_side(self, new_side):
        if not self.unique_side:
            return

        if self.side is None:
            self.side = new_side
            return

        if self.side != new_side:
            self.unique_side = False
            self.side = None

    def _update_stem_geometry(self, stem=None, stem_length=None):
        if not self.unique_stem_desc:
            return

        if stem is None and stem_length is None:
            return

        if stem is not None and stem_length is not None:
            raise ValueError(
                "At least one between stem and stem_length must be not None"
            )

        if self.stem is None and self.stem_length is None:
            self.stem = stem
            self.stem_length = stem_length
            return

        if self.stem is not None or self.stem_length is not None:
            if self.stem != stem and self.stem_length != stem_length:
                self.unique_stem_desc = False
                self.stem = None
                self.stem_length = None

    def update(self, raw_dict):
        if "side" in raw_dict:
            self._update_side(raw_dict["side"])

        if "stem" in raw_dict and "stem_length" in raw_dict:
            raise ValueError("stem and steam_length are mutually exclusive")

        if "stem" in raw_dict or "stem_length" in raw_dict:
            self._update_stem_geometry(
                stem=raw_dict.get("stem", None),
                stem_length=raw_dict.get("stem_length", None),
            )


@dataclass
class RiverDataSource:
    type: str
    longitude: float
    latitude: float
    longitude_index: int
    latitude_index: int

    @staticmethod
    def from_dict(raw_dict):
        return RiverDataSource(**raw_dict)

    def to_json(self):
        data = [
            ("type", self.type),
            ("longitude", self.longitude),
            ("latitude", self.latitude),
            ("longitude_index", self.longitude_index),
            ("latitude_index", self.latitude_index),
        ]
        return OrderedDict(data)


@dataclass
class River:
    id: int
    name: str
    model: Literal["stem_flux", "rain_like"]
    geometry: RiverGeometry
    data_source: RiverDataSource
    biogeochemical_profile: str
    runoff_factor: float | None = None

    @staticmethod
    def from_dict(raw_river: Dict[str, Any]):
        river_id = int(raw_river["id"])
        name = raw_river["name"]
        model = raw_river["model"]
        runoff_factor = raw_river.get("runoff_factor", None)
        geometry = RiverGeometry.from_dict(raw_river["geometry"])
        data_source = RiverDataSource.from_dict(raw_river["data_source"])
        biogeochemical_profile = raw_river["biogeochemical"]["profile"]

        return River(
            id=river_id,
            name=name,
            model=model,
            runoff_factor=runoff_factor,
            geometry=geometry,
            data_source=data_source,
            biogeochemical_profile=biogeochemical_profile,
        )

    def to_json(self) -> OrderedDict[str, Any]:
        data: List[Tuple[str, Any]] = [
            ("id", self.id),
            ("name", self.name),
        ]

        if self.model != "stem_flux":
            data.append(("model", self.model))

        if self.runoff_factor is not None:
            data.append(("runoff_factor", self.runoff_factor))

        data.append(("geometry", self.geometry))
        data.append(("data_source", self.data_source))
        data.append(("biogeochemical_profile", self.biogeochemical_profile))
        return OrderedDict(data)

    def update(self, river_dict):
        if "geometry" in river_dict:
            self.geometry.update(river_dict["geometry"])
        for key in river_dict:
            if key not in ("id", "name", "geometry"):
                raise ValueError(f"Unexpected key: {key}")


@dataclass
class MainFile:
    variables: OrderedDict[str, Variable]
    profiles: Dict[str, OrderedDict[str, float]]
    geometry_defaults: Dict[str, int]
    rivers: List[River]
    physical_defaults: Dict[str, Real] = field(
        default_factory=lambda: OrderedDict(
            [
                ("average_temperature", 10),
                ("temperature_variation", 5),
                ("average_salinity", 5),
            ]
        )
    )

    def to_json(self) -> OrderedDict[str, Dict]:
        data = [
            ("variables", self.variables),
            ("bgc_profiles", self.profiles),
            ("geometry_defaults", self.geometry_defaults),
            ("physical_defaults", self.physical_defaults),
            ("default_model", "stem_flux"),
            ("rivers", self.rivers),
        ]
        return OrderedDict(data)

    def update_rivers(self, raw_dict):
        rivers = {river.id: river for river in self.rivers}
        for river in raw_dict:
            river_id = river["id"]
            rivers[river_id].update(river)


@dataclass
class GeometryUpdate:
    side: Literal["N", "S", "E", "W"] | None = None
    stem: List["float"] | None = None

    @staticmethod
    def from_dict(raw_dict):
        return GeometryUpdate(**raw_dict)

    def to_json(self):
        data = []
        if self.side is not None:
            data.append(("side", self.side))
        if self.stem is not None:
            data.append(("stem", self.stem))
        return OrderedDict(data)


@dataclass
class RiverUpdate:
    id: int
    name: str
    enabled: bool = True
    geometry: GeometryUpdate | None = None

    def to_json(self):
        data = [("id", self.id), ("name", self.name)]
        if not self.enabled:
            data.append(("enabled", False))
        if self.geometry is not None:
            data.append(("geometry", self.geometry))
        return OrderedDict(data)


@dataclass
class DomainSpecificFile:
    rivers: List[River]

    @staticmethod
    def from_dict(dict_content: Dict, main_file: MainFile, domain: Domain):
        enabled_rivers = dict_content["enabled_only"]

        rivers = {}

        for river in main_file.rivers:
            is_inside_domain = domain.contains(
                lon=river.geometry.mouth_longitude,
                lat=river.geometry.mouth_latitude,
            )
            if not is_inside_domain:
                continue

            rivers[river.id] = RiverUpdate(id=river.id, name=river.name)
            for enabled_river in enabled_rivers:
                if (
                    enabled_river["id"] == river.id
                    and enabled_river["name"] == river.name
                ):
                    break
            else:
                rivers[river.id].enabled = False

        for river in dict_content["rivers"]:
            for key in river:
                if key not in ("name", "id", "geometry"):
                    raise ValueError
            for ref_river in main_file.rivers:
                if ref_river.id == river["id"]:
                    break
            else:
                raise ValueError
            current_geometry = GeometryUpdate.from_dict(river["geometry"])

            update_stem = False
            if (
                current_geometry.stem is not None
                and current_geometry.stem != ref_river.geometry.stem
            ):
                update_stem = True
            if update_stem is False and "stem" in river:
                del river["stem"]

            update_side = False
            if (
                current_geometry.side is not None
                and current_geometry.side != ref_river.geometry.side
            ):
                update_side = True
            if update_side is False and "side" in river:
                del river["side"]

            if update_stem or update_side:
                rivers[river["id"]].geometry = current_geometry

        river_list = []
        for river in rivers.values():
            if river.enabled and river.geometry is None:
                continue
            river_list.append(river)

        river_list.sort(key=attrgetter("id"))

        return DomainSpecificFile(river_list)

    def to_json(self):
        return {"rivers": self.rivers}


class RiverJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, "to_json"):
            return o.to_json()
        return super().default(o)


def argument():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--mer-domains", "-d", type=existing_dir_path, required=True, help=""
    )

    parser.add_argument(
        "--main-river-file",
        "-m",
        type=existing_file_path,
        required=True,
    )

    parser.add_argument(
        "--domain-river-dir",
        "-i",
        type=existing_dir_path,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        type=dir_to_be_created_if_not_exists,
        required=True,
    )

    return parser.parse_args()


def read_domain_files(domains_dir: Path) -> dict[str, Domain]:
    domains = {}
    for domain in domains_dir.iterdir():
        if not domain.is_dir():
            continue
        domain_name = domain.name
        domain_file = domain / f"{domain_name}_domain.json"
        domain = Domain.from_json(domain_file)
        domains[domain_name] = domain
    return domains


def read_main_file(main_file_path) -> MainFile:
    data = json.loads(main_file_path.read_text())

    variables_unsorted = {}
    for var_name, var_content in data["variables"].items():
        current_var = Variable(name=var_name, **var_content)
        variables_unsorted[var_name] = current_var
    var_names = sorted(list(var_name for var_name in variables_unsorted))
    variables = OrderedDict([(v, variables_unsorted[v]) for v in var_names])

    profiles: Dict[str : Dict[str, float]] = {}
    for profile_name, raw_profile_content in data["defaults"][
        "profiles"
    ].items():
        profile_content_unsorted = {
            v_name: float(v_values)
            for v_name, v_values in raw_profile_content.items()
        }
        profile_vars = sorted(list(v for v in profile_content_unsorted))
        profile_content = OrderedDict(
            [(v, profile_content_unsorted[v]) for v in profile_vars]
        )
        profiles[profile_name] = profile_content

    geometry_defaults: Dict[str, int] = {}
    for parameter, raw_parameter_content in data["defaults"][
        "geometry"
    ].items():
        geometry_defaults[parameter] = int(raw_parameter_content)

    rivers = []
    for river_raw in data["rivers"]:
        river = River.from_dict(river_raw)
        rivers.append(river)
    rivers.sort(key=attrgetter("id"))

    main_file = MainFile(
        variables=variables,
        profiles=profiles,
        geometry_defaults=geometry_defaults,
        rivers=rivers,
    )
    return main_file


def main():
    args = argument()

    domains = read_domain_files(args.mer_domains)

    main_file = read_main_file(args.main_river_file)

    river_domain_files = {}
    for domain in domains:
        json_file = args.domain_river_dir / f"{domain}.json"
        domain_data = json.loads(json_file.read_text())
        river_domain_files[domain] = domain_data

    for river_domain_data in river_domain_files.values():
        main_file.update_rivers(river_domain_data["rivers"])

    for domain in domains:
        domain_specific_file = DomainSpecificFile.from_dict(
            river_domain_files[domain], main_file, domains[domain]
        )

        domain_json_content = json.dumps(
            domain_specific_file,
            cls=RiverJSONEncoder,
            indent=2,
        )

        output_file = args.output_dir / f"{domain}.json"
        output_file.write_text(domain_json_content + "\n")

    main_data = json.dumps(
        main_file,
        cls=RiverJSONEncoder,
        indent=2,
    )
    output_file = args.output_dir / "main.json"
    output_file.write_text(main_data + "\n")


if __name__ == "__main__":
    main()
